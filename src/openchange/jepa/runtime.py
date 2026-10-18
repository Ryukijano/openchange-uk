"""Single-node torchrun runtime, input preparation, and rank-aware checkpoints."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from openchange.jepa.checkpoints import _numpy_rng_state, _restore_numpy_rng
from openchange.jepa.distributed import DistContext, get_context
from openchange.jepa.views import ViewBatch, make_views, synthetic_observation
from openchange.provenance import sha256_file


def initialize(device_type: str) -> tuple[torch.device, DistContext]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size < 1 or world_size != local_world_size:
        raise ValueError("only a single node with a positive world size is supported")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; no CPU fallback")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError("LOCAL_RANK is outside visible CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        distributed.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(seconds=180),
        )
    return device, get_context()


def wrap_model(model: torch.nn.Module, device: torch.device, ctx: DistContext, norm: str) -> torch.nn.Module:
    if ctx.world_size > 1 and norm == "batch":
        if device.type != "cuda":
            raise ValueError(
                "multi-rank CPU checks require frozen or identity projector norm; SyncBN requires CUDA"
            )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if ctx.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            broadcast_buffers=True,
        )
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def broadcast(value: Any, ctx: DistContext) -> Any:
    objects = [value]
    if ctx.world_size > 1:
        distributed.broadcast_object_list(objects, src=0)
    return objects[0]


def reduce(value: torch.Tensor, ctx: DistContext, op: Any = distributed.ReduceOp.SUM) -> torch.Tensor:
    result = value.detach().clone()
    if ctx.world_size > 1:
        distributed.all_reduce(result, op=op)
    return result


def configure_precision_and_determinism(
    cfg: dict, device: torch.device | str | None = None
) -> None:
    del device
    runtime_cfg = cfg.get("runtime", {})
    if "cpu_threads" in runtime_cfg:
        torch.set_num_threads(runtime_cfg["cpu_threads"])
    if "deterministic" in runtime_cfg:
        torch.use_deterministic_algorithms(bool(runtime_cfg["deterministic"]))
    if cfg.get("precision", "fp32") == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


@contextmanager
def preserve_rng(device: torch.device):
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)


def optimizer_for(model: torch.nn.Module, cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            selected = (
                decay
                if parameter.ndim >= 2 and "cls_token" not in name and "modality_embed" not in name
                else no_decay
            )
            selected.append(parameter)
    spec = cfg["optimizer"]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": spec["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        fused=device.type == "cuda",
    )


class SyntheticDataset:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pool = cfg["synthetic"]["pool_per_stratum"]

    def eligible_indices(self, stratum: int) -> list[int]:
        if stratum in (0, 1):
            return list(range(stratum * self.pool, (stratum + 1) * self.pool))
        return list(range(len(self)))

    def __len__(self) -> int:
        return 2 * self.pool

    def __getitem__(self, key: tuple[int, int]):
        index, stratum = key
        base = synthetic_observation(index, int(index >= self.pool), self.cfg["seed"], self.cfg)[0]
        if stratum in (0, 1):
            return base
        return replace(
            base,
            stratum=stratum,
            s1=None if stratum == 3 else base.s1,
            s2=None if stratum == 2 else base.s2,
            s2_valid=None if stratum == 2 else base.s2_valid,
            synthetic_condition=True,
            intervention="withhold_s2" if stratum == 2 else "withhold_s1",
        )


class StepViewDataset(Dataset):
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __getitem__(self, key: tuple[int, int, int]):
        step, index, stratum = key
        return step, self.dataset[index, stratum]


@dataclass
class ViewCollator:
    cfg: dict
    allow_real: bool

    def __call__(self, samples: list) -> ViewBatch:
        steps, observations = zip(*samples)
        if len(set(steps)) != 1:
            raise ValueError("mixed optimizer steps in an input batch")
        return make_views(observations, self.cfg["seed"], steps[0], self.cfg, allow_real=self.allow_real)


def validate_batch(batch: ViewBatch, quotas: list[int], world_size: int, patch_size: int) -> None:
    counts = torch.bincount(batch.strata, minlength=len(quotas)).tolist()
    if counts != [quota // world_size for quota in quotas]:
        raise ValueError("input batch does not have exact per-rank stratum quotas")
    filled = torch.zeros((batch.n_views, batch.n_samples), dtype=torch.int64)
    for bucket in batch.buckets:
        if bucket.keep_idx.numel() and int(bucket.keep_idx.min()) < 0:
            raise ValueError("negative token index")
        patches = (bucket.resolution // patch_size) ** 2 * len(bucket.sensors)
        # The model still checks widths and crop shapes; validate token bounds on CPU.
        if bucket.keep_idx.numel() and int(bucket.keep_idx.max()) >= patches:
            raise ValueError("token index is out of range")
        filled.index_put_(
            (bucket.view_idx, bucket.sample_idx), torch.ones_like(bucket.view_idx), accumulate=True
        )
    if not torch.all(filled == 1):
        raise ValueError("each view/sample must be represented exactly once")


def save_distributed(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    local_tokens: int,
    digest: str,
    ctx: DistContext,
    device: torch.device,
) -> str | None:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": _numpy_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "local_tokens": local_tokens,
    }
    states: list[Any] = [None] * ctx.world_size
    if ctx.world_size > 1:
        distributed.all_gather_object(states, state)
    else:
        states[0] = state
    outcome = None
    if ctx.rank == 0:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": 2,
                "digest": digest,
                "world_size": ctx.world_size,
                "device_type": device.type,
                "torch_version": str(torch.__version__),
                "step": step,
                "model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rank_states": states,
                "tokens_seen": sum(s["local_tokens"] for s in states),
            }
            temporary = path.with_name(path.name + ".tmp")
            torch.save(record, temporary)
            os.replace(temporary, path)
            outcome = {"sha256": sha256_file(path)}
        except (OSError, RuntimeError) as exc:
            outcome = {"error": str(exc)}
    outcome = broadcast(outcome, ctx)
    if "error" in outcome:
        raise RuntimeError(f"checkpoint write failed: {outcome['error']}")
    return outcome["sha256"]


def load_distributed(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    digest: str,
    ctx: DistContext,
    device: torch.device,
) -> tuple[int, int]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 2,
        "digest": digest,
        "world_size": ctx.world_size,
        "device_type": device.type,
        "torch_version": str(torch.__version__),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    unwrap(model).load_state_dict(record["model"])
    optimizer.load_state_dict(record["optimizer"])
    scheduler.load_state_dict(record["scheduler"])
    state = record["rank_states"][ctx.rank]
    torch.set_rng_state(state["torch"])
    _restore_numpy_rng(state["numpy"])
    random.setstate(state["python"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    return record["step"], state["local_tokens"]
