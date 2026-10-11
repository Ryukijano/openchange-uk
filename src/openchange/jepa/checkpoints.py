"""Atomic, weights-only-compatible checkpoint serialization for JEPA runs."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": torch.tensor(keys.astype(np.int64), dtype=torch.int64),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng(state: dict[str, Any]) -> None:
    keys = state["keys"].cpu().numpy().astype(np.uint32)
    np.random.set_state(
        (
            state["name"],
            keys,
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    config_hash: str,
    cfg: dict,
    metrics_tail: list[dict],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [dict(row) for row in metrics_tail]
    record = {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "sampler_step": int(step),
        "tokens_seen": int(tokens_seen),
        "tokens_last_step": int(metrics[-1].get("tokens_step", 0)) if metrics else 0,
        "config_hash": config_hash,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cpu_threads": int(torch.get_num_threads()),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": _numpy_rng_state(),
        "metrics_tail": metrics,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)
    return _file_sha256(path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config_hash: str,
    cfg: dict,
) -> dict[str, Any]:
    record = torch.load(Path(path), map_location="cpu", weights_only=True)
    if record.get("schema_version") != 1:
        raise ValueError("checkpoint schema_version mismatch")
    if record.get("config_hash") != config_hash:
        raise ValueError("checkpoint config_hash mismatch")
    if record.get("torch_version") != str(torch.__version__):
        raise ValueError("checkpoint torch_version mismatch")
    if (
        record.get("cpu_threads") != cfg["runtime"]["cpu_threads"]
        or torch.get_num_threads() != record.get("cpu_threads")
    ):
        raise ValueError("checkpoint cpu_threads mismatch")
    if record.get("sampler_step") != record.get("step"):
        raise ValueError("checkpoint sampler_step mismatch")
    model.load_state_dict(record["model"])
    optimizer.load_state_dict(record["optimizer"])
    scheduler.load_state_dict(record["scheduler"])
    torch.set_rng_state(record["torch_rng"].cpu())
    random.setstate(record["python_rng"])
    _restore_numpy_rng(record["numpy_rng"])
    return {
        "step": record["step"],
        "tokens_seen": record["tokens_seen"],
        "tokens_last_step": record["tokens_last_step"],
        "metrics_tail": record["metrics_tail"],
    }
