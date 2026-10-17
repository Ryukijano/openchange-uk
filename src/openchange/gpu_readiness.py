"""Bounded synthetic GH200 checks, separate from scientific/model readiness."""

from __future__ import annotations

import argparse
import os
import platform
import re
import time
from datetime import timedelta
from pathlib import Path

import yaml

from openchange.provenance import start_run, summary_lines, write_json

IMAGE = "nvcr.io/nvidia/pytorch:25.05-py3"
CONFIG_KEYS = {"job", "seed", "container", "expect_machine", "expect_device"}


def validate_config(config: object) -> dict:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"config must contain exactly {sorted(CONFIG_KEYS)}")
    expected = {
        "job": "gpu-readiness", "container": IMAGE,
        "expect_machine": "aarch64", "expect_device": "GH200",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"{key} must be {value!r}")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return config


def execution_context(mode: str, env: dict[str, str], machine: str) -> tuple[int, int, int]:
    if machine != "aarch64":
        raise ValueError("requires the aarch64 compute-node container; no CPU/x86 fallback")
    if not env.get("SLURM_JOB_ID") or env.get("SLURM_JOB_NUM_NODES") != "1":
        raise ValueError("requires a one-node Slurm allocation")
    if not env.get("APPTAINER_CONTAINER"):
        raise ValueError("requires Apptainer --nv")
    sha = env.get("OPENCHANGE_GIT_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("missing full git SHA")
    if env.get("OPENCHANGE_EXPECTED_GIT_SHA") != sha or env.get("OPENCHANGE_GIT_DIRTY") != "0":
        raise ValueError("requires the reviewed commit and a clean tree")
    if env.get("OPENCHANGE_IMAGE") != IMAGE:
        raise ValueError("unexpected container image label")
    if not re.fullmatch(r"[0-9a-f]{64}", env.get("OPENCHANGE_SIF_SHA256", "")):
        raise ValueError("missing actual container SHA-256")
    if mode == "single":
        if any(key in env for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
            raise ValueError("single mode must not be launched with torchrun")
        return 0, 0, 1
    if mode != "distributed":
        raise ValueError("mode must be single or distributed")
    rank, local_rank, world = (int(env[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world not in (2, 4) or rank != local_rank or not 0 <= rank < world:
        raise ValueError("distributed mode requires 2 or 4 local ranks on one node")
    return rank, local_rank, world


def run_cuda_checks(config: dict, mode: str, context: tuple[int, int, int], run_dir: Path, report: dict):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel

    rank, local_rank, world = context

    def check(name, function):
        report["active_check"] = name
        write_json(run_dir / "readiness.json", report)
        result = function()
        report["checks"].append({"name": name, "status": "passed", "details": result})
        print(f"rank={rank} check={name} passed", flush=True)

    def hardware():
        if not torch.cuda.is_available() or torch.cuda.device_count() != world:
            raise RuntimeError("CUDA unavailable or visible GPU count differs from requested ranks")
        torch.cuda.set_device(local_rank)
        props = torch.cuda.get_device_properties(local_rank)
        if config["expect_device"] not in props.name or (props.major, props.minor) != (9, 0):
            raise RuntimeError("expected a GH200 with compute capability 9.0")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("native BF16 is required")
        report["runtime"] = {
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "nccl": torch.cuda.nccl.version(),
            "device": props.name, "capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory, "visible_gpus": torch.cuda.device_count(),
            "local_rank": local_rank,
        }
        return report["runtime"]

    try:
        check("hardware_and_runtime", hardware)
        torch.set_num_threads(2)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda", local_rank)
        torch.cuda.reset_peak_memory_stats(device)

        def matmul():
            a = torch.randn(512, 512, device=device) / 512**0.5
            b = torch.randn(512, 512, device=device) / 512**0.5
            reference = a @ b
            actual = a.bfloat16() @ b.bfloat16()
            torch.testing.assert_close(actual.float(), reference, atol=0.002, rtol=0.03)
            return {"shape": [512, 512], "dtype": "bfloat16", "atol": 0.002, "rtol": 0.03}

        check("bf16_matmul_vs_fp32", matmul)

        def attention():
            q = torch.randn(2, 4, 32, 16, device=device)
            k, v = torch.randn_like(q), torch.randn_like(q)
            reference = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            low_q = q.bfloat16().requires_grad_()
            actual = F.scaled_dot_product_attention(low_q, k.bfloat16(), v.bfloat16(), dropout_p=0.0)
            torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.03)
            actual.float().square().mean().backward()
            if low_q.grad is None or not torch.isfinite(low_q.grad).all():
                raise RuntimeError("nonfinite or missing attention gradient")
            return {"api": "scaled_dot_product_attention", "backend": "PyTorch-selected; not certified FlashAttention"}

        check("bf16_attention_forward_backward", attention)

        if mode == "distributed":
            report["active_check"] = "nccl_initialization"
            write_json(run_dir / "readiness.json", report)
            dist.init_process_group("nccl", timeout=timedelta(seconds=120))

            def collective():
                value = torch.tensor(float(rank + 1), device=device)
                dist.all_reduce(value)
                if value.item() != world * (world + 1) / 2:
                    raise RuntimeError("NCCL SUM is incorrect")
                return {"backend": "nccl", "world_size": world, "scope": "single_node_only"}

            check("nccl_all_reduce", collective)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=4, stride=4), torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16, 8),
        ).to(device)
        wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        inputs = torch.randn(8, 6, 32, 32, device=device) + rank

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = wrapped(inputs)
                loss = prediction.float().square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("nonfinite or missing gradient")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("nonfinite parameter after optimizer step")
            return loss.item()

        def training():
            before = [p.detach().clone() for p in model.parameters()]
            losses = [train_step(), train_step()]
            if all(torch.equal(a, b) for a, b in zip(before, model.parameters())):
                raise RuntimeError("optimizer did not update any parameters")
            return {"losses": losses, "precision": "BF16 autocast / FP32 parameters", "synthetic_only": True}

        check("synthetic_image_training", training)

        def restart():
            path = run_dir / "synthetic-checkpoint.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            expected_loss = train_step()
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            saved = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            restored_loss = train_step()
            if expected_loss != restored_loss:
                raise RuntimeError("checkpoint continuation loss differs")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            return {"bitwise_model_continuation": True, "scope": "same_job_fixed_inputs_no_stochastic_layers"}

        check("checkpoint_write_read_continue", restart)

        if world > 1:
            def consensus():
                for parameter in model.parameters():
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
                dist.barrier()
                return {"parameters_equal_across_ranks": True}

            check("ddp_parameter_consensus", consensus)

        torch.cuda.synchronize()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "distributed"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    config = validate_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    rank_label = os.environ.get("RANK", "0")
    if not rank_label.isdecimal():
        raise ValueError("invalid rank")
    run_dir, manifest = start_run(
        job=f"gpu-readiness-{args.mode}-rank{rank_label}", config_path=args.config,
        seed=config["seed"], output_root=args.output_root,
    )
    for line in summary_lines(manifest):
        print(line, flush=True)
    report = {
        "status": "running", "mode": args.mode, "rank": int(rank_label), "checks": [],
        "active_check": "execution_context", "dataset_downloaded": False, "weights_downloaded": False,
        "model_readiness": "not_tested", "multi_node_readiness": "not_tested",
        "syncbn_parity": "not_tested_use_existing_model_specific_suite",
    }
    started = time.monotonic()
    write_json(run_dir / "readiness.json", report)
    try:
        context = execution_context(args.mode, dict(os.environ), platform.machine())
        run_cuda_checks(config, args.mode, context, run_dir, report)
        expected_checks = {
            "hardware_and_runtime", "bf16_matmul_vs_fp32", "bf16_attention_forward_backward",
            "synthetic_image_training", "checkpoint_write_read_continue",
        }
        if args.mode == "distributed":
            expected_checks.update({"nccl_all_reduce", "ddp_parameter_consensus"})
        completed = {item["name"] for item in report["checks"] if item["status"] == "passed"}
        if completed != expected_checks:
            raise RuntimeError("not all required checks completed")
        report["status"] = "passed_synthetic_checks_only"
        report["active_check"] = None
        code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
        code = 1
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(run_dir / "readiness.json", report)
    print(f"status={report['status']} report={run_dir / 'readiness.json'}", flush=True)
    return code
