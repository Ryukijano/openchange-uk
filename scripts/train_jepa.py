#!/usr/bin/env python3
"""Dry-run or train the synthetic-only P1 clean-JEPA pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from openchange.jepa.config import (
    ConfigError,
    config_hash,
    expected_tokens_per_item,
    keep_count,
    load_config,
    lr_at,
)
from openchange.provenance import start_run, write_json
from openchange.splits import pilot_contract


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parameter_count(cfg: dict[str, Any]) -> int:
    dim = cfg["model"]["dim"]
    depth = cfg["model"]["depth"]
    heads = cfg["model"]["heads"]
    ratio = cfg["model"]["mlp_ratio"]
    if dim % heads:
        raise ConfigError("model.dim must be divisible by model.heads")
    patch_area = cfg["patch_size"] ** 2
    stem = sum((len(cfg["sensors"][sensor]) * patch_area + 1) * dim for sensor in ("s1", "s2"))
    block = (4 + 2 * ratio) * dim**2 + (9 + ratio) * dim
    encoder = stem + depth * block + 2 * dim + 3 * dim
    hidden = cfg["projector_hidden"]
    output = cfg["projector_dim"]
    projector = (
        dim * hidden[0]
        + hidden[0]
        + hidden[0] * hidden[1]
        + hidden[1]
        + hidden[1] * output
        + output
    )
    if cfg["projector_norm"] != "identity":
        projector += 2 * (hidden[0] + hidden[1])
    return encoder + projector


def _manifest_extra(cfg: dict[str, Any], mode: str, split_status: str) -> dict[str, Any]:
    torch_version = _installed_version("torch")
    numpy_version = _installed_version("numpy")
    return {
        "mode": mode,
        "config_hash": config_hash(cfg),
        "resolved_config": cfg,
        "split_status": split_status,
        "synthetic_only": True,
        "torch": {
            "version": torch_version,
            "numpy_version": numpy_version,
            "cpu_threads": cfg["runtime"]["cpu_threads"],
            "device": "cpu",
        },
        "torch_version": torch_version,
        "numpy_version": numpy_version,
        "ddp_reduction": "none_single_process",
        "token_keep_rounding": cfg["token_keep_rounding"],
        "lr_scaling": cfg["optimizer"]["lr_scaling"],
        "weight_decay_scope": cfg["optimizer"]["weight_decay_scope"],
    }


def _dry_run(config_path: Path, cfg: dict[str, Any], output_root: Path, split_status: str) -> int:
    torch_available = importlib.util.find_spec("torch") is not None
    run_dir, _ = start_run(
        cfg["job"],
        config_path,
        cfg["seed"],
        output_root,
        extra=_manifest_extra(cfg, "dry_run", split_status),
    )
    global_patches = (cfg["global_crop_pixels"] // cfg["patch_size"]) ** 2
    local_patches = (cfg["local_crop_pixels"] // cfg["patch_size"]) ** 2
    token_table = {
        "per_view": {
            str(cfg["global_crop_pixels"]): {
                str(count): keep_count(global_patches * count, cfg["global_token_drop"])
                for count in (1, 2)
            },
            str(cfg["local_crop_pixels"]): {
                str(count): keep_count(local_patches * count, cfg["local_token_drop"])
                for count in (1, 2)
            },
        },
        "expected_per_item": {
            "both_sensors": float(expected_tokens_per_item(cfg, 2)),
            "one_sensor": float(expected_tokens_per_item(cfg, 1)),
            "quota_weighted": float(
                (
                    expected_tokens_per_item(cfg, 2) * sum(cfg["global_quotas"][:2])
                    + expected_tokens_per_item(cfg, 1) * sum(cfg["global_quotas"][2:])
                )
                / cfg["global_batch"]
            ),
        },
    }
    result = {
        "status": "dry-run",
        "model_check": "skipped_no_torch",
        "config_hash": config_hash(cfg),
        "parameter_count_formula": _parameter_count(cfg),
        "token_table": token_table,
        "note": "plumbing check, not a result",
    }
    if torch_available:
        os.environ["OMP_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
        os.environ["MKL_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
        import torch

        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        from openchange.jepa.model import JEPAEncoder
        from openchange.jepa.views import make_views, synthetic_observation

        observations = [
            synthetic_observation(index, index, cfg["seed"], cfg)[0] for index in range(4)
        ]
        batch = make_views(observations, cfg["seed"], 0, cfg)
        model = JEPAEncoder(cfg).eval()
        with torch.no_grad():
            output = model(batch)
        if output.b.shape != (batch.n_views, 4, cfg["model"]["dim"]):
            raise RuntimeError("dry-run encoder output shape mismatch")
        if output.z.shape != (batch.n_views, 4, cfg["projector_dim"]):
            raise RuntimeError("dry-run projector output shape mismatch")
        result["model_check"] = "ok"
        result["output_shapes"] = {"b": list(output.b.shape), "z": list(output.z.shape)}
        result["parameter_count_actual"] = sum(parameter.numel() for parameter in model.parameters())
    write_json(run_dir / "result.json", result)
    print(json.dumps({"run_dir": str(run_dir), **result}, sort_keys=True))
    return 0


def _optimizer(model: Any, cfg: dict[str, Any]) -> Any:
    import torch

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "cls_token" not in name and "modality_embed" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    optimizer_cfg = cfg["optimizer"]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": optimizer_cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=optimizer_cfg["lr"],
        betas=tuple(optimizer_cfg["betas"]),
        eps=optimizer_cfg["eps"],
    )


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _synthetic_fit(
    config_path: Path,
    cfg: dict[str, Any],
    output_root: Path,
    resume: Path | None,
    split_status: str,
) -> int:
    os.environ["OMP_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
    import numpy as np
    import torch

    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(bool(cfg["runtime"]["deterministic"]))
    from openchange.jepa import losses
    from openchange.jepa.checkpoints import load_checkpoint, save_checkpoint
    from openchange.jepa.model import JEPAEncoder
    from openchange.jepa.views import make_views, synthetic_batch

    digest = config_hash(cfg)
    run_dir, _ = start_run(
        cfg["job"],
        config_path,
        cfg["seed"],
        output_root,
        extra={
            **_manifest_extra(cfg, "synthetic_fit", split_status),
            "resumed_from": str(resume) if resume is not None else None,
        },
    )
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = JEPAEncoder(cfg)
    optimizer = _optimizer(model, cfg)
    peak = cfg["optimizer"]["lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda schedule_step: lr_at(schedule_step, cfg) / peak,
    )
    start_step = 0
    tokens_seen = 0
    metrics_tail: list[dict[str, Any]] = []
    if resume is not None:
        restored = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config_hash=digest,
            cfg=cfg,
        )
        start_step = restored["step"]
        tokens_seen = restored["tokens_seen"]
        metrics_tail = restored["metrics_tail"]

    next_step = start_step
    checkpoint_log = run_dir / "checkpoints.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter()
    step_count = 0
    nonfinite_events = 0
    status = "complete"
    exit_code = 0

    def save_at(next_step: int) -> None:
        checkpoint_path = run_dir / "checkpoints" / f"step_{next_step:06d}.pt"
        checksum = save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=next_step,
            tokens_seen=tokens_seen,
            config_hash=digest,
            cfg=cfg,
            metrics_tail=metrics_tail[-100:],
        )
        _append_json_line(
            checkpoint_log,
            {"step": next_step, "sha256": checksum, "path": str(checkpoint_path)},
        )

    def save_if_needed(next_step: int) -> None:
        if not (run_dir / "checkpoints" / f"step_{next_step:06d}.pt").exists():
            save_at(next_step)

    for step in range(start_step, cfg["schedule"]["total_steps"]):
        if time.perf_counter() - started >= cfg["runtime"]["max_wall_seconds"]:
            save_if_needed(step)
            status, exit_code = "runtime_cap", 3
            break
        model.train()
        step_started = time.perf_counter()
        data_started = time.perf_counter()
        observations = synthetic_batch(cfg["seed"], step, cfg)
        batch = make_views(observations, cfg["seed"], step, cfg)
        data_seconds = time.perf_counter() - data_started
        output = model(batch)
        directions = losses.sigreg_directions(
            cfg["projector_dim"],
            cfg["sigreg_slices"],
            seed=cfg["seed"],
            step=step,
            device=output.z.device,
            dtype=output.z.dtype,
        )
        trained_mode = cfg["sigreg_mode"]
        main_loss = losses.jepa_loss(
            output.z,
            batch.strata,
            n_global=batch.n_global,
            n_strata=len(cfg["strata"]),
            weights=cfg["stratum_weights"],
            lam=cfg["lambda"],
            lam_mu=cfg["lambda_mu"],
            directions=directions,
            mode=trained_mode,
        )
        other_mode = "pooled" if trained_mode == "stratified" else "stratified"
        with torch.no_grad():
            other_loss = losses.jepa_loss(
                output.z.detach(),
                batch.strata,
                n_global=batch.n_global,
                n_strata=len(cfg["strata"]),
                weights=cfg["stratum_weights"],
                lam=cfg["lambda"],
                lam_mu=cfg["lambda_mu"],
                directions=directions,
                mode=other_mode,
            )
        sigreg_pooled = main_loss["sigreg"] if trained_mode == "pooled" else other_loss["sigreg"]
        sigreg_stratified = main_loss["sigreg"] if trained_mode == "stratified" else other_loss["sigreg"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        loss_finite = bool(torch.isfinite(main_loss["loss"]).item())
        grad_norm_value = float("nan")
        clipped = False
        gradients_finite = False
        if loss_finite:
            main_loss["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["optimizer"]["grad_clip_norm"]
            )
            grad_norm_value = float(grad_norm.detach().cpu())
            clipped = math.isfinite(grad_norm_value) and (
                grad_norm_value > cfg["optimizer"]["grad_clip_norm"]
            )
            gradients_finite = math.isfinite(grad_norm_value) and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            )
        tokens_step = int(output.tokens.sum().item())
        tokens_seen += tokens_step
        if loss_finite and gradients_finite:
            optimizer.step()
            scheduler.step()
            step_count += 1
            next_step = step + 1
        else:
            nonfinite_events += 1
            next_step = step

        coordinate_std = output.z.detach().std(dim=(0, 1), unbiased=False)
        valid_fraction = batch.s2_valid_fraction.tolist()
        valid_fraction = [
            None if not math.isfinite(value) else float(value)
            for row in valid_fraction
            for value in row
        ]
        row = {
            "step": step + 1,
            "lr": current_lr,
            "loss": float(main_loss["loss"].detach().cpu()),
            "inv": float(main_loss["inv"].detach().cpu()),
            "sigreg": float(main_loss["sigreg"].detach().cpu()),
            "sigreg_pooled": float(sigreg_pooled.detach().cpu()),
            "sigreg_stratified": float(sigreg_stratified.detach().cpu()),
            "t_mu": float(main_loss["t_mu"].detach().cpu()),
            "ep": main_loss["ep"].cpu().tolist(),
            "ep_over_n": main_loss["ep_over_n"].cpu().tolist(),
            "grad_norm_before_clip": grad_norm_value,
            "clipped": clipped,
            "tokens_step": tokens_step,
            "tokens_seen": tokens_seen,
            "per_bucket_counts": [
                {
                    "resolution": bucket.resolution,
                    "sensors": list(bucket.sensors),
                    "count": int(bucket.keep_idx.shape[0]),
                }
                for bucket in batch.buckets
            ],
            "stratum_counts": torch.bincount(batch.strata, minlength=4).tolist(),
            "synthetic_counts": {
                "data": int(batch.synthetic_data.sum().item()),
                "condition": int(batch.synthetic_condition.sum().item()),
            },
            "s2_valid_fraction": valid_fraction,
            "z_std_mean": float(coordinate_std.mean().cpu()),
            "z_std_min": float(coordinate_std.min().cpu()),
            "b_std": float(output.b.detach().std(unbiased=False).cpu()),
            "data_s": data_seconds,
            "step_s": time.perf_counter() - step_started,
            "max_rss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        _append_json_line(metrics_path, row)
        metrics_tail.append(row)
        metrics_tail = metrics_tail[-100:]

        if not loss_finite or not gradients_finite:
            save_at(next_step)
            status, exit_code = "nonfinite", 4
            break
        if next_step % cfg["checkpoint"]["every_steps"] == 0:
            save_at(next_step)
        if time.perf_counter() - started >= cfg["runtime"]["max_wall_seconds"]:
            save_if_needed(next_step)
            status, exit_code = "runtime_cap", 3
            break
    else:
        next_step = cfg["schedule"]["total_steps"]
        save_if_needed(next_step)

    result = {
        "status": status,
        "note": "plumbing check, not a result",
        "start_step": start_step,
        "next_step": next_step,
        "steps_completed_this_run": step_count,
        "tokens_seen": tokens_seen,
        "nonfinite_events": nonfinite_events,
        "wall_seconds": time.perf_counter() - started,
        "run_dir": str(run_dir),
        "resume_checkpoint": str(resume) if resume is not None else None,
    }
    write_json(run_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--synthetic-fit", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        cfg = load_config(args.config)
        contract = pilot_contract()
        contract.validate()
        split_status = contract.status
    except (ConfigError, ValueError) as exc:
        print(f"configuration refused: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        if args.resume is not None:
            print("--resume is only available with --synthetic-fit", file=sys.stderr)
            return 2
        return _dry_run(args.config, cfg, args.output_root, split_status)
    if cfg["status"] != "synthetic_debug":
        print("--synthetic-fit requires status: synthetic_debug", file=sys.stderr)
        return 2
    return _synthetic_fit(
        args.config,
        cfg,
        args.output_root,
        args.resume,
        split_status,
    )


if __name__ == "__main__":
    sys.exit(main())
