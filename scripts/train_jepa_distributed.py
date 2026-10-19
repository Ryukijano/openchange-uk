#!/usr/bin/env python3
"""Offline clean-JEPA pilot: one process per CPU/GPU, never fetches or submits jobs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openchange.jepa.config import ConfigError, config_hash, load_config, lr_at
from openchange.provenance import git_state, sha256_file, start_run, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--synthetic-fit", action="store_true")
    modes.add_argument("--local-fit", action="store_true")
    result.add_argument("--manifest", type=Path)
    result.add_argument("--sources-config", type=Path, default=Path("configs/sources.yaml"))
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--output-root", type=Path, default=Path("outputs"))
    result.add_argument("--resume", type=Path)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--prefetch", type=int, default=2)
    result.add_argument("--log-every", type=int, default=10)
    result.add_argument("--sigreg-chunk", type=int, default=128)
    result.add_argument("--profile-steps", type=int, default=0)
    return result


def run(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from openchange.jepa import losses, runtime
    from openchange.jepa.distributed import assert_quotas
    from openchange.jepa.gate1 import (
        Gate1Model,
        gate1_loss,
        mix_global_view_batch,
        stratum_preserving_permutation,
    )

    local_pilot = args.local_fit or (args.dry_run and args.manifest is not None)
    cfg = load_config(args.config, local_pilot=local_pilot, cuda=args.device == "cuda")
    if (
        min(args.prefetch, args.log_every, args.sigreg_chunk) < 1
        or args.workers < 0
        or not 0 <= args.profile_steps <= 10
    ):
        raise ConfigError("invalid worker, prefetch, log, chunk or profile setting")
    if cfg["status"] == "planned":
        raise ConfigError("planned configs cannot be fitted")
    if cfg["runtime"]["max_wall_seconds"] <= 0 or cfg["runtime"]["max_wall_seconds"] > 1500:
        raise ConfigError("pilot wall cap must be in (0,1500] seconds")
    if cfg["checkpoint"]["every_steps"] < 1:
        raise ConfigError("checkpoint.every_steps must be positive")
    if not local_pilot and args.manifest is not None:
        raise ConfigError("synthetic mode cannot accept a data manifest")
    if local_pilot and args.manifest is None:
        raise ConfigError("local pilot requires --manifest")
    if args.dry_run and args.resume is not None:
        raise ConfigError("dry-run cannot resume")
    if cfg["model"]["drop_path"] != 0 or cfg["s2_gain_jitter"] != 0:
        raise ConfigError("this pilot supports the selected drop_path=0 and gain_jitter=0 main arm")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.use_deterministic_algorithms(cfg["runtime"]["deterministic"])
    device, ctx = runtime.initialize(args.device)
    runtime.configure_precision_and_determinism(cfg, device)
    if any(q % ctx.world_size for q in cfg["global_quotas"]):
        raise ConfigError("each global stratum quota must be divisible by world size")
    if ctx.world_size > 1 and device.type == "cpu" and cfg["projector_norm"] == "batch":
        raise ConfigError("CPU multi-rank checks need frozen or identity norm; GPU main arm uses SyncBN")
    if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ConfigError("bf16 unsupported on the requested device")
    runtime.seed_all(cfg["seed"], device)
    source_digest = sha256_file(args.sources_config) if local_pilot else None
    manifest_digest = sha256_file(args.manifest) if local_pilot else None
    launch = {
        "recipe": cfg,
        "world_size": ctx.world_size,
        "device": device.type,
        "manifest_sha256": manifest_digest,
        "sources_sha256": source_digest,
        "workers": args.workers,
        "prefetch": args.prefetch,
        "sigreg_chunk": args.sigreg_chunk,
        "git_sha": git_state()["sha"],
        "code_sha256": {
            str(path): sha256_file(Path(__file__).resolve().parents[1] / path)
            for path in [Path("scripts/train_jepa_distributed.py")]
            + [
                Path(f"src/openchange/jepa/{name}.py")
                for name in (
                    "config",
                    "model",
                    "views",
                    "losses",
                    "distributed",
                    "data",
                    "gate1",
                    "checkpoints",
                    "runtime",
                    "guarantees",
                    "wam",
                    "coastal_physics",
                    "stochastic_interpolant",
                    "assimilation",
                    "science_wm",
                )
            ]
        },
    }
    digest = config_hash(launch)
    if local_pilot:
        from openchange.jepa.data import LocalObservationDataset

        dataset = LocalObservationDataset(args.manifest, args.sources_config)
        if dataset.manifest_sha256 != manifest_digest or sha256_file(args.sources_config) != source_digest:
            raise ValueError("manifest or source registry changed during input preflight")
    else:
        dataset = runtime.SyntheticDataset(cfg)
    if args.dry_run:
        if ctx.rank == 0:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "digest": digest,
                        "world_size": ctx.world_size,
                        "local_quotas": [q // ctx.world_size for q in cfg["global_quotas"]],
                        "manifest_sha256": manifest_digest,
                        "source_sha256": source_digest,
                        "synthetic_only": not local_pilot,
                        "note": "no optimizer step; no hardware-performance claim",
                    },
                    sort_keys=True,
                )
            )
        return 0
    from openchange.jepa.data import BalancedStepSampler

    model = runtime.wrap_model(Gate1Model(cfg), device, ctx, cfg["projector_norm"])
    optimizer = runtime.optimizer_for(model, cfg, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_at(step, cfg) / cfg["optimizer"]["lr"],
    )
    start_step, local_tokens = 0, 0
    if args.resume:
        start_step, local_tokens = runtime.load_distributed(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            digest=digest,
            ctx=ctx,
            device=device,
        )
    if not 0 <= start_step < cfg["schedule"]["total_steps"]:
        raise ValueError("checkpoint step is outside remaining pilot steps")
    sampler = BalancedStepSampler(
        dataset,
        cfg["global_quotas"],
        seed=cfg["seed"],
        rank=ctx.rank,
        world_size=ctx.world_size,
        start_step=start_step,
        total_steps=cfg["schedule"]["total_steps"],
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": runtime.ViewCollator(cfg, local_pilot),
        "generator": torch.Generator().manual_seed(cfg["seed"]),
    }
    if args.workers:
        loader_options.update(
            prefetch_factor=args.prefetch, persistent_workers=True, multiprocessing_context="spawn"
        )
    loader = DataLoader(runtime.StepViewDataset(dataset), batch_sampler=sampler, **loader_options)
    run_dir = None
    if ctx.rank == 0:
        with runtime.preserve_rng(device):
            run_dir, _ = start_run(
                cfg["job"],
                args.config,
                cfg["seed"],
                args.output_root,
                extra={
                    "launch": launch,
                    "launch_digest": digest,
                    "resolved_config": cfg,
                    "synthetic_only": not local_pilot,
                    "resumed_from": str(args.resume) if args.resume else None,
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "normalization": "SyncBatchNorm"
                    if ctx.world_size > 1 and cfg["projector_norm"] == "batch"
                    else cfg["projector_norm"],
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "note": "pilot plumbing, not an EO research result",
                },
            )
    run_dir = Path(runtime.broadcast(str(run_dir) if run_dir else None, ctx))
    started = time.perf_counter()
    window_started = started
    window_tokens = 0
    window_steps = 0
    statistics = torch.zeros(5, device=device)
    ep_statistics = torch.zeros(
        cfg["global_views"] + cfg["local_views"],
        4 if cfg["sigreg_mode"] == "stratified" else 1,
        device=device,
    )
    data_seconds = 0.0
    checkpoints: list[dict] = []
    last_metrics: dict | None = None
    saved_steps: set[int] = set()

    def save(step: int) -> None:
        if step in saved_steps:
            return
        path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
        checksum = runtime.save_distributed(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            local_tokens=local_tokens,
            digest=digest,
            ctx=ctx,
            device=device,
        )
        saved_steps.add(step)
        if ctx.rank == 0:
            checkpoints.append({"step": step, "path": str(path), "sha256": checksum})
            write_json(run_dir / "checkpoint_index.json", checkpoints)

    def log(step: int) -> None:
        nonlocal window_started, window_tokens, window_steps, data_seconds, last_metrics
        if not window_steps:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = runtime.reduce(
            torch.tensor(time.perf_counter() - window_started, device=device), ctx, dist.ReduceOp.MAX
        ).item()
        values = (runtime.reduce(statistics, ctx) / (ctx.world_size * window_steps)).cpu().tolist()
        tokens = runtime.reduce(torch.tensor(window_tokens, device=device, dtype=torch.int64), ctx).item()
        seen = runtime.reduce(torch.tensor(local_tokens, device=device, dtype=torch.int64), ctx).item()
        loader_wait = runtime.reduce(torch.tensor(data_seconds, device=device), ctx, dist.ReduceOp.MAX).item()
        peak = runtime.reduce(
            torch.tensor(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                device=device,
                dtype=torch.int64,
            ),
            ctx,
            dist.ReduceOp.MAX,
        ).item()
        reserved = runtime.reduce(
            torch.tensor(
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                device=device,
                dtype=torch.int64,
            ),
            ctx,
            dist.ReduceOp.MAX,
        ).item()
        ep_means = (runtime.reduce(ep_statistics, ctx) / (ctx.world_size * window_steps)).cpu().tolist()
        if ctx.rank == 0:
            row = {
                "step": step,
                "window_steps": window_steps,
                "loss": values[0],
                "inv": values[1],
                "sigreg": values[2],
                "t_mu": values[3],
                "grad_norm_before_clip": values[4],
                "sigreg_null": losses.ep_null_expectation(),
                "tokens_seen": seen,
                "window_tokens": tokens,
                "tokens_per_second": tokens / elapsed,
                "window_seconds_max_rank": elapsed,
                "loader_wait_seconds_max_rank": loader_wait,
                "peak_allocated_bytes_max_rank": peak,
                "peak_reserved_bytes_max_rank": reserved,
                "ep_per_view_stratum": ep_means,
                "stratum_counts": cfg["global_quotas"],
                "phase": "startup_or_warmup" if step <= start_step + args.log_every else "steady_window",
                "lr": optimizer.param_groups[0]["lr"],
            }
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            last_metrics = row
        statistics.zero_()
        ep_statistics.zero_()
        window_steps, window_tokens, data_seconds = 0, 0, 0.0
        window_started = time.perf_counter()

    profiler = None
    if args.profile_steps and ctx.rank == 0:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=3, active=args.profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(run_dir / "traces")),
            record_shapes=True,
            profile_memory=True,
        )
        profiler.start()
    status, next_step = "completed", start_step
    try:
        iterator = iter(loader)
        for step in range(start_step, cfg["schedule"]["total_steps"]):
            before_data = time.perf_counter()
            with torch.profiler.record_function("jepa/input"):
                batch = next(iterator)
                runtime.validate_batch(batch, cfg["global_quotas"], ctx.world_size, cfg["patch_size"])
            data_seconds += time.perf_counter() - before_data
            strata = batch.strata.to(device=device, non_blocking=True)
            if step == start_step:
                assert_quotas(strata, cfg["global_quotas"], ctx)
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.profiler.record_function("jepa/encoder"),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=cfg["precision"] == "bf16",
                ),
            ):
                output = model(batch, validate=False)
            mixed_output = None
            mix_perm = None
            mix_alpha = None
            if cfg["objective"]["name"] == "tessera_mixup":
                mix_perm = stratum_preserving_permutation(
                    strata, seed=cfg["seed"], step=step
                )
                mix_alpha = float(
                    torch.rand(
                        (),
                        generator=torch.Generator().manual_seed(cfg["seed"] + 4099 + step),
                    ).item()
                )
                mixed_batch = mix_global_view_batch(batch, mix_perm, mix_alpha)
                with (
                    torch.profiler.record_function("jepa/mixed_encoder"),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=cfg["precision"] == "bf16",
                    ),
                ):
                    mixed_output = model(mixed_batch, validate=False)
            with torch.profiler.record_function("jepa/sigreg"):
                directions = losses.sigreg_directions(
                    cfg["projector_dim"],
                    cfg["sigreg_slices"],
                    seed=cfg["seed"],
                    step=step,
                    device=device,
                    dtype=torch.float32,
                )
                objective = cfg["objective"]
                directions_s1 = None
                directions_s2 = None
                if objective["name"] == "defensive_split_cfm":
                    directions_s1 = losses.sigreg_directions(
                        objective["s1_private_dim"],
                        cfg["sigreg_slices"],
                        seed=cfg["seed"] + 17,
                        step=step,
                        device=device,
                        dtype=torch.float32,
                    )
                    directions_s2 = losses.sigreg_directions(
                        objective["s2_private_dim"],
                        cfg["sigreg_slices"],
                        seed=cfg["seed"] + 31,
                        step=step,
                        device=device,
                        dtype=torch.float32,
                    )
                loss = gate1_loss(
                    cfg,
                    model=runtime.unwrap(model),
                    output=output,
                    batch=batch,
                    strata=strata,
                    step=step,
                    directions=directions,
                    directions_s1=directions_s1,
                    directions_s2=directions_s2,
                    mixed_output=mixed_output,
                    mix_perm=mix_perm,
                    mix_alpha=mix_alpha,
                    ctx=ctx,
                    chunk=args.sigreg_chunk,
                )
            bad = runtime.reduce(
                (~torch.isfinite(loss["loss"])).to(torch.int32), ctx, dist.ReduceOp.MAX
            ).item()
            if bad:
                status = "nonfinite_loss"
                break
            with torch.profiler.record_function("jepa/backward"):
                loss["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optimizer"]["grad_clip_norm"]
                )
            bad = runtime.reduce((~torch.isfinite(grad_norm)).to(torch.int32), ctx, dist.ReduceOp.MAX).item()
            if bad:
                status = "nonfinite_gradient"
                break
            with torch.profiler.record_function("jepa/optimizer"):
                optimizer.step()
                scheduler.step()
            next_step = step + 1
            step_tokens = int(batch.tokens.sum())
            local_tokens += step_tokens
            window_tokens += step_tokens
            window_steps += 1
            statistics += torch.stack(
                [
                    loss["loss"].detach(),
                    loss["inv"].detach(),
                    loss["sigreg"].detach(),
                    loss["t_mu"].detach(),
                    grad_norm.detach(),
                ]
            )
            ep_statistics += loss["ep"]
            if profiler:
                profiler.step()
            stop = runtime.reduce(
                torch.tensor(
                    int(time.perf_counter() - started >= cfg["runtime"]["max_wall_seconds"]), device=device
                ),
                ctx,
                dist.ReduceOp.MAX,
            ).item()
            if next_step % args.log_every == 0 or stop:
                log(next_step)
            if next_step % cfg["checkpoint"]["every_steps"] == 0:
                save(next_step)
            if stop:
                status = "runtime_cap"
                break
        log(next_step)
        if status == "completed" or status == "runtime_cap":
            save(next_step)
        if ctx.rank == 0:
            result = {
                "status": status,
                "start_step": start_step,
                "next_step": next_step,
                "run_dir": str(run_dir),
                "world_size": ctx.world_size,
                "launch_digest": digest,
                "config_path": str(args.config),
                "git_sha": launch["git_sha"],
                "seed": cfg["seed"],
                "device": device.type,
                "torch_version": str(torch.__version__),
                "metrics_path": str(run_dir / "metrics.jsonl"),
                "last_metrics": last_metrics,
                "checkpoint_paths": [record["path"] for record in checkpoints],
                "wall_seconds": time.perf_counter() - started,
                "note": "synthetic plumbing only"
                if not local_pilot
                else "bounded approved local-data pilot; not a benchmark result",
            }
            write_json(run_dir / "result.json", result)
            print(json.dumps(result, sort_keys=True))
    finally:
        if profiler:
            profiler.stop()
    return {"completed": 0, "runtime_cap": 3}.get(status, 4)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        return run(args)
    except (ConfigError, ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"training refused or failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if "torch.distributed" in sys.modules:
            dist = sys.modules["torch.distributed"]
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
