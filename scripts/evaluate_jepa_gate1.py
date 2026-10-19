#!/usr/bin/env python3
"""Evaluate Gate-1 embedding diagnostics on the approved local pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from openchange.jepa.config import config_hash, load_config
from openchange.jepa.data import BalancedStepSampler, LocalObservationDataset
from openchange.jepa.gate1 import Gate1Model
from openchange.jepa.guarantees import (
    align_sfa_matryoshka_gauge,
    evaluate_matryoshka_prefixes,
    fit_subspace_procrustes,
    lejepa_identifiability_bound,
)
from openchange.jepa.runtime import StepViewDataset, ViewCollator
from openchange.provenance import sha256_file, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--sources-config", type=Path, default=Path("configs/sources.yaml"))
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--eval-steps", type=int, default=5)
    result.add_argument("--output", required=True, type=Path)
    return result


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.eval_steps <= 50:
        raise ValueError("eval_steps must be in [1, 50]")
    cfg = load_config(args.config, local_pilot=True, cuda=True)
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    model = Gate1Model(cfg).to(device)
    record = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if record.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint schema")
    model.load_state_dict(record["model"])
    model.eval()

    dataset = LocalObservationDataset(args.manifest, args.sources_config)
    sampler = BalancedStepSampler(
        dataset,
        cfg["global_quotas"],
        seed=cfg["seed"],
        rank=0,
        world_size=1,
        start_step=0,
        total_steps=args.eval_steps,
    )
    loader = DataLoader(
        StepViewDataset(dataset),
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=ViewCollator(cfg, True),
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    features_a: list[torch.Tensor] = []
    features_b: list[torch.Tensor] = []
    strata_seen: list[torch.Tensor] = []
    samples_seen: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch, validate=False)
            z = output.z.float().cpu()
            features_a.append(z[0])
            features_b.append(z[1])
            strata_seen.append(batch.strata.cpu())
            samples_seen.append(batch.sample_ids.cpu())
    za = torch.cat(features_a)
    zb = torch.cat(features_b)
    strata = torch.cat(strata_seen)
    samples = torch.cat(samples_seen)
    if samples.unique().numel() != samples.numel():
        raise ValueError("diagnostic batch reused pilot locations")

    objective = cfg["objective"]
    procrustes = fit_subspace_procrustes(
        za,
        zb,
        shared_dim=objective["shared_dim"],
        s1_private_dim=objective["s1_private_dim"],
        s2_private_dim=objective["s2_private_dim"],
    )
    sfa = align_sfa_matryoshka_gauge(za, zb)
    identifiability = lejepa_identifiability_bound(za, zb)
    matryoshka = evaluate_matryoshka_prefixes(za, strata, zb, strata)

    result = {
        "status": "completed",
        "diagnostic_scope": "training-pair cross-view diagnostics; not downstream benchmark evidence",
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "config_digest": config_hash(cfg),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "sources_sha256": sha256_file(args.sources_config),
        "eval_steps": args.eval_steps,
        "samples": int(samples.numel()),
        "unique_samples": int(samples.unique().numel()),
        "objective": objective["name"],
        "identifiability_bound": identifiability,
        "sfa_gauge": {
            "eigenvalues": sfa["eigenvalues"].tolist(),
        },
        "subspace_procrustes": {
            "shared_relative_residual": procrustes["shared_relative_residual"],
            "full_relative_residual": procrustes["full_relative_residual"],
            "theoretical_full_floor": procrustes["theoretical_full_floor"],
        },
        "matryoshka_strata_diagnostic": matryoshka,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
