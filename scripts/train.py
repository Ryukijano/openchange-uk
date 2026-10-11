"""Dry-run training entrypoint. It does not download data or start a fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from openchange.provenance import start_run, summary_lines, write_json
from openchange.splits import pilot_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contract = pilot_contract()
    contract.validate()

    if not args.dry_run or config.get("pilot", {}).get("train", False):
        print(
            "Full training is disabled until a clean smoke log is committed "
            "and the split contract is no longer provisional.",
            file=sys.stderr,
        )
        return 2

    seed = int(config["seed"])
    run_dir, manifest = start_run(
        job=str(config.get("job", "train")),
        config_path=args.config,
        seed=seed,
        output_root=args.output_root,
        extra={"mode": "dry-run", "split_status": contract.status},
    )
    print("dry-run")
    for line in summary_lines(manifest):
        print(line)
    print("task:", config.get("task"))
    print("split:", config.get("split"))
    print("train regions:", contract.train_regions)
    print("validation regions:", contract.validation_regions)
    print("test regions:", contract.test_regions)
    print("temporal cutoff:", contract.temporal_cutoff)
    print("backbones reserved for later:", ", ".join(config.get("backbones_later", [])))
    write_json(run_dir / "result.json", {"status": "dry-run", "metrics": None, "checkpoint": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
