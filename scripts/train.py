"""Dry-run training entrypoint. It does not download data or start a fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from openchange.splits import pilot_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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

    print("dry-run")
    print("task:", config.get("task"))
    print("seed:", config.get("seed"))
    print("split:", config.get("split"))
    print("train regions:", contract.train_regions)
    print("test regions:", contract.test_regions)
    print("temporal cutoff:", contract.temporal_cutoff)
    print("backbones reserved for later:", ", ".join(config.get("backbones_later", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
