"""Evaluation stub. It checks the split contract and prints the metric list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from openchange.splits import pilot_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contract = pilot_contract()
    contract.validate()
    if contract.temporal_cutoff == "unset":
        print("No scores: the temporal cutoff is still unset.")
    print("metrics:", ", ".join(config.get("metrics", [])))
    print("earth2_context:", config.get("earth2_context"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
