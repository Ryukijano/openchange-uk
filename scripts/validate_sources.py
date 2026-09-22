"""Refuse network fetches until configs/sources.yaml lists an approved source."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    path = Path("configs/sources.yaml")
    sources = yaml.safe_load(path.read_text(encoding="utf-8")).get(
        "approved_public_sources"
    ) or []
    if not sources:
        print("No approved public sources. Not downloading.")
        return 0
    print(f"{len(sources)} source(s) listed. Fetch logic is not implemented yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
