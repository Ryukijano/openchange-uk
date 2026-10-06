"""Validate configs/sources.yaml. Never fetches; there is no fetch code in this repo yet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openchange.sources import (
    DEFAULT_PATH,
    SourceNotApproved,
    load_document,
    require_approved,
    validate_document,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--require", metavar="SOURCE_ID", help="fail unless this id is approved")
    parser.add_argument("--split", help="with --require, fail unless approved for this split")
    args = parser.parse_args(argv)

    errors = validate_document(load_document(args.config))
    if errors:
        print(f"{args.config}: not valid. Not downloading.")
        for error in errors:
            print(f"  - {error}")
        return 1

    if args.require:
        try:
            source = require_approved(args.require, args.split, args.config)
        except SourceNotApproved as exc:
            print(f"{exc}. Not downloading.")
            return 1
        print(f"approved: {source.id} ({source.license}, {source.expected_size_bytes} bytes)")
        return 0

    entries = (load_document(args.config) or {}).get("approved_public_sources") or []
    if not entries:
        print("No approved public sources. Not downloading.")
    else:
        print(f"{len(entries)} approved source(s) are valid. Fetch logic is not implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
