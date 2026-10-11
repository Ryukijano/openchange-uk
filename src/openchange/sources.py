"""Data-source approval gate.

A fetch is allowed only for a source listed in configs/sources.yaml with every
required field filled in. This module validates; it never downloads anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "sources.yaml"
TOP_LEVEL_KEY = "approved_public_sources"

REQUIRED = (
    "id",
    "url",
    "license",
    "license_url",
    "attribution",
    "expected_size_bytes",
    "use",
    "splits",
    "redistribution",
    "approved_by",
    "approved_on",
)
OPTIONAL = ("notes", "checksum_sha256", "manifest")
USES = {"imagery", "labels", "auxiliary"}
SPLITS = {"pilot", "train", "validation", "test"}
REDISTRIBUTION = {"allowed", "manifest-only"}
PLACEHOLDERS = {"", "tbd", "todo", "unknown", "unclear", "n/a", "na", "none", "?"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceNotApproved(RuntimeError):
    """Raised when code asks to fetch from a source that is not approved."""


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    license: str
    license_url: str
    attribution: str
    expected_size_bytes: int
    use: str
    splits: tuple[str, ...]
    redistribution: str
    approved_by: str
    approved_on: str
    notes: str = ""
    checksum_sha256: str = ""
    manifest: str = ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_placeholder(value: Any) -> bool:
    return _text(value).lower() in PLACEHOLDERS


def validate_entry(entry: Any, today: date | None = None) -> list[str]:
    today = today or date.today()
    if not isinstance(entry, dict):
        return ["entry is not a mapping"]
    errors: list[str] = []
    for key in REQUIRED:
        if key not in entry:
            errors.append(f"missing {key}")
    for key in entry:
        if key not in REQUIRED and key not in OPTIONAL:
            errors.append(f"unknown field {key!r}")

    for key in ("id", "license", "attribution", "approved_by"):
        if key in entry and _is_placeholder(entry[key]):
            errors.append(f"{key} is empty or a placeholder")
    if "id" in entry and not _is_placeholder(entry["id"]) and not ID_RE.match(_text(entry["id"])):
        errors.append("id must be lowercase letters, digits, and hyphens")

    for key in ("url", "license_url"):
        if key in entry and not _text(entry[key]).startswith("https://"):
            errors.append(f"{key} must be an https:// URL")

    if "expected_size_bytes" in entry:
        size = entry["expected_size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append("expected_size_bytes must be a positive integer")

    if "use" in entry and entry["use"] not in USES:
        errors.append(f"use must be one of {sorted(USES)}")

    if "splits" in entry:
        splits = entry["splits"]
        if not isinstance(splits, list) or not splits:
            errors.append("splits must be a non-empty list")
        else:
            bad = [s for s in splits if s not in SPLITS]
            if bad:
                errors.append(f"unknown split(s) {bad}; allowed {sorted(SPLITS)}")
            if len(set(splits)) != len(splits):
                errors.append("splits has duplicates")

    if "redistribution" in entry and entry["redistribution"] not in REDISTRIBUTION:
        errors.append(f"redistribution must be one of {sorted(REDISTRIBUTION)}")

    if "approved_on" in entry:
        raw = entry["approved_on"]
        try:
            approved = raw if isinstance(raw, date) else date.fromisoformat(_text(raw))
        except ValueError:
            errors.append("approved_on must be an ISO date (YYYY-MM-DD)")
        else:
            if approved > today:
                errors.append("approved_on is in the future")

    checksum = entry.get("checksum_sha256")
    if checksum not in (None, "") and not SHA256_RE.match(_text(checksum)):
        errors.append("checksum_sha256 must be 64 lowercase hex characters")
    return errors


def validate_document(doc: Any, today: date | None = None) -> list[str]:
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        return ["sources file must be a mapping"]
    errors = [f"unknown top-level key {k!r}" for k in doc if k != TOP_LEVEL_KEY]
    if TOP_LEVEL_KEY not in doc:
        errors.append(f"missing top-level key {TOP_LEVEL_KEY!r}")
        return errors
    entries = doc[TOP_LEVEL_KEY] or []
    if not isinstance(entries, list):
        return errors + [f"{TOP_LEVEL_KEY} must be a list"]
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = entry.get("id", f"#{index}") if isinstance(entry, dict) else f"#{index}"
        errors.extend(f"source {label}: {e}" for e in validate_entry(entry, today))
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            if entry["id"] in seen:
                errors.append(f"source {entry['id']}: duplicate id")
            seen.add(entry["id"])
    return errors


def load_document(path: Path = DEFAULT_PATH) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_sources(path: Path = DEFAULT_PATH, today: date | None = None) -> list[Source]:
    doc = load_document(path)
    errors = validate_document(doc, today)
    if errors:
        raise SourceNotApproved("invalid sources file:\n  " + "\n  ".join(errors))
    sources = []
    for entry in (doc or {}).get(TOP_LEVEL_KEY) or []:
        values = dict(entry)
        values["splits"] = tuple(values["splits"])
        values["approved_on"] = str(values["approved_on"])
        sources.append(Source(**values))
    return sources


def require_approved(
    source_id: str,
    split: str | None = None,
    path: Path = DEFAULT_PATH,
    today: date | None = None,
) -> Source:
    """The only gate a future fetcher may use. Raises unless the source is approved."""
    for source in load_sources(path, today):
        if source.id == source_id:
            if split is not None and split not in source.splits:
                raise SourceNotApproved(f"{source_id} is not approved for split {split!r}")
            return source
    raise SourceNotApproved(f"{source_id} is not listed in {path}")
