"""Fail-closed local pilot ingestion; no network access or provider archive decoder.

Schema v1 is JSON with exact keys (see validators below). Arrays are unnormalised
float32 S1 dB / S2 reflectance on a common 10 m grid, plus a boolean cloud-valid
mask. Split and normalization JSON sidecars and every NPY payload are SHA-256
pinned. Only accepted pilot-train locations enter this dataset. Source permission
is checked independently of sidecar assertions, before any array is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from openchange.jepa.views import Observation
from openchange.sources import SourceNotApproved, require_approved

S1_BANDS = ("VV", "VH")
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MAX_JSON_BYTES = 4_000_000
_MAX_ARRAY_BYTES = 64_000_000


class ManifestError(ValueError):
    """An unapproved, inconsistent or unavailable local pilot must not be fitted."""


def _keys(value: Any, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ManifestError(f"{label} requires exactly {sorted(expected)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManifestError(f"{label} must be nonempty text without surrounding whitespace")
    if value.lower() in {"unknown", "unset", "tbd", "todo", "none", "proposed"}:
        raise ManifestError(f"{label} must not be a placeholder")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ManifestError(f"{label} must be a finite number")
    return float(value)


def _timestamp(value: Any, label: str) -> datetime:
    value = _text(value, label)
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{label} must be an ISO timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ManifestError(f"{label} must have an explicit timezone")
    return stamp.astimezone(timezone.utc)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ManifestError("JSON file exceeds pilot size limit")
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path.name}") from exc
    if not isinstance(value, dict):
        raise ManifestError("JSON root must be an object")
    return value


def _fingerprint(path: Path) -> tuple[int, ...]:
    try:
        info = path.stat()
    except OSError as exc:
        raise ManifestError(f"local file unavailable: {path.name}") from exc
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _local_path(root: Path, value: Any, suffix: str) -> Path:
    name = _text(value, "local path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute() or "\\" in name or ":" in name or "\x00" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or relative.suffix != suffix
    ):
        raise ManifestError("paths must be relative, traversal-free and use the expected suffix")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ManifestError("symlinks are not allowed in local payload paths")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ManifestError(f"local file unavailable or escapes manifest root: {name}") from exc
    if not resolved.is_file():
        raise ManifestError("local payload must be a regular file")
    return resolved


def _descriptor(root: Path, value: Any, suffix: str, payload: bool = False) -> tuple[Path, str]:
    fields = {"path", "sha256"} | ({"source_id", "revision"} if payload else set())
    descriptor = _keys(value, fields, "payload" if payload else "sidecar")
    checksum = descriptor["sha256"]
    if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
        raise ManifestError("sha256 must contain 64 lowercase hex digits")
    if payload:
        _text(descriptor["source_id"], "source_id")
        if not isinstance(descriptor["revision"], str) or not _REVISION.fullmatch(descriptor["revision"]):
            raise ManifestError("source revision must be an immutable 40-character git SHA")
    return _local_path(root, descriptor["path"], suffix), checksum


def _verify(path: Path, checksum: str, limit: int) -> tuple[int, ...]:
    before = _fingerprint(path)
    if before[2] > limit:
        raise ManifestError(f"{path.name} exceeds pilot size limit")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ManifestError(f"cannot read local file: {path.name}") from exc
    if digest.hexdigest() != checksum:
        raise ManifestError(f"checksum mismatch: {path.name}")
    if before != _fingerprint(path):
        raise ManifestError("local payload changed during verification")
    return before


def _split(value: dict) -> tuple[dict[str, tuple[str, str]], datetime]:
    _keys(value, {"schema_version", "status", "immutable", "accepted_by", "accepted_on",
                  "temporal_cutoff", "geographic_policy", "locations"}, "split")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ManifestError("split.schema_version must equal 1")
    if value["status"] != "accepted" or value["immutable"] is not True:
        raise ManifestError("pilot split must be human-accepted and immutable")
    _text(value["accepted_by"], "split.accepted_by")
    try:
        accepted_on = date.fromisoformat(_text(value["accepted_on"], "split.accepted_on"))
    except ValueError as exc:
        raise ManifestError("split.accepted_on must be an ISO date") from exc
    if accepted_on > date.today():
        raise ManifestError("split acceptance cannot be in the future")
    cutoff = _timestamp(value["temporal_cutoff"], "split.temporal_cutoff")
    policy = _keys(value["geographic_policy"], {"method", "buffer_m", "audit_sha256"}, "geographic_policy")
    if policy["method"] != "footprint_buffer" or _number(policy["buffer_m"], "buffer_m") < 0:
        raise ManifestError("geographic_policy must record a nonnegative footprint buffer")
    if not isinstance(policy["audit_sha256"], str) or not _SHA256.fullmatch(policy["audit_sha256"]):
        raise ManifestError("geographic_policy requires an audit SHA-256")
    locations = value["locations"]
    if not isinstance(locations, list) or not 1 <= len(locations) <= 500:
        raise ManifestError("pilot split must contain 1..500 unique locations")
    membership = {}
    group_roles = {}
    for location in locations:
        _keys(location, {"location_id", "geographic_group", "role"}, "split location")
        identity = _text(location["location_id"], "location_id")
        group = _text(location["geographic_group"], "geographic_group")
        role = location["role"]
        if role not in {"train", "validation", "test"} or identity in membership:
            raise ManifestError("duplicate location or invalid split role")
        if group in group_roles and group_roles[group] != role:
            raise ManifestError("geographic group leaks across split roles")
        group_roles[group] = role
        membership[identity] = (group, role)
    if not {"train", "validation", "test"}.issubset(group_roles.values()):
        raise ManifestError("pilot split requires explicit train, validation and test groups")
    return membership, cutoff


def _normalization(value: dict, split_sha256: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    _keys(value, {"schema_version", "id", "origin", "split_sha256", "s1", "s2"}, "normalization")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ManifestError("normalization.schema_version must equal 1")
    _text(value["id"], "normalization.id")
    if value["origin"] not in {"provider", "train_only"}:
        raise ManifestError("normalization origin must be provider or train_only")
    expected_split = split_sha256 if value["origin"] == "train_only" else None
    if value["split_sha256"] != expected_split:
        raise ManifestError("normalization statistics are not tied to the accepted training split")
    constants = {}
    for sensor, bands, units in (("s1", S1_BANDS, "dB"), ("s2", S2_BANDS, "reflectance")):
        spec = _keys(value[sensor], {"bands", "units", "mean", "std"}, f"normalization.{sensor}")
        if spec["bands"] != list(bands) or spec["units"] != units:
            raise ManifestError(f"{sensor} band order/physical units do not match the local array contract")
        numbers = []
        for name in ("mean", "std"):
            items = spec[name]
            if not isinstance(items, list) or len(items) != len(bands):
                raise ManifestError(f"normalization.{sensor}.{name} requires {len(bands)} constants")
            items = [_number(item, f"{sensor}.{name}") for item in items]
            if any(abs(item) > float(np.finfo(np.float32).max) for item in items):
                raise ManifestError("normalization constants must be representable in float32")
            vector = np.asarray(items, dtype=np.float32).reshape(-1, 1, 1)
            if name == "std" and np.any(vector <= 0):
                raise ManifestError("normalization std must be positive in float32")
            numbers.append(vector)
        constants[sensor] = tuple(numbers)
    return constants


def _normalise(array: np.ndarray, constants: tuple[np.ndarray, np.ndarray], valid=None) -> np.ndarray:
    mean, std = constants
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        output = (array - mean) / std
    if not np.isfinite(output).all():
        raise ManifestError("normalization produced nonfinite values")
    if valid is not None:
        output[:, ~valid] = 0.0
    return output


class LocalObservationDataset:
    """Validate the whole local inventory eagerly; load read-only NPY maps per item.

    ``dataset[index, stratum]`` returns the existing ``Observation``. Its sample_id
    is the manifest's integer ID; metadata(index) retains location/date/source/grid
    provenance. Only withholding is synthetic; no pixels or cloud masks are made up.
    """

    def __init__(self, manifest_path: Path, sources_config: Path):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.sources_config = Path(sources_config)
        self._root = self.manifest_path.parent
        self._manifest = _read_json(self.manifest_path)
        _keys(self._manifest, {"schema_version", "stage", "role", "source_ids", "split",
                               "normalization", "thresholds", "max_pair_gap_seconds",
                               "mask_semantics", "records"}, "manifest")
        if type(self._manifest["schema_version"]) is not int or self._manifest["schema_version"] != 1:
            raise ManifestError("manifest.schema_version must equal 1")
        if self._manifest["stage"] != "pilot" or self._manifest["role"] != "train":
            raise ManifestError("only pilot role='train' can enter a fit dataset")
        source_ids = self._manifest["source_ids"]
        if not isinstance(source_ids, list) or not source_ids or any(not isinstance(s, str) for s in source_ids):
            raise ManifestError("source_ids must be a nonempty list of source IDs")
        if len(source_ids) != len(set(source_ids)):
            raise ManifestError("source_ids must be unique")
        self._approved_revisions = {}
        for source_id in source_ids:
            source = require_approved(source_id, "pilot", path=self.sources_config)
            if source.use != "imagery":
                raise SourceNotApproved(f"{source_id} is not approved for imagery (use={source.use!r})")
            pinned = re.search(r"/resolve/([0-9a-f]{40})/", getattr(source, "url", ""))
            if pinned:
                self._approved_revisions[source_id] = pinned.group(1)

        split_path, split_hash = _descriptor(self._root, self._manifest["split"], ".json")
        _verify(split_path, split_hash, _MAX_JSON_BYTES)
        self._membership, self._cutoff = _split(_read_json(split_path))
        norm_path, norm_hash = _descriptor(self._root, self._manifest["normalization"], ".json")
        _verify(norm_path, norm_hash, _MAX_JSON_BYTES)
        self._constants = _normalization(_read_json(norm_path), split_hash)
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.split_sha256 = split_hash
        self.normalization_sha256 = norm_hash
        thresholds = _keys(self._manifest["thresholds"], {"clear_max", "degraded_min"}, "thresholds")
        clear = _number(thresholds["clear_max"], "thresholds.clear_max")
        degraded = _number(thresholds["degraded_min"], "thresholds.degraded_min")
        if not 0 <= clear < degraded <= 1:
            raise ManifestError("require 0 <= clear_max < degraded_min <= 1")
        gap = _number(self._manifest["max_pair_gap_seconds"], "max_pair_gap_seconds")
        if gap < 0:
            raise ManifestError("max_pair_gap_seconds cannot be negative")
        if self._manifest["mask_semantics"] != "true_is_valid_false_is_cloud":
            raise ManifestError("only an explicit cloud-valid mask with nodata already excluded is supported")
        self._records = self._manifest["records"]
        if not isinstance(self._records, list) or not 1 <= len(self._records) <= 500:
            raise ManifestError("pilot inventory must contain 1..500 observation pairs")
        self._paths: list[dict[str, Path]] = []
        self._fingerprints: list[dict[str, tuple[int, ...]]] = []
        self._eligible: list[list[int]] = [[], [], [], []]
        locations, sample_ids, used_sources, used_paths = set(), set(), set(), set()
        for index, record in enumerate(self._records):
            self._validate_record(record, gap)
            identity = record["location_id"]
            if identity in locations or record["sample_id"] in sample_ids:
                raise ManifestError("pilot requires one pair per unique location and unique sample IDs")
            locations.add(identity)
            sample_ids.add(record["sample_id"])
            paths = {}
            for sensor in ("s1", "s2", "s2_valid"):
                descriptor = record[sensor]
                # All declared sources are approved above; unknown IDs cannot reach np.load.
                path, _ = _descriptor(self._root, descriptor, ".npy", payload=True)
                if descriptor["source_id"] not in source_ids:
                    raise ManifestError("payload source was not declared and approved")
                pinned = self._approved_revisions.get(descriptor["source_id"])
                if pinned is not None and descriptor["revision"] != pinned:
                    raise ManifestError("payload revision differs from the approved source URL")
                used_sources.add(descriptor["source_id"])
                if path in used_paths:
                    raise ManifestError("paired payloads must have distinct local paths")
                used_paths.add(path)
                paths[sensor] = path
            self._paths.append(paths)
        if used_sources != set(source_ids):
            raise ManifestError("declared source_ids must exactly match payload sources")

        # Metadata for every source/record is checked before the first array load.
        for index, (record, paths) in enumerate(zip(self._records, self._paths)):
            fingerprints = {
                sensor: _verify(path, record[sensor]["sha256"], _MAX_ARRAY_BYTES)
                for sensor, path in paths.items()
            }
            self._fingerprints.append(fingerprints)
            arrays = {sensor: self._load(index, sensor) for sensor in paths}
            shape = tuple(record["grid"]["shape"])
            for sensor, channels in (("s1", 2), ("s2", 12)):
                array = arrays[sensor]
                if array.dtype != np.float32 or array.shape != (channels, *shape):
                    raise ManifestError(f"{sensor} must be float32 [{channels},H,W] on the declared grid")
                if not np.isfinite(array).all():
                    raise ManifestError(f"{sensor} contains nonfinite pixels")
                _normalise(array, self._constants[sensor])
            mask = arrays["s2_valid"]
            if mask.dtype != np.bool_ or mask.shape != shape:
                raise ManifestError("s2_valid must be bool [H,W] on the declared grid")
            fraction = float(1.0 - mask.mean())
            if not math.isclose(fraction, record["cloud_fraction"], rel_tol=0, abs_tol=1e-12):
                raise ManifestError("cloud_fraction does not match the cloud-valid mask")
            if fraction <= clear:
                self._eligible[0].append(index)
            if fraction >= degraded:
                self._eligible[1].append(index)
            self._eligible[2].append(index)
            self._eligible[3].append(index)

    def _validate_record(self, record: dict, max_gap: float) -> None:
        _keys(record, {"sample_id", "location_id", "geographic_group", "role", "s1_timestamp",
                       "s2_timestamp", "cloud_fraction", "footprint", "grid", "s1", "s2", "s2_valid"}, "record")
        _integer(record["sample_id"], "sample_id")
        identity = _text(record["location_id"], "location_id")
        group = _text(record["geographic_group"], "geographic_group")
        if record["role"] != "train" or self._membership.get(identity) != (group, "train"):
            raise ManifestError("record is not an accepted pilot-train location/group")
        s1_time = _timestamp(record["s1_timestamp"], "s1_timestamp")
        s2_time = _timestamp(record["s2_timestamp"], "s2_timestamp")
        if abs((s1_time - s2_time).total_seconds()) > max_gap:
            raise ManifestError("S1/S2 timestamps exceed the accepted pairing gap")
        if max(s1_time, s2_time) > self._cutoff:
            raise ManifestError("training pair crosses the immutable temporal cutoff")
        cloud = _number(record["cloud_fraction"], "cloud_fraction")
        if not 0 <= cloud <= 1:
            raise ManifestError("cloud_fraction must be in [0,1]")
        footprint = _keys(record["footprint"], {"crs", "bounds"}, "footprint")
        if footprint["crs"] != "EPSG:4326":
            raise ManifestError("footprint bounds must be in EPSG:4326, not the raster's native CRS")
        bounds = footprint["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ManifestError("footprint requires [west,south,east,north]")
        west, south, east, north = [_number(v, "footprint.bounds") for v in bounds]
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ManifestError("footprint bounds are invalid")
        grid = _keys(record["grid"], {"crs", "shape", "pixel_size_m", "resampling", "nodata_policy"}, "grid")
        _text(grid["crs"], "grid.crs")
        _text(grid["resampling"], "grid.resampling")
        if grid["nodata_policy"] != "excluded" or _number(grid["pixel_size_m"], "pixel_size_m") != 10:
            raise ManifestError("grid must be 10 m with nodata excluded before ingestion")
        if not isinstance(grid["shape"], list) or len(grid["shape"]) != 2:
            raise ManifestError("grid.shape requires [H,W]")
        for dimension in grid["shape"]:
            if _integer(dimension, "grid.shape", 1) > 1024:
                raise ManifestError("pilot grid dimensions cannot exceed 1024 pixels")

    def _load(self, index: int, sensor: str) -> np.ndarray:
        path = _local_path(self._root, self._records[index][sensor]["path"], ".npy")
        fingerprint = self._fingerprints[index][sensor]
        if _fingerprint(path) != fingerprint:
            raise ManifestError("verified local payload changed; reconstruct the dataset with new approvals")
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError, EOFError) as exc:
            raise ManifestError(f"cannot load local NPY array: {path.name}") from exc
        if not isinstance(array, np.ndarray) or _fingerprint(path) != fingerprint:
            raise ManifestError("payload is not a stable NPY array")
        return array

    def __len__(self) -> int:
        return len(self._records)

    def eligible_indices(self, stratum: int) -> tuple[int, ...]:
        if type(stratum) is not int or stratum not in range(4):
            raise ManifestError("stratum must be an integer in [0,3]")
        return tuple(self._eligible[stratum])

    def metadata(self, index: int) -> dict:
        self._check_index(index)
        return json.loads(json.dumps(self._records[index]))

    def _check_index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("observation index out of range")

    def __getitem__(self, key: tuple[int, int]) -> Observation:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ManifestError("dataset key must be (index,stratum)")
        index, stratum = key
        self._check_index(index)
        if index not in self.eligible_indices(stratum):
            raise ManifestError("observation is ineligible for the requested stratum")
        # Missing/changed paired files are refused even if one sensor is withheld.
        for sensor in ("s1", "s2", "s2_valid"):
            path = _local_path(self._root, self._records[index][sensor]["path"], ".npy")
            if _fingerprint(path) != self._fingerprints[index][sensor]:
                raise ManifestError("verified local paired payload changed or became unavailable")
        s1 = None if stratum == 3 else _normalise(self._load(index, "s1"), self._constants["s1"])
        valid = None if stratum == 2 else np.array(self._load(index, "s2_valid"), copy=True)
        s2 = None if stratum == 2 else _normalise(self._load(index, "s2"), self._constants["s2"], valid)
        record = self._records[index]
        return Observation(
            sample_id=record["sample_id"], stratum=stratum, s1=s1, s2=s2, s2_valid=valid,
            synthetic_data=False, synthetic_condition=stratum in (2, 3),
            intervention=("withhold_s2" if stratum == 2 else "withhold_s1" if stratum == 3 else "none"),
            mask_fraction=record["cloud_fraction"],
        )


class BalancedStepSampler:
    """Deterministic global stratum blocks, then rank slicing [rank::world_size].

    Every step draws distinct underlying locations globally, not independently per
    stratum/rank. No replacement or accumulation masquerades as a larger batch.
    Iteration is stateless: resume by setting start_step to the next global step.
    """

    def __init__(self, dataset: LocalObservationDataset, quotas: Sequence[int], *, seed: int,
                 rank: int = 0, world_size: int = 1, start_step: int = 0, total_steps: int = 50):
        self.dataset = dataset
        self.quotas = tuple(quotas)
        if len(self.quotas) != 4 or any(type(q) is not int or q < 0 for q in self.quotas) or not sum(self.quotas):
            raise ManifestError("quotas require four nonnegative integers and a nonempty batch")
        self.seed = _integer(seed, "seed")
        self.world_size = _integer(world_size, "world_size", 1)
        self.rank = _integer(rank, "rank")
        if rank >= world_size:
            raise ManifestError("rank must be smaller than world_size")
        if any(q % world_size for q in self.quotas):
            raise ManifestError("each stratum quota must be divisible by world_size")
        self.start_step = _integer(start_step, "start_step")
        self.total_steps = _integer(total_steps, "total_steps")
        if start_step > total_steps:
            raise ManifestError("start_step cannot exceed total_steps")
        self._pools = tuple(tuple(dataset.eligible_indices(m)) for m in range(4))
        for pool in self._pools:
            if any(type(i) is not int or not 0 <= i < len(dataset) for i in pool) or len(set(pool)) != len(pool):
                raise ManifestError("eligibility pools require unique valid underlying indices")
        if set(self._pools[0]) & set(self._pools[1]):
            raise ManifestError("clear/degraded eligibility must be disjoint")
        all_locations = set(range(len(dataset)))
        if set(self._pools[2]) != all_locations or set(self._pools[3]) != all_locations:
            raise ManifestError("withheld strata must be drawn from the complete paired-location inventory")
        if sum(self.quotas) > len(dataset) or any(len(pool) < quota for pool, quota in zip(self._pools, self.quotas)):
            raise ManifestError("insufficient distinct eligible locations for the global quotas; no replacement")

    def __len__(self) -> int:
        return self.total_steps - self.start_step

    def global_batch(self, step: int) -> list[tuple[int, int, int]]:
        _integer(step, "step")
        rng = np.random.default_rng([self.seed, step, 817])
        chosen = []
        used = set()
        for stratum in (0, 1):
            indices = rng.permutation(self._pools[stratum])[:self.quotas[stratum]]
            chosen.extend((step, int(index), stratum) for index in indices)
            used.update(int(index) for index in indices)
        remaining = rng.permutation([index for index in range(len(self.dataset)) if index not in used])
        offset = 0
        for stratum in (2, 3):
            quota = self.quotas[stratum]
            chosen.extend((step, int(index), stratum) for index in remaining[offset:offset + quota])
            offset += quota
        return chosen

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        for step in range(self.start_step, self.total_steps):
            yield self.global_batch(step)[self.rank::self.world_size]
