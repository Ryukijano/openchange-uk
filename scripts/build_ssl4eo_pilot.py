"""Build the <=500-location SSL4EO-S12 pilot manifest from one shard pair.

Reads train/S1GRD and train/S2L2A tar shards (extracted zarr.zips) plus the approved
train_metadata.parquet, selects <=500 locations under whole-10-degree-cell geographic
groups (smallest cells held for validation/test), pairs S1/S2 by minimum-timestamp-gap
season matched through sample_id suffixes (zarr seasonal order differs per modality),
and writes manifest.json, split.json, normalization.json, geo_audit.txt and checksummed
records/<id>/{s1,s2,s2_valid}.npy for LocalObservationDataset.

Usage: PILOT_DATA_ROOT=/path python scripts/build_ssl4eo_pilot.py
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from pyproj import Transformer

ROOT = Path(os.environ.get("PILOT_DATA_ROOT", "/home/ubuntu/pilot-data"))
OUT = ROOT / "pilot_v1"
OUT.mkdir(exist_ok=True)
(OUT / "records").mkdir(exist_ok=True)
REV = "97a91733db962222e1e6b40a836845ddbbb16b86"
S1_SRC, S2_SRC = "ssl4eo-v11-s1grd-shard-000001", "ssl4eo-v11-s2l2a-shard-000001"
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

df = pd.read_parquet(ROOT / "train_metadata.parquet")
df = df[df.tar == "ssl4eos12_shard_000001.tar"].reset_index(drop=True)
print("locations in shard:", len(df))


def openz(d, z):
    return zarr.open(zarr.storage.ZipStore(str(d / z), mode="r"), mode="r")




def footprint_4326(row):
    crs = int(row.crs)
    b = row.bounds
    tr = Transformer.from_crs(crs, 4326, always_xy=True)
    lons, lats = tr.transform([b[0], b[0], b[2], b[2]], [b[1], b[3], b[3], b[1]])
    return [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]


# group by 10-degree geographic cell
def group_of(lon, lat):
    return f"g{int(np.floor(lon / 10))}_{int(np.floor(lat / 10))}"


items = []
for _, row in df.iterrows():
    zname = row["zarr"]
    g1, g2 = openz(ROOT / "zarr_s1", zname), openz(ROOT / "zarr_s2", zname)
    sid1 = [s.rsplit("_", 1)[-1] for s in g1["sample_id"][:].tolist()]
    sid2 = [s.rsplit("_", 1)[-1] for s in g2["sample_id"][:].tolist()]
    assert sorted(sid1) == sorted(sid2) == ["0", "1", "2", "3"], (zname, sid1, sid2)
    pos1 = {s: j for j, s in enumerate(sid1)}
    pos2 = {s: j for j, s in enumerate(sid2)}
    t1 = np.array(g1["time"][:], dtype="datetime64[ns]")
    t2 = np.array(g2["time"][:], dtype="datetime64[ns]")
    gaps = [
        abs(
            int(t1[pos1[s]].astype("datetime64[s]").astype(int))
            - int(t2[pos2[s]].astype("datetime64[s]").astype(int))
        )
        for s in "0123"
    ]
    s = str(int(np.argmin(gaps)))
    sid_base = str(g1["sample"][()])
    items.append(
        dict(
            row=row,
            z=zname,
            i1=pos1[s],
            i2=pos2[s],
            gap=int(min(gaps)),
            t1=str(pd.Timestamp(t1[pos1[s]]).tz_localize("UTC").isoformat()),
            t2=str(pd.Timestamp(t2[pos2[s]]).tz_localize("UTC").isoformat()),
            group=group_of(row.center_lon, row.center_lat),
            sid=sid_base,
        )
    )

# geographic-group split: whole 10-deg cells to roles; smallest cells -> val/test
groups = {}
for it in items:
    groups.setdefault(it["group"], []).append(it)
ordered = sorted(groups, key=lambda g: (len(groups[g]), g))
val_groups = set(ordered[:2])
test_groups = set(ordered[2:4])


def role_of(g):
    return "validation" if g in val_groups else ("test" if g in test_groups else "train")


train_items = [it for it in items if role_of(it["group"]) == "train" and it["gap"] <= 30 * 86400][:500]
print("train records:", len(train_items), "groups:", len(groups))

train_sids = {it["sid"] for it in train_items}
all_entries = [
    {"location_id": f"loc-{it['sid']}", "geographic_group": it["group"], "role": role_of(it["group"])}
    for it in items
]
non_train = [e for e in all_entries if e["role"] != "train"]
train_entries = [e for e in all_entries if e["location_id"] in {f"loc-{s}" for s in train_sids}]
loc_entries = (non_train + train_entries)[:500]
assert len(loc_entries) <= 500 and {e["role"] for e in loc_entries} >= {"train", "validation", "test"}
records = []
s1_sum = np.zeros(2)
s1_sq = np.zeros(2)
s1_n = 0
s2_sum = np.zeros(12)
s2_sq = np.zeros(12)
s2_n = 0
for n, it in enumerate(train_items):
    row = it["row"]
    rid = f"rec-{n:04d}"
    g1, g2 = openz(ROOT / "zarr_s1", it["z"]), openz(ROOT / "zarr_s2", it["z"])
    s1 = np.asarray(g1["bands"][it["i1"]]).astype(np.float32)  # dB VV,VH
    s2 = np.asarray(g2["bands"][it["i2"]]).astype(np.float32) / 10000.0  # reflectance
    valid = np.asarray(g2["cloud_mask"][it["i2"]]) == 0
    rdir = OUT / "records" / rid
    rdir.mkdir(exist_ok=True)
    for name, arr in (("s1", s1), ("s2", s2), ("s2_valid", valid)):
        np.save(rdir / f"{name}.npy", arr)
    sha = {k: hashlib.sha256((rdir / f"{k}.npy").read_bytes()).hexdigest() for k in ("s1", "s2", "s2_valid")}
    cf = float(1.0 - valid.mean())
    records.append(
        {
            "sample_id": n,
            "location_id": f"loc-{it['sid']}",
            "geographic_group": it["group"],
            "role": "train",
            "s1_timestamp": it["t1"],
            "s2_timestamp": it["t2"],
            "cloud_fraction": cf,
            "footprint": {"crs": "EPSG:4326", "bounds": footprint_4326(row)},
            "grid": {
                "crs": f"EPSG:{int(row.crs)}",
                "shape": [264, 264],
                "pixel_size_m": 10,
                "resampling": "provider-native 10 m grid",
                "nodata_policy": "excluded",
            },
            "s1": {
                "path": f"records/{rid}/s1.npy",
                "sha256": sha["s1"],
                "source_id": S1_SRC,
                "revision": REV,
            },
            "s2": {
                "path": f"records/{rid}/s2.npy",
                "sha256": sha["s2"],
                "source_id": S2_SRC,
                "revision": REV,
            },
            "s2_valid": {
                "path": f"records/{rid}/s2_valid.npy",
                "sha256": sha["s2_valid"],
                "source_id": S2_SRC,
                "revision": REV,
            },
        }
    )
    s1_sum += s1.reshape(2, -1).sum(1)
    s1_sq += (s1.reshape(2, -1) ** 2).sum(1)
    s1_n += s1[0].size
    v = s2.reshape(12, -1)[:, valid.reshape(-1)]
    s2_sum += v.sum(1)
    s2_sq += (v**2).sum(1)
    s2_n += v.shape[1]

audit = (
    "Pilot geographic audit: locations grouped by 10-degree lon/lat cells from SSL4EO-S12 "
    "shard_000001. Whole cells assigned to validation/test before train selection; "
    "0 m buffer requested (cells non-overlapping by construction). Train locations only "
    "in manifest records, one pair per location, season index = minimum |S1-S2| gap."
)
(OUT / "geo_audit.txt").write_text(audit)
audit_sha = hashlib.sha256(audit.encode()).hexdigest()
cutoff = (
    max(
        pd.Timestamp(it["t1"]).tz_convert("UTC")
        if pd.Timestamp(it["t1"]).tzinfo
        else pd.Timestamp(it["t1"]).tz_localize("UTC")
        for it in items
    )
    + pd.Timedelta(days=1)
).isoformat()
split = {
    "schema_version": 1,
    "status": "accepted",
    "immutable": True,
    "accepted_by": "Gyanateet Dutta (Ryukijano)",
    "accepted_on": "2026-10-18",
    "temporal_cutoff": cutoff,
    "geographic_policy": {"method": "footprint_buffer", "buffer_m": 0, "audit_sha256": audit_sha},
    "locations": loc_entries,
}
(OUT / "split.json").write_text(json.dumps(split))
split_sha = hashlib.sha256((OUT / "split.json").read_bytes()).hexdigest()

norm = {
    "schema_version": 1,
    "id": "ssl4eo-shard1-train-only-v1",
    "origin": "train_only",
    "split_sha256": split_sha,
    "s1": {
        "bands": ["VV", "VH"],
        "units": "dB",
        "mean": (s1_sum / s1_n).tolist(),
        "std": np.sqrt(s1_sq / s1_n - (s1_sum / s1_n) ** 2).tolist(),
    },
    "s2": {
        "bands": S2_BANDS,
        "units": "reflectance",
        "mean": (s2_sum / s2_n).tolist(),
        "std": np.sqrt(s2_sq / s2_n - (s2_sum / s2_n) ** 2).tolist(),
    },
}
(OUT / "normalization.json").write_text(json.dumps(norm))
norm_sha = hashlib.sha256((OUT / "normalization.json").read_bytes()).hexdigest()

max_gap = max(it["gap"] for it in train_items)
manifest = {
    "schema_version": 1,
    "stage": "pilot",
    "role": "train",
    "source_ids": [S1_SRC, S2_SRC],
    "split": {"path": "split.json", "sha256": split_sha},
    "normalization": {"path": "normalization.json", "sha256": norm_sha},
    "thresholds": {"clear_max": 0.05, "degraded_min": 0.5},
    "max_pair_gap_seconds": max_gap,
    "mask_semantics": "true_is_valid_false_is_cloud",
    "records": records,
}
(OUT / "manifest.json").write_text(json.dumps(manifest))
print("wrote", len(records), "records; max s1-s2 gap (s):", max_gap)
print("val/test locations:", sum(1 for e in loc_entries if e["role"] != "train"))
