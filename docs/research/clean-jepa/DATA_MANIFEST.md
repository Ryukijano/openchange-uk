# Local pilot manifest and data interface

Implemented only in `src/openchange/jepa/data.py`; existing `Observation` is unchanged.

```python
dataset = LocalObservationDataset(manifest_path: Path, sources_config: Path)
dataset[index, stratum]  # Observation, stratum 0..3
dataset.eligible_indices(stratum)  # tuple[int, ...]
dataset.metadata(index)  # independent JSON copy of source/location/date/grid provenance
len(dataset)  # unique paired train locations
sampler = BalancedStepSampler(dataset, quotas, seed=seed, rank=rank,
                             world_size=world_size, start_step=0, total_steps=50)
for batch in sampler:  # list[(step,index,stratum)], one local batch per step
    ...
```

`manifest_sha256`, `split_sha256`, `normalization_sha256` are exposed for provenance.
The distributed trainer adapts this interface; imagery and compute approvals
remain independent of the manifest's assertions.

## Strict JSON schema v1

All mappings reject missing/extra/duplicate keys. NPY and JSON paths are relative to
the manifest directory: no traversal, absolute paths, symlinks, or remote endpoints.

**Manifest:** `schema_version: 1`, `stage: "pilot"`, `role: "train"`,
`source_ids: [str]`, `split: {path,sha256}`, `normalization: {path,sha256}`,
`thresholds: {clear_max,degraded_min}`, `max_pair_gap_seconds: number`,
`mask_semantics: "true_is_valid_false_is_cloud"`, `records: [record]`.
Require `0 <= clear_max < degraded_min <= 1`; closed endpoints. Grey-zone records
are available only for withheld strata. No synthetic cloud construction.

**Each record:** `sample_id: int >= 0`, `location_id: str`,
`geographic_group: str`, `role: "train"`, `s1_timestamp`, `s2_timestamp` (ISO with
explicit timezone), `cloud_fraction: number`, `footprint: {crs: "EPSG:4326", bounds:
[west,south,east,north]}`, `grid: {crs,shape:[H,W],pixel_size_m:10,resampling,
nodata_policy:"excluded"}`, `s1`, `s2`, `s2_valid`.
Each payload descriptor has exactly `{path,sha256,source_id,revision}`; revision is
40 lowercase hex characters. All payload source IDs must be declared and approved.
S1 is **unnormalized float32 [2,H,W] in dB**; S2 is **unnormalized float32 [12,H,W]
reflectance**, not legacy DN/+1000; mask is bool [H,W]. All values must be finite.
No tar/Zarr/provider decoding, conversion or resampling is performed here.

**Split sidecar:** `{schema_version:1,status:"accepted",immutable:true,accepted_by,
accepted_on:"YYYY-MM-DD",temporal_cutoff:timezone-aware ISO,
geographic_policy:{method:"footprint_buffer",buffer_m:nonnegative,audit_sha256},
locations:[{location_id,geographic_group,role}]}`. Requires explicit train,
validation and test groups, all disjoint by geographic group, <=500 locations.
Manifest has <=500 pairs, one pair per location; all record memberships must be
train and both acquisition dates <= cutoff. This is a **pilot-only** contract, not
acceptance of OpenChange-UK's benchmark proposal. The loader checks group membership,
not actual polygon/buffer overlap; the recorded human curation audit supplies that.

**Normalization sidecar:** `{schema_version:1,id,origin:"provider"|"train_only",
split_sha256:null|accepted_split_sha256,s1:{bands,units:"dB",mean:[2],std:[2]},
s2:{bands,units:"reflectance",mean:[12],std:[12]}}`.
Train-only statistics must name this split SHA; provider statistics use null.
S1 bands `[VV,VH]`; S2 bands `[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]`.
The loader applies `(array-mean)/std`; **all invalid optical pixels become zero
after normalization**, while the mask remains available for view diagnostics.

## Guarantees / limitations

- Every declared source is gated through `require_approved(id,"pilot",path=sources_config)`
  **and `use == "imagery"` before any array access**. Metadata-only approvals fail.
- JSON sidecars and NPY payload hashes are checked before load; arrays use
  `allow_pickle=False, mmap_mode="r"`. Full inventory validated at construction.
- If an approved source URL pins a 40-hex resolve revision, every corresponding
  payload descriptor must match it before any array is opened.
- Changed/unavailable payloads are refused afterward using file identity/size/mtime/ctime;
  reconstruction re-verifies hashes. Payload maps are not cached across worker forks.
- Normalized samples are independent in-memory arrays; withheld modality is `None`,
  never a zero-filled invented sensor. `synthetic_data=False` for all samples;
  `synthetic_condition=True` only for withholding, intervention `withhold_s1/s2`.
- Sampler is stateless/step-seeded. Clear/degraded first, then withheld singles from
  remaining locations; blocks globally and `[rank::world_size]` locally. Quotas
  divisible by world size; no replacement or duplicate underlying location across ranks.
- Withheld eligibility must cover the whole paired inventory. Insufficient global
  or constrained inventory fails, including N=1024 requested from <=500 locations.
- Schema choices are explicit adapter boundaries, not declarations that current
  real sources, pilot policy or normalization have been accepted.
