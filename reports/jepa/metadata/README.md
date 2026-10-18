# SSL4EO metadata-only audit

Status: completed metadata audit, 11 October 2026. Evidence is preserved here.
No imagery or shards downloaded, no training split accepted, no model fit or cluster
job run. The user selected **Approve metadata only** in this session:
https://app.devin.ai/sessions/8371507a5420405593eda30df300d243

## Verified inputs

Pinned dataset: `embed2scale/SSL4EO-S12-v1.1`, revision
`97a91733db962222e1e6b40a836845ddbbb16b86`, CC BY 4.0.

| Table | Bytes | SHA-256 |
|---|---:|---|
| train_metadata.parquet | 37,358,291 | 1a9f413e3b442f01e736bf96edf69a9ae6b08cff1ef611b47ee6cac5a3cec1b1 |
| val_metadata.parquet | 465,458 | e563236c577667ecf62d0f1ea5f4afaf5d7de958ecc65a1f76d37692d3676069 |

Both files passed `require_approved(id, "pilot")`, exact size and SHA-256 validation.
Combined payload: **37,823,749 bytes**, below the approved 40,000,000-byte ceiling.
Only these exact auxiliary endpoints are approved; neither imagery nor `train`
source use is authorized. Raw Parquet stays in gitignored `outputs/` and is not
redistributed in this report. `fetch-result.json` and `manifest.json` preserve the
run and config hash, including the dirty working tree.

## Findings that change the next pilot

### 1. Provider validation is not a strict footprint holdout

There are 246,144 unique location IDs, 243,968 provider-train and 2,176
provider-validation. A WKB footprint intersection audit found **325 cross-split
pairs with positive-area overlap**, affecting **264 validation locations (12.13%)**.
The median overlap is 10.16% of the smaller footprint; the maximum is 92.85%.

Example: training location `0228726` and validation location `0107814` overlap
92.85% of the smaller footprint. WKT polygons are preserved in
`overlap-example.json`. This is patch-footprint overlap, not proof that the
imagery arrays are identical. It is enough to reject these memberships as our
strict geographical split. Build our own geographic groups and boundary buffers.

### 2. Pairing is feasible, but gaps are not negligible

984,576 seasonal index pairs have complete S1/S2 timestamp fields. S1 ranges from
8 December 2019 to 6 October 2021; S2 from 2 December 2019 to 10 October 2021.
Within each location both modality dates are monotonically ordered without repeats.
Nearest among the four S1 dates agrees with the S2 seasonal index for every item.
That does not make the paired acquisitions simultaneous.

Full-instant absolute gap: median **3.695 days**, p95 **19.305**, maximum **49.802**.
Dates are stored as timezone-naive nanosecond instants; this is not a calendar-day
matching rule and timezone interpretation must be documented in the loader.

| Max gap | All candidate pairs | Locations with any pair | Locations outside UK box |
|---|---:|---:|---:|
| 1 day | 168,172 | 119,020 | 117,393 |
| 3 days | 430,355 | 217,969 | 214,946 |
| 7 days | 701,525 | 242,977 | 239,328 |
| 14 days | 877,320 | 245,961 | 241,938 |

These are descriptive candidates, before other benchmark exclusions and accepted
geographic/time/buffer policy. A 7-day limit is feasible for a small pilot; it is
still proposed, not accepted.

### 3. Real clouds are available for a deliberately enriched pilot

80.93% of observations report exactly zero cloud fraction, confirming the
collection bias. However, **124,690 observations (12.66%)** report >5% cloud.
After a <=7-day gap and UK-box exclusion: **84,087 candidate pairs across 69,604
locations** remain above 5%. There are also 9,560 >80%-cloud candidate pairs
across 8,800 locations under those same filters.

Do not claim the real-cloud stratum is absent or that all degradation must be
synthetic. Select real-cloud candidates separately, verify mask class/polarity and
visually inspect actual imagery after approval. A pasted mask remains labelled
synthetic occlusion. Artificial sensor withholding must also retain its intervention
label: complete timestamp fields do not establish natural missing-sensor prevalence.

### 4. Footprints matter; geometry CRS differs from raster CRS

The footprint column is WKB in **EPSG:4326**. `crs` and `bounds` instead describe
native raster UTM grids (75 CRS codes); all native bounds are 2,640 x 2,640 m.
Reprojected geometry bounds agree to <5e-9 m; all geometries are valid and cover
their recorded centers. Do not transform geometry as if it were already in `crs`.

The conservative box [-11, 49, 2.5, 61.5] intersects **4,059 footprints**, versus
4,039 centers: centroid-only exclusion misses 20 items. Exploratory 5/10 km
buffering of a densified box in EPSG:3035 excludes 4,148/4,235 locations. Those
sensitivities are not a signed buffer policy.

### 5. Correct transport and latitude assumptions

The metadata maps **512 locations per full tar**, not 1,000. Train: 477 tars
(last 256); validation: 5 tars (last 128). Four dates per location means 2,048
observation pairs per full tar pair. The first paired validation tars total
2.35 GB according to the earlier API catalogue; their payloads remain unverified.
The 500-item ceiling still requires selective access or an explicitly approved
container overhead. Downloaded observations and retained one-pair items are
separate ledgers.

No locations occur south of 60 S, and only 114 occur north of 60 N. Replace the
earlier equal-six-band quota assumption with fixed feasible quotas before training.

## Next gates

1. Accept a pilot-only geography/time/gap/buffer contract, without accepting the UK
   benchmark split. An exploratory 2020-12-31 cutoff and 30-day embargo leave
   218,366 pre-cutoff and 135,643 post-embargo candidate locations outside the UK
   box at <=7 days. Locations occurring in both eras must still be assigned wholly
   to one geographic split; these counts are not a split.
2. Freeze <=500 one-pair-per-location items and explicitly separate real quality
   from interventions. No benchmark-excluded location may re-enter as a mask donor.
3. Prove bounded tar/ZIP member extraction using offline fixtures before asking
   for imagery permission; no member byte-offset index exists in the catalogue.
4. Inspect approved imagery, implement the local-only adapter, then run a controlled
   pilot. Do not use these descriptive counts as research accuracy results.

## Evidence and reproducibility

`metadata-audit.json` records counts, packages and input hashes. `overlap-example.json`
supports the split warning; `audit_metadata.py` and `fetch_metadata.py` preserve the
one-off scripts actually run (machine-local paths are intentional provenance).
Analysis environment: pyarrow 23.0.1, numpy 2.4.6, shapely 2.2.0, pyproj 3.7.2.
No cloud mask, spectral array, image alignment or downstream label was inspected.

`make check` passes locally (41 discovered tests, 6 torch modules skipped); all
51 optional JEPA tests pass in the CPU environment. The two new regression tests
cover refusal of unsupported quadrature settings and avoiding duplicate checkpoint
records when a runtime cap lands on/after a periodic checkpoint boundary. These
small review fixes are included alongside the evidence. The optional torch suite
is a local check; it is not part of the base CI job.

The user separately approved committing and pushing the metadata approval record,
audit and small P1 review fixes. This does not authorize imagery access, acceptance
of a training split, EO training, a cluster job or merging the pull request.
