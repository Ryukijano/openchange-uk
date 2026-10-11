# Data audit and next implementation handoff

Status: metadata/card audit, 11 October 2026. No imagery, Parquet tables or training
shards were downloaded. `configs/sources.yaml` remains empty. This plan does not
approve a source, accept benchmark splits, or authorize a cluster job.

## 1. What was checked

Provider cards, Hub repository metadata/file sizes, Dataset Viewer schema/size
responses and the SSL4EO provider decoder were inspected. The compact API record is
[audit.json](../../../reports/jepa/data-audit/audit.json). File sizes and advertised
LFS SHA-256 IDs are from the provider; payload checksums have **not** been verified.

| Corpus | Immutable revision | Use next |
|---|---|---|
| [SSL4EO-S12 v1.1 WebDataset](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1/tree/97a91733db962222e1e6b40a836845ddbbb16b86) | `97a91733db962222e1e6b40a836845ddbbb16b86` | Preferred paired S1GRD + S2L2A source; metadata first |
| [SSL4EO-S12 v1.1 legacy Zarr](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1-Zarr/tree/7f74b9a9b8bbb345e4ad7075649f37b58f0fcab8) | `7f74b9a9b8bbb345e4ad7075649f37b58f0fcab8` | Access fallback only, not an interchangeable decoder |
| [Copernicus-Pretrain](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain/tree/477d5dba02388539d491bc8d88175c51f0834534) | `477d5dba02388539d491bc8d88175c51f0834534` | Later location-pool expansion; not the first paired pilot |

All three cards declare CC BY 4.0. Source approval still requires attribution,
licence URL, revision, byte limit, intended split and human approver. The SSL4EO
[provider code](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1/tree/aa364b7a74f7da674b5dafde2db470aa8c06b3df)
declares Apache 2.0; that is separate from the dataset licence. No provider decoder
was copied into our implementation.

### SSL4EO: usable metadata, but a large transport unit

- 246,144 location rows: 243,968 provider-train and 2,176 provider-validation.
  Each location has four seasonal acquisitions. Location rows are not observation
  items. Do not confuse provider membership with our geographic/time contract.
- `train_metadata.parquet`: 37,358,291 bytes; `val_metadata.parquet`: 465,458 bytes.
  Together: **37,823,749 bytes**, about 37.8 MB (36.1 MiB).
- 21 fields: `tar`, `zarr`, `sample_id`, `split`, `center_lon`, `center_lat`, `crs`,
  `bounds`, `geometry`, four S2 times, four S1 times and four cloud-cover values.
  There are no advertised tar member byte offsets in this schema.
- S2L2A: 12 named bands, no B10, int16, 264 x 264; S1GRD: VV/VH, float16 in dB,
  on the optical grid. The source is not a uint16 SAR tensor. Preserve units/dtypes
  when packing. The paper says missing values were filtered and interpolated;
  absence of NaNs does not prove every pixel was originally observed.
- The current release stores one location's four dates in a Zarr ZIP member inside
  a tar, with separate tar files per modality. The provider decoder documents
  1,000 locations per full shard. The first validation S2L2A/S1GRD shard pair is
  **2,352,353,280 bytes**: roughly 4,000 observation pairs, not a 500-item slice.
- Legacy Zarr ZIP blocks contain 64 locations x 4 dates. The first validation
  S2L2A/S1GRD block pair is **292,306,366 bytes**. Even this loads unselected dates
  and locations; fetching a container is not approval to use every member.
- Legacy optical values include a +1000 offset; the current WebDataset removes it.
  The legacy card also records a repaired S1 date-order bug. Never mix normalization
  statistics, revisions or temporal indices across these formats.

The public card's example includes S1/S2 gaps of about 12 and 22 days. Four matching
seasonal indices are **not** a same-day pairing guarantee. Keep both timestamps and
reject pairs outside an approved maximum gap.

### Copernicus: defer until the pilot works

The pinned card reports 4,227,387 S1 and 4,218,065 S2 images, with S2 TOA containing
13 bands. Its 18,713,054 total includes S3, S5P and DEM; it is not 18.7M S1/S2 pairs.
The API lists raw GeoTIFF and WebDataset variants, filename catalogues and a
100-grid example. The first six example tars range from 451 MB to 1.01 GB; the
first six aligned full tars range from 4.90 to 6.40 GB. The examples are not a tiny
paired EO fixture. Spatially aligned does not establish temporal matching; retain
this source for expansion only after timestamp and overlap audits. Do not use the
grid embeddings as if they were raw paired imagery.

## 2. Next work, in order

### A. Metadata-only approval and curation

First request: approve only the two pinned SSL4EO metadata tables, with a **40 MB
payload ceiling**, `use: auxiliary`, `splits: [pilot]`, and manifest-only publication.
Give each exact file endpoint its own source entry/checksum. No imagery permission
is implied. Fetching code must call `require_approved()` before network access and
verify the pinned file size and SHA-256. Keep raw tables outside git.

Build an offline `scripts/curate_jepa_pilot.py` that takes local metadata plus an
explicit pilot contract and writes a manifest and rejection ledger. It must:

1. Validate keys, coordinates, CRS, footprint geometry, timestamps and cloud fields;
   quarantine malformed/ambiguous records, not silently infer them.
2. Group every date of a location together. Transform footprints into a common CRS
   for overlap checks; use geometry, not centroids alone. Exclude the conservative
   UK box from IMPLEMENTATION.md plus an approved metric buffer. Track other
   benchmark footprints separately; UK exclusion alone is not decontamination.
3. Join S1/S2 by location and explicit acquisition timestamps with no reuse of an
   acquisition in incompatible roles. Proposed first gap limit: **7 days**, pending
   approval. Report attrition at 1/3/7/14 days for feasibility; do not choose a new
   limit because training results look better.
4. Apply predeclared geographic blocks and a pilot-only temporal policy. Regions,
   cutoff, embargo and buffer must be explicit and signed before imagery fitting.
   This does not accept the separate OpenChange-UK benchmark proposal.
5. Select at most **one acquisition pair per location**, at most 500 items total.
   Hash ranking only selects within eligible geographic groups; it is not an image
   split. Report achievable clear/degraded coverage and counts after each exclusion.
6. Output deterministic IDs, dates, gap, footprint/CRS, source revision, provider
   membership, tar/member paths and expected bytes. Cloud thresholds remain
   provisional until mask encoding is verified. No source labels are training targets.

**Exit:** deterministic manifest, rejection counts and contract hash. Missing policy
or too few eligible pairs is a failed gate, not permission to widen the data silently.

### B. Bounded access proof before imagery approval

Build the extractor against local fake tar/ZIP fixtures first. Verify tar member
names and modality keys, reject duplicate/missing keys, unsafe paths and excessive
expanded sizes, and stop at the byte/item ceiling. Test HTTP byte ranges separately:
require valid 206/Content-Range and immutable file identity; refuse a server that
returns the entire file to a bounded range request. Pin a member index if needed.

A location member still holds four dates. Record **downloaded dates/locations**
separately from **retained one-pair items**. Any unavoidable extra imagery needs an
explicit overhead approval before transfer, even if it is discarded immediately.
Do not start a full-shard stream then call its first 500 examples a bounded download.
If selective access is impractical, price a legacy block fallback with its offset
contract or request a provider subset; do not automatically fetch either.

**Exit:** reviewed route, byte estimate and checksum strategy. Only then request
bounded imagery approval for the selected pilot manifest.

### C. Local decoder and real-data adapter

Implement `openchange.jepa.data` as a local-only reader, separate from fetching.
Its observation record includes location/observation IDs, revision, file hashes,
S1/S2 acquisition times and IDs, footprint/CRS, band order, physical units,
normalization version, pixel validity and cloud-mask semantics. Match modality keys
explicitly; never zip two independently shuffled streams or invent a withheld sensor.

Check mask class IDs/polarity and distinguish cloud, snow and nodata. SEnSeI masks
are model-derived quality estimates, not ground-truth clouds. Keep original
condition and every synthetic intervention/donor ID separate. Use aligned crop
geometry; fit normalization on pilot-train only. Fixed training donors never cross
into validation. Hand-check a fixed 16-item contact sheet and a range/shape report.

Add a **new explicit real-pilot config/entry path** only after adapter tests; do not
weaken `p1_synthetic.yaml` or its synthetic-only refusal checks. Validate source and
split/manifest hashes before reading arrays. Training workers have no network I/O.

**Exit:** decode, leakage and round-trip tests; verified units/offsets; committed
manifest and inspection evidence. No claim of downstream utility yet.

### D. Controlled local pilot, then human-run target-hardware smoke

Run the same tiny model/views on the accepted local pilot for plumbing and throughput.
Log per-stratum counts, unique locations, repeated items, data/decode/step times,
post-drop tokens, loss memory and non-finite events. Test restart. This pilot cannot
establish Q1 ranking, Q2 5,000-location calibration or Q3 scientific performance.

Only after those gates: prepare a human-reviewed one-GPU <=30-minute Isambard
smoke, measure GH200 I/O/memory/throughput, reprice the award ledger, and implement
evaluation adapters/BT/MAE controls before any scale grid. The agent never connects
to the cluster. EO remains a representation study, not an action-conditioned world model.

## 3. Approval boundary

Current decision needed: the **37.8 MB metadata-only fetch**. Imagery transfer,
pilot gap/split/buffer/mask policy and cluster work require later explicit decisions.
No array-level quality, pairing yield or remaining eligible corpus size has been
measured in this audit.
