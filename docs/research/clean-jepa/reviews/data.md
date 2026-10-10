# Clean-JEPA on EO: data corpus and I/O pipeline (data subagent report)

Date: 2026-10-10. Desk research only. Nothing downloaded except metadata parquets and one 195 MB Copernicus-Pretrain example shard (`example_100_grids/example_100_webdataset/example-000009.tar`), opened to check dtypes. "Measured" here means read from a file listing, a metadata table or that one sample. It never means a training or I/O benchmark.

## 0. What changes in the plan (short version)

1. **The "100%" data axis is about 1.07M S1/S2 patch locations, not 1B samples.** Copernicus-Pretrain has 1,067,267 S1/S2 patches in 247,723 grid cells, about 4 timestamps each, so 4.22M S2 images ([card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain)). A "1B-sample" run is about 240 passes over every S2 image, or about 940 per location. Plan v2's own deduplicated-grid-cell axis puts epochs on a separate axis, which helps, but the largest cell must be described as "about 1M unique locations seen many times", not "1B samples".
2. **The 1.1 MB/sample figure is roughly total repo storage divided by all 18.7M images of all eight modalities**, so it should not be used to price S2 reads. (The HF repo holds 22.67 TB, which counts both the raw GeoTIFF copy and the WebDataset copy.) Measured from the example shard, one S2 TOA patch time series is int16 [4, 13, 264, 264] = 7.25 MB, i.e. **1.81 MB per S2 image**. S1 is float32 [4, 2, 264, 264] = 2.23 MB, i.e. **0.56 MB per S1 image**.
3. **The <150 KB/sample target cannot be met losslessly at 264×264.** Packed at native resolution (10 m and 20 m bands, uint16), S2 10-band comes to 0.77 MB raw. Lossless compression gets it to about 0.38 MB (the 2.0x ratio measured from the HF listing, §3). With S1 added, the honest figure is **about 0.6 MB per S1+S2 image pair**. 150 KB is reachable only with 128×128 tiles (§3).
4. **Token dropping does not cut bytes read, and a patch-major layout does not help much.** LeJEPA's global views cover most of a 264² tile. Bytes per token are cut by drawing many views from each read (read once, make V views), not by partial reads. §5 gives the numbers.
5. **Copernicus-Pretrain's "4 timestamps" are not four seasons of one year.** In the sample I opened, the S2 dates are 2019-10-13, 2020-09-22, 2021-09-25, 2022-10-05: the same season in four years. Meanwhile S1 for the same patch is 2021-02/06/08/11. The download script searches ±30 days around 2022 equinox/solstice reference dates, then the same windows 1 and 2 years earlier ([GEE_download_ssl4eo_s2.py](https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Pretrain/data_collection/GEE_download_ssl4eo_s2.py)). S1 and S2 in Copernicus-Pretrain are aligned in space, not in time. The Q3 "same-scene" S1↔S2 pairs and the temporal-positive row therefore have to come from SSL4EO-S12 v1.1. There the median |S1−S2| gap is 3 days (99th percentile 25 days, max 49), measured from `train_metadata.parquet`.
6. **SSL4EO-S12 overlaps Copernicus-Pretrain in space.** Copernicus-Pretrain's 1M S1/S2 locations include the 251K SSL4EO-S12 locations ([data_collection README](https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-Pretrain/data_collection)). Training on the union counts those places twice.
7. **The corpora are mostly city-centred and cloud-filtered at collection time.** This hurts M (availability-stratified SIGReg) and Q3. Copernicus-Pretrain samples around the 10K largest cities (σ = 50 km) and includes the SSL4EO-S12 locations. SSL4EO-S12 v1.1 keeps scenes under 10% cloud: 80.9% of its S2 images have SEnSeI cloud fraction exactly 0 and only 5.2% exceed 0.3. Major TOM keeps cloud_cover ≤ 25 (max in metadata 24.9997). The "S2 partly cloudy" and "S1 only" strata barely exist in the data. They have to be **synthesised** (drop S2; paste a real cloud mask from another sample). The plan has to say this, or the claims about "real cloud masks" do not hold.
8. **OpenChange-UK regions and cutoff are not set** (repo `reports/SPEC.md`, `docs/PLAN.md`: "Region names and the time cutoff are still unset"). The holdout below therefore removes all UK land plus a buffer from pretraining. That keeps it valid whatever regions are frozen later.
9. **The compute numbers disagree.** OpenChange-UK SPEC.md says "Project u6xn, Isambard-AI Phase 2, 5000 NHR through 15 Mar 2027". Plan v2 budgets against 15,000. Resolve this before pricing anything.
10. **The leakage-safe date cutoff depends on which corpus is used.** Copernicus-Pretrain S2 runs to early 2023 (polar windows around 2023-03-20). SSL4EO-S12 v1.1 ends 2021-10-10. Major TOM Core runs to 2024 (468 S2 products in 2024). OpenChange-UK's future window must start after the latest acquisition in the chosen corpus. I recommend a cutoff of 2025-01-01, with pretraining kept strictly before it.

## 1. Corpus fact sheets

All counts come from the dataset card or paper unless marked "measured" (from metadata or file listings I read today). Sizes come from the HF tree API and are decimal TB.

### 1.1 Copernicus-Pretrain (SSL4EO-S)
- **Source:** [HF card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain), revision `477d5dba`; [paper 2503.11849](https://arxiv.org/abs/2503.11849); [code](https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-Pretrain).
- **Grid:** ERA5 0.25°×0.25° cells. 312,567 cells have at least one modality ("union", ~310K); 219,543 have all eight ("joint", ~220K) (paper Table 7).
- **S1/S2:** 247,723 cells, 1,067,267 patches, ~4 timestamps, 4,227,387 S1 and 4,218,065 S2 images (card). The joint subset has 996,978 patches and 3,948,217 images per sensor. Per the paper, cells hold about 4 local S1/S2 time series on average. Patches are placed by Gaussian sampling (σ 50 km) around the top 10K cities, plus 40K uniform polar locations, plus 1–2 patches to fill empty cells.
- **S2:** GEE `COPERNICUS/S2`, i.e. L1C TOA, all 13 bands resampled to 10 m, 264×264. Stored as **int16** (measured: torch ShortStorage, 7,248,384 B for [4,13,264,264]). The default scene filter is `CLOUDY_PIXEL_PERCENTAGE ≤ 20` (script `--cloud-pct 20`). **No per-pixel cloud mask is shipped** for S2 (UNVERIFIED: none appears in the README file structure).
- **S1:** GEE `COPERNICUS/S1_GRD`, VV+VH, 10 m, stored as **float32** (measured). GEE's S1_GRD values are log-scaled dB ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)); the script only calls `toFloat()`. Instrument mode filter (IW vs EW at the poles): UNVERIFIED.
- **Auxiliaries:** S3 OLCI 96×96×21 (~8 timestamps, 2021); S5P CO/NO2/SO2/O3 monthly means 28×28 at 1 km (2021); Copernicus DEM GLO-30 960×960 per cell. **No ERA5 or land cover is included.** The cells are ERA5-aligned, so ERA5 can be joined by cell id. Plan v2 says "DEM and ERA5 alignment are already in Copernicus-Pretrain"; only DEM is, and ERA5 is a join.
- **Dates:** "around anchor year 2021" (paper). In practice S2 windows fall in 2020–2022 (or 2020–2023 for polar), as in the sample above.
- **Format:** two copies. (a) `raw_geotiffs_*`: `tar.zst` chunks of 1k grids per modality, chunk ids aligned across modalities, 6.30 TB joint + 0.58 TB union-only (measured). Per modality, joint: s2 3.524 TB, s1 1.947 TB, dem 0.605 TB, s3 0.204 TB, s5p ~0.02 TB. (b) `ssl4eo_s_*` WebDataset: 2,255 + 955 tars, 11.66 + 2.06 TB, average 5.2 GB per tar, one `.pth` per modality per grid plus `.json`. Repo `usedStorage` is 22.67 TB.
- **Licence:** "CC-BY-4.0." (card). Code is Apache-2.0.
- **Known problems:** 4 timestamps per patch but spread across years; "most" patches have 4 timestamps and "most" S1/S2 are paired (data_collection README; histograms in paper Figs. 8–11, numbers UNVERIFIED); SSL4EO-S12 locations are included; the README says the collection scripts are not yet cleaned. Patches within a cell can overlap (UNVERIFIED for this corpus; measured for SSL4EO below).
- **Download:** `hf download wangyi111/Copernicus-Pretrain --include "raw_geotiffs_220k_aligned/s2_toa_mix_chunk*"` etc. The fnames json files (125–151 MB) list every file.

### 1.2 SSL4EO-S12 v1.1
- **Source:** [HF card](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1), revision `97a91733`; [report 2503.00168](https://arxiv.org/abs/2503.00168); [GitHub](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1). A Zarr version is at [embed2scale/SSL4EO-S12-v1.1-Zarr](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1-Zarr) (revision `7f74b9a9`, 19,234 files, CC-BY-4.0).
- **Size:** 246,144 locations × 4 timestamps = 984,576 samples. Measured: 243,968 train + 2,176 val unique `sample_id`. The val split follows TerraMesh's 99/1 partition "to avoid spatial overlap".
- **Modalities:** S2L1C (13 bands), S2L2A (12 bands), S2RGB, S1GRD (VV, VH), NDVI, LULC (ESRI, augmented with cloud and snow masks), Copernicus DEM. 264×264 at 10 m, native UTM. The S2 cloud mask is **SEnSeI v2** (uint8 per pixel per timestamp, in the L2A zarr), not SCL. S1 is dB (−50 to +1 typical, per the report); the storage dtype is UNVERIFIED (I believe float16, but did not confirm). S2 is DN 0–10k.
- **Dates (measured):** S2 from 2019-12-02 to 2021-10-10 (2019: 3.7k, 2020: 622k, 2021: 350k images). Four timestamps "from different seasons", selected 2019–2021 with cloud < 10%. The loader's `reindex_seasonal=True` sorts by quarter.
- **S1↔S2 alignment (measured):** |Δt| median 3 days, p90 14, p99 25, max 49. An S1 time-ordering problem is reported and still open ([issue #4](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1/issues/4), opened 2025-04-02), so P0 must check S1 date order against `S1_time_i`.
- **Cloud (measured, `cloud_cover_i`, SEnSeI fraction):** 80.9% exactly 0; 2.7% in (0, 0.01]; 3.7% in (0.01, 0.05]; 2.5% in (0.05, 0.1]; 5.0% in (0.1, 0.3]; 5.2% in (0.3, 1]. 38.6% of locations have at least one timestamp > 0.05.
- **Latitude (measured):** 36.1% in 15–35°N, 37.3% in 35–50°N, 5.9% in 50–60°N, 0.05% north of 60°N, 6.0% in 15°S–15°N, 12.9% in 15–35°S, 1.7% in 35–60°S. Strongly mid-latitude and city-biased.
- **Duplicates (measured):** nearest-neighbour centre distance median 3.9 km. **28% of samples have another sample within 2.64 km** (the patch width), so their footprints overlap.
- **Size on disk (measured):** S2L2A 0.903 TB, S2L1C 0.882 TB, S1GRD 0.180 TB, NDVI 0.094, S2RGB 0.166, DEM 0.009, LULC 0.009. 477 train tars per modality, about 512 samples each (UNVERIFIED, from 243,968/477). The card says "2.3TB"; repo usedStorage is 4.84 TB, which includes history.
- **Licence:** CC-BY-4.0 (card YAML).

### 1.3 Major TOM Core (S2L2A, S2L1C, S1RTC)
- **Source:** [Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A) (rev `b7612203`), [Core-S2L1C](https://huggingface.co/datasets/Major-TOM/Core-S2L1C) (`aeea03d5`), [Core-S1RTC](https://huggingface.co/datasets/Major-TOM/Core-S1RTC) (`93185d1e`); [paper 2402.12095](https://arxiv.org/abs/2402.12095).
- **Grid:** Major TOM points at 10 km nominal spacing. Rows are equal-distance in latitude; columns are recomputed per row. Ids look like `922D_249L`. A patch is 1,068×1,068 at 10 m (10.68 km), held in its own UTM CRS, so neighbouring cells "overlap slightly" (paper §3.1).
- **Counts (measured from `metadata.parquet`):** S2L2A has 2,245,886 rows over 2,238,689 unique cells (7,197 repeat rows). S1RTC has 1,469,955 rows over 1,458,481 cells, **all of which are also S2 cells**.
- **Temporal depth:** 1 per cell ("global monotemporal"). Years (measured): S2 2016–2024, mostly 2019–2023. S1 2015–2024. **S1 and S2 are separate products with unrelated dates**, matched by cell only.
- **Bands:** S2L2A 12 bands at native resolution (B01/B09 at 60 m, the 20 m bands at 20 m) plus the SEnSeI cloud mask at 10 m; L1C adds B10. S1RTC is VV/VH in **linear power**. Each band is a binary GeoTIFF blob inside parquet rows (about 500 rows per parquet); 4,492 (S2) and 2,930 (S1) parquet files.
- **Cloud (measured, `cloud_cover` in %):** filtered to < 25. 53.9% are 0; 18.8% in (10, 25].
- **Latitude (measured):** close to uniform over land. 25.8% in 15°S–15°N, 11.7% north of 60°N, 4.5% south of 60°S.
- **Size (measured):** S2L2A 22.96 TB, S2L1C 23.55 TB, S1RTC 15.69 TB.
- **UK:** 8,174 S2 cells inside a coarse UK bounding box (49.8–60.9°N, 8.7°W–1.8°E). That box includes some Ireland and France, so this is an upper bound.
- **Licence:** `cc-by-sa-4.0` (card YAML). This is ShareAlike: derived datasets (packed shards, embeddings) carry the same licence if redistributed.

### 1.4 MMEarth
- [MMEarth-data README](https://github.com/vishalned/MMEarth-data), [paper 2405.02771](https://arxiv.org/abs/2405.02771). 1.2M tiles at 128×128 (597 GB) or 64×64 (152 GB), in a single HDF5. Sentinel-2, Sentinel-1, ERA5 (temperature and precipitation), ASTER GDEM, Dynamic World, canopy height and ESA WorldCover. Tiles are sampled by RESOLVE biomes and ecoregions. Licence: "CC BY 4.0" (LICENSE-data). Duplicates were removed in v001. Monotemporal. S2 level, S1 orbit handling and the year range are UNVERIFIED (not read today). It is the only candidate with biome-stratified sampling and ERA5/WorldCover built in, which suits the Q3 probe variables. 128² tiles fit the 150 KB target (§3).

### 1.5 SatlasPretrain
- [SatlasPretrain.md](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md). Web-Mercator zoom-13 tiles, 512×512 PNG per band per tile per image; Sentinel-2 (L1C product names), Sentinel-1, Landsat, NAIP. Labels are ODC-BY; Sentinel imagery is under the Copernicus legal notice. Images are 8-bit PNGs (for `tci`; the bit depth of other bands is UNVERIFIED). Total aligned S1/S2 sample count: UNVERIFIED. **Not recommended:** Web-Mercator distortion at high latitude, PNG quantisation, and a label-centred US/global mix.

### 1.6 Others at > 1M aligned S1/S2
- **TerraMesh** (TerraMind's corpus): about 9M S1+S2 samples at 264×264, built from Major TOM Core and SSL4EO-S12 v1.1, subsampled by ecoregion and LULC, with SEnSeI v2 masks ([TerraMind 2504.11171](https://arxiv.org/abs/2504.11171) §8). It is a curated re-packing of 1.2 and 1.3, not new data. Public release and licence are UNVERIFIED today.
- **AlphaEarth's training sites** are published, but the imagery is not released as a dataset ([2507.22291](https://arxiv.org/abs/2507.22291) S16.1).

## 2. Recommendation: pretraining corpus and data axis

**Primary corpus: Copernicus-Pretrain S1+S2 raw GeoTIFFs (joint plus union), plus SSL4EO-S12 v1.1 for time-matched pairs, deduplicated on one grid.** Optionally add Major TOM S2L2A for high-latitude and tropical cells at the 100% point.

Reasons. It is CC-BY-4.0, which avoids ShareAlike on derived shards. It is the corpus behind Copernicus-FM and Copernicus-Bench, so frozen baselines and the benchmark share a data lineage. Its ERA5 cell id gives a ready strata key. The raw download is 6.3–6.9 TB, not 23 TB.

What it costs: L1C only (no L2A, no cloud mask), and the city bias. Major TOM fixes the latitude bias but is CC-BY-SA, single-date and mismatched in date between S1 and S2. Add it as a separate, labelled arm only if the 100% point turns out data-limited.

**Unique-location definition.** Use one key everywhere: the **Major TOM 10 km grid cell id** of the patch centre. Major TOM is a published deterministic grid, works on any corpus from (lat, lon), and is close to equal-area. The ERA5 0.25° cell (about 28×28 km at the equator) is kept as the strata unit and holdout unit. For every patch from every corpus, record `mt_cell = majortom_cell(lat, lon)`, `era5_cell`, `utm_footprint`.

**Dedupe rule:**
1. Assign every patch to `mt_cell`.
2. Within a cell, keep patches whose footprints overlap by less than 25% in IoU. Same-location patches from different dates are kept as extra timestamps of one location, not as new locations.
3. Across corpora, SSL4EO and Copernicus-Pretrain patches in the same cell with footprint IoU > 0.25 are merged into one location, and their timestamps are pooled.

The data axis counts `mt_cell`s.

Expected counts are estimates. 1.07M Copernicus patches; 28% of SSL4EO patches have a neighbour within 2.64 km, and Copernicus-Pretrain uses the same Gaussian city sampling. A 10 km cell holds about 14 non-overlapping 264² footprints. So I expect about 0.5–0.8M unique 10 km cells. Count this exactly in P0 from `fnames_union_310k.json.gz`, which has every file path with coordinates.

**Stratified subsampling (1% / 10% / 100%).** Strata are the RESOLVE ecoregion biome (14 classes) × latitude band (6 bands: <−35, −35..−15, −15..15, 15..35, 35..50, >50) × S1 availability. The 1% and 10% subsets sample ERA5 cells (not patches) within each stratum, proportionally but with a floor of max(1% share, 20 cells) per non-empty stratum. All patches of a selected ERA5 cell are taken. That keeps neighbouring patches on the same side and stops 1% becoming Europe-plus-China. Subsets are nested (1% ⊂ 10% ⊂ 100%), using a fixed hash of `era5_cell`.

| Axis | ERA5 cells | Unique 10 km cells (est.) | S1/S2 patches | S2 images (≈4/patch) |
|---|---|---|---|---|
| 1% | ≈2.5k | ≈5–8k | ≈10.7k | ≈42k |
| 10% | ≈25k | ≈50–80k | ≈107k | ≈422k |
| 100% | 247,723 | ≈0.5–0.8M | 1,067,267 | 4,218,065 |

The cell and patch columns scale card numbers; the 10 km column is an estimate. Adding Major TOM S2L2A at 100% adds up to 2.24M 10 km cells, but at one date each, so "100%" would no longer mean the same thing as in the subsets. Keep it as a separate "+MajorTOM" point.

## 3. Sample representation and bytes

**Dynamic range.**
- S2 L1C/L2A DN is reflectance × 10,000 (the +1,000 offset introduced with processing baseline 04.00 and whether the GEE collection used here is harmonised: UNVERIFIED). Values fit in uint16 without loss. Clamp to [0, 65535], store raw, and keep the offset in metadata.
- S1 dB: SSL4EO-S12 documents a typical range of −50 to +1 dB. Mapping: \(q = \mathrm{round}((\mathrm{dB}+50)\cdot 1000)\) clipped to [0, 65535], which covers −50 to +15.5 dB at 0.001 dB steps. That step is far below speckle (about 1 dB for single-look GRD at 10 m, my estimate) and below float16 precision near −30 dB (0.0156 dB), if the source is float16. So it is effectively lossless relative to the source. Major TOM S1RTC is linear: convert with \(10\log_{10}\) first and send 0 and NaN to q=0 with a nodata flag.

**Bytes per image, uint16, before compression.** A 264² band is 69,696 px; 20 m is 132² = 17,424; 60 m is 44² = 1,936.

| Content | All bands upsampled to 10 m | Native resolution |
|---|---|---|
| S2 4×10 m bands | 557,568 B | 557,568 B |
| + 6×20 m (B5, B6, B7, B8A, B11, B12) = **S2 10-band** | 1,393,920 B (1.39 MB) | 766,656 B (0.77 MB) |
| + 3×60 m (B1, B9, B10) = 13-band | 1,812,096 B (1.81 MB; matches the measured Copernicus-Pretrain int16) | 778,272 B (0.78 MB) |
| S1 VV+VH | 278,784 B | 278,784 B |
| Cloud mask (SEnSeI, uint8 / 2-bit) | 69,696 / 17,424 B | same |
| **S2 10-band + S1 + mask (uint8)** | 1.74 MB | **1.12 MB** |

**Compression.** End-to-end ratio from the file listing: Copernicus-Pretrain joint S2 GeoTIFF `tar.zst` is 3.524 TB for 3,948,217 images, i.e. 0.89 MB per image against 1.81 MB as int16, **2.0x**. Whether the GeoTIFFs inside are themselves compressed is UNVERIFIED, so treat this as the ratio for the source format, not as zstd on raw uint16. S1 is 1.947 TB for 3.95M images, 0.49 MB against 0.56 MB float32, 1.14x, because float noise does not compress. On uint16 dB, S1 should compress better (UNVERIFIED; measure in P4).

**Honest bytes per sample, losslessly:** S2 10-band native about 0.38 MB, plus S1 about 0.2 MB, plus mask about 0.01 MB, **≈ 0.6 MB per S1+S2 image pair**. Without the 60 m bands that changes by less than 2%, so dropping them barely matters for bytes. Upsampling 20 m bands to 10 m before storing nearly doubles S2 bytes and should not be done. Upsample on the GPU.

**The <150 KB target is not reachable losslessly at 264².** It can be reached by:
- (a) 128² tiles, i.e. four non-overlapping 128² crops per 264² patch, about 0.14 MB per tile lossless, with the dedupe rule applied per tile;
- (b) lossy coding, which should be rejected for a scaling paper unless ablated;
- (c) S2 only with 10 m bands at native size (0.56 MB raw, about 0.28 MB compressed).

Recommendation: store 264² native-resolution uint16 with zstd level 3–9 (or blosc-zstd with byte shuffle), and **budget 0.6 MB per pair**. Plan v2 used 1.1 MB, so its I/O cost was overstated by about 1.8x, but its target was optimistic by 4x.

## 4. Shard format and read path on Isambard-AI

**Facts** ([storage docs](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)): Phase 2 $PROJECTDIR is 200 TiB and 50,000,000 inodes; $SCRATCHDIR is 5 TiB and 1,000,000 inodes; compute-node $LOCALDIR is 48 GiB, "typically implemented as a tmpfs RAM disk". Parallel areas are Lustre or VAST. Not backed up; retention runs to project end plus a 30-day grace period. The page says both "60 days since access" and "Project End Date" for scratch, so treat it as unresolved. The [SquashFS tutorial](https://docs.isambard.ac.uk/user-documentation/tutorials/datasets-in-squashfs/) says parallel filesystems are "optimised for large sequential reads and writes" and do poorly with many small files. **No published per-node or aggregate bandwidth figure found: UNVERIFIED.** The plan's 5 GB/s per node is an assumption.

| Format | Random access | Files for 4.2M pairs | Sequential | Resumable | Verdict |
|---|---|---|---|---|---|
| WebDataset tar (uint16 npy + json per sample) | none inside a shard; shard-level shuffle plus a buffer | ≈1,250 at 2 GB | best (one stream per worker) | resume by (epoch, shard index, offset); `wids` gives indexed access | **recommended for training** |
| numpy memmap (one big array per field, fixed-size records) | O(1) per sample | ≈10–100 | good if read in contiguous blocks; random 0.6 MB reads go through Lustre RPCs (≥1 MB) | trivial (index) | **recommended for eval and the pilot**; fixed size only fits 264² tiles |
| zarr v2/v3 (chunk = 1 sample or 1 tile) | yes | 1 file per chunk: 4.2M+ inodes unless zarr v3 sharding / zip store | poor with small chunks | yes | only with sharding; SSL4EO's zarr.zip per sample inside tar is the worst of both |
| parquet/arrow (Major TOM style) | row-group level (~500 rows) | ≈4.5k | good | yes | fine for ingest from Major TOM, not for training |
| SquashFS image of any of the above | as the inner format | 1 | good; docs recommend it for many-file datasets | — | wrap the eval memmaps / small benchmarks |

**Design.**
- WebDataset shards of **1–4 GB** (≈2,000–6,000 pairs at 0.6 MB), one shard ⊂ one data-axis split, sample order shuffled at build time.
- Shards grouped by `era5_cell` hash so that 1% ⊂ 10% ⊂ 100% are shard lists, not filters.
- Lustre stripe count about 4–8 for the shard directory (`lfs setstripe -c 4 -S 4M`; values UNVERIFIED for this site, ask BriCS).
- Total files about 1–3k, far under 50M inodes.
- Each record holds `s2.u16` (native-res bands concatenated, or three arrays by resolution), `s1.u16`, `cloud.u8`, `meta.json` (ids, dates, strata, licence, source revision), and optional `aux.npz` (DEM crop, WorldCover fractions).

**Throughput needed (estimate).** ViT-B, LeJEPA multi-crop with 2×224² global and 6×96² local views at patch 16 is 2×196 + 6×36 = 608 tokens per image. Training FLOPs ≈ 6 × 86M × 608 ≈ 0.31 TFLOP per image. At about 160 TFLOP/s effective per GH200 (≈40% MFU of dense bf16; assumption), that is ≈ 510 images/s/GPU. At 0.6 MB, that means **≈0.3 GB/s per GPU, 1.2 GB/s per node**. ViT-S at ρ=0.9 is about 40x fewer FLOPs per image, so **≈12 GB/s per node, which is probably not feedable**. The I/O wall is at small models with heavy token dropping, not at 1B. For small models: (i) the 1% subset (≈42k images × 0.6 MB ≈ 25 GB) fits in $LOCALDIR (48 GiB) and should be staged there per job; (ii) take more views per read (§5).

**Throughput test plan for the first 30-minute 1-GPU smoke** (human-run; Devin prepares scripts only):
1. `dd`/`fio`-style sequential read of one 2 GB shard, cold then warm, from $PROJECTDIR, $SCRATCHDIR and $LOCALDIR (copy first).
2. WebDataset loader with 0, 8, 16, 32 workers on 72 Grace cores, decoding zstd, no GPU, then images/s and GB/s.
3. The same loader feeding a ViT-S and a ViT-B forward/backward at ρ ∈ {0, 0.75, 0.9}, to get GPU-util and step time.
4. Random-access memmap read of 10k samples.
5. Record node, stripe settings, time of day, and the `lfs getstripe` output.

Pass rule: the loader sustains at least 1.5× the GPU's consumption at ViT-B ρ=0 on one GPU. Extrapolating to 4 GPUs and multi-node needs a second smoke, because shared-filesystem contention does not show at 1 GPU.

**Storage ledger (estimate, decimal TB; 200 TiB ≈ 220 TB).**

| Item | Size |
|---|---|
| Raw Copernicus-Pretrain S1+S2+DEM GeoTIFF tar.zst (joint + union) | 6.9 |
| Raw SSL4EO-S12 v1.1 S2L2A + S2L1C + S1GRD + LULC + DEM | 2.0 |
| (optional) Raw Major TOM S2L2A + S1RTC | 38.6 |
| Packed shards, Copernicus + SSL4EO (≈5.2M pairs × 0.6 MB) | 3.1 |
| (optional) Packed Major TOM in 264² tiles (2.24M × ~16 tiles × 0.4 MB S2, + S1 1.46M × 16 × 0.2 MB) | ≈19 |
| Checkpoints: 20 per run kept for Q1. Full state (fp32 + Adam, 16 B/param) for the last only, bf16 weights (2 B/param) for the rest. S: 22M, B: 86M, L: 300M, 1B. ≈45–60 runs | ≈1.5 |
| Eval embeddings: pooled (1024-d bf16) for ≈300k eval images × ≈1,200 checkpoints | ≈0.7 |
| Patch-level embeddings: do not store; recompute per probe | 0 |
| Eval datasets (Copernicus-Bench 0.155 TB repo, PANGAEA subset, OpenChange-UK) | ≈0.5 |
| **Total without Major TOM / with Major TOM** | **≈15 / ≈72** |

Storage is not the limit. Delete raw files after packing once checksums are recorded, unless redistribution rules say otherwise.

## 5. Pre-tokenisation vs on the fly

**Can be precomputed once** (model-independent): band selection and order; uint16 packing; native-res arrays; S1 dB quantisation; per-band mean and std (from the 1% subset only, frozen); cloud masks (SEnSeI from SSL4EO/Major TOM; for Copernicus-Pretrain L1C, run s2cloudless or SEnSeI v2 once, per 2.6); per-pixel valid mask; strata labels (biome, latitude band, availability, cloud bin); dedup ids (`mt_cell`, `era5_cell`, footprint hash); holdout flags; date; source revision; licence; sha256.

**Cannot be precomputed:** patch embedding (depends on the model, and on patch size if varied); crops and views (random per step); normalisation if it is learned; token masks if they are resampled each step, which the LeJEPA recipe needs.

**Does a patch-major layout save bytes?** It can store each 264² image as 16×16-px patches contiguously, [n_patch=16.5²≈272, bands, 16, 16]. One 16² patch of 10 bands at 10 m is 5,120 B uncompressed. At ρ=0.9 a 196-token view keeps about 20 patches, i.e. about 100 KB of a 1.39 MB image. But:
- (i) LeJEPA uses ≥2 global views of 224² from a 264² tile, which together cover 72–100% of the tile. Kept tokens of different views are spread across the tile, so the union of kept patches over all views is most of the tile. If each of two global views keeps a random 10% of its tokens, the union touches about \(1-(1-0.1)^2 \approx 19\%\) of the tile's patches; with 6 local views added, about 25–40% (estimate).
- (ii) 5 KB reads are far below Lustre's efficient request size (≈1 MB), so partial reads only pay off once the record is in node memory anyway.

**Honest saving: at most about 2–3x in bytes**, and only with block-structured masking where the dropped unit is a 64×64-px block (82 KB for 10 bands), not single tokens. That changes the augmentation, so it becomes a confound for the token-dropping ablation.

**What I recommend instead:**
- (a) store at native resolution (1.8x saving, §3);
- (b) zstd (2x);
- (c) **draw many views per read**: V views per image read, each with its own mask. Bytes per *visible token* fall by V. With LeJEPA's V≈8, a run of N views reads N/8 images;
- (d) stage the 1% and 10% subsets into node RAM ($LOCALDIR 48 GiB holds 1%; host memory per node on Isambard: UNVERIFIED).

So: report "images read" and "views seen" as two separate quantities in the paper.

## 6. Availability strata

**Operational definitions per (location, timestamp)**, with c = SEnSeI cloud plus cloud-shadow fraction inside the crop:
- m1 *S2 clear*: S2 present, c ≤ 0.01.
- m2 *S2 partly cloudy*: bins (0.01, 0.1], (0.1, 0.3], (0.3, 0.7].
- m3 *S1 only*: S2 absent or c > 0.7, S1 present.
- m4 *both*: S2 with c ≤ 0.3 and S1 within ±7 days.

**Frequencies in the data (measured, SSL4EO-S12 v1.1, per S2 image):**
- c=0: 80.9%; (0, 0.01]: 2.7%; so m1 ≈ 83.6%.
- (0.01, 0.1]: 6.2%; (0.1, 0.3]: 5.0%; > 0.3: 5.2%.
- S1 within ±7 days: 72.5% (measured).

Mean cloud fraction by latitude band is flat, 0.040–0.070, highest at 15°S–15°N (0.070), because of the < 10% selection filter. Copernicus-Pretrain ships no per-pixel masks (UNVERIFIED: none in its README layout); its scene filter is ≤ 20%. Major TOM is < 25% (scene-level, % units); per-pixel SEnSeI masks exist but I did not read the distribution. The real-world frequency of cloudy UK scenes is therefore **not represented** in any of these corpora (UNVERIFIED as a number, but follows from the filters).

**Consequence.** m2 and m3 have to be **constructed**:
- (i) *mask transplant*: apply the SEnSeI mask of a random other S2 image of the same biome and season to the S2 input (zero or mark the pixels; keep c from the transplanted mask);
- (ii) *sensor drop*: drop S2 to make m3, drop S1 to make S2-only.

Real cloudy samples come only from the Major TOM (10, 25]% tail (18.8% of its rows) and SSL4EO's > 0.1 tail. Use them as a held-out "real cloud" test of M, not as the training source.

**≥256 per stratum per step.** The global batch is drawn as K strata × 256 slots. A sampler takes clean records from the shard stream and *assigns* each to a stratum, by applying the transform before the encoder. Every stratum is filled by construction. The assignment is logged in the batch so SIGReg is computed per stratum. With 4 strata the minimum global batch is 1,024 images. That fits ViT-S and B on a node; for L/1B it needs ≥2 nodes or gradient accumulation with SIGReg over the accumulated set (an all-gather of embeddings, not of gradients).

## 7. Leakage protection

**Benchmarks: footprint, dates, overlap with pretraining.** Copernicus-Bench task details are from its [card](https://huggingface.co/datasets/wangyi111/Copernicus-Bench).

| Benchmark | Footprint | Acquisition dates | Overlap risk with Copernicus-Pretrain |
|---|---|---|---|
| EuroSAT-S2 / -S1 | European cities ([EuroSAT 1709.00029](https://arxiv.org/abs/1709.00029); "34 countries" from the paper body, UNVERIFIED today); S1 version re-downloaded by the Copernicus-FM group | S2 ~2017 (UNVERIFIED) | **high spatially** (city-centred sampling on both sides), low temporally |
| BigEarthNet v2 (reBEN) S1/S2 | 10 European countries (from BigEarthNet; UNVERIFIED today) | 2017-06 to 2018-05 (UNVERIFIED); CLC 2018 labels ([2407.03653](https://arxiv.org/abs/2407.03653)) | high spatially in Europe; different years |
| DFC2020 S1/S2 | SEN12MS-based scenes on all inhabited continents ([1906.07789](https://arxiv.org/abs/1906.07789)); test-scene list UNVERIFIED | 2016–2017 (UNVERIFIED) | medium |
| LCZ-S2 (So2Sat LCZ42) | 42 urban agglomerations plus 10 smaller areas ([1912.12171](https://arxiv.org/abs/1912.12171)) | ~2017 (UNVERIFIED) | **very high**: both sample cities |
| Flood-S1 (Kuro Siwo) | 43 flood events globally ([2311.12056](https://arxiv.org/abs/2311.12056)) | 2015–2022 (UNVERIFIED) | medium; 2020–22 events may share dates |
| Sen1Floods11 (PANGAEA) | 11 flood events, chips per event ([README](https://github.com/cloudtostreet/Sen1Floods11)) | event dates in its geojson, 2016–2019 (UNVERIFIED) | low–medium |
| Cloud-S2 (CloudSEN12) | global | 2018–2020 (UNVERIFIED) | medium |
| Cloud-S3, LC100-S3, Biomass-S3, AQ-S5P | built by the Copernicus-FM authors ("new"), probably on the same ERA5 grid and 2021 | 2021 (UNVERIFIED) | **potentially direct**: same group, same grid, same year. Check grid ids before use; these are not in our S1/S2 headline tasks anyway |
| OpenChange-UK | UK coast; regions and cutoff unset | future window after cutoff | UK patches exist in all corpora (SSL4EO 3,448 in a coarse UK box; Major TOM 8,174) |

**Exclusion rule.** For every benchmark *test and validation* sample with coordinates, compute its footprint polygon. Drop any pretraining patch whose footprint is within **B = max(2 km, one patch width)** of it. Use 2.64 km for 264² patches, and add 5 km for city-scale tasks (LCZ, EuroSAT), because spatial autocorrelation of land cover reaches past one patch. Implementation: precompute the set of `mt_cell`s whose 10 km cell intersects any buffered test footprint. Exclude at cell level (coarser and safer than footprint level), and log the number of pretraining patches removed per benchmark. Benchmarks without coordinates in the released files (check EuroSAT-S1 and the S3/S5P tasks) cannot be filtered spatially. Report them as "overlap unknown", or drop them from the headline set.

**Temporal rule.** Leakage on single-date benchmarks is mostly spatial (memorising a place), so the date rule only matters for change and flood tasks. For those, drop pretraining images within ±30 days of any test acquisition at overlapping cells.

**UK holdout (recommended).** Exclude **all of Great Britain and Northern Ireland plus a 25 km buffer** from pretraining. Use the Natural Earth admin-0 polygon for GBR, buffered in EPSG:27700, and exclude every `mt_cell` whose 10 km cell intersects it. Cost: under about 1.4% of SSL4EO locations and about 0.4% of Major TOM cells. Since OpenChange-UK regions are not frozen, a whole-UK exclusion is the only rule that cannot be invalidated by a later split choice. Rejected alternative: exclude only the frozen test regions. Mark it PROPOSED (OpenChange-UK rule: splits stay PROPOSED until human review). It also keeps the "UK is OOD" claim honest for Q1–Q3.

**Temporal cutoff for OpenChange-UK future dates.**
- Latest acquisitions: SSL4EO-S12 v1.1 2021-10-10 (measured); Copernicus-Pretrain about 2023-04 (from script windows; measure in P0 from fnames json); Major TOM 2024 (measured, 468 S2 products).
- Rule: OpenChange-UK future window starts **≥ 2025-01-01**, and every pretraining image dated ≥ 2025-01-01 is dropped (none exist today).
- The UK exclusion above removes UK pixels of every date, so the future window is protected twice over.

## 8. P0 pilot slice

**Corpus:** SSL4EO-S12 v1.1 (CC-BY-4.0; time-matched S1/S2; SEnSeI masks; per-sample zarr.zip in the Zarr repo). That repo has 19,234 files; whether they are per-sample or chunked is UNVERIFIED and must be checked before selecting. If they are not per-sample, use the 5 val shards (2,176 samples; S2L2A + S1GRD + S2L1C ≈ 5 × ~4 GB ≈ 20 GB). That is acceptable but larger.

**Selection (deterministic, from `train_metadata.parquet` at revision `97a91733`):**
1. Drop samples in the box 49–61.5°N, 11°W–2.5°E (UK holdout plus margin).
2. Stratify by latitude band. Available counts after the UK drop: (−60, −35] 4,225; (−35, −15] 31,567; (−15, 15] 14,515; (15, 35] 88,093; (35, 50] 90,446; (50, 61] 11,062.
3. Take 84 per band (6 × 84 = 504, trimmed to 500) in ascending `sha256(sample_id)` order.
4. Force at least 25% of each band to have max cloud_cover over the 4 timestamps > 0.05 (38.6% of locations qualify), so masks and m2 are exercised.
5. Reject any candidate within 2.64 km of an already-chosen one (dedupe test).

**Size:** 500 locations × 4 timestamps. From the file listing (train totals divided by 243,968 locations), the compressed source is S2L2A ≈ 3.7 MB, S2L1C ≈ 3.6 MB and S1GRD ≈ 0.74 MB per location (4 timestamps). That gives ≈ 4.0 GB for 500 locations with all three, or ≈ 2.2 GB for S2L2A + S1 only. After packing to native-res uint16 + zstd: ≈ 500 × 4 × 0.6 MB ≈ 1.2 GB.

**Provenance fields per sample:** `source_repo`, `source_revision` (HF commit sha), `source_file`, `sample_id`, `sha256` of the source blob and of the packed record, `licence` (string from the card YAML), `attribution` (citation key), `center_lat/lon`, `crs`, `bounds`, `mt_cell`, `era5_cell`, `S2_time_i`, `S1_time_i`, `cloud_cover_i`, `pack_version`, `pack_git_commit`, `holdout_flag`, `benchmark_exclusion_flags`. Register the source in `configs/sources.yaml` (URL, licence, attribution, size, split/use) before download, following the OpenChange-UK gate.

**Pipeline:**
1. Source approval.
2. Download the selected files with `hf download --include` at the pinned revision.
3. Verify shapes and dtypes, NaN share, S1 date order (open issue #4), and |Δt|.
4. Reproject nothing; keep native UTM.
5. Pack: S2 bands into native-res uint16 arrays, S1 dB to uint16 (§3), mask to uint8.
6. Compute strata and dedup ids.
7. Write 1 WebDataset shard (≈1.2 GB) and 1 memmap copy.
8. Checksum both, then round-trip test: decode and compare to source within quantisation tolerance.
9. Record bytes per sample and CPU decode images/s (P4).

## 9. Closest prior work: data pipelines

| Work | URL | What it stored and how | What it does not do |
|---|---|---|---|
| Copernicus-FM / Copernicus-Pretrain | [2503.11849](https://arxiv.org/abs/2503.11849) | ERA5-grid cells; GeoTIFF tar.zst chunks of 1k grids per modality; WebDataset `.pth` per modality per grid (S2 int16, S1 float32), ~5 GB tars | no cloud masks, no time-matched S1/S2, no dedupe across overlapping patches, no I/O measurements |
| SSL4EO-S12 v1.1 | [2503.00168](https://arxiv.org/abs/2503.00168) | zarr.zip per sample (time × band × y × x), int16 S2, dB S1, SEnSeI masks, inside WebDataset tars per modality; TerraMesh-defined spatial val split | city-centred only; ≤10% cloud only; overlapping patches (28% within 2.64 km, measured) |
| Major TOM Core | [2402.12095](https://arxiv.org/abs/2402.12095) | 10 km grid; parquet rows with per-band GeoTIFF blobs at native resolution, SEnSeI mask; metadata parquet for filtering | monotemporal; S1/S2 dates unrelated; CC-BY-SA |
| TerraMind / TerraMesh | [2504.11171](https://arxiv.org/abs/2504.11171) | ~9M 264² S1+S2 samples from Major TOM + SSL4EO v1.1, ecoregion/LULC subsampling, SEnSeI masks; tokenises each modality with FSQ/VQ tokenisers before pretraining | its tokenisers are generative targets, not a byte-saving read path; no deduplicated data-scale axis |
| TESSERA v2 | [2607.03949](https://arxiv.org/abs/2607.03949) | pixel-wise time series; 395 Barlow Twins runs on GH200; data axis in millions of d-pixels; int8 embeddings released (TESSERA v1, [2506.20380](https://arxiv.org/abs/2506.20380)) | storage format of the training corpus not extracted here (UNVERIFIED); no patch-level ViT, no JEPA |
| AlphaEarth Foundations | [2507.22291](https://arxiv.org/abs/2507.22291) | 8,412,511 video sequences from 5,145,244 sites, 3,047,520,515 frames of 128×128 at 10 m, UTM; ~1.1% of land area; nine gridded sources | training imagery not released; storage format not public |
| Prithvi-EO-2.0 | [2412.02732](https://arxiv.org/abs/2412.02732) | 4.2M HLS time series at 30 m, 6 bands common to Landsat and S2, MGRS tiles | no S1; 30 m; no deduplicated scale axis |

**What changes in plan v2 (data and pipeline only):**
- §1.5: replace "about 1.1 MB per Copernicus-Pretrain sample, about 1 PB per 1B-sample run" with "1.81 MB per S2 image as shipped; ≈0.6 MB per S1+S2 pair packed losslessly; and reads are per *image*, not per view, so a run of N views with V views per image reads ≈0.6 MB × N/V".
- §3 I/O plan: drop the "< 150 KB per sample" target. Use "≈0.6 MB per pair lossless at 264², or ≈0.14 MB with 128² tiles". Stage the 1% subset in $LOCALDIR.
- §3 Data axis: 100% = 247,723 ERA5 cells / 1.07M patches / ≈0.5–0.8M unique 10 km cells (estimate; count in P0). "1B samples" means views seen.
- Q3: Copernicus-Pretrain S1/S2 are not time-matched, so Q3 pairs come from SSL4EO-S12 v1.1 (|Δt| median 3 d). "ERA5 already in Copernicus-Pretrain" is wrong; it is a join by cell id.
- M: cloudy and S1-only strata are mostly synthetic (mask transplant, sensor drop). "Real S2 cloud masks" needs a separate real-cloud test set from the Major TOM 10–25% tail.
- Temporal row: SSL4EO-S12 v1.1 has four timestamps within about a year; Copernicus-Pretrain's four timestamps span multiple years at one season. Pick per experiment and say which.
- Leakage: whole-UK exclusion plus 25 km buffer; future cutoff ≥ 2025-01-01; cell-level buffer exclusion for every benchmark test footprint; S3/S5P Copernicus-Bench tasks flagged as possible same-grid leakage.
- Compute: reconcile 5,000 NHR (OpenChange-UK SPEC) vs 15,000 (plan v2).
