# forecast_observe: training plan audit

**Topic:** forecasting a future land-surface state from incomplete Sentinel-1/Sentinel-2 history; later, choosing which archived observation to process next.
**Checked:** 13 Oct 2026. Research only: nothing trained, no datasets or weights downloaded. The only data read were listings, headers and ranged byte reads of 13 EarthNet2023 NetCDF cubes (total read about 30–40 MB, my estimate).
**Verdict: FINETUNE_EXISTING, gated.** The first step is a CPU-only check of whether Sentinel-1 helps. Observation selection is premature and should not be trained yet.

## 0. What changed from the prior dossier

1. **Answer to the main question: EarthNet2023-Africa has native S1 and S2 acquisition days.** I range-read cube headers. Each cube has a daily time axis, `s1_avail` and `s2_avail` flags, and `s1_vv`/`s1_vh`/`s2_B02…B8A` values that are present *only* on flagged days (fill value everywhere else; checked on two pixel time series). So "interpolated" is not true at pixel level for this release. The dossier marked this UNVERIFIED; it is now VERIFIED for the cubes I read.
2. **Train and test windows differ.** All 12 train cubes I sampled (Kenya, South Africa, Ethiopia, Zimbabwe) span the same 450 days, 2016‑01‑0x to 2017‑03‑2x. All 4 test cubes I sampled span 2016‑01‑02 to 2021‑12‑31 (2,191 days). Training data therefore come almost entirely from before Sentinel‑2B was operational, so test periods in 2018–2021 have denser revisits than training. That is a real shift in how many observations are available, and it is directly relevant to this topic.
3. **The release's licence is stated inside each file:** "EarthNet2023 - Africa is shared under CC-BY-NC-SA 4.0" (global attribute). The Zenodo toolkit record says CC BY‑NC‑SA 2.0. The in-file statement is the more specific one.
4. **Every cube includes a non-learned climatology baseline:** `ndviclim_mean`/`ndviclim_std`, monthly Landsat NDVI for 1984–2020.
5. **No other public dataset I checked has S1+S2 time series and future targets at this scale.** GreenEarthNet, EarthNet2021 and DeepExtremeCubes are S2 only. SeasFire is a 0.25° climate cube. SEN12MS‑CR‑TS covers one year and is built for cloud removal.

## 1. The single model most worth training

**Job:** at a cutoff date, forecast vegetation greenness (NDVI) for a 1.28 km patch of African shrubland or grassland 5–30 days ahead, with a quantile band. Sentinel‑2 history is gappy because of clouds, so the model should also use Sentinel‑1 radar history. The analyst wants to know early whether the patch is heading below its seasonal norm.

- **Input (at cutoff *t*):** up to 365 days of S2 L2A (B02,B03,B04,B05,B06,B07,B8A) with `s2_mask`/`s2_SCL`, plus S1 RTC VV/VH, both at native acquisition days with day-offset encoding. Also static DEM, ESA WorldCover and SoilGrids, and ERA5/ERA5‑Land up to *t* only. All 128×128 at 10 m, as stored in the cubes.
- **Output:** per-pixel NDVI quantiles (0.1/0.5/0.9) for each S2 acquisition day in (*t*, *t*+30]. Also a patch-level summary: the shrub/grass-pixel median NDVI and its anomaly against `ndviclim_mean`.
- **Target source:** future S2 NDVI computed from bands on flagged acquisition days after *t*. Only pixels with `s2_mask==0` (clear) and vegetation land-cover classes count. It is independent of the inputs **in time only**: it is the same sensor as the optical history, but it is never shown to the model. Sentinel‑1 is an independent sensor. The cloud mask (sen2flux) is the main source of label noise.
- **Weather setting:** report two variants. (a) No future weather: this is the honest forecast. (b) Realised future ERA5: an oracle upper bound only. GreenEarthNet/Contextformer and the local time-series baselines in that paper are conditioned on future weather (paper §3.5, "given future weather as covariates"), so their skill is not operational skill.

The secondary decision task (stop / reveal one S1 scene / reveal one S2 scene) should **not** be trained now; see §6.

## 2. Strongest existing methods and the measured gap

The numbers below are the sources' own numbers, read in the source.

- **Contextformer, GreenEarthNet, Europe, S2 only** ([CVPR 2024 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.pdf), Table 2, OOD‑t test 2021–22):

| Model | R² | RMSE | NSE | Beats climatology on |
|---|---|---|---|---|
| Contextformer (6.1M parameters) | 0.62 | 0.14 | 0.09 | 66.8% of cubes |
| Climatology | 0.58 | 0.18 | −0.34 | – |
| Persistence | 0.00 | 0.23 | −1.28 | – |
| Prophet | 0.57 | 0.16 | −0.05 | 60.6% of cubes |
| PredRNN | 0.62 | 0.15 | 0.03 | 64.7% of cubes |

  The 25-day RMSE is 0.08 for Contextformer and 0.16 for climatology. Contextformer‑16M is no better than 6M (66.8% vs 67.3% of cubes). Gap: the gain over climatology is modest, it needs future weather, it is Europe only, and it uses no S1.
- **ConvLSTM, Africa** ([arXiv 2210.13648](https://arxiv.org/abs/2210.13648)): S2 NDVI with ERA5/SMAP/DEM on an earlier, different EarthNet-Africa variant. No S1.
- **DeepExtremeCubes ConvLSTM** ([arXiv 2410.01770](https://arxiv.org/abs/2410.01770)): the source reports R² = 0.9055 on its test set for reflectance/kNDVI. That figure is for S2-only, extreme-event minicubes and is not comparable to NDVI-anomaly skill.
- **EO‑WM** ([arXiv 2606.27277](https://arxiv.org/abs/2606.27277), preprint): the abstract reports a 5.63% relative reduction in NDVI decline-amplitude error. Weights are "will be made open-source", so not available yet.
- **AgriMatNet / probabilistic NDVI** ([arXiv 2602.17683](https://arxiv.org/abs/2602.17683); [HF card](https://huggingface.co/ireneiele/agrimatnet-vegetation-forecasting), MIT): European field-level quantile forecaster, 28 tabular features. The abstract says "target history is the primary driver".
- **S1+S2 densification and forecasting** ([arXiv 2605.04239](https://arxiv.org/abs/2605.04239), preprint): generative S2 forecasting from S1+S2. I read the abstract only; no numbers, data or weights checked. **This is the closest prior work on the S1 angle and must be read in full before any novelty claim.**
- **Observability forecasting** ([arXiv 2607.13651](https://arxiv.org/abs/2607.13651)): LeWorldModel, 18.0M parameters, trained from scratch on EarthNet2021. The abstract says LightGBM is stronger on the binary any-usable task and more robust on OOD.

**Remaining gap:** I found no public, protocol-clean measurement of whether Sentinel‑1 history improves 5–30 day NDVI forecasts in African rangeland without future weather. No EarthNet2023-Africa benchmark paper exists yet; the toolkit README says to cite the Zenodo DOI "until there is an original publication". Whether S1 adds value is an open empirical question, and it is the one worth answering.

## 3. Data contract

| Item | Value | Status |
|---|---|---|
| **EarthNet2023-Africa** sample count | train 44,141 NetCDF cubes (0.797 TB); test 4,534 cubes (0.353 TB); 50/49 country folders | VERIFIED, bucket listing `s3.bgc-jena.mpg.de:9000/earthnet?prefix=earthnet2023/` |
| Cube shape | 128×128, daily time axis; static layers | VERIFIED, h5py header of `test/Algeria/30RTU1877.nc` |
| Sensors / variables | S2 B02–B8A, `s2_SCL`, `s2_mask`; S1 `s1_vv`, `s1_vh` (linear backscatter); ERA5 and ERA5‑Land daily; SRTM/ALOS/COP DEM; ESA WorldCover; Geomorpho90m; SoilGrids; Landsat NDVI climatology | VERIFIED, same header |
| Native timestamps | `s1_avail`/`s2_avail` flags. Values exist only on flagged days. Example test cube: 427 S1 days and 358 S2 days over 2016–2021; 70 days have both. S2 gaps are multiples of 5 days (5-daily filter); S1 gaps are 2–16 days | VERIFIED (2 cubes, pixel-level check) |
| S1 processing | Minicuber default: DEA `s1_rtc` (or Planetary Computer `sentinel-1-rtc`), Lee filter size 9, `groupby(date).last()` | Code VERIFIED ([sentinel1.py](https://raw.githubusercontent.com/earthnet2021/earthnet-minicuber/main/earthnet_minicuber/provider/sentinel1.py)); which provider the release used is UNVERIFIED |
| Temporal coverage | train cubes 2016‑01→2017‑03 (450 days); test cubes 2016‑01‑02→2021‑12‑31 | VERIFIED for 12 train and 4 test cubes; UNVERIFIED for all 44k |
| Spatial independence | No cube ID appears in both splits. 1,087 of 1,121 test MGRS tiles also appear in train. At about 10 km (my reading of the ID as tile + km offsets), 36 of 3,567 test cells are shared with train | Listing VERIFIED; the ID interpretation is my inference |
| Tabular extract | `en22_tabular_train_tiny.csv` (0.74 GB) and `_large.csv` (34.6 GB): ERA5, soil and `s2_NDVI` columns, **no S1 columns** | VERIFIED (header via range read) |
| Licence | CC BY‑NC‑SA 4.0 (file attribute); Zenodo toolkit record says CC BY‑NC‑SA 2.0 | VERIFIED ([Zenodo 10659371](https://zenodo.org/records/10659371)) |
| Upstream licences | PC `sentinel-1-rtc`: CC‑BY‑4.0; PC `sentinel-2-l2a`: Copernicus Sentinel terms | VERIFIED (PC STAC JSON); DEA s1_rtc and ERA5 terms UNVERIFIED |
| On HF | No | VERIFIED (HF search) |
| **GreenEarthNet** | S2 B02/B03/B04/B8A + SCL at 20 m plus E‑OBS weather. Train 23,816 cubes (2017–20); val 245; OOD‑t 2021–22; OOD‑s/OOD‑st 800 cubes. **No S1** | VERIFIED ([README](https://github.com/vitusbenson/greenearthnet), paper) |
| GreenEarthNet licences | weights MIT ([Zenodo 10793870](https://zenodo.org/records/10793870), 2.45 GB) | Weights licence VERIFIED; dataset licence UNVERIFIED |
| **EarthNet2021** | S2 at 20 m, 32,000 samples, S2 only | VERIFIED ([abstract](https://arxiv.org/abs/2104.10066)); an unofficial `Little-Liu/earthnet2021-npz` exists on HF (CC‑BY‑4.0 card, provenance unchecked) |
| **DeepExtremeCubes** | about 40–42k S2 L2A minicubes, 2016–2022, 5-daily, plus ERA5‑Land; **no S1**; 2.3 TB | VERIFIED ([OPARA](https://doi.org/10.25532/OPARA-703) page shows a CC BY 4.0 badge; [HF video version](https://huggingface.co/datasets/isp-uv-es/DeepExtremeCubes-video), 270 GB, cc‑by‑4.0) |
| **SeasFire** | 0.25°, 8-daily, 2001–2021, CC BY 4.0; no S1/S2 imagery; wrong scale for this task | VERIFIED ([Zenodo 7108392](https://zenodo.org/records/7108392), [HF](https://huggingface.co/datasets/orion-ai-lab/seasfire_monthly)) |
| SEN12MS‑CR‑TS | S1+S2, 2018 only, cloud removal | Taken from the dossier, not rechecked |

Practical notes. Python `requests` failed against the EarthNet S3 endpoint on an expired-certificate error, while curl worked (recheck before scripting). The GitHub API rate-limits unauthenticated calls from the VM.

## 4. Recommended model

- **Initialisation:** Contextformer‑6M GreenEarthNet checkpoint (MIT, Zenodo 10793870). It is a European, S2-only, 20 m checkpoint. Add an S1 input pathway: VV/VH in dB plus availability and day-offset tokens fused into the per-patch temporal transformer. Feed new S2 bands through the existing vision encoder by adapting the input projection. Always also train a **same-architecture model from scratch** as a control. At 6M parameters scratch training is cheap, and pretraining in Europe may not help in Africa. Use the from-scratch model only if it is not worse.
- **Why not a large EO foundation model (Prithvi‑EO‑2.0, Apache‑2.0; Galileo, MIT; TerraMind, Apache‑2.0; Clay, Apache‑2.0, per HF metadata):** none of them forecasts NDVI. Using one would mean training a forecasting head on frozen per-date embeddings, at about 100× the inference cost, and the published Contextformer ablation shows a frozen PVT encoder does not beat the small model. This is a fair later ablation, not the default.
- **Objective:** pinball (quantile) loss on clear-sky vegetated pixels at native S2 target days, weighted towards days 5–30. Add an auxiliary binary cross-entropy on the patch-level event "median NDVI anomaly at least 0.10 below climatology". The 0.10 threshold is a frozen operational choice, not an ecological standard.
- **Training with missing observations:** randomly drop S1 and S2 dates in the history at training time. This reproduces the 2016 vs 2018+ density shift and is what a later observation-selection study would need.
- **Parameter count:** about 6.2M (6.1M per the source, plus an estimated ~0.1M for the S1 path).
- **Compute (my estimate, not measured; single H100/GH200):**
  - Pilot fine-tune on about 5k train cubes (about 90 GB): 3–8 GPU-hours, plus CPU hours to convert NetCDF to a 5-daily array store.
  - Full run on 44k cubes (0.8 TB), 3 seeds × {S2-only, S2+S1} × {init, scratch}: about 60–150 GPU-hours. Input/output, not floating-point compute, is likely the bottleneck.

## 5. Baselines to beat (same cutoffs, same masks)

1. Persistence (last clear NDVI).
2. In-cube Landsat climatology (`ndviclim_mean`).
3. Climatology plus persisted anomaly (last observed anomaly decaying to zero; my addition, likely the strongest cheap baseline).
4. Previous year.
5. LightGBM on tabular per-patch features: S2 history only, and S2 + S1 history (S1 dB statistics, VH/VV ratio, trend). With and without the future-weather oracle.
6. Contextformer zero-shot (European weights, S2 only).
7. Contextformer retrained S2-only (to isolate the S1 gain).

Prophet and the Kalman filter are optional, since the source reports about 3 hours per cube on 8 CPUs.

## 6. Pilot and kill criteria

**Step A: CPU-only S1 check, before any GPU work. Units: 400 forecast instances.** 200 test cubes whose dominant WorldCover class is shrubland (20) or grassland (30), drawn from 10-km cells absent from train. Two cutoffs each, one in 2020 and one in 2021. This gives 400 instances, each a patch-median NDVI trajectory over (*t*, *t*+30] that can be checked by hand from a plot. Hand-inspect 50 for cloud-mask errors.

- Train LightGBM on features from about 2,000 train cubes (2016–17).
- Stratify by the S2 clear-observation count in the 60 days before the cutoff (≤2 vs >2).
- **Hold-outs:** temporal (train 2016–17, test 2020–21), spatial (unseen 10-km cells), plus one country block (Kenya/Ethiopia/Somalia withheld entirely from feature training) as a stress test.
- **Metrics:** RMSE and NSE at days 5–30; outperformance over climatology (paper's 3-of-4-metrics definition); pinball loss and 80% coverage; Brier score for the decline event. Paired bootstrap resampled by cube.

**Step B: fine-tune.** Only if step A passes K1 or K2. Same 400-instance evaluation plus the full held-out test set.

**Kill criteria, declared now (thresholds are my choices):**
- **K1 (S1 value):** in the sparse-S2 stratum, adding S1 to LightGBM must lower days 5–30 RMSE by ≥0.005 NDVI, with the 95% bootstrap interval excluding 0. If not, drop S1 and drop observation selection. Continue optical-only only if K2 passes.
- **K2 (worth learning at all):** without future weather, the best learned model must beat climatology-plus-anomaly on ≥55% of held-out cubes (outperformance definition). If not: NO_TRAINING_NEEDED; ship the climatology/anomaly tool.
- **K3 (worth deep learning):** the fine-tuned Contextformer must beat LightGBM by ≥0.005 RMSE or give better-calibrated quantiles (coverage within 0.75–0.85 where LightGBM's is not). If not, ship LightGBM.
- **K4 (target observable):** if more than 30% of target windows have no S2 date with at least 20% clear vegetated pixels, the target is too sparse; redesign the horizon.
- **K5 (leakage):** any gain that disappears when future weather is removed is not reported as forecast skill.

**Observation selection: premature.** The flags make an archive replay possible: hide a scene, reveal it, re-forecast. But the value of revealing one more S1 scene is bounded by K1. If S1 adds nothing, the policy reduces to "use the newest clear S2". Revisit only after K1 passes and the quantiles are calibrated. Even then, compare against newest, clearest, always‑S1, always‑S2 and variance-reduction greedy policies before claiming anything.

## 7. Public HF Space

**What a non-specialist could do:** pick one of a few dozen curated African test patches on a map, choose a past cutoff date, and see:
- the S2 and S1 history, with gaps visible;
- the forecast band against the patch's long-term normal;
- a plain sentence such as "likely below normal in 3–4 weeks: 70%";
- then a reveal of what actually happened, with the baseline scores beside the model's.

Everything is labelled as historical replay. There are no live forecasts and no land-management advice. A CSV export gives patch ID, dates, forecast and observed values.

- **Compute:** a 6M-parameter model on one 128×128 cube should run on CPU Basic (estimate; latency not measured). ZeroGPU is not needed. Precompute forecasts for the curated patches.
- **Rights risk:**
  - Cube contents are CC BY‑NC‑SA 4.0. A free, non-commercial Space with attribution looks compatible, but ShareAlike may apply to derived artefacts; this needs a reading of the licence or a question to the data authors.
  - Fine-tuned weights trained on NC‑SA data have unclear licensing; do not label them MIT.
  - Fetching live imagery from Planetary Computer for new locations is technically possible (PC S1 RTC is CC‑BY‑4.0) but out of scope for a first, bounded demo.

## 8. Critical unknowns and the cheapest check

1. **Do all 44k train cubes share the 2016–17 window?** Range-read the `time` coordinate of about 300 random train cubes (a few MB). If yes, there is no in-distribution data from the Sentinel‑2B era for training.
2. **Which S1 provider and orbit mix (DEA vs PC; ascending/descending)?** Match one cube's `s1_avail` dates against the PC and DEA STAC catalogues.
3. **Does S1 help at all?** Step A, on CPU (the existing tabular CSV lacks S1, so features must be extracted from cubes).
4. **Does the European Contextformer transfer to Africa?** Run the 2.45 GB checkpoint zero-shot on the 400 pilot instances.
5. **Is the test set really spatially independent?** Convert cube IDs to coordinates with `earthnet.get_coords_from_cube` and measure train–test distances.
6. **Cloud-mask quality in arid and savanna scenes:** hand-label 50 S2 frames.
7. **Overlap with arXiv 2605.04239 (S1+S2 S2-forecasting):** read the full paper, data and code.
8. **What NC‑SA permits for derived weights and the Space:** read the CC BY‑NC‑SA 4.0 legal code; if unclear, ask the dataset authors (the user's decision).
