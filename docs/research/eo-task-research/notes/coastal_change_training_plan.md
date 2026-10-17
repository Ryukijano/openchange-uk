# coastal_change: training plan audit

**Problem:** deciding whether an apparent shoreline change between satellite images is real, or caused by tide, waves, a different shoreline proxy, or misregistration, checked against field surveys.
**Verdict: LIGHT_HEAD_ON_FROZEN, conditional on the pilot.** Train a small per-observation error and abstention model on top of a frozen CoastSat/CoastSeg extraction, with optional frozen EO-encoder chip features. Do not train a segmenter or encoder from scratch, and do not fine-tune a foundation model for segmentation.

Method: online reading of primary sources only (repos, dataset cards, Zenodo/Dryad/OSF/HF metadata, papers). Nothing was downloaded beyond README/JSON metadata, and nothing was trained. Numbers in quotes are the source's numbers, not mine.

---

## 1. Corrections to the prior dossier

- SDS_Benchmark now lives at `SatelliteShorelines/SDS_Benchmark`, licence GPL-3.0, last push 2026-03-31 (VERIFIED, GitHub API). The dossier said the repo had no licence. That was wrong: the code has one, though the survey data keeps its upstream licences.
- The repo already holds **per-transect time series from five teams** (CoastSat, CASSIE, SHOREX, Deltares, UQMAO) at all four sites under `submissions/`, plus 5-minute tide and ERA5 wave CSVs for each site (VERIFIED from the git tree). That is enough to fit and test an error model **without downloading any imagery**. The dossier missed this, and it is the cheapest pilot available.
- A fifth site, CALAMILLOR (Mallorca), has been added with only a polygon and reference shoreline. No survey data is in the repo (VERIFIED from the tree and `sites_info.txt`).
- Late-March 2026 commits are titled "update tide levels [part 3–5]", but the README still says FES2014. Which tide model the CSVs now use is UNVERIFIED.
- **Learned QC already exists in CoastSeg** and the dossier did not mention it. `ImageRGBClassifier` (good/bad image, ONNX, GPL-3.0) and `ShorelineFilter` (good/bad segmentation, ONNX, GPL-3.0) are by M. Lundine (USGS) and called from `coastseg/classifier.py` with a default threshold of 0.40 (VERIFIED: [HF](https://huggingface.co/2320sharon/ImageRGBClassifier), [HF](https://huggingface.co/2320sharon/ShorelineFilter), CoastSeg source). Neither card gives training data or any accuracy, so how well they work against surveys is UNVERIFIED. Any new model has to beat these, not just hand screening.
- The dossier's 48-hour matching rule is stricter than the benchmark's own "closest satellite-derived [point] within a window of 10 days" ([Vos et al. 2023](https://www.nature.com/articles/s43247-023-01001-2), Methods). Keep 48 h for test labels, but expect far fewer matches than the benchmark reports.

## 2. Which component actually benefits from learning

| Component | Existing non-learned / learned solution | Is there a learnable, independent target? | Verdict |
|---|---|---|---|
| Waterline segmentation | Sub-pixel MNDWI + Otsu (CoastSat) with a small 4-class NN; Doodleverse SegFormer/Res-UNet zoo | Labels are hand-drawn on the same image, so they are not independent of the input. LICS (Irish Landsat): "NDWI benchmark outperformed U-NET with an average accuracy of 97.2%" vs 95.0% ([arXiv 2409.15311](https://arxiv.org/abs/2409.15311)). Vos 2023: ~10 m RMSE at microtidal sites "(1/3 of a pixel)". | Not the bottleneck. Do not train. |
| Tide correction | FES2014/FES2022 + beach slope (CoastSat) | Physics | Already solved where slope is known. Do not learn. |
| Wave setup | Stockdon parameterisation | Physics; results are mixed | Vos 2023: setup improved Duck CoastSat RMSE "from 9.2 to 7.6 m" but made the Narrabeen bias worse for SHOREX/CASSIE "(from ~5 to ~10 m)". This is a residual a model could learn per site, but with only ~5 sites it will overfit. Use it as a feature, not a target. |
| Georegistration | Landsat RMSE>10 m discard; S2 pass/fail flag; SHOREX/CoastSeg co-registration | Vos 2023: SHOREX, the only algorithm that co-registers, had the best S2 STD "6.9 and 5.8 m" at Duck/Narrabeen | Use a non-learned tool (co-registration) as a baseline and a feature. |
| **Per-image, per-transect error / bad-observation detection** | Manual screening; CoastSat `check_detection`; despiking (`max_cross_change` 40 m, Otsu range [-0.5, 0]); CoastSeg good/bad classifiers | **Yes.** Target = tide-corrected SDS position minus surveyed MSL-contour position. The survey is a physical measurement, independent of the imagery. | **Train this.** It is the step that still costs people time: USGS reports "As much as a full workday (8–10 hours) was needed just to manually screen and filter out bad imagery and initial detections" at Barter Island ([OFR 2025-1054](https://pubs.usgs.gov/publication/ofr20251054/full)). |
| Proxy ambiguity (which feature the waterline sits on) | None automated | Waikīkī: mid-resolution SDS "aligned with the waterline in only ~50% of cases" ([Mikkelsen et al. 2026](https://doi.org/10.1038/s41598-026-54336-z)) | It is captured inside the error target. A separate proxy classifier needs per-image proxy labels, which only Waikīkī has. Later work. |

So the useful job is a calibrated per-observation error interval that decides keep/drop and whether a change between two dates is larger than the combined uncertainty. Vos 2023 frames that question with one constant error per site: the share of surveyed changes larger than SDS accuracy was "32% at Duck, 55% at Narrabeen, 41% at Torrey Pines and 18% at Truc Vert". A per-observation model should raise that share by narrowing intervals on good images and dropping bad ones.

## 3. The model to train

- **Input (per image × transect observation):** (a) metadata and physics features: sensor, tide level at acquisition, offshore Hs/Tp/direction (ERA5), Stockdon setup, beach slope, Landsat geometric RMSE or S2 geometric flag, cloud fraction and distance to cloud, Otsu threshold, distance to reference shoreline, shoreline length, sand-pixel area, CoastSeg classifier scores, and the spread across the five benchmark algorithms on the same date. (b) Optional: a frozen EO-encoder embedding of a ~2.5 km chip around the transect (Sentinel-2/Landsat bands).
- **Output:** predicted signed error (m) and a 90% interval, a keep/abstain flag, and for a pair of dates a signed change interval or "cannot tell".
- **Label:** the surveyed MSL (or site-datum) contour position on the same transect within ±48 h (test) or ±10 days (train, with the time gap as a feature). The label comes from GNSS/lidar/DEM field surveys, so it is independent of the image.
- **Model:** gradient-boosted quantile regression (LightGBM/XGBoost) on the metadata features, plus an ablation that adds frozen embeddings through a 2-layer MLP head (~0.1–1 M trainable parameters). Pairwise change intervals come from the two per-date intervals, with split-conformal calibration on a held-out site. Frozen encoder candidates: Prithvi-EO-2.0-300M (Apache-2.0), Clay (Apache-2.0), Galileo (MIT), TerraMind-1.0-base (Apache-2.0); licences VERIFIED from HF model metadata. Their fit for 10–30 m beach chips is UNVERIFIED.
- **Loss:** pinball loss at q = 0.05/0.5/0.95, plus binary cross-entropy on |error| > 15 m (the "bad observation" event).
- **Why not from scratch or full fine-tuning:** there are only ~5–8 surveyed sites. The data cannot support learning visual features, and the segmentation evidence above shows learned segmenters do not beat index thresholds on this task.

## 4. Data contract

| Item | Details | Status |
|---|---|---|
| SDS_Benchmark (sites, transects, tides, waves, 5-team SDS time series) | Duck, Narrabeen, Torrey Pines, Truc Vert; "4–5 shore-normal transects with the highest frequency of surveys" per site; Landsat 5–9 and S2; tides every 5 min; ERA5 waves; GPL-3.0 code; not on HF | VERIFIED: [repo](https://github.com/SatelliteShorelines/SDS_Benchmark), tree, README; Vos 2023 text |
| Duck FRF surveys | "monitored on a monthly to fortnightly basis since 1974"; NAVD88 (offset -0.128 m to MSL in README) | Cadence VERIFIED (Vos 2023). Licence UNVERIFIED (FRF portal returned HTTP 410; probably US-government public domain). |
| Narrabeen profiles | 5 profiles, monthly since 1976; Dryad CC0, ~98 MB | VERIFIED: [Dryad API](https://doi.org/10.5061/dryad.28g01) (CC0, 2017-03-24) |
| San Diego / Torrey Pines | 2001–2016; Dryad CC0, ~9.8 GB in total (download only the `torrey_*` subset) | VERIFIED: [Dryad API](https://doi.org/10.5061/dryad.n5qb383) |
| Truc Vert DEMs | 2003–2019, fortnightly since 2004; OSF CC0 1.0 | VERIFIED: [OSF API](https://osf.io/jftw8/) |
| Waikīkī | 128 weekly surveys over 3.5 years, 500 m reef-lined beach; items CC0 per dossier; article CC BY-NC-ND | Survey count VERIFIED (abstract). Item licences UNVERIFIED by me (dossier claim). |
| USGS OFR 2025-1054 sites (incl. Barter Island, Elwha) | Matched SDS vs in-situ comparisons | Report VERIFIED. Whether its matched data is released is UNVERIFIED. |
| Landsat / Sentinel-2 imagery | Open reuse (USGS; Copernicus legal notice) | Per dossier, not re-checked |
| Doodleverse/Coast Train labels | 1,018 4-class S2/Landsat RGB labels from 10 beach sites; 3,649 2-class labels (579 Doodler + 3,070 SWED); CC BY 4.0 | VERIFIED: Zenodo [7335647](https://doi.org/10.5281/zenodo.7335647), [7384263](https://doi.org/10.5281/zenodo.7384263). Only needed if segmentation is retrained, which it should not be. |
| SWED (UKHO) | Licence is the "Geospatial Commission Data Exploration license": 12-month evaluation/R&D terms | VERIFIED ([openmldata.ukho.gov.uk](https://openmldata.ukho.gov.uk/)). **Exclude from any public model.** Note that the CC BY Zenodo 7384263 re-packages 3,070 SWED pairs, so the rights to that subset are unclear. |
| FES2022 tide grids | Separate AVISO terms | UNVERIFIED. Do not redistribute grids; extracted time series also need checking. |

Sample count: matched (image, transect, survey within 10 days) pairs per site are in Vos 2023 Table 3, which I could not extract from the HTML (UNVERIFIED). The ±48 h count is unknown. All observations from one scene share georegistration and water-level error, so the effective sample size is closer to the number of scenes times sites than to the number of rows.

## 5. Strongest existing methods and the remaining gap (source numbers)

- **Vos 2023 benchmark:** RMSE "below 10 m" at Duck; "13 and 18 m" at Torrey Pines; "28 and 48 m" at Truc Vert. Truc Vert images below +0.2 m AMSL tide were discarded. With visual screening and runup correction, Castelle et al. reached "RMSE of 10 m, 7 m bias and R² of 0.78" at the same site, which shows that site-specific screening is what closes the gap at macrotidal sites.
- **Mikkelsen 2026 (Waikīkī):** after co-registration, outlier removal and water-level correction, RMSE was "2.83 m (PlanetScope) to 5.54 m (Sentinel-2)". Proxy inconsistency remains.
- **USGS OFR 2025-1054:** manual screening of 8–10 h per site set, with weeks of setup.
- **CoastSeg ImageRGBClassifier / ShorelineFilter:** these exist, but I found no published survey-based evaluation.
- **Remaining gap:** nobody has published a per-observation, survey-calibrated error model validated on held-out sites (as far as I found; search not exhaustive). One unread 2026 paper may overlap: "When is satellite-derived shoreline change interpretable? A signal-to-noise framework…" ([RSMA 105374](https://doi.org/10.1016/j.rsma.2026.105374)); I could not get its abstract. NatShore (dossier) claims pixel confidence but is not survey-calibrated per the dossier.

## 6. Baselines the model must beat

1. No filtering, tide-corrected (FES + site slope).
2. CoastSat defaults: `cloud_thresh` 0.5, `dist_clouds` 300 m, `max_dist_ref` 100 m, `min_length_sl` 500 m, plus `reject_outliers` (`max_cross_change` 40 m, Otsu range [-0.5, 0]).
3. 2 + Landsat geometric RMSE > 10 m discard + S2 geometric fail discard.
4. 3 + site tide-level cut-off (e.g. Truc Vert < +0.2 m AMSL), + Stockdon setup.
5. 4 + CoastSeg ImageRGBClassifier + ShorelineFilter at 0.40.
6. 4 + co-registration (CoastSeg `6_zoo_workflow_with_coregistration.py` / SHOREX-style).
7. Inter-algorithm spread across the five benchmark submissions as an uncertainty score.
8. Constant site-RMSE interval (the Vos 2023 Fig. 8 framing) and a Monte Carlo physical-error envelope (tide, slope, georef, survey).
9. Always-abstain and a seasonal-mean reference.

## 7. Pilot and kill criteria

**Stage 0 (no imagery, CPU, ~1 day):** use the existing `submissions/` time series, tide CSVs and wave CSVs. Compute residuals against the surveyed MSL series, then fit metadata-only GBM quantile models with **leave-one-site-out** (4 folds) and a **temporal block** (train ≤2016, test ≥2017 where surveys exist). This shows whether error can be predicted at all before any imagery is touched.

**Stage 1 (hand-checkable test set, ≤500 units):** 4 sites × 4 transects × ~30 observations ≈ 480 image–transect observations with a survey within ±48 h, frozen before fitting. Test sites are held out: Truc Vert (macrotidal) and Waikīkī (reef, external). Development and calibration use Duck, Narrabeen and Torrey Pines. Every test observation gets a manual QC note (cloud, proxy, misregistration) so failures can be read, not just scored. Pairs for change claims are non-overlapping endpoint pairs on the same transect and sensor.

**Metrics:** retained-RMSE vs coverage curve; AUROC/AUPRC for |error| > 15 m; 90% interval coverage and width per site; for pairs, the signed-claim false-direction rate and the share of surveyed changes detected at matched coverage. Results are reported per site with scene-cluster bootstrap CIs.

**Kill criteria (pre-declared):**
- K1: at 70% retained coverage, the model reduces retained RMSE by <15% relative to the best of baselines 2–7 on ≥2 of the held-out test sites → **kill the learned component**. Ship baselines + physical envelope as a no-training tool (NO_TRAINING_NEEDED).
- K2: 90% intervals cover <85% on a held-out site after conformal calibration → no claim-capable output; abstain-only.
- K3: false-direction rate >10% of issued change claims, or <25% of surveyed pairs get a claim → no "real change" verdicts in the Space.
- K4: frozen-embedding features add <5% relative RMSE gain over metadata-only → drop the encoder (the model stays a metadata model, which is fine but means it is not an EO-representation result).
- K5: <100 matched ±48 h observations at either test site → the pilot design fails; widen to ±5 days with the gap as a feature, or stop.

## 8. Compute (estimates, not measured)

- Stage 0: CPU only, minutes per fit.
- Stage 1 with frozen embeddings: about 20–50k chips through a 300 M-parameter encoder plus head training ≈ **1–3 GPU-hours** on one H100/GH200.
- Full run (8–10 sites, all sensors, 3 encoders × 5 seeds × LOSO folds): **~20–40 GPU-hours**. The real cost is GEE imagery retrieval and survey preprocessing, not GPU time.

## 9. Public HF Space

What a non-specialist could do:
- Pick a benchmark beach and see each satellite shoreline marked keep/flag, with its predicted error next to the survey truth (archived replay).
- Upload a CoastSat/CoastSeg transect CSV (dates, positions, sensor, tide) and get per-row flags plus, for two chosen dates, "retreat of X ± Y m", "advance", or "cannot tell from these images".

Feasibility:
- The metadata model runs on CPU in milliseconds; ZeroGPU is not needed.
- The image-feature variant only works on curated sites with precomputed embeddings. User imagery would need GEE credentials, which a public Space should not handle.

Rights risks:
- CoastSat/CoastSeg code and the two QC ONNX models are GPL-3.0, so a Space that bundles them must be GPL-compatible.
- Do not redistribute FES grids.
- Exclude SWED-derived data.
- Survey truth shown in replays is CC0 (Narrabeen, San Diego, Truc Vert).
- Abstain on inlets, cliffs, estuaries and mixed sand/gravel.
- No accuracy claim outside the validated site types.

## 10. Critical unknowns and the cheapest check for each

| Unknown | Cheapest check |
|---|---|
| Is per-observation error predictable across sites at all? | Stage 0 on the existing submission CSVs (no imagery) |
| How many ±48 h matches exist per site | Run `2_check_shoreline_accuracy.ipynb` with the window set to 2 days on the repo's CoastSat submission |
| What CoastSeg's good/bad classifiers were trained on, and how they score against surveys | Ask or read the CoastSeg wiki and Lundine's repos; score them on the Stage 1 images |
| Which tide model the 2026 SDS_Benchmark CSVs use | Read the "update tide levels" commit diffs |
| Overlap with the 2026 RSMA signal-to-noise framework and NatShore | Read both full texts before claiming novelty |
| Licences for Duck FRF and FES-derived series | FRF metadata or USACE data policy; AVISO FES2022 licence text |
| Whether more surveyed sites with released data exist (USGS OFR sites, UK regional monitoring programmes, CoastSnap) | Check data releases; the UK coastalmonitoring.org terms page returned 404 (UNVERIFIED) |

## Bottom line

The learnable gap is screening and per-observation uncertainty, not segmentation. The target comes from field surveys and is independent of the imagery. It needs a small head, not a new encoder. A no-imagery pilot can falsify it within a day. If it does not beat CoastSat QC + co-registration + CoastSeg classifiers + inter-algorithm spread, the right deliverable is a non-learned uncertainty and abstention tool.
