# Coastal measurement model: measured screen and training plan

13 October 2026. **Recommendation: pursue a coastal measurement encoder only through controlled transfer tests. Do not start backbone pretraining yet.** The first repository-only screen was negative. That is a useful falsifier, not evidence that a bigger model is needed.

## 1. What actually ran

SDS_Benchmark commit `c0747c52b310433411fc4ab96822ee67441a8d2e`. Local CPU, scikit-learn quantile gradient boosting; no external survey archives, satellite imagery or weights fetched. The clone itself contains published illustrations. No GPU/cluster access, commits, push or deployment. Research reviewers fetched source text and catalogue metadata, including a small CoastTrain metadata CSV—not image arrays.

The repository has Narrabeen raw profiles, but gitignores the processed survey pickles. Duck, Torrey Pines and Truc Vert surveys must be acquired externally. **The approved four-site experiment is incomplete. This is a one-site temporal feasibility screen, not geographic transfer.** It cannot exercise all nine planned baselines: cloud/QC/geometry features and a complete physical registration/noise budget are absent.

Protocol saved before outcome computation in `coastal-stage0/protocol.md`. Fixed splits: fit through 2009; calibration 2010–2016; test 2017–2019. Match each CoastSat observation to the nearest survey within **exactly 48 hours**, without interpolating satellite positions. Extract the 0 m MSL contour from the survey profile; exclude unobserved/ambiguous crossings. Date-only surveys introduce roughly +/-12 h timestamp ambiguity. A 36 h match-count sensitivity is logged. Residuals include survey/matching uncertainty; they are not pure extraction-error truth.

| Split | Transect observations | Independent acquisition groups | Sensors |
|---|---:|---:|---|
| Fit | 138 | 31 | L5, L7 |
| Calibration | 194 | 45 | L5, L7, L8, S2 |
| Test | 329 | 68 | L7: 20 rows; L8: 52; S2: 257 |

Five transects from one scene are correlated. There are not 329 independent test images. The newer sensors were absent during fitting. This is a demanding sensor/time shift with a very small training set; it does not settle general learnability. The original >=100 matched test-observation gate is not satisfied if observations mean independent scenes.

Slope was fixed to the benchmark Narrabeen assumption **0.1**, not estimated from each image or survey. It is therefore not an informative variable in this one-site model. Stockdon setup here uses wave height/peak period and that fixed slope; it is not the full physical runup/registration noise budget.

### Measured results

Retain whole acquisition groups, ranked without test labels; actual scene coverage **70.6%** (48/68 scenes), row coverage about 70%. Units are metres; lower RMSE is better.

| Route | All-row RMSE | Retained RMSE |
|---|---:|---:|
| Published tide-corrected CoastSat positions | 6.855 | 6.735 |
| Fit-only median bias correction | 7.934 | 7.749 |
| Stockdon setup correction | 8.239 | 8.120 |
| Absolute tide ranking, unchanged positions | 6.855 | 6.638 |
| CoastSat–SHOREX disagreement ranking, unchanged positions | 6.855 | 6.485 |
| Learned metadata quantiles | 8.826 | 8.434 |
| Learned metadata + disagreement quantiles | 8.838 | 8.410 |

The primary comparison was against the cheap route chosen **on calibration data**: setup correction. Metadata + disagreement gave **-3.57% RMSE reduction** (i.e. worse); scene-cluster bootstrap 95% interval **[-11.49%, +4.96%]**. The unchanged published route is also substantially better on test. Disagreement ranking's test result is descriptive, not a post-hoc replacement for the registered primary comparator.

The metadata + disagreement model has **95.1% marginal row interval coverage**, **80.9% simultaneous scene coverage**, and **36.7 m mean endpoint interval width**. These are different quantities. Retained-row coverage is 93.9%. Conservative exploratory pair intervals produced **0 claims out of 138 pairs**; metadata-only produced 1/138. Zero false claims with zero claims is not useful predictive success. Ninety-percent endpoint intervals do not imply ninety-percent joint pair coverage.

Seeds 0/1/2 gave identical outputs in this small deterministic HGB fit; this is not independent replication. Five focused unit checks and Ruff passed. Existing OpenChange `make check` passed with eight skips; it does not validate coastal performance. No full static type checker was configured for this standalone research script. Local logs are preserved but not committed because publication was not authorised.

### Important input corrections

- Deltares uses composite shorelines; UQMAO uses annual high-tide composites. They are not individual-image predictions. We did not invent five-method scene disagreement.
- CASSIE's sampled timestamps differ by 12 hours from CoastSat's matching dates. Its count audit is provisional; timestamps were not silently shifted.
- Disagreement here means two same-acquisition methods, CoastSat and SHOREX, matched within 10 minutes by sensor. It is missing for 55.8% of matched rows, particularly S2.
- Published submissions already reflect algorithm filtering. They do not represent every rejected/cloudy image. A manual-screening model needs rejected cases too.
- The raw Narrabeen file extends to 2019 and came through the benchmark/WRL route. A CC0 2016 Dryad snapshot does not automatically settle the rights to every later extension. Resolve lineage before redistribution.

## 2. What a model worth training would learn

**A representation of coastal appearance and observation reliability, conditioned on physical state, that transfers with few labels.** Not an unconditional “all dates of this beach are the same” representation, and not a model that invents surveyed topography from a satellite pixel.

Proposed system:

`multispectral coastal chip + acquisition metadata → EO encoder → local coastal features + physics-conditioned heads → observation error / proxy / quality → calibrated two-date measurement or abstention`

Keep CoastSat/CoastSeg extraction, stable-land registration and tide corrections as explicit components until a controlled test shows why replacement helps. Align on stable inland features, never on the changing waterline. A generated missing modality is a prediction, not an observation.

Three genuinely distinct transfer families would support a broader encoder claim:
1. **Independent-survey measurement/reliability:** residual distributions and supported change decisions.
2. **Coastal semantic/boundary segmentation:** water, whitewater, sediment, vegetation; independently labelled masks, metre-based boundaries.
3. **Expert coastal attributes:** sediment/coast type, built environment and defences, using a region-separated dataset such as CoastBench after its assets pass rights review.

Segmentation, its extracted contour and its IoU are not three tasks. Four sandy beaches and a single error head do not establish a coastal foundation model.

## 3. Pretraining: concrete, conditional recipe

### First test the existing representations
Use the same metadata, chip extent, labels, splits and small head with frozen [Prithvi EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) and [TerraMind](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base). Add [AnySat](https://huggingface.co/g-astruc/AnySat) and [Galileo](https://huggingface.co/nasaharvest/galileo) as resources allow. Compare a small randomly initialised task model too. Prithvi/TerraMind cards say Apache-2.0; AnySat/Galileo say MIT. Audit pinned code and data separately.

Prithvi's pretraining uses six-band HLS at 30 m; native S2 has different spectral/resolution conventions. Resampling to 10 m adds no detail. TerraMind handles S1/S2 and missing modalities; initially disable generated-modality inputs. Common-band optical comparisons and native/multisensor comparisons are separate tracks. Copernicus-FM is a research comparator, but its permissive weight licence does not erase mixed/NC code components.

### Only if a measured image-feature gap survives
Continue pretraining the best compatible encoder on licensed Landsat/Sentinel coastal sequences, not on the four benchmark beaches. Catalogue and rights audit first; entire calibration/test coastal cells excluded at all dates, with a proposed 50 km buffer enlarged where needed. Unknown upstream training geography remains a disclosed limitation, not a “never seen this coast” claim.

Start with the encoder's **native pretraining objective** (masked reconstruction/token prediction for the Prithvi/TerraMind track), balanced around boundaries rather than mostly empty ocean. Preserve physical GSD, bands, timestamps, validity masks and source provenance. Balance continents, tide ranges, coast types, turbidity, seasons and sensors. Include general-EO replay and a forgetting probe. Proposed first corpus: at most 100 coastal blocks x 20 acquisitions = 2,000 chips, after a separate 100–500-example hand-checked pilot and approval. Six-band uint16 256x256 arrays alone are about 1.57 GB; this is not a full-scene transfer estimate or approved download.

Compare ONE added objective at a time:
- **Local/geometry-aware representation:** preserve boundaries; known translations should transform coordinates predictably, not look like erosion.
- **Physical conditioning:** retain tide/waves/sensor state explicitly. Predicting tide from metadata already supplied is not a useful learning objective. Do not pretend a coarse DEM is intertidal survey truth.
- **Cross-sensor training:** only align compatible states/footprints with explicit time/tide differences; SAR and optical need not show the same instantaneous waterline. Train missing-sensor handling without claiming hallucinated pixels are evidence.
- **Conditional temporal prediction:** later, only if labels and short-event sampling justify it. Use elapsed time and information available at inference; preserve actual erosion/recovery and nourishment. No blanket temporal invariance.

JEPA is an ablation, not a commitment. Random-initialised large-scale pretraining waits until controlled adaptation and small task models demonstrably plateau and data diversity/budget justify it. No GPU-hour estimate is credible until a chosen model and loader are profiled.

## 4. Post-training: grounding in real measurements

Post-training here means supervised task adaptation and calibration—not RLHF.

1. Fit proxy-specific boundary/semantic heads on audited masks; record if labels are human drawn, teacher generated or field measured.
2. Fit signed residual quantiles or a probabilistic head against compatible-datum, close-time field surveys, using pinball/proper-scoring losses. Include survey uncertainty, matching gap and shared scene effects in analysis; sparse surveys do not become dense pixel truth.
3. Add independent quality/proxy labels and exceedance probability where they help, with losses validated separately. Teacher CoastSat outputs may be weak training labels, never the gold test target.
4. Try frozen head, adapters/LoRA and full fine-tuning before paying for continued pretraining. Match label counts, searches and measured compute; report upstream sunk pretraining separately.
5. Freeze model selection, then calibrate on disjoint geography/time. Evaluate empirical per-site coverage and width. Four sites cannot provide finite distribution-free 90% new-site group-conformal guarantees: even nine independent calibration units merely remove the order-statistic obstruction; they do not prove exchangeability.
6. Calibrate pair residuals directly when possible, or use conservative simultaneous endpoint bounds. Report false direction AND claim coverage on locked test pairs. Never narrow intervals to make a demo more impressive.

## 5. What “better” must mean

Retain original gates rather than quietly replacing them with reviewer thresholds:
- Learned retained-RMSE reduction >=15% at 70% coverage on at least two held-out sites; per-site and all-row results shown.
- Nominal 90% intervals must achieve >=85% empirical coverage; width also shown.
- Pair false-direction rate <=10%, with >=25% of eligible pairs receiving a claim.
- >=100 close-time matched independent observations per test site; count scenes, surveys and rows separately.
- Frozen imagery must add >=5% beyond metadata to justify its complexity.

Additional proposed gate, to register before a later experiment: coastal continuation improves the same original encoder under matched post-training by >=10% on the chosen primary task-risk metric, without >5% degradation on event recall/general-EO probes. Otherwise remove continued pretraining.

Only consider “coastal foundation model” after reproducible low-label transfer across >=3 independent task families, >=3 held-out coastal regions/regimes and >=2 sensors. These are proposed study criteria, not a universal definition. Name the exact winning metric/regime; do not claim globally better than every EO model.

## 6. Prior art and release gates

Coastal continuation and adapters already exist ([Ma et al., abstract checked](https://doi.org/10.1109/TGRS.2025.3573159)). [NatShore](https://github.com/isaacgbhk/NatShore) already does tide-aware selection, extraction and pixel confidence. [Prithvi on two Maldivian islands](https://arxiv.org/html/2511.10177v1) reports strong low-label segmentation; that is not unseen-coast survey-error calibration. None of these results is a measurement from our project. Our RSMA review remains abstract-only. A fuller literature review is needed before claiming novelty.

Open Landsat/Sentinel can support an unlabeled corpus. Original [USGS CoastTrain](https://doi.org/10.5066/P91NP87I) and [CoastBench](https://zenodo.org/records/15800285) are candidate distinct-task labels, with asset provenance still checked. Do not indiscriminately use the 1,018-image [Doodleverse set](https://zenodo.org/records/7335647): it overlaps all four benchmark beaches. Exclude SWED/repackages. Published [USGS QC data](https://doi.org/10.5066/P1N4VI7H) includes PlanetScope restrictions; select verified open-sensor portions only. FES/TPXO grids have separate terms and must not be redistributed by default.

The eventual Space: curated dated observations → supported position/change interval or “cannot tell” → physical/QC evidence → downloadable provenance. Clearly separate retrospective survey comparisons from unsurveyed inference. No arbitrary URLs, pickles or archives; bounded uploads only, no silent retention. CPU precomputed features first; bounded raw inference later. No release is authorised now.

## 7. Smallest next step needing approval

Add **Truc Vert** (294 monitoring NCs plus required grid, **162,611,806 bytes**) and **Torrey Pines** (mapped elevation ZIP, **115,224,176 bytes**). Total **277,835,982 bytes**, under a proposed **300 MB transfer cap**. Both source records state CC0. No imagery or weights. Uncompressed size and successful preprocessing remain unverified; enforce file/size allowlists and do not follow notebook downloads blindly.

This gives three sites for a further CPU screen, not the promised complete four-site test or foundation validation. Duck adds roughly 579 MB through 2020 (catalogue-rounded sizes) and needs its public redistribution terms resolved. Its live archive must not silently change the benchmark cutoff.

**Next decision: approve the two bounded CC0 survey downloads, or stop here with the negative one-site screen and the gated model plan.**

## Evidence files

`coastal-stage0/`: protocol, script, five unit checks, per-input hashes/environment, `results.json`, run log, matched rows and predictions. Local analysis only; raw/derived surveys are not bundled for public redistribution. Reviewer reports: [method](https://app.devin.ai/sessions/da4f3d076101499ba5523349ea526544), [data](https://app.devin.ai/sessions/f57646f97b0349629f1d3f58e4528373).
