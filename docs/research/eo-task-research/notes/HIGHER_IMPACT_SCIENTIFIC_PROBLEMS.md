# Higher-impact scientific problems with usable Hugging Face demos

13 October 2026. Research recommendation, not experimental results or a build authorization.

## Decision

Do not make encoder-upgrade continuity the flagship scientific claim. It is useful infrastructure, but the current experiment does not establish useful task performance, saved work or adoption.

Two stronger research bets survive:

1. **Forecast a changing Earth state and estimate which observation would actually improve the forecast.** This is the best match to the user's world-action-model ambition, but has the weakest verified data path of the finalists.
2. **Determine when apparent coastal movement supports a physical-change claim, rather than a tide/wave/registration artefact.** This has the stronger near-term validation path and documented analyst burden. It is not merely coastal segmentation or hosting CoastSat.

The independent sceptic preferred coastal claim validity for investment now; forecasting/observation selection for eventual WAM research; and historical analogue search for an immediately understandable interface. My synthesis retains those distinctions: **highest strategic fit is not the same as highest present feasibility or demonstrated impact**. There is no evidence that any candidate will attain AstroLoc-like adoption.

For the user's ambition, I would investigate forecast/observe as the primary scientific aim, with a strict metadata/rights/chronology gate before implementing a model. If that gate fails, the coastal question is the better evidence-grounded alternative. Do not build both full systems or pretrain first.

## Comparison: five independent audits and one consuming red team

| Problem | Scientific capability, not UI feature | Space action | Current verdict |
|---|---|---|---|
| Forecast + observation value | Predict a future land-state proxy, then choose information that improves a scored prediction under cost | Inspect history, compare forecasts, reveal recommended evidence, verify against withheld future observations | Conditional research bet; data chronology/rights/target support unresolved |
| Coastal physical change | Establish or refuse a signed mean-sea-level contour-change claim despite nuisance explanations | Inspect paired observations, uncertainty and survey support; export a reproducible claim or refusal | Strongest near-term scientific instrument; matched data and benefit over physical bounds unproved |
| Rare-event analogue retrieval | Find cross-site event patterns with spatial verification and genuine no-match cases | Query a dated example; inspect a small set of historical analogues and source footprints | Useful interface, narrower/crowded scientific gap |
| Restoration recovery | Separate field-verified water-table recovery from a wet year at unseen sites | Compare wells, weather and references; decide where field verification is still needed | Coordinates/dates/external measurements block it; no causal-effect claim |
| Flood-area evidence audit | Quantify newly inundated area using a small independent probability sample and a transferred map | Audit AOI/date/map against sampled evidence and export supported area bounds | Pivot: established statistics plus unresolved independent synchronized truth |

Scores are judgment aids, not measurements or adoption probabilities. The independent red team rated coastal impact/openness/feasibility/demo/WAM fit **4/3/3/4/2**, forecast/observe **4/3/2/3/4**, and retrieval **3/2/3/5/1** on a five-point scale. They should not be summed into an artificial winner.

## 1. Forecast, then choose useful evidence

### Scientific question

**Can a model predict the task-specific value of an additional radar or optical observation better than recency, predicted usability or generic uncertainty when forecasting vegetation decline?**

Start with sustained rangeland greenness decline 20–30 days ahead, not arbitrary Earth simulation, famine prediction or vegetation damage diagnosis. The audit proposes NDVI falling at least 0.10 below its latest reliable value on both verification dates. That is a proposed operational label, not an ecological damage threshold; seasonality, harvest and observation quality require explicit controls.

At the cutoff, choose stop or process one *already acquired* concealed S1/S2 scene. The action changes available information, not Earth. Its pixels are hidden until selection; future targets stay hidden until scoring. This is archive-reveal evaluation, not satellite tasking. Cost initially means measured bytes/runtime or explicit cost ratios, not invented analyst savings.

Potential scientific contribution: learn and validate **task-specific observation value under shift and missingness**, with a temporally honest comparison between predicted and realized forecast-score improvements. Value-of-information theory, NDVI forecasting and uncertainty estimates are not new. A paper needs a substantive method/result or reusable benchmark that exposes a failure existing evaluations miss—not only a new loss or Gradio wrapper.

### Prior art and the gap

- [GreenEarthNet/Contextformer, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html): multimodal vegetation forecasting already exists.
- [Probabilistic NDVI forecasting, 2026 preprint](https://arxiv.org/html/2602.17683v3): calibrated distributional forecasts are not the new capability.
- [Observability forecasting, 2026 preprint](https://arxiv.org/html/2607.13651v1): forecasting whether an observation will be usable is different from whether it helps a particular scientific task.
- [EO-WM, 2026 preprint](https://arxiv.org/html/2606.27277): weather-conditioned probabilistic surface forecasting and extreme-weather response tests already exist. Do not claim to introduce EO world models.
- [DAWP, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Abstract-Conference.html): irregular multimodal assimilation already exists.

My additional primary-source check found [MIT/JPL/Planet dynamic cloud-avoidance tasking, IW PSS 2025](https://ai.jpl.nasa.gov/public/documents/papers/kacker-iwpss-2025.pdf). Sensor webs and dynamic tasking are established. The possible delta is **realized task-specific predictive information**, not inventing observation selection or claiming NASA demand.

[JRC ASAP](https://agricultural-production-hotspots.ec.europa.eu/about.php) establishes a consequential vegetation-warning investigation workflow. It does not establish demand for this proposed local tool or quantify its savings.

### Existing data and gates

[GreenEarthNet](https://github.com/vitusbenson/greenearthnet) supports optical vegetation forecasting and geographic/temporal splits, but cannot demonstrate radar selection. [EarthNet2023 Africa](https://deepcube-h2020.eu/earthnet-2023-revolutionizing-vegetation-greenness-forecasting-for-africas-ecosystems/) describes multisensor/weather minicubes; its exact released raw scene timestamps, pairing, target support and redistribution rights remain unverified. The older Africa formulation uses different resolution/interpolation and future weather: do not silently treat the variants as identical.

SEN12MS-CR-TS supplies temporal S1/S2 pairs but may pair observations two weeks apart and targets cloud reconstruction, not independent future-state truth. It is not an automatic substitute. Toolkit/code licences do not establish dataset rights. No robust released African multisensor forecasting checkpoint was verified.

Use only weather forecasts actually issued by cutoff, or no future weather. Realized future reanalysis is appropriate for an explicitly conditional hindcast, **not evidence of a deployable forecast**. Missing future optical truth means unknown, not no decline. Do not evaluate interpolated targets that leaked future context.

### First falsifier and public workflow

The agent's 384-*scored*-slot proposal excluded histories from its count. Before approval, replace it with a full input ledger: **12 sites × one cutoff × four availability replays × ten input/verification slots = at most 480 records**, counting masked/repeated slots conservatively. Example slot allocation: five past acquisitions, two concealed candidate acquisitions, two future optical verification dates and one timestamped weather-summary record. This is a proposed counting contract, not confirmed data availability; if adequate context needs more, reduce episodes rather than hide additional acquisitions. Static covariates and provenance must be declared too.

Four development, four calibration and four sealed test sites, nonoverlapping footprints and separately frozen time blocks, are only a small feasibility screen. Do not train a large policy on four sites or claim reliable calibration from four test sites. Data rights, timestamps and raw target support must pass first.

Compare persistence, seasonal/history and small weather-history predictors; stop, random, newest, predicted-usability, uncertainty×usability, always-S1/S2 and greedy information-gain policies. Same observations and budgets for all routes. Evaluate proper forecast scores, paired score improvement, calibration/risk–coverage and performance by sensor/availability. Bootstrap sites, not pixels. Hindsight best action is an upper bound, not an available policy.

Kill scaling if gains disappear against cheap policies, require future-weather/cloud leakage, target truth is sparse, or the sensor/data contract fails. Inconclusive intervals are not a win.

**Space:** no-login, rights-cleared historical examples → timestamped forecast and uncertainty → recommended archived scene/stop → controlled reveal → updated forecast → separate future-truth reveal and downloadable evidence/baseline ledger. That interaction lets people contest an information decision. New-input inference comes only after supported geography/preprocessing are validated; do not call a curated replay a live global service.

## 2. Physical coastal change, not moving pixels

### Scientific question

**When can satellite views support a correctly signed displacement of a surveyed local-mean-sea-level beach contour, after accounting for tide, waves, competing shoreline proxies and shared registration errors?**

An interval spanning zero means direction unresolved, not unchanged. Waterline motion is not necessarily erosion; contour displacement does not alone establish sediment loss, long-term trend or climate/storm attribution.

The plausible contribution is an identifiability/selective-inference procedure with independent survey validation: issue more supportable signed claims at the same false-claim rate than conservative physical-error propagation. A learned selector must beat that physical baseline; a prettier coastline or confidence raster is insufficient.

### Evidence and direct competition

I verified the full [USGS 2025 report](https://pubs.usgs.gov/publication/ofr20251054/full): difficult Barter Island manual screening took as much as **8–10 hours**. It describes multiweek setup, partly learning software and working with developers; completed processing is fast. These are reported workflow burdens, **not our measured savings**. Barter Island is not automatically validation for a sandy-beach model.

[Vos et al. 2023](https://www.nature.com/articles/s43247-023-01001-2), [CoastSeg 2024](https://joss.theoj.org/papers/10.21105/joss.06683), [Mikkelsen et al. 2026](https://doi.org/10.1038/s41598-026-54336-z), [CoastXplain 2026](https://doi.org/10.1109/ACCESS.2026.3656095) and [NatShore 2026](https://doi.org/10.1016/j.jag.2026.105277) narrow the gap. Tidal correction, extraction, uncertainty displays and accessible tools already exist. Evaluate the physical conclusion, not only extraction RMSE.

### Data and pilot

Published surveys include Narrabeen, Torrey Pines and Truc Vert, with Duck source pointers in [SDS_Benchmark](https://github.com/kvos/SDS_Benchmark). I independently confirmed **CC0-1.0** in the official Dryad metadata for [Narrabeen](https://doi.org/10.5061/dryad.28g01) and [San Diego beaches](https://doi.org/10.5061/dryad.n5qb383). The coastal auditor verified Truc Vert OSF CC0; my independent OSF node check found its licence reference, but full licence verification was not needed to declare the whole pipeline approved. Imagery, tides/waves, datum transforms and derivative packaging still need source-specific clearance.

The original coastal audit incorrectly treated reef-lined Waikīkī as a primary sandy-beach holdout. The red team corrected this: it can only be a separately labelled OOD stress case. Do not silently include it to complete the sample budget.

Proposed screen: four sandy sites × four fixed transects × twenty acquisition endpoints = 320 endpoint records; 160 disjoint-pair change records; **480 scored records**, but at most **80 acquisition clusters**, not independent pixels. All additional calibration observations must stay within the declared ledger. Require compatible datum and surveys within 48 hours, or stop/revise the *proposal* before acceptance—not after results. Exact feasible matches are not established. These are already-known benchmark coasts: protocol development is not blind new-coast validation.

Compare CoastSat/CoastSeg QC, tide/slope/wave corrections, correlated Monte Carlo physical bounds, no-change/trend and always-abstain. Measure paired-change interval coverage/width, false signed claims and retained coverage by site/sensor. Kill a learned component if it cannot improve useful coverage at matched error; fail the instrument if physical truth/timing is insufficient. Provisional thresholds in the audit are research choices, not safety certification; uncertainty may be too wide for a verdict.

**Space:** curated paired observations and surveys, or bounded numeric endpoint inputs → physical uncertainty propagation → supportable signed-change interval or refusal → nuisance explanations and provenance export. Initially a CPU/static instrument, not a globally robust raw-image model. Raw-image automation is a later earned capability. No SAR shoreline substitution claim without independent validation.

## 3. Historical rare-event analogue search

Question: can a before/after/later burn example retrieve geographically disjoint, spatially verified analogues better than metadata, dNBR and shape search, while rejecting a gallery without that phenotype?

[MTBS](https://www.mtbs.gov/mapping-methods) offers a plausible Landsat-first archive and strong existing filters. [OpenDPR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_OpenDPR_Open-Vocabulary_Change_Detection_via_Vision-Centric_Diffusion-Guided_Prototype_Retrieval_for_CVPR_2026_paper.html), composed retrieval and [Earth Index](https://docs.earthindex.ai/concepts/running-predictions) already cover substantial retrieval/prototype functionality. A hosted temporal-search Space exists but declares a small synthetic corpus: hosting is not validation.

Space: query an example → a few dated, localized historical analogues → source scene IDs/footprints and no-match when justified. Possible scientific value lies in independently judged relevance, disjoint events and genuine absence—not generic text search or an AI narrative.

The proposed 460-unit screen contains only four queries and two absent-gallery queries. Even perfect rejection cannot establish reliable abstention. Kill if cheap quality-matched search ties, relevance cannot be independently defined or weight/asset provenance fails. This has strong demonstration appeal but lower confidence in a distinct scientific contribution than the two main bets.

## Why not lead with restoration effects or floods?

**Restoration:** promising environmental value, but the current released field tables lack verified satellite joins; separate NatureScot wells are request-gated. Existing Finnish work already predicts peatland water tables from S1/S2/Landsat. Weather-adjusted greenness/wetness is not a causal intervention effect or biodiversity/carbon measurement. Randomized plot treatment contrasts might support a separate causal question if assignment, timing and interference assumptions pass; generic satellite monitoring does not.

**Flood:** new-event transfer, foundation-model probing and OOD abstention already exist. A map-assisted probability-sampling area audit is potentially useful, but the estimator is established. GEOID permanent-water labels partly derive from annual embeddings; its card has different AOIs of one activation in train/test. Expert SAR labels in Kuro Siwo are not independent field/sensor truth. Time-matched independent evidence is the limiting resource, not the encoder.

## Where Clean-JEPA and WAM belong

Clean-JEPA may supply robust sensor-specific features, but its Gaussian regularizer does not establish semantic alignment, correct future dynamics or information value. Q2 remains a deployment safeguard; Q3 becomes an actual sensor-ablation question; Q1 reduces development costs only if empirically demonstrated.

The WAM-adjacent path is grounded representation → probabilistic temporal state → calibrated information-action value → closed-loop replay evaluation. It does not imply physical control, weather intervention or autonomous satellite acquisition. Start with cheap/frozen models; new GH200 training is justified only by a measured failure those models cannot solve.

## Release and next decision

All five audits are bounded searches, not proof that a gap is universally open. No candidate has proven demand, savings, robustness or adoption. No benchmark performance was run here.

Before public release: source-specific rights, immutable geographic/time splits, checkpoint-training overlap audit, supported input contract, independent calibration/test evidence, failure examples and exportable provenance. Public Spaces expose code; hosting entitlement and latency remain unverified. Static examples/numeric inference avoid a large always-on encoder; eligible CPU or bounded Gradio ZeroGPU inference can follow after profiling. Preserve no-login curated replay; unsupported/new inputs must not inherit replay accuracy claims. No arbitrary URLs, archives, executable checkpoints or unbounded rasters; no persistent private coordinate/upload logging.

Immediate recommendation: **a metadata-only feasibility check for forecast/observe**, not a new trainer or Space. If native sensor chronology, future truth or rights fail, stop that multisensor formulation; an optical-only pilot would test a different claim. Coastal matched-survey feasibility is the separate alternative, not a fabricated substitute result.

Six independent sessions ran in the recorded workflow `wfr-4db8a7c042ee4af6b6b4c62b292cc0bb`. The five complete dossiers and sceptical review are supporting attachments. The repeatable audit workflow can be saved as a research skill if the user wants it; no repository skill was installed or committed.

Research only: no dataset/imagery/embedding/weight fetch, model training, HPC access, repository source change, publication or deployment. Primary-source documentation and metadata were read. Local repository checks ran using the documented Python 3.11 virtual environment; optional Torch suites were skipped, so this is not training validation. Pre-existing untracked CUDA failure logs were left untouched.
