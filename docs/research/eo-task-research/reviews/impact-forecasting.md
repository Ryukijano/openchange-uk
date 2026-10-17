# Forecast changing land, then choose one observation

**Topic:** `forecast_observe` · **Evidence checked:** 13 October 2026 · **Verdict: CONDITIONAL-GO for a small falsifier, not a live product or new world-model programme.**

## 1. One problem and contribution

**Can task-specific observation value beat cheap sensor-selection rules when forecasting sustained rangeland greenness decline 20–30 days ahead, under missing Sentinel-1/Sentinel-2 observations?**

Restrict the task to African non-crop rangeland patches. Proposed target: the vegetation-area median NDVI is at least 0.10 below its latest reliable value at both day +20 and +30. Freeze this operational threshold before evaluation; it is not an established ecological damage threshold. Harvest, seasonality and cloud artifacts are confounders, so do not call the label drought, forest loss or restoration failure.

At a historical cutoff, choose **stop, process one already-acquired S1 scene, or process one already-acquired S2 scene**. Their contents remain concealed until selection. Predict future decline before and after reveal. This is prediction, not detecting an already-visible change; archive processing/reveal, not ordering a satellite; and an information action, not a land-management intervention. It cannot identify causal restoration effects.

Hypothesis: task-specific expected improvement in a proper forecast score, penalized by processing cost, is more useful than choosing the newest, clearest or most uncertain scene. Contribution would be a model-independent, causally ordered replay protocol and calibrated observation-value evaluation—not JEPA, multimodal fusion or value-of-information theory itself. A negative result would identify when radar or sophisticated planning adds nothing. Researchers might adopt its fixed manifests and paired action outcomes to compare policies; adoption remains untested.

## 2. Real job and potential impact

The prospective user is an analyst investigating vegetation-performance warnings, especially rangeland anomalies relevant to food security. [EU JRC ASAP](https://agricultural-production-hotspots.ec.europa.eu/about.php) provides ten-daily subnational warnings, monthly expert hotspot narratives and high-resolution hotspot inspection, supporting IPC, Cadre Harmonisé and GEOGLAM workflows. This establishes consequential work, **not measured analyst hours, processing expenditure or demand for this particular tool**. No practitioner was contacted.

Success could connect a forecast to a checkable next evidence request, rather than maintaining task heads through an encoder upgrade. That offers potentially greater operational relevance than the continuity reviewer. It is not proven greater impact: local NDVI is a proxy, ASAP operates at larger scales, and no adoption or savings evidence was found.

## 3. Five closest primary works

Methods/data/evaluation passages were read for works 1–4; DAWP's primary proceedings text was also inspected.

1. **Benson et al., GreenEarthNet/Contextformer, CVPR, June 2024.** [Primary paper](https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html). Fine-resolution NDVI forecasting with weather, masked optical history and geographic/temporal holdouts already exists. **Delta:** value of choosing one observation, not another forecast architecture.
2. **Iele et al., Probabilistic NDVI Forecasting, 2026 preprint; inspected v3, 25 June.** [Primary text](https://arxiv.org/html/2602.17683v3). Handles sparse satellite series, weather and quantiles; evaluates CRPS/pinball and strong time-series baselines. **Delta:** calibrated sensor-choice utility on future decline, not probabilistic NDVI alone.
3. **Albughdadi, Observability Forecasting, 2026 preprint; v1 dated 15 July.** [Primary text](https://arxiv.org/html/2607.13651v1). Predicts next usable observation/recovery timing; LightGBM is stronger on the simpler any-usable task and more robust OOD. Uses provided future weather and explicitly disclaims planning. **Delta:** vegetation-task value beyond cloud usability, without privileged future weather.
4. **Luo et al., EO-WM, 25 June 2026 preprint.** [Primary text](https://arxiv.org/html/2606.27277). Weather-conditioned stochastic forecasts already assess decline severity and weather-response direction. **Delta:** selecting information under cost, not claiming that decline forecasting is new. The repository claims NeurIPS 2026; an accepted proceedings publication was not independently verified.
5. **Gong et al., DAWP, NeurIPS 2025.** [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Abstract-Conference.html). Assimilates irregular multimodal satellite observations for atmospheric forecasting. **Delta:** land-specific observation selection, not irregular-input assimilation itself.

Also screened [TAMMs, ICLR 2026](https://openreview.net/forum?id=PXDFTeIqMd): its temporal descriptions/image generation are not calibrated observation-value tests. Classical active sensing already motivates entropy/variance reduction and cost-aware greedy selection; those are mandatory comparators, not novelty claims. No direct matching S1/S2 rangeland policy study was verified within six search rounds. This is a bounded search, not proof of absence; satellite-tasking prior art needs a dedicated review before claiming novelty.

## 4. Data, weights, rights and overlap

**GreenEarthNet, Europe:** 128×128 at 20 m; 30 five-daily S2 frames, 10 context/20 target, four reflectance bands and 150 daily E-OBS weather values. Training: 23,816 cubes, 2017–2020; temporal holdouts 2021–2022; 800 spatial-holdout cubes support geographic tests. Labels are clear-sky vegetation NDVI, not damage truth. [Repository](https://github.com/vitusbenson/greenearthnet) documents split access and a 2.3 GB pretrained-model archive at [Zenodo](https://zenodo.org/records/10793870), whose software/weights record declares MIT. This is not proof of dataset redistribution rights. Contextformer has known European training support; never call its original locations independent tests. It cannot validate S1 selection or African rangeland transfer.

**EarthNet2023–Africa, toolkit release:** [Official project description](https://deepcube-h2020.eu/earthnet-2023-revolutionizing-vegetation-greenness-forecasting-for-africas-ecosystems/) describes 128×128 at 10 m, five-daily imagery, one-year context/three-month forecasts at ten-daily steps, S1/S2, ERA5 and static covariates. Released calendar years, raw scene timestamps, pairing and native label provenance remain unverified. The [earlier 2022 paper](https://arxiv.org/html/2210.13648) instead describes 30 m, 45 ten-daily steps, 2016–2021 and interpolated optical targets; its locations can cross train/test years. **Do not equate these variants.**

The [toolkit](https://github.com/earthnet2021/earthnet-toolkit) exposes `earthnet2023` train/test downloads and recommends 1.5 TB. A metadata-only [bucket listing](https://s3.bgc-jena.mpg.de:9000/earthnet?list-type=2&prefix=earthnet2023/&max-keys=5) confirmed individual country/patch NetCDF objects, so selective object access exists; exact scene support inside them was not inspected. [Toolkit DOI](https://zenodo.org/records/10659371) declares CC BY-NC-SA 2.0; source code LICENSE says CC BY-NC-SA 4.0. Neither establishes the archive's upstream imagery/derived-data redistribution rights. No suitable African multisensor checkpoint was verified.

**SEN12MS-CR-TS:** [Primary paper](https://arxiv.org/html/2201.09613) and [official access page](https://patricktum.github.io/cloud_removal/sen12mscrts/) describe 53 global ROIs, 30 intervals in 2018, 15,578 patch sequences, 256×256 at 10 m, S1 VV/VH and 13-band S2 L1C. Pairs may differ by two weeks. It targets cloud reconstruction, not future-state truth. Upstream redistribution rights remain unresolved; mirrors cannot settle them. Its pretrained cloud-removal models are not independent forecasting baselines.

**Existing forecast weights:** [AgriMatNet model card](https://huggingface.co/ireneiele/agrimatnet-vegetation-forecasting) declares MIT and exposes safetensors; 28-feature input, three quantiles, width 128/eight layers. European EarthNet-derived training support may overlap tests; exact patch/year overlap needs checking. Availability is verified, robustness is not. [EO-WM](https://github.com/Luo-Z13/EO-WM) currently promises full weights later. No data or weights were downloaded.

## 5. Falsifier: at most 384 scored observation-slots

Pre-register 12 nonadjacent sites, two cutoffs each and four history-availability strata: both sensors, S1-only, S2-only, neither. Count two candidate sensor/date observations plus two future optical target dates for **every** stratum: 12×2×4×4 = **384** scored slots. Repeated sites, dates, sensors and masked replays are counted, not advertised as independent samples. Histories are inputs, not additional scored targets; pixels are aggregated, not extra replicates. All policies share these slots. No extra scoring for tuning.

Freeze a metadata-derived manifest before viewing outcomes: four geographically separated development sites/earliest year, four calibration sites/later year, four test sites/latest year; no overlapping footprints, histories or target windows. Exact years depend on verifying the released archive. Disclose pretrained overlap; exclude it from independent claims. Small site count makes this a falsifier, not publication-grade validation.

Compare persistence/seasonal trend, tabular weather+history forecasting, and compatible frozen pretrained forecasts; policies: stop, random, newest, catalog-confidence, uncertainty×predicted usability, always-S1/S2, and cost-aware greedy information gain. Use only information timestamped by cutoff. No retrospective cloud masks, target-guided episode selection, interpolated future context or realized future weather. Missing target truth is unknown, not no-decline.

Report Brier/log loss, quantile pinball/coverage/width, paired net-score improvement across processing-cost ratios, and sensor/stratum performance. Calibrate only on calibration sites; bootstrap by site, not pixels. Costs are measured runtime/bytes or explicit hypothetical ratios—not money saved. Hindsight best-action/full-reveal results are upper bounds only. **Kill** if rights/chronology fail, truth is mostly unobservable, gains vanish against cheap policies, calibration fails held-out, or improvement requires future weather/cloud leakage. Wide intervals mean no demonstrated advantage, not success.

## 6. A useful, bounded Space

**Input → forecast → evidence request → reveal → verification.** No-login curated licensed replays show timestamped history, decline probability/interval, recommended archived scene and processing-cost sensitivity. Reveal updates the forecast; reveal withheld future truth separately. Export a provenance manifest, scene IDs, masks, forecast and paired baseline scores so users can check the decision independently—not merely view a map.

New input: validated numeric JSON/CSV patch summaries with fixed schema, dates, sensor units and provenance; cap rows/file size, reject URLs, archives, executables and arbitrary rasters. No persistent uploads or precise private-location logs. Abstain for unsupported geography, absent history, bad quality or uncalibrated shift. Distinguish unsupported-input abstention from no-decline.

CPU supports replay/tabular inference plausibly; latency is unmeasured. A small quantile model could use bounded [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) calls after profiling, not a 387M-parameter video diffusion default. Label everything **historical archive replay**; no live forecast, acquisition booking or intervention recommendation without a separate verified integration.

## 7. Subjective scores and decision

**Potential impact 4/5:** consequential warning-investigation job, proxy and adoption unproven. **Scientific openness 3/5:** task-conditioned observation value may remain open; forecasting/usability are crowded. **Feasibility 2/5:** rights, chronology, weather leakage and target sparsity are serious gates. **Demo usefulness 3/5:** independently checkable reveal decisions; bounded replay, not operational coverage. **WAM fit 4/5:** future state plus information action; no physical control or causal restoration.

**CONDITIONAL-GO only for the falsifier.** Resolve exact African archive variant/rights, raw temporal provenance, checkpoint overlap and active-sensing prior art first. Otherwise pivot to an explicitly optical-only protocol, not pretend GreenEarthNet proves S1/S2 utility. No training, implementation, deployment or adoption claim is justified yet.
