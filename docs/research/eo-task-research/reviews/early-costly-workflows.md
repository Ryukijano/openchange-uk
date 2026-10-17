# Clean-JEPA: an operational task worth testing

## Recommendation

**Lead: evidence triage for forest-disturbance alerts that keeps working after an encoder upgrade and when optical imagery is unavailable.** The user-facing task is: “Which candidate disturbances should an expert inspect first, and which existing observation would resolve the uncertainty?” Not another forest detector, compliance classifier, or forecast-residual alert system.

Forest triage is the best match to Q2, Q3, and later observation decisions. **Coastal triage has stronger quantified labour evidence. Neither is fully green today:** MultiEarth’s manual-mask licence needs clarification; coastal quality labels do not support a paired SAR/optical geospatial replay. The recommendation is conditional, not a claim of demonstrated savings or novelty.

The proposed work preserves the existing predictor-free Clean-JEPA architecture. Its present encoder is neither temporal nor causal. A temporal predictor and observation policy remain later experiments.

## 1. Forest alert verification and evidence triage

**Verified bottleneck.** Global Forest Watch describes visual verification using high-resolution Planet imagery to determine whether forest loss represents deforestation [1]. This establishes a commercial-imagery/expert-review workflow, but not a measured cost per case. RADD already provides cloud-penetrating Sentinel-1 alerts: approximately 6–12-day observations, 0.1-ha minimum mapping unit, and confirmation within a maximum 90 days [2]. Its Congo Basin validation reports roughly 2% false positives and 5% false negatives for disturbances above 0.2 ha; these are not universal rates. Moisture-sensitive swamp forests and natural disturbances remain caveats. Generic “detect change despite clouds” is therefore a weak contribution.

**Hypothesis.** Preserve an existing optical-trained disturbance head and historical evidence index; map a newer encoder’s optical and SAR representations into that head’s canonical space. Return a ranked review queue, matching historical examples, sensor/date provenance, and abstentions. SAR provides task evidence—not a fabricated optical image. Gains must exceed cloud/quality rules and ordinary uncertainty ranking.

**Existing outcomes.** MultiEarth includes S1 from 2014–2021, S2 from 2018–2021 with QA60, and eleven manually labelled deforestation time slices from 2016–2021, derived from cloud-free Planet mosaics [3]. Ten label dates were produced by Scale AI and one by the authors. Labels are reference masks, not day-exact event times or legal/driver attribution. Known training-file endpoints responded successfully; container listing failed. Its licence table documents source imagery terms but does not clearly state an independent redistribution licence for the manual masks [4].

**Bounded replay.** Use only public labelled material; make new geographical-block and time holdouts rather than rely on hidden challenge truth. Replay candidate disturbances with acquisitions revealed in timestamp order. Score expert-review priority against held-out masks. Join actual historical alert footprints only if coverage and timestamps can be verified; otherwise explicitly call this a *candidate-alert surrogate*, not an operational RADD audit.

## 2. Coastal shoreline-evidence triage

**Verified bottleneck.** USGS reports weeks of workflow setup, then minutes-to-hours analysis; difficult sites can require 8–10 hours of manual screening/filtering [5]. Visual QA and parameter trials are material work. That figure is not a universal per-site cost or an established saving achievable by Clean-JEPA.

**Better task than another QC classifier:** identify which extracted shoreline observations need expert review because they might distort a persistent-change conclusion; show supporting acquisitions and tide/wave context. Do not classify every tidal displacement or cloudy segmentation as erosion.

USGS already releases suitability labels and trained filtering approaches. Its segmentation-QC release contains dated RGB crops, including S2, Landsat and PlanetScope, with geospatial information deliberately removed [6]. These labels cannot independently evaluate a full S1/S2 reveal policy. The SDS benchmark supplies in-situ shoreline series, satellite-derived results, and tide/wave context for Narrabeen, Duck, Truc Vert and Torrey Pines [7].

**Bounded replay.** Leave one benchmark site out; reveal existing dated shoreline estimates sequentially. Compare uncertainty/context ranking, existing QC rules, and proposed evidence ranking at identical review budgets. Evaluate retained shoreline error and long-term trend error against surveys, after consistent water-level handling. Separately evaluate USGS QC classification; do not pretend the two datasets have already been joined. Actual expert-minutes saved requires a timed reviewer study. “Which field survey can safely be skipped?” is unsupported by these labels alone.

**Why it loses as lead:** strongest labour evidence, but generic QC is already addressed, SAR pairing is absent, and released labels do not establish expert-review actions or survey value. It remains the strongest operational fallback if a defensible evidence-to-survey join is available.

## 3. Wildfire perimeter review—not flood mapping

FLOGA covers 326 Greek fires during 2017–2021, with pre/post S2 and MODIS, cloud/water/land-cover masks, and Hellenic Fire Service reference perimeters [8]. The paper explicitly describes manual delineation as time-consuming and unscalable, but supplies no analyst-hour measurement. Crucially, its reference polygons were generated **semi-automatically with manual photointerpretation**, not independent field-survey truth.

The public GeoTIFF dataset declares CC BY 4.0 [9]. Annotation metadata supplies fire start/end dates, optical acquisition identifiers, and candidate pre/post Sentinel-1 identifiers/orbit directions [10]. Candidate S1 identifiers do **not** establish that registered, analysis-ready S1 arrays are already bundled.

**Bounded replay.** Hold out whole fires and a later year; rank boundary segments requiring expert refinement under clear versus cloudy/smoky evidence. Compare spectral burn indices, the released burnt-area baseline, uncertainty ranking, and frozen-head SAR transfer. Reveal only archived acquisitions available at the replay clock; assess perimeter/area error, abstention, observation count and latency. Check actual acquisition gaps before promising early response.

**Why it loses:** attractive bounded disaster demo, but weaker quantified workflow costs, coarse pre/post sampling, label dependence on image interpretation, and an uncompleted SAR data gate. It is a robustness benchmark before an operational savings story.

## Availability and licence gates

| Dataset | Time, outcomes and quality information | Access/licence; remaining gate |
|---|---|---|
| MultiEarth [3–4] | Dated S1/S2 chips; monthly manual deforestation masks; S2 QA60 | Known public files reachable; listing brittle. Source terms documented; manual-mask licence unclear. Verify coordinate/date joins and valid-pixel handling. |
| USGS coastal QC [6] | Dated RGB suitability examples; no retained georeferencing or SAR | Public-domain USGS products with attribution; retain metadata and Planet-derived-image conditions. Not full multispectral arrays. |
| SDS benchmark [7] | Four sites; dated shoreline/survey series; tides/waves | Public repository and evaluation notebooks; GPL-3.0 code. Retain individual survey-source terms rather than infer a blanket raw-imagery licence. |
| FLOGA [8–10] | 2017–2021 fires; pre/post optical arrays and masks; fire/acquisition dates | HF declares CC BY 4.0. Check annotation provenance, revised cloud flags and actual S1-array availability. |
| PASTIS-R [11] | 2,433 aligned S1/S2 patches; dates, crop/parcel labels; five folds | Public 54-GB archive, CC BY 4.0. Best Q2/Q3 technical control; no disturbance, reviewer-action or field-cost outcomes. |

ForestNet offers expert driver labels but Landsat rather than the required paired S1/S2 foundation. FLAIR-2 supplies optical time series and land-cover outcomes, not disturbance-review outcomes. DynamicEarthNet offers monthly semantic change but not paired SAR. SpaceNet-7 concerns Planet building footprints; WorldStrat chiefly pairs resolution levels, not verification actions. These should not displace a task-matched benchmark.

## Smallest useful pilot and honest cost accounting

**After licensing/metadata approval and availability of two frozen checkpoints:** cap the MultiEarth pilot at 1,000 eligible site chips. Train the optical head and paired alignment on earlier dates/blocks, calibrate on separate 2020 blocks, and lock a disjoint 2021 test. Require explicit pairing gaps, validity masks, real-cloud versus synthetic-masking separation, and enough repeated labels to test persistence. Do not infer exact disturbance dates from monthly mosaics.

1. **Q1:** select encoder/checkpoint using predeclared label-free invariance/collapse diagnostics; report downstream results once after selection. This tests—not assumes—that the diagnostics predict usefulness.
2. **Q2:** compare unchanged old head/index, unaligned upgrade, paired alignment, head refitting, and full archive re-embedding. Gate reuse on held-out task calibration, recall and retrieval quality. GeoTessera explicitly forbids interchange of independently learned versions/variants [12]; alignment is a falsifiable experiment, not permission to mix them.
3. **Q3:** apply the frozen optical task head to mapped SAR representations; compare SAR-specific supervision and ordinary paired training. Report missing/cloudy strata separately and abstain when transfer fails.
4. **Evidence budget:** show one case at a time with dates, available sensors, scores and retrieved examples. Evaluate review budgets such as 10%, 20% and 50%; compare random, quality-rule and uncertainty queues. Later policies may reveal cached observations or wait for the next archived acquisition. They cannot access future images/labels prematurely or claim causal action effects.

Measure disturbance recall at equal review budget, calibration, missed-event audits, retrieval relevance, image reads and encoder calls. Count these first; measure reviewer minutes separately. Mapping new queries into an old index can avoid rebuilding it **while preserving old representation content**. It cannot upgrade old archive information. Reusing a head alone does not eliminate archive re-embedding, new-image inference, or adapter-training costs.

**Settled choice:** pursue forest evidence triage across version/sensor changes, with a mandatory MultiEarth licence-and-join gate. Use PASTIS-R only for early technical feasibility. Do not present any of the three as already operationally cheaper. No imagery, training or HPC was used for this research.

## Primary sources — 12 exact URLs

1. https://pro-news.globalforestwatch.org/news-events/how-gfw-pro-supports-eudr-compliance
2. https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert
3. https://ar5iv.labs.arxiv.org/html/2306.04738
4. https://docs.google.com/document/d/1oJhCmR32lTH2TZN43ADXS4x8O7FtTzcI0rOhrCHdvTM/edit?usp=sharing
5. https://pubs.usgs.gov/publication/ofr20251054
6. https://cmgds.marine.usgs.gov/catalog/pcmsc/DataReleases/CMGDS_DR_tool/DR_P1N4VI7H/segmentation_filter_metadata.html
7. https://github.com/SatelliteShorelines/SDS_Benchmark
8. https://ar5iv.labs.arxiv.org/html/2311.03339
9. https://huggingface.co/datasets/orion-ai-lab/FLOGA-GeoTIFFs
10. https://github.com/Orion-AI-Lab/FLOGA-annotations
11. https://zenodo.org/records/5735646
12. https://geotessera.readthedocs.io/en/stable/index.html
