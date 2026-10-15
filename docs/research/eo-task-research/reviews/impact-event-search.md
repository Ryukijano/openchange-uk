# event_search — verified historical burn-transition analogue search

**Research audit, 13 October 2026. Verdict: CONDITIONAL-GO for a falsifier only.** Do not start a new foundation model or promise archive-wide deployment. No experiments, data/weight downloads, repository changes or practitioner contact were performed.

## 1. One question and contribution

**Can one georeferenced before/after/later burn exemplar retrieve cross-site analogues with independently verified footprints, while rejecting galleries without that event phenotype, better than spectral/shape and frozen-feature search at five inspected results per query?**

Restrict “analogue” to preregistered observable burn morphology and severity pattern—not shared cause, restoration effect or ecological outcome. Candidate rare phenotypes are fragmented burn mosaics with unburned islands and narrow elongated footprints; their actual scarcity must be measured, not assumed. Clear 30 m Landsat observations are the first scope; SAR/cloud penetration is not claimed.

Hypothesis: exemplar-conditioned change features plus temporal/quality verification improve *verified* precision@5 and abstention over an equally quality-gated cheap baseline. A matching heatmap alone is insufficient. Contribution would be a disjoint-site benchmark combining query relevance, footprint verification and genuine gallery-level absence. A negative result that ordinary dNBR/shape retrieval suffices is valuable only if labels, splits and failure cases are reusable. No universal novelty is claimed from this bounded audit; Clean-JEPA is neither required nor established as suitable.

## 2. User, job and possible impact

The proposed user is a fire-effects researcher or imagery analyst assembling comparable historical disturbance cases. [MTBS mapping methods](https://www.mtbs.gov/mapping-methods) document image selection/preprocessing, on-screen boundary delineation, severity interpretation and cloud/shadow/snow/anomaly masking. This verifies a manual evidence-analysis workflow, **not demand for this particular analogue-search tool**, nor measured hours or money saved.

If useful, this addresses the substantive job of finding evidence across archives rather than the continuity reviewer's narrower encoder-migration maintenance job. That is potentially greater reach, not demonstrated impact. MTBS already identifies fires; simply finding a named fire adds little. The value must be finding an unusual cross-site pattern that existing fire metadata filters cannot identify efficiently. Compare inspected candidates, not invented labour savings; analyst utility remains unverified.

## 3. Closest primary work and incumbents

Methods/data/evaluation were inspected for the five works below, particularly Text-ITSR, composed EO retrieval and OpenDPR.

| Primary work; date/status | What it already establishes; remaining delta |
|---|---|
| [RSICRC](https://github.com/rogerferrod/RSICRC), Discovery Science 2024 | Bi-temporal change captioning/text retrieval with a RemoteCLIP-derived backbone on LEVIR-CC. Delta: cross-site visual event analogues with independent footprint/null verification, not captions. |
| [Text-ITSR](https://arxiv.org/abs/2501.19043), January 2025 preprint | Text↔bi-temporal retrieval on LEVIR-CC and Dubai CCD, **including no-change pairs**; evaluation uses caption metrics over top-five matches. Delta: visually queried analogues and calibrated gallery-level “none,” not merely adding no-change examples. |
| [Benchmarking Composed Image Retrieval for Applied Earth Observation](https://arxiv.org/abs/2605.24442), May 2026 preprint | Scene-plus-target-state retrieval, including xView2-CIR same-scene post-disaster retrieval and training-free baselines. Delta: geographically disjoint historical analogues, not recovering the same scene. |
| [OpenDPR](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_OpenDPR_Open-Vocabulary_Change_Detection_via_Vision-Centric_Diffusion-Guided_Prototype_Retrieval_for_CVPR_2026_paper.html), CVPR 2026 | Separates change proposals from category identification and uses visual prototypes. Delta: archive ranking and absence calibration; proposal localization/prototype matching are not new. |
| [Finding Change in Satellite Archives from Text](https://arxiv.org/abs/2607.28571), July 2026 preprint | Subtraction prefilter plus fusion reranking reduces query cost 10–15× in its benchmark. Delta: verified analogue relevance/nulls, not another efficient text-search index. |

[Earth Index](https://docs.earthindex.ai/concepts/running-predictions) already searches embedding neighbours from positive/negative labels with confidence filtering and iterative refinement. [GRAFT](https://graft.cs.cornell.edu/), ICLR 2024, establishes ground-to-remote open-vocabulary transfer. [GeoQuery](https://arxiv.org/html/2605.05405v3), May 2026 preprint, adds text-aligned global EO search using frozen CLAY embeddings, but reports difficulties with ephemeral wildfire scars in RGB. Neither generic example search nor text search is an open gap.

The public [Open Vocabulary Temporal Change Retrieval Space](https://huggingface.co/spaces/panagiotis427/Open_Vocabulary_Temporal_Change_Retrieval) was reported RUNNING by Hub metadata. Its README offers before/after views, heatmaps and export, but explicitly uses a synthetic two-site/eight-month corpus. Its [upstream documentation](https://github.com/Panagiotis427/msc-gbda-ov-temporal-change-retrieval) distinguishes relative scores from calibrated probabilities and acknowledges patch-token and site-leakage problems. This is a direct interface incumbent, not proven product performance; UI interaction was not tested. TEE/GFW/Whisp and flood-mask Spaces remain adjacent incumbents, not evidence of this specific retrieval gap.

## 4. Data, rights, weights and leakage

[MTBS](https://www.mtbs.gov/faqs) maps large US fires from 1984 onward across CONUS, Alaska, Hawaii and Puerto Rico: generally ≥1,000 acres west/≥500 east. [Products](https://www.mtbs.gov/product-descriptions) have 30 m pre/post Landsat or Sentinel imagery, NBR/dNBR/RdNBR, severity classes, boundaries, masks and scene metadata. Prefire seasonality is matched to postfire phenology; initial/extended assessments differ. These are analyst-interpreted products, not perfect independent ecological truth.

Subset path: [official download entry](https://www.mtbs.gov/direct-download), now consolidated into the Burn Severity Site; select individual fire IDs and Landsat-only products, then identify an additional Landsat Collection 2 surface-reflectance observation. No subset was acquired. Freeze fire IDs, scene IDs/dates, sensor, collection/processing versions, assessment type, masks and hashes. Restrict the pilot to Landsat 8 OLI to avoid treating cross-sensor appearance differences as events. A later clear observation is **not guaranteed** for every selected case.

[USGS policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits) places USGS-produced information in the US public domain but excludes some third-party material. Check each asset's origin/metadata; “freely downloadable” and a mirror's licence are not redistribution clearance. Exclude Sentinel-derived assets initially. TS-SatFire/FLOGA are alternatives, not selected: convenient Kaggle/HF access and FLOGA's MIT code do not settle upstream imagery rights.

[RemoteCLIP](https://github.com/ChenDelong/RemoteCLIP) supplies RN50/ViT-B/32/ViT-L/14 checkpoints; code is Apache-2.0, while the inspected HF card did not declare a weight licence. [RSICRC's HF card](https://huggingface.co/RogerFerrod/RSICRC) declares MIT and LEVIR-CC fine-tuning; that does not license LEVIR imagery. Geographic overlap between pretrained data and MTBS sites is unresolved. Neither is a validated checkpoint for this task; audit checkpoint-specific terms and training lists before use, report opaque overlap explicitly. Annual fused TESSERA features cannot substitute for dated observations or SAR-only inputs.

## 5. Bounded falsifier: 460 scored units

Proposed—not executed—manifest: 64 unique site windows, each with exactly three observations: 192 site/date/sensor units. Four separate one-example query sites also have three observations: 12 units. Score all four queries against all 64 windows: 256 adjudicated relevance decisions. **Total 460.** Masks, quality and localization use those same observations/decisions, not extra scenes; second sensors/additional dates require replacing units. Models reuse identical decisions, not independent replication.

Freeze development: 16 California windows, 2015–16; calibration: 16 Oregon windows, 2017–18; test: 32 Colorado windows, 2019–20; queries: four Utah windows, 2014. Require ≥100 km separation between partitions and no shared fire/perimeter. All observations remain within their partition's time range. Deterministic selection rules and the full manifest precede model scoring; if support is insufficient, stop rather than move test boundaries after seeing results. These geographic shifts also test ecosystem transfer, a possible confound to report.

Two queries have gallery analogues; two are absent from the test gallery. Include confirmed other-fire patterns and independently reviewed cloud/phenology/misregistration/non-fire negatives. **Not in MTBS ≠ no fire.** Blind dual review plus adjudication establishes observable relevance using imagery/perimeters; MTBS severity/dNBR cannot alone define both baseline and truth. Gallery labels never enter retrieval features.

Baselines: MTBS metadata filtering; raw dNBR/NBR-distribution nearest neighbours; spectrally derived change-component perimeter/shape descriptors (not MTBS masks); frozen RGB and before/after-feature differences; temporal similarity with identical quality gates. Compare verified precision@5, accepted-mask IoU, false acceptance and abstention coverage. Predeclare IoU≥0.5 as the localization criterion, not an observed result.

Report paired differences, site-cluster uncertainty and per-query results; do not bootstrap correlated pairs as 256 independent cases. With only two absent test queries, even zero false accepts gives a one-sided 95% binomial upper bound near 78%: **this pilot cannot validate reliable no-match calibration**. Kill scaling if either positive query fails to gain at least one verified top-five hit over the best cheap baseline, any absent query is accepted, or gains disappear after leakage/quality controls. Insufficient labels or wide uncertainty means inconclusive/no scaling, not a positive claim. Thresholds are proposed decisions, not results.

## 6. Space: input → bounded inference → verifiable evidence

No-login curated examples query a fixed licensed historical index; display “historical replay,” never “live global search.” CPU cached-vector ranking covers at most 10,000 precomputed windows, with three exemplar crops maximum. Optional new-input inference is bounded to three ≤512×512 supported six-band GeoTIFF crops, ≤20 MB total, explicit CRS/dates/sensor/version; malformed, remote-referencing, executable, archive and unsupported files are rejected. ZeroGPU embedding is capped at 30 seconds; timeout abstains. These are design limits, not measured latency or assured hosting entitlement.

Return ranked coordinates, scene IDs/links, before/after/later dates, sensor/version, independently interpretable evidence masks, scores, calibration status and cloud/registration flags. No VLM story or unsupported causal explanation. New uploads stay session-local, are deleted after inference, excluded from training/logs/index, and carry a privacy warning. Unknown sensors/poor registration abstain. Small calibration displays counts/intervals, not probability theatre. Replay remains available when GPU quotas prevent new-input inference.

## 7. Judgments and decision

**Subjective 1–5:** potential impact **4** (direct evidence-search job, adoption unproven); scientific openness **3** (crowded retrieval literature, narrow protocol gap); feasibility **3** (public imagery, difficult relevance/null truth and overlap); demo usefulness **4** (inspectable reusable search, not a map dashboard); WAM fit **2** (information action, no dynamics/control demonstrated).

**CONDITIONAL-GO only for the above screen.** The strongest unknowns are natural analogue-search demand, phenotype rarity, upstream weight rights/overlap, independent nulls, and relevance beyond cheap metadata/spectral/shape matching. Larger archives and trustworthy abstention remain untested. If simple search suffices, **PIVOT to a reusable benchmark/negative result**, not an embellished retrieval product. Researchers would adopt it only if the manifests and labels expose failures their caption/same-scene benchmarks miss; adoption is a hypothesis too.
