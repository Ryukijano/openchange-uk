# Keep forest evidence usable when the sensor or model changes

Research decision, 13 October 2026. Literature/documentation review only. No imagery or embedding arrays fetched, no EO experiment run, no repository source changed, no cluster access.

## Decision

**Lead workflow: forest-alert evidence triage with continuity across encoder upgrades and missing optical observations.** Help an analyst decide which candidate disturbances need closer inspection, using dated radar/optical evidence and a previously validated task head/history. Refuse unsupported transitions rather than silently changing the meaning of the map.

**Verdict: PIVOT toward a cheap compatibility falsifier, not GO for full pretraining.** Generic SAR deforestation detection, optical-to-SAR transfer, and Procrustes alignment are not new. The potential contribution is measurable reduction in lifecycle/review cost while preserving task decisions and avoiding transition-induced false alerts. Neither that benefit nor architecture-specific superiority has been demonstrated.

The smallest first test uses **already-published TESSERA embeddings**, not new satellite pretraining. It isolates our original Q2. PASTIS-R then supplies the controlled Q3 test. Forest monitoring is the eventual utility, not something a static crop benchmark has already validated.

## 1. The costly job and what would become easier

**User:** an environmental monitoring analyst or developer maintaining a validated local forest-monitoring workflow.

**Current job:** inspect flagged regions against satellite evidence, decide whether further imagery or investigation is needed, and keep a consistent historical monitoring record when data/model versions change.

Two primary sources establish different parts of the problem:

- [Global Forest Watch Pro](https://pro-news.globalforestwatch.org/news-events/how-gfw-pro-supports-eudr-compliance) says visual verification is crucial, uses monthly 5 m Planet mosaics, and identifies cases requiring very-high-resolution imagery or field investigation. This establishes a real expert/evidence workflow. **It does not establish a dollar or minute cost per case.** Planet imagery is available within GFW Pro; do not assume every verification requires a separate purchase. Tree-cover loss, deforestation, disturbance cause and legal compliance are different targets.
- [GeoTessera's official documentation](https://geotessera.readthedocs.io/en/stable/index.html#dataset-versions) explicitly forbids mixing embeddings from different version/variant pairs. Even the same coordinates/year have incompatible channel meanings; old heads, clustering and cosine comparisons cannot simply consume new features. **This establishes incompatibility, not the measured cost of a particular forest operator's model migration.**

Proposed simplification: preserve the old validated interface; translate supported new features into its coordinate system; show a bounded evidence-review queue; retain date/sensor/version provenance; abstain when the translation or available sensor is inadequate.

The initial savings hypothesis is reduced head retraining/revalidation and historical archive processing. A later hypothesis is fewer observations or cases reviewed at the same disturbance recall. Neither implies that a field survey can safely be skipped, that SAR substitutes for all optical information, or that the model certifies EUDR compliance.

**Input:** a candidate footprint, replay date, archived acquisitions available by that date, sensor/quality metadata, encoder version, and an existing head/history.

**Output:** spatial disturbance evidence, confidence/abstention, dated supporting and contradictory examples, and a review priority. No invented clear optical image; no precise event date unless reference data supports it.

## 2. Why our existing Clean-JEPA design fits

Preserve the current recipe: separate S1/S2 16-pixel stems, shared ViT, modality and spatial embeddings, two global and six local views, availability-stratified SIGReg, 128-dimensional projector, no teacher/EMA/predictor in the clean main arm. BF16 model computation and FP32 SIGReg/DDP remain engineering choices, not workflow novelty.

Our three original questions have distinct roles:

1. **Q2, head/history continuity — first:** can a small unlabeled paired-anchor set preserve an existing task head across encoder changes without processing the whole historical archive again?
2. **Q3, sensor fallback — next:** can that task retain useful, calibrated decisions from S1 when optical observations are naturally cloudy or absent? Natural clouds and synthetic withholding must be separate results.
3. **Q1, checkpoint selection — supporting:** can label-free diagnostics select an encoder without repeatedly fitting labeled probes? Keep this selection experiment independent of downstream safety labels. TESSERA v2 already reports weak association between pretraining loss and downstream quality; rediscovering that alone is not a contribution.

Alignment must operate on **the features the deployed head actually consumes**. Aligning the pooled 128-D projector does not establish compatibility of a dense patch-token head. The present CLS/view objective is not proof that small local disturbances survive crop/sensor invariance.

For an old linear head with weight matrix W, the same-acquisition logit error obeys

    ||W T z_new - W z_old|| <= ||W|| · ||T z_new - z_old||.

Small feature residuals can still change a decision near a threshold. Fit on paired outputs from the same acquisition and coordinate, then evaluate class-wise margins, calibration and map decisions on held-out geography. Do not align different dates as though actual Earth change were model drift.

Gaussian marginal distributions do not establish semantic correspondence or a universal rotation. Separate sensor strata can have different orientations. Fewer than 128 independent centered anchors cannot identify a full 128-D orthogonal map; even more anchors can be rank-deficient. Procrustes, ridge and a small nonlinear adapter are competing baselines, not assumed solutions.

## 3. Existing data and the exact roles

| Release | Verified content/access/licence | Use and limitation |
|---|---|---|
| **TESSERA / GeoTessera** | [Current project README](https://github.com/ucam-eo/tessera) declares embeddings/weights CC0 and software MIT. [Docs](https://geotessera.readthedocs.io/en/stable/index.html#dataset-versions) list 1.0/vultr and 1.1/dclimate, 128 channels, annual 10 m grids; grid geometry matches but channels do not. v2 coverage is still rolling out. | Cheap same-year Q2 migration test. Verify actual region/year overlap, checkpoint provenance, quantization/scales and preprocessing before choosing a pair. A variant change is not necessarily an encoder change. Annual fused vectors cannot test within-year early detection or SAR-only fallback. |
| **PASTIS-R** | [Official benchmark](https://github.com/VSainteuf/pastis-benchmark), [Zenodo 5735646](https://zenodo.org/records/5735646): 2,433 patches; S2 and ascending/descending S1 time series; crop/parcel labels; CC BY 4.0. Released ZIP is 53,695,489,854 bytes, about 54 GB. | Clean technical Q2/Q3 control. Its crop labels are not forest event or expert-action outcomes. Do not promise selective access to a small subset until the archive/range-access path is verified; no full archive is approved. |
| **MultiEarth 2023** | [Paper](https://arxiv.org/html/2306.04738v1): S1 2014–2021, S2 2018–2021 with QA60; 17,215 training chips of 256×256; eleven deforestation truth slices across 2016–2021. Ten slices manually labeled by Scale AI, one by authors, using cloud-free Planet mosaics. [Data addendum](https://docs.google.com/document/d/1oJhCmR32lTH2TZN43ADXS4x8O7FtTzcI0rOhrCHdvTM/edit) gives Sentinel source terms and unrestricted Landsat with attribution. | Best forest candidate, conditional on coordinate/date joins, validity masks and **explicit manual-mask rights**. The addendum does not resolve those label rights. Eleven sparse dates are not continuous monthly truth or daily onset labels. Known download endpoints exist; exact capped extraction remains unverified. |
| **SSL4EO-S12 v1.1** | Existing approved metadata audit: 246,144 locations × four seasons, paired S1/S2; [HF release](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1), CC BY 4.0. Only metadata access is approved. | Later representation pretraining, not disturbance/review truth. Four seasonal samples cannot validate a dense observation policy. No imagery authorization follows from metadata approval. |
| **DISFOR** | [HF card](https://huggingface.co/datasets/JR-DIGITAL/DISFOR): 3,823 unique S2 time series, disturbance timing/agents, confidence/interpreter/cluster metadata; card declares EUPL-1.2. | Useful forest timing stress test, not an already-paired S1/S2 dataset or tropical dense map. Preserve spatial/temporal clusters and annotation uncertainty. |

Dataset licence is not code licence. Existing labels and published features reduce new collection requirements; they do not remove source approvals or automatically allow redistribution of every derived display.

## 4. Novelty audit

Search covered recent CVPR 2025/2026, ICCV 2025 and ICLR 2026 sources, plus older directly relevant transfer/compatibility work. Absence of a direct match in this review is not proof that none exists.

| Five closest works | Already done; remaining proposed delta |
|---|---|
| [CR-JEPA, 2026 preprint](https://arxiv.org/html/2606.00706) | Modality-specific stems, shared transformer, cross-modal latent prediction, SIGReg and SAR/optical retrieval. Shared-backbone/SIGReg retrieval is not our novelty; test lifecycle-safe task use instead. |
| [AnySat, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.pdf) | Multimodal JEPA and dense transfer across resolutions/sensors. Teacher-free implementation is insufficient; evaluate frozen-head/history continuity and transition risk. |
| [TESSERA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Feng_TESSERA_Temporal_Embeddings_of_Surface_Spectra_for_Earth_Representation_and_CVPR_2026_paper.pdf) | Annual S1/S2 embeddings and lightweight task heads. Our question concerns safe migration of already-validated heads, not another foundation map. |
| [When Embedding Models Meet: Procrustes Bounds and Applications](https://arxiv.org/abs/2510.13406) | Generic cross-model alignment, partial upgrades and interoperability. Procrustes itself is prior art. Proposed delta: task-local spatial safety under combined version/availability shifts. No acceptance claim is made here. |
| [Learning Source Domain Representations for Electro-Optical to SAR Transfer, ICLR 2024 ML4RS workshop](https://ml-for-rs.github.io/iclr2024/camera_ready/papers/56.pdf) | Optical-to-SAR distribution alignment and shared classifier; reduced effective dimensionality helps. Foundation features alone can be inadequate. Our full-dimensional Gaussian projector does not automatically solve this. Test no-new-label spatial continuity and explicit failures. |

Additional overlap: [CROMA, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) for optical/SAR pretraining; [SkySense V2, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_SkySense_V2_A_Unified_Foundation_Model_for_Multi-modal_Remote_Sensing_ICCV_2025_paper.pdf) for shared multimodal backbones; [DyMo, ICLR 2026](https://openreview.net/forum?id=PWhDUWRVhM) for inference-time missing-modality selection; [frugal change-detection active learning](https://arxiv.org/abs/2312.16965) for learned analyst selection. None makes generic budgeted sensing new.

**One-sentence candidate delta:** Unlike generic embedding alignment or optical-to-SAR transfer, preserve a previously validated spatial monitoring interface and historical evidence through model/sensor transitions, with measured transition-induced error, abstention and total lifecycle cost.

**Open gate:** provisional yes for that precise evaluation problem; no for generic alignment/retrieval/all-weather detection. **Contribution gate:** unclear until strong efficiency/safety results or an informative negative result. **Feasibility gate:** bounded released-feature pilot plausible under a week after approval; operational forest experiment not green until label rights and chronology pass.

## 5. Why not the alternatives?

These scores are project-selection judgments, not measurements; 1 low, 5 high.

| Candidate | Novelty | Feasibility | Potential impact | Learning value | Agenda fit |
|---|---:|---:|---:|---:|---:|
| Forest evidence continuity/triage | 3 | 2 | 4 | 5 | 5 |
| TESSERA head-migration falsifier | 2 | 4 | 3 | 5 | 5 |
| Coastal quality filtering | 1 | 3 | 4 | 3 | 2 |
| Generic cross-sensor retrieval | 1 | 4 | 2 | 3 | 3 |

[USGS OFR 2025-1054](https://pubs.usgs.gov/publication/ofr20251054/full) gives the strongest quantified labour evidence: up to an 8–10-hour day screening imagery/detections at challenging sites and weeks establishing/refining the workflow. But USGS already released [segmentation suitability data](https://cmgds.marine.usgs.gov/catalog/pcmsc/DataReleases/CMGDS_DR_tool/DR_P1N4VI7H/segmentation_filter_metadata.html) and trained filtering approaches. Its RGB crops lack the geospatial/SAR inputs needed here. Do not turn an existing QC classifier into a novelty pitch.

[RADD](https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert) already uses cloud-penetrating S1 with 6–12-day revisits and a maximum 90-day confirmation window. It is an operational comparator, not a weak straw man. Its known moisture/baseline/natural-disturbance limitations do not show that our system will improve it.

## 6. Smallest falsifiable pilot

**Proposed, not authorized or run.** Use an existing labelled crop-map region/year and two provenance-verified TESSERA releases for the same coordinates/year. Start with at most **500 distinct geographic sample sites total**, with explicit non-overlapping blocks for head fitting, unlabeled alignment, calibration and held-out testing. Allocate enough rank-supported anchors for a 128-D map; reduce supported dimension or abstain when rank is insufficient. Freeze exact sample IDs, labels, year semantics, spatial buffers, licences and byte cap before any fetch. A same-year static pilot does not satisfy the future-date robustness requirement for a later monitoring claim.

Train one lightweight old-feature task head and freeze it. Compare:

1. old features + old head;
2. new features + unchanged head, no map;
3. Procrustes/ridge/small nonlinear map + frozen old head;
4. new head fitted on new features at the same label budget;
5. later, old encoder retained versus full history re-encoding/retraining, where raw data is actually available.

**Falsifiable prediction:** paired-anchor migration retains held-out task performance within 2 percentage points of matched head refitting, reduces same-year transition disagreement, and costs less locally after anchor preparation and safety calibration. The 2-point tolerance is a proposed engineering gate to freeze before results, not an established field standard. A small pilot may have confidence intervals too wide to establish it.

Measure macro-F1/IoU as applicable, class/boundary errors, wrong transitions against truth, calibration, risk–coverage, bytes read, encoder/head/adapter calls and elapsed local cost. Bootstrap spatial blocks, not neighboring pixels. Same-year map churn is **not** a validated forest alert rate. Only claim savings for the measured workload.

If this fails for TESSERA, report that failure; it does not logically disprove every Clean-JEPA encoder. It prevents treating generic alignment as an already-solved benefit. New Clean-JEPA training needs a separate rationale and approval.

**Next Q3 control:** raw PASTIS-R sensor time series, distinct availability strata, actual timestamps and geographic holds. Do not hide coordinates of fused annual embeddings and call that SAR-only evidence. **Then forest:** legally usable MultiEarth blocks and sparse truth dates, candidate-alert surrogate first unless real alert footprints/times can be joined. Never label surrogate ranking as operational RADD verification.

## 7. Budgets, uncertainty and kill criteria

For a retained old archive of N vectors and n paired anchors, compare measured rebuild cost against anchor preparation, adapter fitting, new-query inference and mapping. An adapter can leave old vectors in place, but cannot improve their old information. Published embeddings already externalize upstream inference: do not count avoided global pretraining/inference as our local saving.

For triage, use identical candidate pools/features/actions for random, chronological/quality-first, uncertainty and greedy expected-error-reduction-per-cost baselines. Report true-disturbance recall versus cases reviewed or archived observations revealed, missed-event audits, coverage/calibration and total replay cost. Report review-count proxies until a timed expert study measures minutes. Stop processing cost must be charged before evaluating a cached feature, not after consuming it for free.

**Abstain** for unsupported sensor/version/dimension, inadequate anchor coverage, uncertain calibration, invalid/misregistered inputs, information unavailable from radar, or decision risk above the held-out threshold. Abstention must not become a way to hide difficult events; report coverage and missed-event audits by regime.

Pre-register these stop rules:

- **One-week pilot box:** stop the compatibility claim if held-out migration is more than 2 points behind matched refitting, fails to reduce transition errors, or uncertainty is too wide to support the claim. Do not change the holdout or tolerance after results.
- **Cost box:** stop the savings claim if anchor creation/mapping/revalidation eliminates the measured local saving. Require a later matched triage policy to achieve at least 20% fewer reviewed cases at fixed 95% disturbance recall, with uncertainty intervals; these are proposed decision thresholds, not results.
- **Data/policy box:** stop forest/policy expansion if rights, replay chronology or sensor joins remain unresolved, or if simple matched rules achieve equal safety at equal/lower cost. Stop any daily onset claim now: the proposed truth does not support it.
- **Architecture box:** if ordinary multimodal models meet the same safety/cost targets, do not claim that Clean-JEPA is necessary. Test a clean objective advantage only at matched data, label and compute budgets.

| Risk | Likelihood | Mitigation |
|---|---|---|
| Representation changes are not a rotation | High | Ridge/nonlinear/retained-encoder controls; held-out task gate, not Gaussianity claims |
| Clouds correlate with geography/season/events | High | Natural-cloud strata and future/geographic holds; separate synthetic interventions |
| Local disturbance information erased | Medium/high | Actual spatial head/boundary metrics and small-event strata |
| Sparse/interpretive truth or unresolved rights | High | Exact truth-date contract; no daily claims; PASTIS technical fallback |
| Review metric does not equal human savings | High | Count proxies honestly; timed expert study before minutes/dollars claims |

## 8. Public demo and WAM path

First demo: select a held-out region; display unchanged-ground map decisions under old features, naïve upgrade and supported migration; show version/provenance, errors and abstentions. This proves compatibility behaviour, not operational forest competence.

Eventual forest demo: select a candidate and replay date; see only then-available S1/S2 observations, georeferenced evidence, persistent-change support and an inspect-more/insufficient-evidence result. Cloud and upgrade controls distinguish synthetic interventions from actual historical conditions. Include cases where fallback fails. Do not display an invented optical reconstruction as evidence.

WAM progression remains representation → separately validated elapsed-time/context dynamics → budgeted reveal/annotation policy → crossed encoder/predictor/policy update gates. PointWAM contributes a common spatial reference, not literal EO point trajectories. RSI-Anything contributes feedback allocation and component-pair testing. Selecting an observation is an information action, **not an intervention on Earth**. Current Clean-JEPA is not a world-action model.

**Next decision:** prepare a bounded metadata/licence/provenance contract for the released-feature pilot. Do not start a large pretraining run to answer a question that existing embeddings can test first.
