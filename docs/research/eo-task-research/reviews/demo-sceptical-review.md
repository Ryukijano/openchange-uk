# OpenChange-UK: sceptical product and research review

**Reviewed 13 October 2026. Research only.** No repository edits, commits, publication, secrets, cluster access, imagery/embedding-array downloads, or training. Sources below are primary documentation, repository code, model cards and papers; author-reported performance is not independently reproduced.

## Verdict

**Conditional go: a visible, fail-closed continuity gate for a user's custom mapping task. No-go: another polygon mapper or generic radar/optical confidence map.** TEE already supplies the former workflow; GFW's January 2026 integrated disturbance alerts supply a strong incumbent for the latter. Neither reviewed source establishes the proposed combination of a custom task, a fixed labelled acceptance cohort, negative-flip/coverage tests, and an explicit approve-or-reject encoder migration. That is a plausible product gap, not a claim of universal novelty or a safety guarantee.

Start with **Q2, retaining validated heads and prediction history through an encoder update**. Treat Q1, cheaper checkpoint selection, as a supporting experiment. Keep Q3, missing-optical reliability, a separately tested sensor contract: published fused TESSERA annual embeddings cannot be made SAR-only by hiding a channel. Budgeted next-look selection is a promising later direction, not a justified launch claim.

## 1. What TEE actually establishes

Audit snapshot: [TEE commit c787b6b](https://github.com/ucam-eo/TEE/tree/c787b6bd908c79e72d2362b85db0e850861a6313), dated **10 October 2026**, README v1.9.0. Dependency checks below use the tags pinned by that snapshot, not merely dependency `main`.

| Question | Verified evidence | Boundary of the claim |
|---|---|---|
| Does polygon-to-map already exist? | The [repository guide](https://github.com/ucam-eo/TEE/blob/c787b6bd908c79e72d2362b85db0e850861a6313/public/user_guide.md) documents polygon labels/uploads, classification, spatial holdout, year comparison, model download and single-band GeoTIFF export with class/year/CRS metadata. Evaluation uses the user's compute server. | Do not sell these capabilities as differentiation. A managed HF workflow might reduce setup, but its feasibility/cost has not been demonstrated. |
| Is embedding version recorded? | [Label code](https://github.com/ucam-eo/TEE/blob/c787b6bd908c79e72d2362b85db0e850861a6313/public/js/labels.js#L1583) exports `embedding_version` from dataset metadata, or null. | A provenance field is not an acceptance gate; this field alone does not identify the inference variant, checkpoint, sensor pattern or preprocessing. |
| Does import enforce compatibility? | The same code [imports `data.labels`](https://github.com/ucam-eo/TEE/blob/c787b6bd908c79e72d2362b85db0e850861a6313/public/js/labels.js#L1774) and [reconstructs label entries](https://github.com/ucam-eo/TEE/blob/c787b6bd908c79e72d2362b85db0e850861a6313/public/js/labels.js#L1994), including stored embedding vectors. No comparison against the exported top-level embedding version is visible in these paths. | This is a scoped code observation, not proof that no guard exists anywhere or that deployed behaviour was tested. Coordinate-only labels can remain useful when embeddings are recomputed; stale stored embeddings require a different contract. |
| Is the downloaded model self-describing? | Pinned [tessera-eval v1.18.0 server](https://github.com/ucam-eo/tessera-eval/blob/v1.18.0/tessera_eval/server.py#L3535) writes `{"model": clf, "class_names": valid_class_names}` through joblib. | That outer payload does not supply a complete task/data migration passport. We did not inspect every classifier object's internals or run an import. |
| Are versions pinned? | [TEE requirements](https://github.com/ucam-eo/TEE/blob/c787b6bd908c79e72d2362b85db0e850861a6313/requirements.txt) pin tessera-eval v1.18.0 and tessera-vq v0.9.0, with geotessera >=0.11.0. Pinned [evaluation dataset code](https://github.com/ucam-eo/tessera-eval/blob/v1.18.0/tessera_eval/dataset.py#L35) fixes v1.1/dclimate. The [VQ client](https://github.com/sk818/tessera-vq/blob/v0.9.0/tessera_vq/client.py#L274) can query dataset version. | Software/package version, embedding dataset version, and inference variant are different identities. Pins and version queries do not validate a downstream head after migration. |

[GeoTessera's README](https://github.com/ucam-eo/geotessera) explicitly says variants are separate inference runs, do not interoperate even within one version, and training/prediction must use the same `(version, variant)`. The [TESSERA v1.1 model card](https://huggingface.co/geotessera/TESSERA-V-1.1/blob/main/README.md) warns that checkpoint and `data_source` must match because normalization differs. Both read on 13 October 2026. **Equal vector dimensions are not compatibility.**

There is a concrete communication ambiguity: [Anil Madhavapeddy's release note, 12 June 2026](https://anil.recoil.org/notes/tessera-v11-out), calls v1.1 a “drop-in replacement” because the 128-dimensional tile format is unchanged, and says “Accuracy of your tasks should improve in all cases.” The note provides visual/temporal-quality explanations, not a frozen-old-head acceptance study. Same format can support the same loading pipeline while the semantic coordinate space changes. **Do not cite the blog as validation for swapping embeddings beneath an existing classifier.** The operational contract should distinguish format compatibility, same-version temporal transfer, and cross-version/variant head compatibility.

### Material confound in the earlier Vienna result

The [hosted guide](https://tee.cl.cam.ac.uk/user_guide.html) described Austria/2024; the newer repository guide states the sample declarations are **2022**, reporting weighted F1 approximately **0.78 using 2022 versus 0.30 using 2024**. These are documentation-reported values, not our measurements, and not directly comparable to the earlier macro-F1 figures.

The supplied 348-parcel experiment had old RF macro-F1 .127, unmapped .061, Procrustes .060, ridge .124 and new-head refit .111. Its label provenance is unresolved; head/alignment sites overlap; there is no independent calibration set; the base model is weak and the region compact. **Do not interpret this as safe ridge compatibility or evidence for a migration product.** Verify label year first; even with correct years, the design is exploratory.

## 2. Five closest primary-source works/tools

These five cover the direct product competitor, Q1 selection, Q3 sensor flexibility, Q2 regression, and the operational forest incumbent. Dates are publication/revision dates where available; live documentation was accessed 13 October 2026.

| Work/tool and date | Exact supported claim | Implication and caveat |
|---|---|---|
| **[TEE](https://github.com/ucam-eo/TEE)**; audited commit 10 Oct 2026 | Public guide/code cover polygons, classifiers, holdouts, year comparison and exports. | Strongest direct product comparator. The reviewed paths do not demonstrate a labelled migration acceptance gate; absence across all deployments is unverified. |
| **[TESSERA v2](https://arxiv.org/html/2607.03949v2)**; revised 6 Aug 2026 | Authors report 395 Barlow-Twins-family runs evaluated on 15 tasks; pooled loss/performance correlation has magnitude <0.2. Within-budget loss ranking is also weak. They report interannual stability improvements and distilled, dimension-flexible embeddings. | Q1 is not novel merely because loss is a bad selector. Beat a cheap downstream-probe selector on held-out tasks and equal total cost. Findings are family-specific; temporal embedding stability does not prove old-head compatibility. The release/readme does not establish ready coverage for every requested v2 region. |
| **[AnySat](https://arxiv.org/html/2412.14123v3)**; revised 9 May 2025, CVPR 2025 | Multimodal, scale-adaptive JEPA trained across heterogeneous EO datasets/11 sensors; paper reports generalization to unseen sensor configurations. [Official code](https://github.com/gastruc/AnySat) documents modality-specific inputs. | Mandatory strong baseline for the shared-S1/S2-encoder agenda. It does not establish calibrated safety on the proposed custom tasks; channel, resolution and normalization contracts still matter. |
| **[Positive-Congruent Training](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)**; CVPR 2021 | Defines negative flips: old-correct to new-wrong. Error reduction and regression reduction are distinct; focal distillation targets regression, unlike ordinary imitation. | Q2 needs correctness-aware transition metrics, not agreement alone. Experiments are image classification, not EO or sensor loss; zero negative flips is trivially achieved by copying the old model. |
| **[GFW integrated disturbance alerts](https://www.globalnaturewatch.org/blog/data-and-tools/integrated-deforestation-alerts/)**; 13 Jan 2026 | Combines GLAD-L, GLAD-S2, RADD and DIST-ALERT; expands to global vegetation and repeated disturbances; cross-system detections raise confidence. Events within 180 days can be grouped. | Generic radar+optical confidence triage is already served. Cross-system benefits remain limited to tropical forests; elsewhere coverage may come only from DIST-ALERT. Resampling to 10 m does not turn native 30 m evidence into 10 m observations. Its strong false-positive language is a provider claim, not a universal error guarantee. |

**Supporting sources:** [Backward-Compatible Representation Learning, CVPR 2020](https://openaccess.thecvf.com/content/CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html) concerns interoperable old/new visual-search embeddings, not EO task-head guarantees. [CROMA, NeurIPS 2023](https://github.com/antofuller/CROMA) exposes SAR, optical and joint representations: an appropriate sensor baseline, subject to matching input contracts. [RankMe, ICML 2023](https://proceedings.mlr.press/v202/garrido23a.html) supplies a label-free representation-selection baseline; rank alone does not establish semantics. [EO conformal uncertainty, 13 July 2024](https://www.nature.com/articles/s41598-024-65954-w) supports calibrated prediction sets under stated assumptions, not arbitrary spatial/time-shift guarantees. [RADD paper, 2021](https://doi.org/10.1088/1748-9326/abd0a8) and [official product page](https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert) establish the specialized Sentinel-1 forest-alert incumbent.

## 3. Best public demo: “Does this update preserve my mapping task?”

1. Load a small, licensed public example or upload labelled polygons **within a clearly stated supported area/year**. Validate CRS, label year, taxonomy and overlap. A live arbitrary-world/date service is not promised.
2. Fit a lightweight head on the current encoder; show geographically held-out performance, not a training score. Export the baseline map and task passport.
3. Choose a supported candidate encoder. Run the same fixed acceptance cohort; separately report genuinely time-changing land cover rather than calling all map differences model regressions.
4. Display **accepted / rejected / insufficient evidence**, with negative flips, corrected errors, old errors retained, abstentions, coverage and weakest geography/time/sensor subgroup. Unknown contracts block inference or promotion; mere distributional similarity cannot authorize it.
5. Export a ZIP: candidate GeoTIFF with explicit no-data/abstain values, review GeoJSON, site transition CSV and machine-readable passport. Preserve prior predictions and the candidate decision. Require explicit confirmation to promote; do not silently replace history.

The passport binds checkpoint/hash, version/variant, sensors, preprocessing, temporal window, feature dimension, head/class mapping, label provenance, splits, calibration identity and output hashes. Never load arbitrary uploaded pickle/joblib as trusted executable input; use a constrained task format or server-created model IDs.

**User value:** a mapping task that survives a proposed upgrade without silently changing its meaning, plus a reviewable downloadable decision. **Engineering, not research novelty:** metadata checks, version pins, map overlays, ZIP exports, managed hosting and promotion buttons. A research contribution would require a genuinely better label-efficient transition policy, demonstrated against refitting, compatibility training and selective prediction across several tasks/transitions.

## 4. Falsifiable evaluation

### Baselines

For the same fixed cohort and permitted data, compare: (a) old encoder/old head, (b) old encoder/refit head, (c) new encoder/old head only if contract permits—otherwise record refusal, (d) new encoder/fresh refit, (e) ridge and orthogonal-Procrustes adapters using **disjoint** alignment sites, (f) refit/adapter with calibrated abstention, (g) a suitable simple raw-feature classifier. Include AnySat/CROMA for the paired raw-sensor branch; do not compare incompatible observation windows or pretend fused annual features provide a SAR-only ablation. PCT/BCR are training baselines only in a later permitted encoder-training study.

### Metrics that cannot reward preserving mistakes

On labelled sites define `N_CW` = old-correct/new-wrong and `N_WC` = old-wrong/new-correct. Report both counts, conventional negative-flip rate `N_CW/N`, and regression among previously correct cases `N_CW/N_C`. Separately count old-correct→abstain, old-wrong→abstain and wrong→still-wrong. Copying the old model has zero flips but zero corrected errors: **it cannot pass an improvement claim**. Abstaining on everything cannot pass a useful-coverage gate.

Report macro-F1/per-class recall, correction-minus-regression counts, Brier/calibration diagnostics, coverage, selective risk at matched coverage, and worst geography/time/availability stratum. Give both site-weighted and area-weighted results; pixel counts are not independent samples. Use paired spatial-block uncertainty estimates. Map disagreement area is a review statistic, not ground-truth error.

For temporal continuity use same-site/date old/new representations to isolate an encoder update, plus separately labelled future-time observations to test monitoring generalization. True crop rotation or disturbance must not be penalized as a model regression. Keep every site/time observation within one partition.

Pre-register utility/noninferiority margins, maximum regression, minimum coverage and review cost **before** viewing the acceptance cohort; never select sites or tune candidates based on its outcomes. Repeatedly reusing it for development invalidates its holdout status. Example feasibility targets—not universal safety limits—are ≤2 percentage points macro-F1 deterioration, ≥90% useful coverage, and ≤2% negative flips, while correcting old errors or reducing label/compute cost against refit at matched quality. Require uncertainty bounds to satisfy the chosen gate; otherwise return **insufficient evidence**, not acceptance.

### Geography/time shift and Q1/Q3

- Calibrate on disjoint blocks/periods; test on unseen geography and future time. Report subgroup failures. Confidence, OOD scores and conformal sets are triage aids, not certified correctness under arbitrary shift. Unknown sensor/normalization contracts fail closed.
- Q1: compare proposed SIGReg/spectral/other checkpoint proxies with loss, random/final checkpoint, RankMe and cheap fixed downstream probes. Measure ranking on held-out tasks/regions, top-choice regret and **total** proxy/probe/label cost. Gaussian marginals/effective rank do not establish semantic alignment; arbitrary rotations can preserve them.
- Q3: on a separately approved paired S1/S2 dataset, test joint, optical-only and SAR-only inputs plus realistic availability/timing gaps. Compare independent SAR heads, AnySat and CROMA at matched coverage. Cloud absence is not random; radar fallback must not claim it recreates unavailable optical semantics.

## 5. Smallest useful pilot and gates

**Start with 240 unique sites, one binary or small-taxonomy task, one encoder transition:** 40 head-training, 40 adapter-development, 40 calibration, 120 locked acceptance sites in four geographic/time blocks. This is a feasibility screen, not adequate certification of rare failures. At zero failures in 120 *independent* trials, the approximate 95% upper bound is still 2.5%; spatial dependence makes effective evidence weaker. It cannot certify the illustrative 2% gate. Expand only after the screen and separate approval, never above 500 unique sites in this pilot.

1. **Provenance go/no-go:** verify label year and task usefulness of the old model; an old classifier near .127 macro-F1 is not a convincing continuity target. Resolve split leakage and normalize the comparison contract before any performance claim.
2. **Technical go/no-go:** demonstrate one invalid-version refusal, one rejected candidate and one evidence-backed candidate decision, with downloaded artifacts matching displayed metadata. “Insufficient evidence” is a valid result.
3. **Adoption go/no-go:** have a small set of intended users try the polygon→baseline→upgrade decision→export workflow. Continue only if the gate changes a real deployment/review decision and saves effort relative to TEE plus manual validation. Merely finding the maps attractive is insufficient.
4. **Research go/no-go:** after feasibility, require multiple tasks/transitions, geographically/time-held-out improvement and lower labelled validation cost than simple refit plus abstention. A one-region adapter result is not enough.

**Kill or narrow the claim if:** label years remain unresolved; variants/sensors cannot be identified; alignment needs test overlap; a weak old model supplies the only apparent success; gains disappear under geographic/time holdout; regression exceeds tolerance; useful coverage collapses; improvements merely copy old errors; or simple refit+abstention matches the method at equal cost. If encoder continuity works but sensor transition does not, release only the supported encoder audit, not a radar-fallback claim.

**No data fetch is authorized.** A follow-up proposal must specify exact URLs/files or tiles, dates, bytes, license/redistribution terms, transformations, storage and deletion policy for separate approval. PASTIS-R's stated 54 GB monolithic packaging may undermine a tiny pilot; sparse MultiEarth forest truth and unclear mask redistribution undermine a broad forest demo; DISFOR's S2-only scope cannot validate radar fallback. These are supplied constraints requiring source/access verification before selection. No imagery, arrays or model weights were fetched here.

## 6. Ranked risks and the next-look route

| Rank | Risk | Decision |
|---|---|---|
| 1 | **Invalid evidence:** label-year mismatch, overlapping sites and too few independent blocks make continuity look better or worse for the wrong reason. | Resolve before another experiment; show counts/uncertainty rather than a safe headline. |
| 2 | **Adoption overlap:** TEE and updated GFW already provide maps and confidence triage. | Lead with the visible custom-task continuity decision. If users do not need upgrades, stop. |
| 3 | **Research overlap:** backward compatibility, negative flips, multimodal JEPA and cheap selection have substantial prior art. | Demonstrate a cost/quality frontier or a new validated protocol; adding losses or metadata is not novelty. |
| 4 | **Data/hosting friction:** paired-sensor preparation, calibration labels, licensing, coverage and managed inference costs. | Launch bounded examples; verify permissions and HF limits before promising uploads at arbitrary scale. |
| 5 | **Safety/monitoring mismatch:** annual features are not real-time alerts; confidence can fail under shift. | State cadence and supported distribution; abstain/refuse explicitly; compare forest workflows to GFW, not to no incumbent. |

Forest evidence review could be an example of the continuity gate, but **generic confidence-ranked forest triage is not the best standalone bet** after the GFW update.

For information-action WAM, first replay historical observation opportunities under a fixed budget. At each step the policy sees only evidence available then and can request another date/sensor observation from a realistic availability menu. Compare random, recency-only, uncertainty-only, diversity-aware and cost-aware expected-value rules. Evaluate error/risk reduction per observation or byte, time to confident resolution, coverage and calibration at **equal budgets** on held-out regions/periods; prevent future-information leakage. Distinguish requesting data from acquiring a label, and include their different costs. A learned dynamics/world model is justified only if it improves this frontier over simple policies. Observation selection is an information action, not an intervention on the landscape.

## Evidence status

**Verified in primary text/code:** the scoped TEE paths/version pins; GeoTessera's incompatibility warning; the cited papers' methods and author-reported claims; the January 2026 GFW capabilities and geographic limits.

**Recommendation/inference:** continuity-gate product, evaluation design, budget/splits, illustrative thresholds, ranked risks and information-action progression.

**Not established:** universal novelty; any candidate's actual reliability; Vienna label provenance; available/licensed paired pilot subsets; arbitrary-region v2 availability; HF latency/cost; user demand; SAR-only task performance; calibration validity under new geography/time. No training, deployment or performance reproduction was undertaken.
