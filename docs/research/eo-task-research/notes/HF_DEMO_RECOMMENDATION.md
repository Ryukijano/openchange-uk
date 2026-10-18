# A usable EO demo: keep a mapping task usable through model changes

**Decision brief — 13 October 2026. Research only; nothing deployed.**

## Recommendation

Build a **task-continuity reviewer** for ecological GIS analysts: “Can I still use this habitat/land-cover map after an embedding upgrade?” The browser workflow should preserve a user's labels, task definition and previous results; compare the old pipeline with candidate replacements; and export the evidence for retaining, refitting or retiring the pipeline. Do not silently substitute the newest encoder.

The point is not an adapter demonstration. It is making an existing mapping project reusable without losing track of which decisions remain supported. An adapter that copies old mistakes is not success. If fresh small-head refitting wins, show that and retain the historical record.

**Conditional first bet**, not established customer demand or research novelty. Product and novelty reviewers independently ranked it first; the data reviewer found released annual features the lowest-friction backend. Forest next-evidence selection is the second bet, not the initial alert product.

## The costly workflow and what is actually documented

Ecologists already combine scarce survey labels, satellite features, a small classifier and GIS outputs. The [Cambridge practitioner workshop](https://www.clr.conservation.cam.ac.uk/news/TESSERA-Workshop), 16 June 2026, reports expensive, fragmented monitoring, labour-intensive surveys, inconsistent field/remote-sensing standards and demand for accessible interfaces and uncertainty reporting. Participants included statutory agencies and restoration practitioners. This establishes a real setting, not demand for our particular product or a measured saving.

[GeoTessera](https://github.com/ucam-eo/geotessera) explicitly says independently learned versions and inference variants cannot be mixed. A user changing embeddings must revisit the task pipeline and validation. Current [TEE](https://github.com/ucam-eo/TEE) supplies mapping, spatial evaluation and exports, but classifier evaluation requires the user's compute server. The reviewed code/guide does not establish a held-out task-level migration acceptance workflow; this is a scoped observation, not proof none exists elsewhere.

**No sourced hours or monetary saving was found.** A lightweight head may be cheap to refit. Measure full feature access, adapter fitting, calibration, revalidation, historical re-encoding and analyst time against TEE plus ordinary versioned reports. Preserving a head does not necessarily avoid re-encoding.

## Ranked bets

| Bet | User's job | Decision |
|---|---|---|
| Task-continuity reviewer | Reuse a labelled mapping project and evaluate an upgrade without losing provenance | First, conditional on useful baseline and user demand |
| Forest next-evidence selector | Choose the next archived observation to inspect for an existing alert under a budget | Later/fallback; needs chronology, rights and equal-budget evaluation |
| Generic custom mapper, anomaly map or radar/optical alert map | Make another map | Do not lead with it: strong incumbents already exist |

[TEE](https://github.com/ucam-eo/TEE) already has polygons, classifiers, spatial holdouts, year comparisons and GeoTIFF exports. [TerraMind-AD](https://github.com/edornd/terramind-ad) already supplies temporal anomaly heatmaps/dashboard tooling. [GFW's January 2026 update](https://www.globalnaturewatch.org/blog/data-and-tools/integrated-deforestation-alerts/) combines four disturbance systems, confidence and repeat disturbances, with coverage-specific limits. [FAO Whisp](https://www.openforis.org/whisp/) already has free plot-risk analysis, evidence convergence, “more information needed” and CSV/GeoJSON export. Adding uncertainty colours or a download button is not differentiation.

## What someone would do on Hugging Face

1. **Open a supported example, or upload their dated labels.** GeoJSON/CSV, a fixed class schema, supported region/year and a declarative task manifest. No terminal, visitor API key, executable model upload or arbitrary remote URL.
2. **Review a proposed update.** Compare old features/old head, raw new features/unchanged head, alignment plus old head, and a new-head refit. Same observation year first, so actual landscape change is not confused with model migration. Display negative flips, corrected errors, calibration, coverage and the scope of the available evidence.
3. **Download a reusable project bundle.** Per-site GeoJSON/CSV, a report, head/adapter numeric state where safely supported, and immutable versions, preprocessing, label years, splits, licences and validation records. Keep old results readable. A later year's use requires its own evidence; earlier approval is not permanent validity.

Outputs: **retain old route / candidate meets this limited validation contract / refit preferred / insufficient evidence**. A passing screen is not certification under arbitrary distribution shift. Unsupported geometry, feature version, sensor or geography must produce an explicit refusal, not zeros or a plausible-looking map.

Packaged cases must be real recorded outcomes, including rejected and abstained cases. Do not manufacture a successful migration for the landing page. Separate replay from new-input inference. Private/sensitive locations should not persist by default; disclose server processing and clean transient uploads after export.

## Existing data and model path

**Q2 first:** released TESSERA v1.0/vultr and v1.1/dclimate annual features, which authors release under CC0. Pin both version and variant, normalization, year and spatial aggregation. This is combined model-and-inference-run migration, not a pure encoder experiment. Label source, year, rights and selected-site coverage still need approval and verification. Annual fused S1/S2 vectors cannot be converted into SAR-only inputs.

**Q3 separately:** [TorchGeo PASTIS-R-100](https://huggingface.co/datasets/torchgeo/PASTIS-R-100), CC BY 4.0, provides 100 paired crop time-series samples in a 2,038,579,777-byte archive rather than the 53.7 GB full release. AnySat base is a candidate MIT-licensed pretrained baseline; its weights are 503,756,453 bytes. Combined proposed transfer is 2,542,336,230 bytes before extraction. Exact revisions/checksums and contracts are in the data audit. No download is authorized by this brief.

PASTIS subset geography, dates and class coverage remain uninspected. AnySat pretraining includes PASTIS-HD, so possible imagery/location overlap must be disclosed; this is not automatically an unseen-region benchmark. Simulated optical withholding must be labelled simulated. Crop fallback cannot establish forest reliability.

**Forest is not release-ready:** DISFOR is optical-only, has EUPL-1.2 obligations and centre-point rather than dense-chip labels. MultiEarth manual-mask redistribution remains unresolved. Sen1Floods11 official source/label STAC licences conflict with a permissive HF mirror. Do not resolve conflicts by choosing the convenient mirror. Check ancillary-source obligations too.

**Austria pilot:** the 348-parcel experiment remains exploratory, with weak absolute task quality, overlapping training/alignment sites and no independent calibration cohort. Its label year/rights remain unresolved. TEE's current 2022 warning applies to a DIFFERENT 42,789-polygon `austria.zip`, not proven to be the source of our 349-parcel `austria_crops.geojson`. We have not established a wrong-year mismatch. Neither the ridge result nor published pixel-CV numbers prove safe migration or deployment savings.

## The first experiment and the research contribution

**Proposed feasibility screen:** one useful coarse task, one same-year transition, 240 unique site-year pairs: 40 head-training, 40 unlabelled paired-anchor, 40 labelled calibration and 120 locked test sites across multiple separated blocks. Old/new vectors yield 480 feature records; do not add repeated years or additional observations silently. Site/group roles stay disjoint. If 40 training sites are inadequate, redesign counts BEFORE results and human split acceptance, still within the 500-sample cap.

Use spatial/event-group uncertainty, not neighbouring-pixel random splits. A 120-site test is small: even zero failures in independent trials has an approximate 95% upper bound of 2.5%, and dependence weakens it further. It cannot certify a 2% rare-regression limit. The pilot can reject the idea, not establish operational safety. A later paired-sensor screen must explicitly cap and count all site-date-sensor observations at 500, rather than calling hundreds of long time series 100 independent observations.

Baselines: old pipeline, naive head reuse, identity/Procrustes/ridge, fresh refit, refit plus abstention. Metrics: task quality, calibration/risk-coverage, correct-to-wrong and wrong-to-correct transitions, churn, subgroup failures, feature bytes, total runtime and labelled/analyst effort. Fix acceptable quality and coverage thresholds before scoring; wide intervals mean insufficient evidence.

**Falsifiable Q2 hypothesis:** under matched total budgets, a compatibility-gated route can retain a useful old task head with no practically important held-out degradation relative to refitting, while reducing measured lifecycle cost. Stop if copying old errors is the only apparent benefit, adaptation needs test leakage, refitting is equally cheap/reliable, or coverage collapses. A proposed two-percentage-point non-inferiority margin needs adequate evidence; it is not a result.

Q1 later compares loss/SIGReg-based checkpoint selection with final/random checkpoints and cheap labelled probes, measuring selection regret and TOTAL cost on held-out tasks. Q3 later tests actual modality-specific inputs and gap/recovery regimes against AnySat/CROMA and independent SAR heads. Align the spatial features consumed by the task, not only a global projector; Gaussian marginals do not establish semantic compatibility. Clean-JEPA pretraining is justified only if matched pretrained baselines leave a meaningful gap.

## Build and hosting sequence after approval

1. Freeze the exact labelled task, source contract and independent splits; establish a useful old route.
2. Implement offline comparison, strict manifests and honest recorded cases. No arbitrary pickle/joblib upload.
3. Build the browser workflow and test export fidelity, refusal paths, privacy and capped requests.
4. Launch only after publication approval and checked rights/runtime. Start with static precomputed review if needed; use bounded CPU inference for genuinely new inputs when account eligibility is verified. Static replay is not new inference.
5. Test with three relevant analysts. If at least two see no repeat-use need or prefer ordinary cheap refitting, stop adapter-product work. This is a proposed demand gate, not a survey result.

[HF documentation](https://huggingface.co/docs/hub/spaces-overview) documents free Static hosting and CPU Basic resources around 2 vCPU/16 GB RAM, subject to current compute-plan eligibility. Verify account entitlement before promising Gradio/Docker hosting. Stream bounded features; disk is ephemeral; do not require ZeroGPU for the main flow. At 10 m, a 10 km² region has about 100,000 pixels: two 128-D float32 arrays alone need about 102.4 MB. This is arithmetic, not measured latency. Raw-encoder inference and [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) queues/quotas are optional later.

**WAM progression:** after a validated evidence task, evaluate next-observation policies on a finite dated archive at equal cost. Compare confidence/newest-clear, random, uncertainty and cost-aware policies. Hide future observations, charge requests and labels separately, measure risk reduction per observation/byte and analyst minutes. Only add learned dynamics if it improves this frontier. This is information-action modelling, not intervention on Earth or a current world simulator.

## Evidence and next decision

Three independent reports were read: product workflow, data/model feasibility and novelty/adoption red team. Primary-source facts are distinguished above from proposed gates. User demand, HF runtime, cost savings, validated migration, forest fallback and Clean-JEPA superiority are NOT established.

No new imagery, embedding arrays or weights were fetched; no source code, commits, pushes, cluster access or deployment. Existing local `make check` passed using the documented virtualenv; eight optional Torch modules were skipped. This is repository hygiene, not EO/model validation. Previously uncommitted CUDA logs remain untouched; the earlier SyncBatchNorm failure remains unresolved.

**Next decision:** approve continuity-first local prototyping, or pursue the riskier forest next-evidence bet. Any new data fetch still needs its own exact source/size/rights/split approval; deployment remains separate.
