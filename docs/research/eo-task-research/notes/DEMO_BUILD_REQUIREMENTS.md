# Requirements shared by every demo bet

Status: research/design only. Proposed engineering budgets and gates, not performance claims.

## Product contract
- Primary visitor: an EO practitioner with candidate areas or trusted labels, not a foundation-model developer. Entry journey requires no terminal, cloud-account setup or HPC. Packaged examples must work without provider credentials.
- Show a clear answer, supporting dated observations, explicit limitations and a GIS-usable export. Return insufficient evidence when unsupported. No claim of legal compliance, event cause or future date precision absent valid truth.
- Packaged examples are historical replay; real user inputs must be distinguished from replay. A gallery alone is not ready-to-use for new work.
- Strict supported domain/year/version matrix; cap request area, sample count, files, band count, network bytes and timeout. Coverage and licences checked before loading. Unknown source or user-specified remote URL rejected; prevent SSRF, archive bombs and malicious serialized model uploads.
- Keep user labels/private locations isolated between sessions; do not publish uploads, persist by default or log private geometry. Document retention/cleanup limits. Export a versioned bundle for returning visits; no promise that ephemeral Space disk is durable.
- First result should use a pinned baseline; experimental adapters or fallback must be opt-in with gate outcome and provenance. Unsupported migration retains old model or abstains; it does not silently replace the head.

## Research must change a visible capability
Q2: do upgrades introduce false decisions/alerts? Fit adapters without test labels but validate against independent labelled truth. Report both correct-to-wrong and wrong-to-correct transitions, not churn alone. Retaining a wrong prediction is not success.
Q3: does actual S1 input support the task when S2 is unavailable? Annual fused TESSERA vectors cannot implement SAR fallback. Natural clouds, synthetic missingness and model-version changes must be separated. No generated optical image counts as observed evidence.
Q1: develop cheaper checkpoint selection internally, but never replace release safety validation with loss/SIGReg metrics.
WAM: later use georeferenced uncertainty/dynamics to choose an archived scene or annotation under explicit cost; compare to quality-first, random and simple greedy rules. Cannot inspect unseen pixels/features when ranking candidate actions. No physical Earth intervention claim.

## Evaluation and release gates
- Freeze data licence, schema, geography, dates, labels and split IDs before the next scored run. Pilot at most500 samples with head-training, paired-anchor, calibration and geographically separated test roles; reduced supported map dimension if anchors cannot identify full128D mapping. No unvalidated year-transfer crop labels.
- Establish a useful base task BEFORE interpreting compatibility. Existing Vienna ten-class macroF1~0.13 is not a ready public crop map. Diagnose labels/year, parcel-vs-pixel consumption, feature quality and class support; do not merge classes or move test region merely to improve scores.
- Primary user-work metrics: decisions/errors at matched coverage and review budget, missed-event audits, actual requests/bytes, cold+warm p50/p95 runtime, failure rate. Review counts are proxies, not minutes/dollars. Timed comparison with real practitioners needed for labour savings.
- Candidate migration gate from prior dossier: within2percentage points of matched new-head refit, with paired spatial-block uncertainty. This is a proposed engineering tolerance, not a universal standard; if intervals are too wide, inconclusive. Include pinned-old route and full refit cost.
- Calibration thresholds chosen on calibration sites only. Report risk-coverage and error by sensor, cloud, region, class/event size. With distribution shift, probabilities are not guarantees.
- Demo readiness includes fresh-session example API/UI, new supported input, unknown CRS/bands/version rejection, upstream outage, cancellation, concurrent private sessions, cold restart, mobile layout and export re-import. UI testing approval is separate.

## HF hosting (official docs checked2026-10-13)
- Gradio+CPU is sufficient for precomputed feature/head/adaptor operations, not large EO encoder inference. CPUbasic has2vCPU16GBRAM50GBephemeral disk and no hourly charge, but creation requires applicable paid plan. Static is free; free qualifying personal accounts may host up to2ZeroGPUGradioSpaces. Do not promise freeCPU hosting to any account.
- ZeroGPU optional only for bounded PyTorch encoder inference, not training or arbitrary-area jobs. Gradio-only; queues/daily quotas apply; current anonymous quota2minutes. Keep packaged CPU examples usable when GPU is unavailable.
- A10km2 ROI at10m has100,000pixels. One128Dfloat32 feature array is51.2MB; two are102.4MB before overhead. These are arithmetic RAM estimates, NOT measured transfer sizes or request latency. Stream/quantize and cap bytes rather than eagerly load whole0.1degree tiles.
- Versioned public examples/weights live in pinned Hub repos or approved read-only sources; local cache disposable. User-return bundle contains manifest, safe numeric state and hashes; never execute arbitrary pickle/joblib supplied by visitors. Paid buckets/dedicated GPUs require separate approval.

## Verified sources
- TEE competitor and own-compute limitation: https://github.com/ucam-eo/TEE ; https://tee.cl.cam.ac.uk/user_guide.html
- GeoTessera streaming/grid/caching: https://geotessera.readthedocs.io/en/stable/zarr_quickstart.html
- HF plan/resources: https://huggingface.co/docs/hub/spaces-overview
- ZeroGPU quotas: https://huggingface.co/docs/hub/spaces-zerogpu
- Ephemeral disk: https://huggingface.co/docs/hub/spaces-storage
- Forest visual review: https://pro-news.globalforestwatch.org/news-events/how-gfw-pro-supports-eudr-compliance
- Strong incumbent4-system fusion: https://www.globalnaturewatch.org/blog/data-and-tools/integrated-deforestation-alerts/

No publication/deployment or source approval implied by this design.

## Minimal component boundary for either product
1. Source adapter: reads only approved/pinned public inputs or uploaded numeric/geospatial files; validates CRS, time, masks, bands, units, hashes and class mapping.
2. Task bundle: pinned feature-version ID plus safe numeric head state, label schema, training-region/time support, calibration record and provenance. A changed feature dimension/version is a hard error, not an automatic reinterpretation.
3. Evidence model: predicts only the validated task; returns spatial probabilities, quality flags and support limitations. Optional new-version adapter is fitted only on paired same-coordinate/same-date anchors. Fitting without labels does not remove the labelled safety-gate requirement.
4. Gate: return experimental-migration status, tested support and metrics; retain old route/abstain when unvalidated. No automatic data-version updates.
5. Review/action module: baseline ordering first; later candidate reveal policy reads metadata/previously revealed observations only. Cached unseen features cannot be consulted for free.
6. Browser view/export: map plus dated evidence, errors/unknowns, source/version card, reusableGeoTIFF/GeoJSON/CSV and manifest. Advanced model controls secondary to the main user question.

A compatibility adapter cannot update information in an old archive or guarantee new model improvements. Claims must name the actual avoided operation: some head refitting/history re-embedding, with matched task quality and the full anchor/calibration overhead counted. If the user needs new information from the new encoder, re-encoding may still be necessary.

## Cost evidence cautions
Wageningen's RADD product page quotes approximately$0.0075/km2 for monthly10m EECUbatchcompute (https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert). This is a narrow production-compute statement, not total monitoring/verification cost, and should not be compared directly with our unmeasured demo budget. It argues against assuming generic detection compute is the expensive bottleneck; substantiate expert review/validation costs separately.
The TESSERA v1.1 release blog calls128D tiles a drop-in replacement (https://anil.recoil.org/notes/tessera-v11-out), whereas stableGeoTessera docs warn independent semantic feature spaces cannot mix. Reuse of the disk format is not evidence a frozen classifier remains valid.
