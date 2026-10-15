# OpenChange-UK: product/workflow scout

**Research date: 13 October 2026.** Research only; four targeted search rounds plus primary-source reading. No repository edits, agents spawned, imagery/embedding downloads, training, cluster access or deployment. Documentation and public application source were read, not executed. The demo and fetches below are proposals, not delivered software or approvals.

## Decision

**First bet: “Can I still use this habitat map?” — a bounded task-continuity review, not another classifier/map builder.** Target an ecological GIS analyst who already has survey labels, a validated small classifier, and a map/report that must remain interpretable after an embedding upgrade. Show *keep the pinned old pipeline / revalidate an adapted head / refit / insufficient evidence*. Preserve old outputs and their provenance; never silently substitute the newest model.

**Fallback: “Which evidence should I inspect next?” for existing forest-disturbance candidates.** Select the next useful observation from a finite, approved archive under a review budget, then export the investigation ledger. Do not compete with GFW on alert generation, radar/optical fusion or confidence colours.

This ranking favours a concrete documented incompatibility over an already mature forest-alert product. It is **not evidence of customer demand**, and light-head refitting may be so cheap that adaptation adds no value. Test that before investing in the shared ViT.

## What incumbents already do

- **TEE is a serious baseline** [S1]. Pins/polygons, imported ground truth, k-NN/RF/XGBoost/MLP and spatial classifiers, geographic holdout, cross-year inference, trained-model downloads, and classification GeoTIFFs already exist. Its *hosted* site serves exploration/labelling, not classifier evaluation or map generation: those require the user's `tee-compute` process. The all-local route needs Python installation and two local services; remote GPU/SSH is optional, not mandatory. Hosting this evaluation is a real convenience improvement, **not new research**. It also sacrifices TEE's local-data privacy unless processing stays in-browser.
- **GeoTessera explicitly forbids mixing versions and inference variants** [S2]. The default changed to `1.1/dclimate`; `1.1/cambridge` is another non-interoperable inference run. The release already prevents mixed-dataset output directories/GeoTIFF merges [S3]. A version-warning UI alone is therefore insufficient. TEE also warns that cross-year class changes can be embedding drift rather than real habitat change [S1]. A *held-out task-level continuity decision* is not documented in the guide reviewed; that is not a claim no competitor has it.
- **GFW/Global Nature Watch already fuses four systems**—GLAD-L, GLAD-S2, RADD and DIST-ALERT—with repeated disturbance tracking, global vegetation coverage, and cross-system confidence [S4]. RADD addresses cloud-obscured optical observations. Multiple-system advantages are concentrated in tropical forests; outside their coverage, global DIST-ALERT is not global radar redundancy. A new generic “cloud-aware change map” is not enough.
- **Places to Watch already prioritises and adds human context** [S5]. The provider describes millions of alert pixels daily, automated screening, then manual contextual/imagery review, resulting in 5–10 quarterly priority locations. These are provider workflow figures, not measured savings available to this project, nor counts of independent disturbance events.
- **FAO Whisp already supplies plot-level evidence convergence and exports** [S12]. Free browser/API/QGIS paths combine public layers, classify low/high risk or “more information needed”, and return CSV/GeoJSON. Users of the App/API/plugin do not need their own GEE account; notebooks do. This further rules out generic polygon-risk review as the gap. It does not establish that every route is anonymous: the FAQ also describes API-key limits. Software's MIT licence does not automatically settle each input-layer's redistribution rights.
- **Flood mapping is similarly crowded**: Copernicus Rapid Mapping publishes delineation/grading and situational reports [S7]; GFM includes flood/water masks, exclusion masks, likelihood, advisory flags, metadata, downloads and report generation [S8]. IBM/NASA has a public Prithvi flood Space [S9]. Its application source expects prepared Sentinel-2 imagery and returns image/prediction/overlay; a plain water-mask demo would duplicate it. Runtime was not tested.

## Ranked concepts: exact users and journeys

### 1. Task-continuity review for an existing labelled habitat/land-cover project

**User:** the GIS analyst maintaining a small conservation/landscape-recovery project's classification and annual evidence pack—not a member of the public expecting arbitrary worldwide mapping. Cambridge's habitat programme explicitly describes sparse/costly survey labels and UK recovery baselines for Cumbria/Cairngorms [S6]; that supports the setting, not adoption of this proposed tool.

**Three steps**
1. Open a bundled public example, or upload labelled GeoJSON **inside supported, pre-approved areas** plus a declarative old-pipeline manifest. Select old/new embedding releases and the task's fixed class schema. No executable pickle/joblib uploads in a public service.
2. Compare the old pipeline, naive old-head/new-feature use, candidate adaptation, and fresh small-head refit on separated calibration sites; inspect disagreements and unsupported locations. Same-year old/new comparison first, to avoid conflating habitat change with model change.
3. Download `site_review.geojson`, prediction/disagreement CSV, and a self-contained HTML/JSON report recording versions/variants, head identity, split IDs, uncertainty, validity scope and the decision. A supported map export is useful later, but GeoTIFF generation alone is already solved by TEE.

**Workaround/cost evidence:** install/run TEE locally or use one's own GPU server; rerun validation and preserve separate GIS outputs. The installation/services are documented [S1]. No sourced number of analyst hours, retraining spend or migration incidence was found. The survey-cost statement [S6] is qualitative.

**Return/output use:** import the old review bundle at the next model or annual update, see which previous conclusions changed, and hand the exported evidence to colleagues. Keep the old pipeline reproducible when replacement fails. **Retaining history is not retaining validity**: an adapted head is a new pipeline requiring independent revalidation.

**No-login path:** bundled example → review → download. Optional bounded polygon upload must disclose whether labels leave the browser and delete transient server data. Sensitive species/landowner locations should not be uploaded casually. No user API key or compute server should be required.

**Differentiator beyond a map:** a task-level compatibility decision and immutable validation/history contract, compared against the cheap-refit alternative. Primary research link is Q2; Q1 can later reduce validation/checkpoint-selection cost. Annual TESSERA features do **not** test Q3.

**Visible failure:** “old head not validated here”, “version/variant mismatch”, “insufficient independent labels”, “coverage missing”, “refit preferred”. Never label a parcel “unchanged” solely because adapted predictions agree.

### 2. Next-evidence selector for existing forest-disturbance candidates

**User:** an NGO monitoring analyst or environmental reporter preparing a small investigation shortlist from GFW alerts—not an autonomous enforcement agent or patrol-routing tool.

**Three steps**
1. Choose a bundled, dated case set or upload ≤50 candidate polygons/alert IDs from a supported region; choose a budget of observations to inspect, not a promise of satellite tasking.
2. For each candidate, see the currently unresolved question and the next available archived S1/S2 observation suggested to address it; review dated evidence and mark “disturbance supported / unresolved / alternative explanation”. Record why radar, optical, or neither is useful.
3. Export a ranked review queue plus GeoJSON/CSV/HTML evidence ledger, with observation IDs/dates, missingness, attribution, analyst notes and unresolved cases. The useful output is a handoff, not another heatmap.

**Workaround/cost evidence:** GFW integrated alerts + Places to Watch/context layers + manual imagery review [S4–S5]. The latter explicitly describes time-consuming prioritisation and manual contextual screening. WRI's 2017 Leuser case reports that the pre-Forest-Watcher workflow supplied new deforestation data only about monthly and covered fractions of the park [S10]; this is historical, and Forest Watcher already addressed offline navigation/report collection. **Do not claim those old shortcomings persist today or invent hours saved.**

**Return/output use:** reopen unresolved cases after new observations; retain what was reviewed and why it remained unresolved. A reporter can cite the dated ledger; a monitoring team can pass the shortlist to human decision-makers. Do not certify legality, cause, or safe field access.

**No-login path:** pre-curated examples and locally downloadable queue. Upstream API access/redistribution may still constrain live custom cases; no-login visitor access does not make upstream data unrestricted.

**Differentiator beyond a map:** better *next evidence to inspect* than “newest clear optical” or GFW's confidence ranking at the same review budget. Utility must be measured against those baselines, not inferred from uncertainty scores. This is an information action, not intervention. Q3 needs actual paired S1/S2 observations; the future WAM link is budgeted observation selection. Q1/Q2 can support reliability across pipeline changes but must earn their place.

**Visible failure:** “no suitable observation in supported archive”, “radar geometry/seasonality ambiguous”, “optical obscured”, “insufficient evidence”; abstain rather than convert no observation into no disturbance.

### 3. Post-event flood-footprint disagreement review

**User:** a local flood-recovery GIS analyst/researcher comparing footprints for a documented past event—not real-time emergency dispatch.

**Three steps**
1. Pick a bundled event and district boundary (or bounded supported polygons), including permanent-water context.
2. Inspect where radar/optical footprints disagree, with acquisition timing, exclusion/no-observation flags and a fixed head's validation scope.
3. Download a district review GeoJSON/CSV and dated evidence report listing supported inundation candidates and unresolved areas; no property-level safety verdict.

**Workaround/cost evidence:** download CEMS/GFM products into GIS and inspect exclusions/dates. Rapid Mapping says delivery is on the order of hours–days and its situational report begins within four hours of activation [S7]. These are *existing service timelines*, not this demo's measured latency, demand, or avoidable labour cost. No direct cost-of-review figure found.

**Return/output use:** compare revisions during recovery and annotate discrepancies for a later event report. Public examples: a finite CEMS activation portfolio, or the India/Spain/USA examples named in the Prithvi Space source [S9], subject to separate licence checks—none downloaded here.

**No-login path:** bundled event → bounded review → export. Do not promise anonymous live GFM downloads: reviewed access guidance describes account/token flows, while the manual also lists STAC catalogues [S8]; actual chosen access route remains unverified.

**Differentiator/failure:** calibrated *task continuity and sensor-disagreement* review, not flood-mask generation or generic confidence maps. GFM already provides advisory/likelihood/exclusion layers, so the burden of proof is high. Display “outside validation”, “unknown in excluded area”, “observed at this time, not maximum event extent”. It fits Q3 but brings stronger safety expectations and operational incumbents; rank third.

## Smallest first milestone and pilot

**First usable milestone after separate data/publication approval:** a public, no-login continuity reviewer for one fixed three-class land-cover task and one same-year embedding transition. Ship three *real, recorded* example cases: pipeline retained, refit/retirement indicated, and unsupported/insufficient evidence. If no migration passes, ship honest failed/abstained cases—not manufactured successes. Downloads are the report, per-site review and complete immutable manifest; arbitrary worldwide coordinates/dates are explicitly unsupported. Optional user input is bounded polygons, not setup instructions or API credentials.

**240-site feasibility pilot, PROPOSED pending human split review:** 80 head-training polygons; 80 separate same-site old/new feature-pair alignment polygons; 40 distinct calibration polygons; 40 untouched test polygons. Use a coarse task with enough labels per class, geographic blocks separating all four roles (including ≥2 test blocks), no overlapping parcels, and grouped analysis. Fit adapters only on alignment sites; choose methods/thresholds on calibration; test once. Use the same **label-matched observation year** on both sides—2022 if using TEE's supplied Austrian crop example, not its older hosted guide's 2024 suggestion. This isolates the *combined model + inference-run transition*, not a pure encoder effect. A future-year test requires dated labels independently valid for that year; never carry old labels forward by assumption. Forty test sites give weak per-class uncertainty; the pilot can reject an idea, not establish operational safety.

Compare old head/old features, unmapped old head/new features, Procrustes/ridge + old head, and fresh head/new features. Start with released baselines; do not wait for, or presume superiority of, teacher-free training. Preserve/version old checkpoints and heads even when refitting wins. Only subsequently compare a cheap Q1 selection rule to the full validation procedure, accounting for feature IO, fits and wall time. Do not use final test sites to select checkpoints.

**Inherited evidence warning:** the supplied 348-parcel Vienna experiment gave RF macro F1 old .127, unmapped .061, Procrustes .060, ridge .124, new-head refit .111. These were supplied to this scout, not rerun. One compact region, overlapping training/alignment locations, no independent calibration sites, a weak base classifier and parcel-IID bootstrap make this a tentative compatibility signal, **not evidence that ridge safely preserves a useful classifier**. The proposed independent-role pilot is meant to address these weaknesses.

**Additional validity gate verified during this scout:** the current TEE repository guide explicitly dates `austria.zip` crop declarations to **2022** and warns that evaluating them against 2024 embeds a different planting. It reports weighted F1 ~.78 (2022) vs ~.30 (2024) in its own run [S1]—not numbers comparable to the inherited macro-F1 experiment. If that experiment used these 2022 labels with 2024 features, its task-performance interpretation must be withdrawn pending corrected label-year evaluation. Actual provenance of the inherited experiment is being checked by the parent; this scout did not inspect its arrays or labels.

**Fallback first milestone:** a finite forest case pack with dated existing alert evidence, a next-observation queue and downloadable investigation records. Use ≤40 independent disturbance/control candidate sites across geographic blocks and available time windows. Compare blinded analyst-supported resolution at a fixed review budget against GFW confidence ordering and newest-clear-optical ordering. If suitable licensed raw paired observations/truth are unavailable, label it an evidence-browser milestone, not validated SAR fallback or a forecasting model.

**Dataset friction, inherited rather than independently rechecked here:** the task brief flags PASTIS-R's 54 GB monolithic archive, MultiEarth 2023's sparse forest truth (11 slices)/unclear manual-mask redistribution, and DISFOR's optical-only inputs. A parallel audit subsequently reported a **PASTIS-R 100-sample, ~2.04 GB, CC BY 4.0** option: integrate its verified source/manifest before assuming the 54 GB route is necessary. It is a paired controlled crop task, not forest-disturbance truth. TESSERA annual CC0 features are a low-raw-imagery migration route, not a substitute for Q3 observations. None of these facts alone clears a public-example licence or source-approval gate.

## Kill criteria and remaining approvals

- **Demand kill:** show the continuity export to three relevant analysts. If at least two do not identify a real repeat workflow or instead prefer a cheap clean refit with ordinary versioned reports, stop adapter-product work. This is a proposed product gate, not survey evidence.
- **Technical kill:** if the old head is not useful on independent geography, do not “preserve” it. If adapters offer no advantage over refitting at comparable labels/compute, retain history/contracts but stop claiming migration is the contribution. Insufficient confidence intervals mean *no go*, not proven safe.
- **Forest fallback kill:** if a blinded review finds no benefit over existing GFW confidence/newest-clear rules, or the queue merely repeats existing advice, do not build another map. Ground-truth disagreement or missing evidence must remain visible.
- **Access/product kill:** no public example/export until source rights are checked. If the main journey requires users to configure compute, find band-correct imagery, supply credentials, or wait for arbitrary large feature downloads, shrink to supported cases rather than call it ready to use.
- **Agenda boundary:** separate S1/S2 stems and availability-stratified SIGReg require modality-specific validation; Gaussian marginals do not establish alignment. Fused annual TESSERA cannot become SAR-only by hiding a channel. No present evidence establishes Clean-JEPA superiority over AnySat/CROMA/TESSERA or a working geospatial WAM.
- **Fetch approval remains separate:** the next request must name exact label source URLs/licence/attribution, selected polygon/site IDs, source release/variant, years, spatial bounds, asset/chunk count, maximum bytes, train/alignment/calibration/test uses, and permitted public derivatives. “TESSERA is CC0” does not approve arbitrary sampling or the parcel-label licence. No label archive, feature array or imagery was fetched here. TEE's `austria.zip` is an example's existence, not proof of redistribution rights.
- **HF constraint:** official docs currently give CPU Basic 2 cores/16 GB RAM/50 GB ephemeral disk, no hourly hardware charge but a paid-plan requirement for Gradio/Docker compute creation; Static is free [S11]. Public app access is documented. Use a static/precomputed no-login review as the first hosting option or verify compute-plan eligibility later. Do not require ZeroGPU for the principal path; queue/quota and sleeping hardware should not block exports. No hosting entitlement, live app, latency, licence or performance was tested here.

## Primary-source ledger

All sources accessed **2026-10-13**. Undated pages are identified as such. Exact claims above are bounded to the portions read; project/provider claims are not independent user studies.

| ID | Source, date | Supported claim / caveat |
|---|---|---|
| S1 | [TEE hosted guide](https://tee.cl.cam.ac.uk/user_guide.html), [current repository guide](https://github.com/ucam-eo/TEE/blob/main/public/user_guide.md), undated | Hosted-only has no classifier evaluation; optional all-local or remote compute; labels, models, spatial splits, GeoTIFF and cross-year use exist. Current raw guide was read, including compute, splits, model/map exports, privacy and drift caveat. Default random/pixel splitting and geographic-split options must not be conflated. |
| S2 | [GeoTessera docs](https://geotessera.readthedocs.io/en/stable/index.html), undated | S1/S2 fused 128-D annual embeddings; independently learned non-interchangeable versions/variants; default 1.1/dclimate, 1.0 frozen. Does not prove an adapter can preserve predictions/validity. |
| S3 | [GeoTessera v0.11.0](https://github.com/ucam-eo/geotessera/releases/tag/v0.11.0), **2026-10-04** (GitHub release metadata checked) | Global 1.1/dclimate default; anonymous HTTPS point/region reads documented; version/variant tags and mixed-output safeguards. No data request performed. |
| S4 | [Integrated disturbance alerts](https://www.globalnaturewatch.org/blog/data-and-tools/integrated-deforestation-alerts/), **2026-01-13** | Four systems, radar cloud advantage, confidence fusion and repeated disturbance; multiple-system benefits limited by coverage. Article's older Amazon nine-day result used three systems, not evidence for a new model/four-system universal gain. |
| S5 | [Places to Watch workflow](https://www.globalnaturewatch.org/blog/data-and-tools/places-to-watch-investigating-deforestation/), **2026-08-12** | Millions of alerts, manual contextual review, 5–10 quarterly selections; aimed at media/advocacy investigations. Not a quantified labour-saving study; alerts do not prove illegal deforestation. |
| S6 | [Cambridge habitat mapping programme](https://ucam-eo.github.io/habitat-mapping/), undated | Field labels described as costly/sparse; embeddings + survey labels and UK long-horizon landscape-recovery settings. No numeric survey cost or migration-product demand. |
| S7 | [CEMS Rapid Mapping portfolio](https://mapping.emergency.copernicus.eu/about/rapid-mapping-portfolio/), page references manual **2025-01-27** | Hours–days service, FEP/delineation/grading, situational reporting within four hours of activation; public products unless sensitive. Not a guarantee for OpenChange. |
| S8 | [GFM v4 Product User Manual](https://extwiki.eodc.eu/gfm_assets/gfm4.0_pum_2025.pdf), **2025-12-01** version; [access guidance](https://global-flood.emergency.copernicus.eu/media/events/4/accessing_and_using_gfm_data.pdf), undated | Read extracted introduction/product/access contents; strong radar incumbent with ancillary uncertainty/coverage layers and reports. Full asset licence, chosen STAC authentication and download performance not verified. |
| S9 | [Prithvi flood Space](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Sen1Floods11-demo), [application source](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Sen1Floods11-demo/blob/main/app.py), undated current code | Prepared 6/13-band S2 input, images/overlay outputs, three country examples. Source read; runtime and example redistribution not tested. |
| S10 | [WRI Leuser/Forest Watcher case](https://wri-indonesia.org/en/insights/no-service-no-problem-bringing-online-forest-monitoring-tools-offline-and-field), **2017-06-12** | Historical monthly manual-data availability, limited patrol coverage, offline app addressed navigation/collection. Not evidence of today's unresolved product gap. |
| S11 | [HF Spaces overview](https://huggingface.co/docs/hub/spaces-overview), [ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu), undated live documentation | Public app access, basic hardware/compute-plan/static distinction, ephemeral disk, sleep lifecycle and GPU quotas. Account eligibility and anonymous end-to-end journey not tested. |
| S12 | [FAO Whisp](https://www.openforis.org/whisp/), [Whisp FAQs](https://www.openforis.org/whisp-faqs/), undated | Free/open-source plot-risk review, convergence of evidence, CSV/GeoJSON; no own GEE account for App/API/plugin, notebooks require own credentials. Existing “more information needed” outcome. Does not imply credential-free API calls or unrestricted layer licences. |
