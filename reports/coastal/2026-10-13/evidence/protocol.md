# Coastal three-site CPU screen (independent, exploratory extension)

Written 2026-10-13 before viewing cross-site model outcomes. Earlier Narrabeen outcomes are known; this is not a new independent replication of that beach. No four-site or foundation-model claim. Duck remains excluded.

## Hypothesis H-C1
Metadata plus two-algorithm disagreement predicts signed per-observation shoreline error on an unseen sandy beach better than a calibration-selected non-learned/constant-bias route. Prediction: at least 15% retained-RMSE reduction at 70% whole-scene coverage on both newly acquired test beaches. Negative findings are retained. Imagery is absent; failure cannot justify imagery pretraining.

## Labels and acquisition
Only the user-approved 294 Monitoring_TrucVert files, Grids.nc, and torrey_mapped_sand_elevations.zip; combined survey payload cap 300000000 bytes, expected 277835982 bytes. Source licences, declared sizes, checksums and actual transfer ledger preserved. No downloads inside modelling. Reuse local Narrabeen profiles and benchmark submissions/metadata at fixed SDS commit. Torrey mapped elevations are independent survey-derived, but spatially smoothed products, not exact truth.

MSL: Narrabeen/Truc Vert 0 m in their provided datum; Torrey 0.774 m NAVD88 (benchmark conversion). Exactly one downward crossing after sorting/merging chainage; no extrapolation or bridging missing elevation cells. Exclude profiles with ambiguous/missing crossings. Preserve survey date-only uncertainty; assign UTC midnight as benchmark does, not an assertion of sub-day precision. Primary match ±48 h using total_seconds; count-only sensitivity ±36 h. No widening after seeing counts.

Reconstruct transect origin/direction against supplied benchmark geometry, not against residual minimisation. Validate origins in projected local CRS with grid coordinates. No learned/label-fitted translation. If geometry/datum fails independently, stop that site and report it. Dense neighbouring transects do not count as independent scenes. Grid coordinates, valid ranges and mask semantics will be inspected before labels are accepted.

## Splits
Three fixed folds: test Narrabeen / calibrate Truc Vert / fit Torrey; test Truc Vert / calibrate Torrey / fit Narrabeen; test Torrey / calibrate Narrabeen / fit Truc Vert. Fit dates <=2009-12-31, calibration 2010-01-01..2016-12-31, test >=2017-01-01. No held-out-site labels for fitting/calibration/selection. Cutoffs unchanged from the local temporal screen. No reverse fold added post hoc. Sensors that are absent in fit are reported.

All observations of a site/date/sensor acquisition stay together. Distinct survey-date groups are checked across partitions; sites already disjoint. Treat transects, survey reuse and temporally adjacent scenes as dependent. Optional pilot training size cap: at most 500 rows, achieved only by retaining whole fit scenes in ascending deterministic SHA256 rank until adding a scene would exceed the cap. No label-dependent sampling. Full-scale neural training is out of scope.

## Primary outcome
Equal-site mean of retained-RMSE relative reductions for the two **new** test sites (Truc Vert, Torrey), metadata+disagreement seed 0 versus the calibration-selected baseline, using their own 70% scene rankings. Raw RMSE is row-weighted within site; report actual row/scene coverage. Averaging relative improvements is explicit; Narrabeen, other seeds and modes are exploratory. No hypothesis-test/p-value or broad confidence claim from two test sites. The 15% per-site engineering gate is a secondary decision rule, not universal statistical significance.

## Conditions, in execution order
Baselines first: published tide-corrected CoastSat, fit-site median residual correction, Stockdon setup correction, tide-magnitude ranking, absolute CoastSat–SHOREX disagreement ranking. Select baseline on calibration retained RMSE, fixed name-order tie break. Also report uncorrected observations when present. Missing QC/cloud/georef information means no all-nine-baseline claim. Setup uses fixed benchmark slopes (Narrabeen .1, Truc Vert .05, Torrey .035), not measured image-specific slopes.

Then metadata head and metadata+disagreement head, seeds [0,1,2], HGB quantiles [.05,.5,.95], max_iter100, max_leaf_nodes15, learning_rate .07, min_samples_leaf20, L2=1, early_stopping=False. Equal scene total training weight. Features: tide, swh/mwp/pp1d, wave direction sin/cos, season sin/cos, setup, fixed slope, sensor indicators; optional algorithm disagreement. No site ID, transect ID, coordinates, absolute chainage, year, survey position or matching gap in the model.

Quantiles sorted before calibration. Calibration padding uses nonnegative maximum nonconformity across each scene, finite-sample order statistic at .9. Test width ranks whole scenes; fixed hash breaks ties. Report measured marginal, retained and scene-simultaneous interval coverage and width. Exchangeability across sites/time is not assumed; there is no guaranteed unseen-beach coverage.

## Exploratory analysis and stopping
Per-sensor RMSE, raw position RMSE, ≥100 independent matched test-scenes/site data gate, date-only sensitivity counts, all-row error, pair-claim coverage/false directions using non-overlapping chronological endpoints per transect/sensor and distinct survey dates. Pairs across transects remain correlated; endpoint .9 intervals do not imply .9 joint-change coverage. Scene bootstrap1000, RNG42, is exploratory and ignores additional serial/survey-date dependence. Insufficient split (<10 scenes or <30 rows) stops the affected fold; no cutoff change. No outlier removal based on residual magnitude.

## Reproducibility and checklist
SDS git hash, script/protocol/raw-input hashes, package versions, data provenance, explicit parameters and elapsed/exit status logged. Scripts live outside the Git repo and are uncommitted; no formal publication/operational-readiness claim until the committed-log rule is resolved with user approval. No commit/push/PR/deployment authorised.

- [x] Hypothesis, falsifier and one primary outcome defined.
- [x] Labels/inclusion, geographic and temporal splits fixed.
- [x] Baselines, ablations, seeds, hyperparameters and CPU-only hardware specified.
- [x] No model-outcome peeking or post-hoc condition selection.
- [x] Analysis, budget/stopping rules and negative reporting specified.
- [x] Acquisition versions/hashes saved before label/model acceptance; environment manifest saved before execution.
- [x] Prior Narrabeen test exposure disclosed; this is an exploratory extension.

Protocol ID coastal-three-site-20261013; author Devin; no outcome-driven amendments.

## Pre-outcome implementation clarification, 2026-10-13
Use the benchmark's fixed accuracy-evaluation transects, not all dense neighbouring profiles: Narrabeen [PF1,PF2,PF4,PF6,PF8], Truc Vert [-400,-300,-200,-100], Torrey [PF525,PF535,PF585,PF595]. These choices come from 2_check_shoreline_accuracy.ipynb, not our label results. All site slopes/datum settings remain fixed. Audit all survey files but accept labels only for those transects.

Truc Vert's elevation attributes explicitly state MSL after subtracting .4 m from NGF; do **not** subtract it twice. Grid local cross-shore coordinates start at 20 m; provided satellite transect geometries start at that first grid point. Use direct projected along-transect chainage from the supplied origin, not the untouched grid-x number. Preserve both numbers and the geometry-only offset for audit. No residual-fitted registration.

Torrey public archive download currently fails (API 401, publisher's public HTML download link 403). Truc Vert acquisition completed at 162611806 payload bytes. Without Torrey no registered three-site model fold or primary aggregate will be run. Continue label/geometry/count auditing only while awaiting the archive. Preserve this failed acquisition log; do not silently call a two-site substitute the registered experiment.
