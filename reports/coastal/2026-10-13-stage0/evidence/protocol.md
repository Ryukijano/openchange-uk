# Coastal Stage 0: scoped repository-only screen

Registered 2026-10-13 before residuals or model scores were computed. Exploratory feasibility experiment, not independent public registration or a full validation.

## Amendment to the proposed four-site experiment (input audit, before outcomes)
SDS_Benchmark c0747c52b310433411fc4ab96822ee67441a8d2e includes Narrabeen raw profiles only. Groundtruth pickles are gitignored; Duck, Truc Vert and Torrey Pines require external archives. Waikiki is absent. Deltares and UQMAO are composites, not image observations (UQMAO also uses a high-tide proxy). CASSIE timestamps differ by 12 hours from CoastSat on sampled identical dates. Do not shift those silently. No external archives or imagery will be downloaded in this screen. No all-nine-baseline claim is possible; missing image metadata and registration noise terms are not manufactured.

## Hypothesis and primary outcome
H0-screen: tide/waves/season/sensor and same-acquisition CoastSat–SHOREX disagreement predict errors on later Narrabeen dates. Primary outcome: retained RMSE reduction at 70% scene coverage versus the cheap baseline selected on calibration data. A >=15% reduction is a feasibility signal only; the original K1 requires multiple unseen beaches and cannot be decided here.

## Data and labels
Five Narrabeen profiles (PF1, PF2, PF4, PF6, PF8). Extract the MSL contour at elevation 0 m on the chainage axis. Sort chainages, average duplicate coordinates, interpolate to the benchmark's 1 m grid, retain exactly one downward MSL crossing. No extrapolation below an unobserved contour; omit ambiguous multiple crossings. Also compare direct piecewise-linear crossings as a label sensitivity diagnostic. Profile dates have no time of day: interpret as midnight UTC as the benchmark does, flag the +/-12 h uncertainty, and report a <=36 h match sensitivity count. Nearest survey within exactly +/-48 hours using total_seconds, not truncated integer days. No temporal interpolation of the satellite positions.

Individual corrected MSL submissions: CoastSat Landsat and Sentinel-2, SHOREX, CASSIE. Audit available samples/counts, but the primary model is CoastSat only. Deltares/UQMAO composites excluded from image-error modelling. Match CoastSat and SHOREX by sensor within 10 minutes, not same calendar date; disagreement is not five-method spread. Tide/waves sampled by nearest time within 20 min/60 min respectively. Only observed features; no survey gap, transect identifier, year or survey position as model inputs.

## Frozen splits and controls
- Fit: scene date <=2009-12-31.
- Calibration: 2010-01-01 through 2016-12-31.
- Test: >=2017-01-01. No test tuning.
- All transects from one acquisition remain together; scene equally weighted.
- Baselines: published tide-corrected position unchanged; fit-only median bias correction; Stockdon setup correction; tide-magnitude ranking; CoastSat–SHOREX disagreement ranking. Choose the best cheap method by calibration retained RMSE, not by test.
- Physical RSMA noise baseline unavailable: no positional/registration uncertainty components. Setup alone is not baseline 9.
- Models: scikit-learn HistGradientBoostingRegressor quantiles 0.05, 0.5, 0.95, 100 iterations, 15 leaves, learning rate 0.07, L2=1, min samples leaf=20; seeds [0,1,2]. This substitutes a documented existing CPU quantile implementation for uninstalled LightGBM. Ablations: metadata-only and metadata plus disagreement. No hyperparameter search.
- Features: sensor one-hot, tide, Hs, wave periods/direction, seasonal sin/cos, slope and computed setup; optional two-method disagreement. Missing values remain NaN. No absolute chainage.
- Retain 70% of test scenes ranked by interval width or baseline QC proxy; count actual row coverage. Fixed hash breaks tied rankings; no test labels used.
- Calibration: CQR non-negative widening based on the per-scene maximum nonconformity over available transects, at finite-sample 90% quantile. A temporal shift can invalidate exchangeability; empirical test coverage must be reported, not guaranteed.

## Secondary exploratory outcomes
Bias, all-row RMSE, raw and calibrated interval coverage/width, per-sensor errors, count of independent scenes and survey dates, uncorrected raw-position RMSE, and same-sensor non-overlapping two-endpoint change claims on test dates. Endpoints must match distinct surveys; union-bound pair interval from each endpoint's error bounds; report empirical coverage/false direction and claim coverage, no assumed independent errors. Ninety-percent endpoint intervals do not imply ninety-percent pair intervals. The full K2/K3 cannot pass on one site.

## Analysis and stopping
Scene-cluster bootstrap 1000 replicates on fixed predictions/selections for the primary paired reduction CI; no row-independent CI. Single primary comparison, all other comparisons exploratory. No clipping errors or dropping bad predicted cases. Three fixed seeds reported. Run all fixed conditions; stop only on invalid input or missing features. Negative outcomes and failed runs retained in logs. No GPU job, deployment, repo-source edits, commit or push.

## Checklist sign-off
Hypothesis, primary metric, exclusions, splits, controls, seeds, fixed budget, scene-level analysis, negative-result policy and data commit are specified above. Environment versions and input hashes saved by the script before scoring. Output directory is this folder. Original four-site and baseline 9 gates remain blocked, not marked passed or failed.

## Output-only amendment after first run
Add simultaneous per-scene interval coverage and retained-row interval coverage so marginal row coverage is not mistaken for either. Expand input hashes to all relevant local CSVs. Suppress Pandas mixed-type warning with explicit low_memory=False. No labels, thresholds, splits, model settings or selection were changed; outputs rerun and preserved. HGB early_stopping=False throughout (no internal random validation split).
