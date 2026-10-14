# Registered protocol: narrabeen-duck-20261014

Status: NEW experiment, registered 2026-10-14 BEFORE any model code or realised
Duck counts. It does not amend the failed coastal-three-site-20261013 protocol,
which remains stopped at its data gate and preserved unmodified. This protocol
was designed after an exploratory counts-first audit (counts_first_audit_
20261014.md) that returned only sample-size feasibility numbers — no model
outcomes were seen and none exist to peek at.

## Hypothesis and prediction

H1: a metadata + algorithm-disagreement head predicts per-observation shoreline
error well enough that retaining ~70% of whole scenes by predicted quality cuts
retained RMSE by >=15% on the held-out site, vs the calibration-selected cheap
baseline. This is a project gate, not a universal standard.

## Design: two-site geographic transfer

Sites: DUCK (fit source), NARRABEEN (held-out test). Only two sites have
verified licences, payloads and (upper-bound) sample sizes:

- Fold A: fit DUCK / calibrate DUCK-temporal / test NARRABEEN.
- Fold B: fit NARRABEEN / calibrate NARRABEEN-temporal / test DUCK.

Temporal partitions unchanged: fit <=2009-12-31, cal 2010-2016, test >=2017.
Whole-scene grouping preserved; transects from one scene never split.

Fixed transects (from benchmark accuracy evaluation, unchanged): NARRABEEN
[PF1,PF2,PF4,PF6,PF8], DUCK [-91,1,1006,1097].

## Data contract

- Truth: single downward MSL contour per profile/date, no upward crossing, no
  bridging of missing cells, no extrapolation (unique_crossing, unchanged).
- Duck elevations are NAVD88. MSL contour = -0.128 m NAVD88 (benchmark
  notebook; sites_info contour_level is not used). Survey datum not fitted
  from residuals. Narrabeen profiles are already MSL.
- Duck geometry: FRF profileNumber identifies each line; local xFRF/yFRF
  coordinates projected onto the benchmark transect origin/direction exactly
  as for Truc Vert (no residual-fitted registration). Chainage and off-axis
  coordinates stored separately.
- Matching: nearest survey date per profile/scene, abs gap <=48 h primary,
  <=36 h sensitivity. Survey dates are date-only (midnight UTC) — same
  contract as the failed experiment; disclosed limitation.
- Sources: openchange sources.yaml entries with licence + attribution.
  Duck licence verbatim from file metadata: "may be redistributed and used
  without restriction... properly acknowledge the USACE FRF and the
  supporting investigator(s)". Attribution recorded in sources.yaml.

## Sample gate (hard stop)

Each fold's fit/cal/test partitions need >=30 realised matched rows AND
>=10 whole scenes AFTER contour-quality exclusions — verified on realised
counts, not the catalog upper bound. Failures stop the affected fold; no
window widening, transect change, or region change after seeing counts.

## Baselines and model conditions (unchanged from failed protocol)

Cheap baselines at ~70% retention: unfiltered, tide-only, wave-only,
sensor-only, disagreement-only. Learned: HistGradientBoostingRegressor
quantiles [0.05,0.50,0.95], seeds [0,1,2], max_iter 100, max_leaf_nodes 15,
lr 0.07, min_samples_leaf 20, l2 1, no early stopping, equal scene weighting.
Features: tide, swh, mwp, pp1d, dir sin/cos, seasonal sin/cos, Stockdon setup,
site slope, sensor indicators, optional CoastSat-SHOREX disagreement. No site
ID, transect ID, coordinates, chainage, year, survey position or gap.

Primary outcome: retained-RMSE relative reduction of metadata+disagreement
vs calibration-selected baseline on the held-out site, at ~70% whole-scene
retention, both folds; equal weight per fold. Secondary/exploratory as
before (per-sensor RMSE, pair-claim coverage, scene bootstrap n=1000 seed 42).

## Reproducibility

Same manifest/lock discipline as the failed audit: file hashes, git SHAs,
environment lock, run manifest with model_fit_performed flag, failed-run
logs preserved. No imagery, weights, GPU or cluster work in this phase.

## Pre-registration checklist

- [x] Hypothesis, falsifier, one primary outcome
- [x] Labels/inclusion, geographic and temporal splits fixed
- [x] Baselines, ablations, seeds, hyperparameters, CPU-only hardware
- [x] No model-outcome peeking (none exist); exploratory audit disclosed
- [x] Analysis, budget and stopping rules written before downloads
- [x] Licences verified in-file (Duck) and via snapshots (Narrabeen via SDS repo)
- [x] Prior Narrabeen test exposure disclosed (Stage-0 exploratory screen)
