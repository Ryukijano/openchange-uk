# Narrabeen+Duck: eligible data, failed learned-head gate — 2026-10-14

## Completed model comparison

The [new protocol](code/protocol_narrabeen_duck_20261014.md) required at least 15%
retained-RMSE reduction on **both** held-out beaches, at roughly 70% scene
retention. The saved [fold results](evidence/fold_results.json) fail that gate:

| Fold | Calibration-selected baseline | Baseline RMSE | Metadata + disagreement RMSE | Reduction |
|---|---|---:|---:|---:|
| A: fit Duck, test Narrabeen | `unfiltered` | 7.210 m | 6.830 m | +5.3% |
| B: fit Narrabeen, test Duck | Wave-only | 6.961 m | 7.960 m | −14.3% |

Mean reduction −4.5%. The metadata-only condition improves A by 3.4% and worsens
B by 27.0%. Seeds 0/1/2 are identical under these deterministic settings, not
independent replications. Nominal 90% quantile intervals cover 68.3% and 79.1% of
retained rows in A/B. No conformal widening is applied by this runner. Do not
interpret marginal row coverage as scene, pair or new-site coverage.

The [archived code](code/run_folds.py) fits on the source beach through 2009,
selects the baseline on that beach's 2010–2016 calibration partition, then tests
the other beach from 2017. It selects whole scenes using absolute predicted
median error, but scores the original shoreline residuals without learned bias
subtraction. Fitting is scene-weighted; reported RMSE is row-weighted. No test
outcomes select the baseline.

Implementation caveats preserved rather than changed post hoc:

- `unfiltered` is a constant-score, fixed-order subset at ~70% retention, not
  all-scene RMSE or a random baseline. Ties follow the archived sort order.
- Planned bootstrap, per-sensor and pair secondary analyses were not emitted
  by this runner. There is no claimed confidence interval or full protocol
  completion for those secondary outcomes.
- The saved JSON is the original machine-readable run evidence. A full original
  stdout log, original fold-run manifest and per-scene predictions are not in
  this archive; publication-time hashes do not retroactively supply them.

This is a negative result for this fixed metadata/disagreement configuration,
not proof that imagery will solve the problem or justification for pretraining.
[Current interpretation and limitations](../../../docs/PROGRESS.md).

## Original registration and audit notes

Exploratory counts-first audit, then a NEW registered protocol
(code/protocol_narrabeen_duck_20261014.md). The failed three-site protocol is
preserved unmodified under 2026-10-13.

## Gate outcome

| Site | ±48 h rows/scenes | fit | cal | test | Gate |
|---|---:|---:|---:|---:|---|
| Narrabeen | 661/144 realised | 138/31 | 194/45 | 329/68 | PASS |
| Duck | 679/204 realised | 252/77 | 79/22 | 348/105 | PASS |
| Truc Vert | 10/5 realised | — | — | — | FAIL |
| Torrey | payload unavailable | — | — | — | UNKNOWN |

Duck figures are REALISED (contour-quality exclusions applied): 561 accepted
contour rows, 115 of 676 profile-dates rejected. Survey dates are date-only
(midnight UTC contract). Caveats kept: mean off-axis 9.5 m, max 230.6 m;
CoastSat-SHOREX disagreement missing for 29% of Duck rows.

## Provenance

- Duck: USACE CHL THREDDS public catalog; licence verbatim from file metadata
  (unrestricted use, must acknowledge USACE FRF + supporting investigators).
  Only the 169 files matching scene dates ±48 h on transects [-91,1,1006,1097]
  were fetched: 157,193,250 bytes (corrects an initial 25 MB estimate).
  Per-file SHA-256 in evidence/duck_downloads.jsonl.
- sources.yaml: updated allowlist, appended duck-frf-surveys entry.
- No imagery, weights, GPU or cluster work. Payloads and derived CSVs are
  excluded from this archive.
