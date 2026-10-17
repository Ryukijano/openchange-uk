# Narrabeen Stage 0: one-beach temporal screen

Historical CPU-only run, 13 October 2026, archived for publication on 15 October.
This is not a four-site test. SDS_Benchmark commit:
`c0747c52b310433411fc4ab96822ee67441a8d2e`.

See the original [protocol](evidence/protocol.md), [stdout log](evidence/run.log),
[result JSON](evidence/results.json), [environment snapshot](evidence/environment.lock)
and [code](code/run_stage0.py). `code/` is byte-identical to the original working
scripts. No experiment has been rerun or retuned for this publication.

## Result

31 fit / 45 calibration / 68 test scenes; whole-scene grouping. About 70% scene
retention, quantile HGB seeds 0/1/2:

| Condition | Retained test RMSE | Reduction vs calibration-selected setup baseline |
|---|---:|---:|
| Published position, unchanged | 6.735 m | Context, not the registered comparator |
| Setup baseline selected on calibration | 8.120 m | Reference |
| Metadata | 8.434 m | −3.87% |
| Metadata + disagreement | 8.410 m | −3.57% |

Unlike the later two-site selection-only runner, this screen evaluates corrected
positions and calibrated intervals. Do not pool their scores as one experiment.
The metadata+disagreement intervals average 36.68 m wide and support zero of
138 exploratory pair-direction claims. Metadata-only supports one; the pairs
are correlated and are not independent validation examples.

## Replay boundaries

Raw surveys, matched residual/prediction tables and pair tables are not
redistributed here. The script expects a writable `coastal-stage0` workspace
next to the pinned `sds-benchmark` clone and regenerates output files there.
Restore externally held approved inputs and copy the archived scripts plus
protocol into that layout; do not execute into this evidence directory:

```bash
"$PYTHON" "$WORKSPACE/coastal-stage0/run_stage0.py"
```

Dependencies are recorded by the original environment file; they are not added
to the repository's minimal test environment. This is a historical evidence
archive, not a portable, independently clean-room-reproduced release. Original
repository/visual check logs describe their historical checks, not this PR.
