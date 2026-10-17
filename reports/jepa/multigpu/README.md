# Multi-GPU pilot: local synthetic evidence

This directory preserves CPU/Gloo validation of the offline-array and distributed
pilot implementation. It contains no EO imagery, model weights, GPU benchmark or
accepted scientific split.

`VALIDATION.md` is the original handoff note: its references to "uncommitted"
describe the state when those measurements were captured, not the publication
state of this directory. The fit/restart logs and manifests are copied unchanged;
their dirty base checkout and old HEAD are recorded honestly. Every training-code
hash in those manifests matches the implementation published with this evidence.

- `multigpu-base-check.log`: torch-free base checks.
- `multigpu-jepa-tests.log`: 98 tests discovered, 96 passed, two CUDA checks skipped.
- `multigpu-cpu-pilot.yaml`: the exact 10-step FP32/frozen-normalization recipe.
- `multigpu-fit.log`, `fit/`: two-rank ViT-Ti synthetic fit with worker prefetch and profiling.
- `multigpu-resume.log`, `resume/`: restart from step 5, finishing at step 10.
- `multigpu-parity.json`: exact final model/optimizer/scheduler/RNG/counter comparison.
- `publication-base-check.log`, `publication-jepa-tests.log`: fresh pre-publication checks.

Checkpoint paths and profiler paths in the historical logs identify local files;
the large binaries are not redistributed. CPU timings are not GH200 performance.
See [the runbook](../../../docs/research/clean-jepa/MULTIGPU.md) for the remaining
imagery, curation and human-run hardware gates. No cluster command was executed.
