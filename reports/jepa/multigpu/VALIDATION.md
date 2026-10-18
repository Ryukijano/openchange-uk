# EO Clean-JEPA multi-GPU pilot implementation

Status: local, uncommitted implementation and validation evidence. No imagery
was fetched, no EO model was trained, and no cluster job was submitted. This is
not a research result, a formal GPU smoke pass, or a GH200 throughput claim.

## What is implemented

- Offline S1/S2 array ingestion with explicit imagery-source approval, immutable
  split/normalization sidecars, payload hashes, and unique-location sampling.
  Existing metadata-only approvals fail before array access.
- One-node torchrun/DDP with CUDA/NCCL or CPU/Gloo, bf16 model/autocast and fp32
  SIGReg, CUDA SyncBatchNorm, fused CUDA AdamW, packed characteristic-function
  reductions, pinned-memory workers, and bounded positional caches.
- Nonfinite/wall-cap consensus and rank-zero checkpoints containing rank-specific
  CPU/NumPy/Python/CUDA RNG state, sampler step and token counters. Resume rejects
  world-size, device, PyTorch-version, configuration and training-code changes.
- Profiling, throughput/loader/memory diagnostics, exact config/code hashes and
  checkpoint paths. A human-only four-GPU template defaults to dry-run and stays
  within the 30-minute limit; it refuses dirty or wrong-commit checkouts.

The Clean-JEPA objective and selected scientific arm are unchanged. The pilot
uses global batch 32, not 1024: a <=500-location inventory cannot provide 1024
distinct locations. Temporal prediction, georeferenced change targets and
observation-selection policies remain later stages, not implemented WAM claims.

## Parent-checkout validation

- `make check PYTHON=.venv/bin/python`: passed; 43 base tests discovered, eight
  optional-torch modules skipped in the torch-free environment.
- `make test-jepa PYTHON=.venv-jepa/bin/python`: 98 tests discovered; 96 passed,
  two CUDA-only checks skipped. Tests include source/split/path/payload rejection,
  deterministic sampling, one-vs-two-rank updates, exact restart, worker-count
  parity, stop consensus and checkpoint-write failure propagation.
- Ruff, Python compile checks, `git diff --check`, Slurm policy checks and
  `bash -n` passed. No static type checker is configured in this project.
- The local-array adapter completed a two-rank fit on temporary **synthetic**
  array fixtures with temporary test-only approvals; this did not authorize or
  validate actual imagery. Changing fixture approval to auxiliary was refused.
- ViT-Ti, eight views, batch 32, FP32, frozen projector normalization: 10 synthetic
  CPU/Gloo steps completed with two ranks and two loader workers per rank.
  The reported training-loop wall time was 9.357 seconds, excluding input/model
  setup. Profiling was enabled; this is not a performance benchmark.
- Restarting that run from step 5 reproduced the step-10 model, optimizer,
  scheduler, per-rank RNG/counters, global tokens and launch digest **exactly**.
  A CPU profiler trace was emitted and parsed (117,892 events).

Environment: PyTorch 2.14.1+cpu. Base HEAD:
`52d6bf61304e4b2aaa486c003865751120579d0d`, with local changes. Training manifests
record hashes for the actual uncommitted training code; HEAD alone does not
identify this implementation. Logs are not committed.

## Remaining gates

1. Review and separately approve publication of these changes.
2. Freeze and hand-audit geography, time cutoff, pairing gap and masks; separately
   approve bounded imagery extraction and training-only normalization.
3. Build the raw-provider-to-array extraction path. The loader does not decode
   tar/Zarr, perform radiometric conversion/resampling or verify geographic buffer
   geometry; its human-accepted curation audit must establish those facts.
4. Human-run the synthetic dry-run, CUDA FP32 SyncBN parity, bf16 restart check,
   and bounded local pilot in the documented Arm64 GH200 container. Neither CUDA
   test ran here, and NGC 25.05 compatibility has not been validated on hardware.
5. Profile the real pipeline before selecting chunk size, worker count or further
   kernel changes. Do not infer GH200 throughput or scaling from the CPU timings.

The newer local PyTorch emitted a `broadcast_buffers` deprecation warning; the
existing argument remains supported here. The CPU profiler also warned about an
allocation predating the profiling window. Neither interrupted these local runs.

Supporting archive contains the base/JEPA logs, fit/resume logs, tested CPU config,
exact-state comparison, metrics and provenance manifests. Synthetic checkpoints
and the large profiler trace are not bundled.
