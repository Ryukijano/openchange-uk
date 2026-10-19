# BriCS GPU readiness: what to check and what is still unknown

## Bottom line

GPU environment validation can proceed **in parallel with the coastal reference
audit**. No new imagery is needed to test CUDA, BF16, attention, basic training,
checkpoint I/O or same-node DDP. The earlier recommendation to audit labels first
applies to *scientific experiments and imagery acquisition*, not to synthetic
runtime checks.

This checkout now contains a human-run readiness probe. **It has not run on
BriCS.** Local lint, unit tests and shell syntax checks cannot establish that a
remote GPU environment is ready. No agent connection, job submission, library
installation on BriCS, data download or model-weight download was performed.
The kit is published for human review and execution, not as a completed GPU test.

BriCS is the Bristol Centre for Supercomputing, not a single machine. The existing
project documentation targets **Isambard-AI Phase 2**, with Arm64/aarch64 CPUs and
GH200 GPUs. That is the target below. Isambard 3 Grace is a different, CPU-only
system; the MACS comparison service also has some GPU partitions. Do not reuse a
GH200 job unchanged on another BriCS service.

## 1. What counts as ready?

| Layer | Required evidence | Current conclusion |
| --- | --- | --- |
| Access and allocation | Human login works; correct project; current balance and storage quota checked | Not checked by the agent |
| Image identity | Arm64 NGC SIF staged; actual SHA-256 matches the reviewed image | Historical image exists in supplied logs; current staged file not checked |
| One GPU | Correct architecture/device; BF16 arithmetic and attention; finite gradients and updated parameters; checkpoint continuation | New probe prepared, not run on BriCS |
| Same-node multi-GPU | Correct rank binding; NCCL SUM; DDP parameter agreement; per-rank checks all complete | New probe prepared, not run on BriCS |
| Actual JEPA trainer | Its CUDA tests, SyncBN parity, BF16/restart and synthetic trainer stages | Historical supplied run contains an unresolved failure; current checkout not GPU-validated |
| Actual frozen encoder | Approved local weights; native bands/radiometry; dependencies and real forward pass on GH200 | Not tested; no weights or imagery acquired |
| Actual task adaptation | Approved data, stable labels, fixed protocol; model-specific finite updates/resume and evaluation | Not ready to claim |
| Multi-node training | Separate networking, Slingshot/plugin, launcher and throughput validation | Out of scope under current one-node rules |
| Public Hugging Face demo | Separate deployment environment, inputs, latency, memory and useful predictions | Not implemented or tested |

There is no honest single switch called “GPU ready for everything.” A pass at one
layer is a prerequisite, not evidence for later layers. Do not force survey joins,
licence checks or a tiny metadata model onto GPUs: GPU work belongs in neural
inference, adaptation and training. JEPA remains optional, not the goal.

## 2. Existing evidence: do not lose the failure

The supplied historical files are:

- `reports/cuda/slurm-7365198/jepa-cuda-checks-7365198.out`
- `reports/cuda/slurm-7365198/jepa-cuda-checks-7365198.err`

They identify job `7365198`, code commit
`66eb56cdcc080118da66d3ebeac3334dde4cdbc7`, aarch64, four GH200s and image
`nvcr.io/nvidia/pytorch:25.05-py3`. The recorded SIF hash is
`41bc95eb3ad44a361cfd795d03fa15a954a4afe0b924a2a69cc1c9d335c2a2a8`.
That is historical provenance, **not a requirement that a fresh conversion of the
same OCI tag must yield the identical SIF bytes**. A separately built SIF needs its
own reviewed identity and hash.

The log ends with `test_cuda_syncbn_single_vs_two_rank` failing and the suite
reporting `FAILED (failures=1)`. The corresponding historical batch template runs
the test suite before the four-GPU dry-run and fit under `set -euo pipefail`.
Therefore the log does **not** prove those later stages completed.

These supplied logs remain uncommitted in this working tree. This is an inventory
of their contents, not a new committed GPU result. Do not report a new readiness
pass or scientific result until its reviewed log has been committed.

The JEPA implementation is in the separate local checkout
`/home/ubuntu/repos/openchange-jepa`, associated with
[PR #2](https://github.com/Ryukijano/openchange-uk/pull/2). Its lifecycle was not
checked here. It has pre-existing uncommitted code changes. This readiness work
does not alter, commit or discard those changes. This documentation/archive
checkout does not contain the JEPA trainer source; do not assume that copying a
JEPA command here makes that trainer available.

## 3. Stack to use

Use the repository's documented **Arm64 NGC PyTorch 25.05** image through
`apptainer exec --nv`. Current BriCS ML-package documentation uses that image, and
the known-issues page lists it among versions compatible with the documented
multi-node host-library adaptation. This is a compatibility choice, not a claim
that it is the latest available container.

NVIDIA documents 25.05 as PyTorch `2.8.0a0+5228986c39`, CUDA 12.9 and Python 3.12.
Record what the *actual container* reports. Do not replace its PyTorch with an x86
wheel or install the VM's CPU environment on GH200. BriCS documents alternative
Arm64 pip/conda routes, but switching routes would require a separately tested
environment; it is unnecessary for the first checks.

- `nvidia-smi` records the host driver/device. Its displayed “CUDA Version” is not
  the version of the loaded toolkit or proof of a usable PyTorch runtime.
- `torch.version.cuda` records the framework's build/runtime family. A successful
  kernel and backward pass are needed in addition to version strings.
- `nvcc` and a system CUDA module are needed for compiling custom kernels, not
  automatically for an NGC PyTorch run. Do not stack arbitrary toolkit/compat
  library paths onto the image to make a check appear to pass.
- Keep inherited user-site Python packages out (`PYTHONNOUSERSITE=1`); the new job
  clears local `PYTHONHOME`/`PYTHONPATH` and provides the repository source path.
- No package installs, image pulls, datasets, weights or external telemetry inside
  the readiness job. Offline flags are set, and the probe has no download code.
  These flags are not a firewall or a replacement for source approval.

For eventual **multi-node** jobs the official Apptainer guide requires
`brics/apptainer-multi-node` and `/host/adapt.sh`, with the correct host networking
libraries. The new checks are **one node only**. An intra-node NCCL pass does not
validate Slingshot, RDMA or multi-node bandwidth. No multi-node template is added.

## 4. The new checks

Files:

```text
configs/gpu_readiness.yaml
src/openchange/gpu_readiness.py
scripts/check_gpu_env.py
slurm/06_gpu_readiness.sbatch
tests/test_gpu_readiness.py
```

Both modes use fixed, small synthetic tensors. No pretrained architecture is
selected by these tests.

### Single mode: one GPU, at most ten minutes

1. Require a clean reviewed git commit, explicit config, actual SIF hash, one-node
   Slurm allocation, aarch64 and Apptainer. No local CPU fallback.
2. Require exactly one visible GH200, compute capability 9.0 and BF16 support.
3. Compare a 512×512 BF16 matrix product with an FP32 result, with TF32 disabled.
4. Test PyTorch-selected scaled-dot-product attention in BF16 against FP32,
   including a finite backward pass. This is **not** a certificate that a
   particular FlashAttention implementation was selected or benchmarked.
5. Train a tiny six-channel synthetic image network for two optimizer steps;
   require finite losses/gradients/parameters and a real parameter update.
6. Save and reload model/optimizer state; compare a further step with uninterrupted
   execution. Require identical loss and bitwise model-state continuation within
   the same job with fixed inputs and no stochastic layers. This does not prove
   arbitrary model, RNG, data-worker or cross-job resume correctness.
7. Record versions, device, peak allocated memory and per-check outcomes.

It is a correctness/packaging check, not a memory-capacity stress test or a
representative throughput benchmark. Numerical tolerances are frozen in code;
failures must be investigated, not automatically loosened.

### Distributed mode: two GPUs first, four only afterwards

The same template is used with a reviewed `--gpus=2` or `--gpus=4` override and
`OPENCHANGE_GPU_MODE=distributed`. It launches one `torchrun` parent inside the
single-node allocation, with one worker per visible GPU; no MPI launcher or
second Slurm task layer.

Each rank binds its own GPU, runs the checks, verifies an NCCL SUM with a known
answer, trains through DDP on rank-specific synthetic inputs, checks continuation
and verifies that all model parameters match rank zero. Process-group operations
have a 120-second timeout. `torchrun` has no automatic worker restarts.

There must be a successful Slurm exit **and** a complete passing report for every
rank. A rank-zero JSON alone is insufficient. Missing reports, killed workers,
timeouts, nonzero exits, a leftover `running` status or failed checks mean no pass.

All jobs remain one node and at most ten minutes; the existing 30-minute repository
ceiling is unchanged. Running 1-, 2- and 4-GPU stages at the full time limits would
request at most 70 GPU-minutes in total, excluding queue time and image staging.
Verify actual billing and remaining allocation yourself; the planning figure in
the old specification is not a current balance.

## 5. Human-run steps

**The cloud agent may start `clifton auth`; the human completes SSO approval. Never copy certificates, keys, tokens, passwords or authenticated tunnels into the repository or chat.**

Before using the probe:

1. Review the published preparation commit. Stage that exact code on BriCS using
   your normal human-operated transfer workflow. Publication of the kit does not
   constitute a GPU pass or submit a job.
2. Confirm the project really targets Isambard-AI Phase 2. Check the official
   [service status](https://status.isambard.ac.uk), access, allocation and quota.
3. Let the agent start `clifton auth`, then approve the displayed device link/code
   yourself following the official login guide. Do not copy an SSH certificate,
   key, token, password or authenticated tunnel into this session.
4. On the login node, stage or reuse the documented Arm64 NGC SIF in project
   storage. Prefer the existing approved image if its provenance is known. Review
   any new pull against the official container instructions; no image pull is
   needed inside a test job.
5. Use a clean checkout at the reviewed commit. Do not delete unrelated work to
   satisfy the clean-tree check. Use a separate reviewed checkout when necessary.
   Check that the checkout's `.git` metadata resolves correctly on BriCS, rather
   than copying an invalid worktree pointer from another machine.
6. Set the **reviewed** SHA and SIF hash below. Do not automatically bless whatever
   happens to be present; compare the actual hash with your staged-image record.

On the login node, from that repository root:

```bash
# HUMAN ONLY. Replace every placeholder before use.
export OPENCHANGE_EXPECTED_GIT_SHA="REVIEWED_40_CHARACTER_COMMIT_SHA"
export OPENCHANGE_SIF="${PROJECTDIR}/sif-images/pytorch_25.05-py3.sif"
export OPENCHANGE_EXPECTED_SIF_SHA256="REVIEWED_64_CHARACTER_SIF_SHA256"

git rev-parse HEAD
git status --short
sha256sum "$OPENCHANGE_SIF"
python3 scripts/check_sbatch.py
mkdir -p logs

# First submission only. Review its logs before any distributed submission.
export OPENCHANGE_GPU_MODE=single
sbatch --job-name=gpu-ready-1 slurm/06_gpu_readiness.sbatch
```

The placeholders intentionally cannot pass validation. Do not paste them unchanged.
The job computes and checks the actual SIF hash, rather than trusting a sidecar.
It refuses tracked **or untracked** changes. Ignored output directories are fine.

After a complete, reviewed one-GPU pass:

```bash
# HUMAN ONLY. Same reviewed code/config/container. Update SHA only after review
# if committing the previous log changed HEAD.
export OPENCHANGE_GPU_MODE=distributed
sbatch --job-name=gpu-ready-2 --gpus=2 slurm/06_gpu_readiness.sbatch
```

After reviewing that pass, if four-GPU work is actually planned:

```bash
# HUMAN ONLY. Optional; do not run before the two-GPU report is clean.
sbatch --job-name=gpu-ready-4 --gpus=4 slurm/06_gpu_readiness.sbatch
```

Use the default general-purpose queue; no reservation, QOS, partition override or
`--exclusive` is needed here. These are not interactive-reservation jobs. Do not
run the Python probe directly on a login node.

Monitor manually, no faster than the documented 60-second interval:

```bash
# HUMAN ONLY. Replace JOB_ID with the actual ID.
squeue --me
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
```

Do not put these commands in `watch` or a tight loop.

## 6. After generic runtime checks: test the actual workload

### JEPA implementation, if we still intend to use it

Do not submit its existing synthetic fit as if the old suite passed. First review
the separate checkout's pre-existing changes and produce a pinned, clean revision.
Use its model-specific CUDA tests on real allocated GPUs. Keep the failing SyncBN
comparison at its registered tolerance and capture the named parameter/buffer,
per-step differences, gradients and BN statistics. Determine whether the cause is
an implementation problem or an explained precision/collective-order effect.
Adding LayerNorm or relaxing a tolerance is not a substitute for resolving the
original test. No root cause is established by the archived traceback alone.

Only after those checks pass should the existing `05_jepa_cuda_checks.sbatch`
proceed through its four-GPU dry-run and 50-step synthetic fit. Require logs from
each stage, not only the test-suite header. Real-data training additionally needs
approved arrays, split/normalization manifests and hand-checked pilot inputs.

### Coastal frozen encoder or fine-tuning

Choose the task and an encoder that actually supports the sensors. The current
fit period contains Landsat 5/7, so a Sentinel-2/HLS-only input contract cannot be
silently substituted. Separately approve the model-weight source/licence,
revision, dependencies, byte cap and native band/radiometry/QA contract. Stage
weights before a job and require offline local loading.

Then check the **actual** encoder on GH200: imports and extension compatibility,
an input-contract-correct forward pass, finite outputs, expected feature shapes,
precision comparison, memory and representative batch throughput. For adaptation,
add real backward/update and interrupted-versus-continuous resume tests with that
model. The tiny network in the generic probe does not certify Prithvi, TerraMind,
Clay, Contextformer or another model.

The scientific experiment still needs audited references, a risk-aligned target,
whole-scene grouping and appropriate controls. GPU readiness is not evidence that
imagery improves the negative coastal result or that pretraining is justified.

### Other training tracks and the Space

Forecasting or another selected task gets its own model/data/environment checks;
there is no reason to install unrelated JAX, TensorFlow, vLLM or robotics stacks
pre-emptively. The useful Hugging Face Space remains the product goal. Test its
portable inference environment separately rather than making users depend on a
BriCS container or an active cluster allocation.

## 7. What to return for review

For each completed stage, return:

- Slurm stdout **and** stderr from `logs/gpu-ready-N-JOBID.*`.
- Job state and exit code from the human's accounting check.
- Every rank's `manifest.json` and `readiness.json` under
  `outputs/gpu-readiness-MODE-rankR-slurmJOBID/`.
- The reviewed git SHA and staged SIF hash; these are also in the manifest/header.

Do not send a full environment dump, authentication material, raw imagery, labels,
weights or `.pt` checkpoints. The synthetic checkpoint is local test output and
is not needed for review. Review logs for accidental sensitive values before
publication. Commit the reviewed log/manifest records, not large payloads, before
making a formal readiness claim. Returning logs here does not by itself approve
committing or pushing them.

A failed run is useful evidence: preserve the failing stage and environment, fix
the cause, and rerun the smallest affected check. Do not hide a failure by moving
directly to a longer job or a new model.

## 8. Official sources checked

Live official documentation was read for this runbook. Local skills were used for
CUDA, responsible AI-generated HPC code, containers, Slurm and NCCL. The live
documentation takes precedence where a skill is outdated; notably, Isambard 3
Grace and the MACS comparison service are not the same hardware.

- [Using AI agents](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/)
- [BriCS ML packages](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/)
- [Containers and Arm64 compatibility](https://docs.isambard.ac.uk/user-documentation/guides/containers/)
- [Apptainer host-library adaptation for multi-node](https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/)
- [Distributed-training tutorial](https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-training/)
- [Scheduling, limits and interactive billing](https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/)
- [Known issues, including NGC/GLIBC and overlapping allocations](https://docs.isambard.ac.uk/service-status/known_issues/)
- [Login](https://docs.isambard.ac.uk/user-documentation/guides/login/)
- [Storage](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)
- [NVIDIA PyTorch 25.05 contents and compatibility references](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html)

These sources establish documented configuration, **not current account access,
service health or a runtime pass for this project**.
