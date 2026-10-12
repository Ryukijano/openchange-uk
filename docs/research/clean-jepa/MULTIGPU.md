# Offline EO training on one GH200 node

Status: implementation under local review. No imagery source or real-data split is
approved, no cluster job has run, and no GH200 throughput has been measured.
The benchmark split in `reports/SPLIT_PROPOSAL.md` remains proposed.

## 1. What is being trained

The first executable milestone remains the selected clean representation arm:

\[
\mu_n=\frac1{V_g}\sum_{v<V_g} z_{vn},\qquad
L_{\mathrm{inv}}=\frac1{NVK}\sum_{v,n}\|z_{vn}-\mu_n\|^2,
\]
\[
L=(1-\lambda)L_{\mathrm{inv}}+
\lambda\frac1V\sum_{v,m}w_m EP_{vm}+\lambda_\mu T_\mu.
\]

Use the canonical definitions in `FORMULATION.md`: equal availability weights,
live global-view mean, Gaussian SIGReg, 17 quadrature knots, 1,024 directions,
K=128, lambda=0.05 and lambda_mu=0. No teacher, predictor or EMA in this arm.
Two global/six local views, separate sensor stems, one shared ViT, and
global/local token drop 0.9/0.5 remain selected defaults.

The local pilot uses N=32, not the scientific N=1024. Its EP sensitivity and loss
scale differ; never interpret its loss as evidence for the full study. A <=500
unique-location inventory cannot supply a location-unique N=1024 batch. Refuse
that batch rather than repeat locations or manufacture independence through
sensor-withholding copies. The production batch needs a separately approved
larger inventory and a new run contract.

## 2. Research sequence, not an uncontrolled joint objective

1. Validate the clean representation and missing-sensor failure tests.
2. Freeze a selected encoder. Train temporal predictions at registered spatial
   locations, with elapsed time and acquisition metadata available at decision
   time. Compare with persistence and seasonal baselines. A later candidate is
   `p(h_t, delta_t)` against `stopgradient(h_future)`, using verified positive
   time differences and validity masks. Do not use future-only quality metadata
   as context.
3. Test grounded local change targets as an ablation inspired by PointWAM.
   Georeferenced EO cells are not automatically persistent material points;
   clouds, sensor changes and registration error must not be labelled surface
   dynamics. This does not import PointWAM weights or code.
4. Only with a defined decision environment, test budgeted observation selection,
   calibrated feedback and crossed old/new component-retention checks inspired
   by RSI-Anything. Sensor acquisition is an information action, not a physical
   intervention on Earth. No arbitrary action-conditioned causal claims from
   passive SSL4EO observations.

Only stage 1 has a trainer here. Stages 2–4 require separate accepted experiments.

## 3. Data boundary

`LocalObservationDataset` accepts a bounded, explicitly reviewed local manifest
of preprocessed S1/S2 `.npy` arrays and a Boolean optical validity mask. It does
not decode tar shards, download data, infer optical offsets, or silently estimate
normalization statistics from held-out data. This is an interchange format, not
an end-to-end raw SSL4EO ingestion pipeline.
The exact schema and sidecar contracts are in [DATA_MANIFEST.md](DATA_MANIFEST.md).

An accepted pilot must name imagery sources with `use: imagery` and split
`pilot` in the source registry, pin revisions and preprocessing, record payload
checksums and geographic groups/timestamps, and freeze an accepted pilot split.
The existing metadata-only sources cannot authorize these arrays. Actual source
approval, normalization values, mask polarity, alignment and split acceptance
remain human tasks. Do not add fictional approvals to make the example runnable.

Sample unique locations globally, assign availability conditions, lay out quota
blocks, then shard by rank. Sensor withholding is recorded as a synthetic
condition but does not turn actual imagery into synthetic data. Shuffle samples
*inside* the accepted training pool; never shuffle locations into new splits.

Read each selected pair once on its rank, make its eight views in workers, and
prefetch those views. Use mmap-compatible arrays for this <=500-pair pilot; a
larger-scale packed/sharded IO format requires a measured IO study first.

## 4. Distributed semantics

One node only, one torchrun process per GPU. Bind `LOCAL_RANK` before NCCL
initialization and use standard DDP mean gradient reduction. Each rank must
contain exactly the same per-stratum counts. Keep gradient accumulation at one.

SIGReg sums empirical characteristic functions globally through the existing
autograd SUM reduction. Replicated global EP yields a world-size factor in local
backward contributions; DDP's mean reduction cancels it. **Do not divide EP by
world size**, detach its sums or evaluate independent microbatch EP losses.
Direction chunking/recomputation preserves the objective; microbatch averaging
does not, because characteristic-function discrepancy is nonlinear.

Convert the projector's two BatchNorm layers to SyncBatchNorm before wrapping
CUDA DDP. Normalize once over the restored `[V*N_local, hidden]` rows. CPU/Gloo
comparisons use frozen or identity normalization and explicitly do not validate
the production SyncBN path. Do not convert the frozen-normalization ablation.

Keep FP32 master weights, bf16 autocast through encoder/projector attention,
and FP32 SIGReg outside autocast. Sine/cosine sums share one packed collective
per chunk (and one in backward) without changing the loss. Directions are created in FP32 directly;
upcasting already-rounded bf16 directions would not repair their geometry.
Use PyTorch SDPA dispatch and CUDA fused AdamW, not new custom kernels by default.
Positions have a bounded per-device cache; view-index coverage is checked on CPU
before transfer rather than forcing GPU scalar reads in each encoder bucket.
No hard-coded NCCL transport-disable flags or invented partition/QOS settings.

## 5. Entry points

Local synthetic preflight (requires the existing optional CPU torch environment):

```bash
PYTHONPATH=src .venv-jepa/bin/python scripts/train_jepa_distributed.py \
  --config configs/jepa/p1_synthetic.yaml --dry-run
make test-jepa PYTHON=.venv-jepa/bin/python
```

`tests/test_jepa_runtime.py` runs a tiny one-versus-two-rank Gloo comparison,
same-world restart and deterministic worker-prefetch checks. It uses frozen BN.
The CUDA SyncBN comparison is a separate test requiring two actual GPUs.

After imagery and an immutable pilot contract are accepted, local-array preflight:

```bash
PYTHONPATH=src python3 scripts/train_jepa_distributed.py \
  --config configs/jepa/gh200_local_pilot.yaml --dry-run --device cuda \
  --manifest /approved/local/pilot.json --sources-config configs/sources.yaml
```

The human-reviewed four-GPU template is `slurm/04_jepa_pilot_4gpu.sbatch`.
Create `logs` and `outputs` before submitting: Slurm opens its log before the
script can create those directories.
It defaults to synthetic dry-run, not fitting. Actual synthetic/approved local
fit selection is explicit. It never downloads an image, package or dataset.
Review the documented aarch64 NGC 25.05 container and existing one-GPU smoke
before using it. The 30-minute ceiling is unchanged; the trainer's internal cap
leaves time for checkpointing but cannot guarantee a slow checkpoint fits.

## 6. Restart and failure semantics

One artifact writer, rank zero. Checkpoints store model, optimizer, schedule,
next sampler step, local token counters and each rank's Python/NumPy/CPU/CUDA RNG
states. Loading uses `weights_only=True`. Launch/config/manifest/source digests,
world size, device type and torch version must match. Elastic world-size restart
is intentionally unsupported. Changing worker/prefetch/chunk settings also
requires a new launch contract. CPU replay is tested exactly; CUDA numerical
replay still requires hardware validation and is not promised bitwise.

Ranks agree on nonfinite loss before backward, nonfinite gradient before update,
and wall-cap termination. A failed update is not saved as an accepted checkpoint.
Rank-zero checkpoint write failures are broadcast. Unrecoverable rank errors are
handled by torchrun's fail-fast process supervision and the finite collective
timeout, not by continuing with partial gradients.

## 7. Measure before optimization claims

Logs include the resolved recipe, git state, environment, source/manifest hashes,
world size, GPU name, window-mean losses, exact Gaussian null, tokens/sec,
max-rank window duration, loader wait and CUDA peak memory. Token counts exclude
CLS consistently with P1. `--profile-steps 5` records a bounded rank-zero PyTorch
trace after three warmup iterations; trace runs are not throughput runs.

On hardware, first establish one-vs-two GPU SyncBN update agreement, finite bf16
loss/gradients and restart. Then measure one/two/four GPUs at fixed global batch,
report max-rank steady-window timing, and sweep workers 0/2/4/8 and SIGReg chunks
64/128/256. Record allocated and reserved memory and variance across repeats.
Do not report FLOP estimates as actual throughput or MFU. Only after identifying
the bottleneck consider further collective fusion,
GPU-native crops, attention checkpointing, or shape-bucketed compile. Preserve
full-batch gradients and repeat the numerical tests after each optimization.

Official references:
- https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/
- https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
- https://docs.pytorch.org/docs/stable/data.html
- https://docs.pytorch.org/docs/stable/profiler.html
- https://arxiv.org/abs/2610.02840
- https://arxiv.org/abs/2610.13912

NGC 25.05's torch build may be a development version. Its actual installed
behavior must pass the smoke/correctness gates; stable online docs alone do not
establish compatibility. Neither paper's results were reproduced here.
