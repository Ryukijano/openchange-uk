# Implementation runbook: local gates before scale

This is a work specification, not a list of already working commands. Implement in
small reviewed changes. Do not download anything, fit EO models, submit cluster jobs,
or claim readiness merely because this document exists. Decisions are in [README](README.md),
loss definitions in [FORMULATION](FORMULATION.md), source records in [SOURCES](SOURCES.md).

## 0. Verify the existing repository first

From the repository root, use the documented development environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
make check PYTHON=.venv/bin/python
```

If an existing environment is present, use it rather than recreating it. Current checks
cover the benchmark scaffolding, not JEPA. No PyTorch runtime is required for these
checks. The source list is deliberately empty. Before any future fetch:

```bash
make validate-sources PYTHON=.venv/bin/python
```

A successful validation of an empty source list approves **no source**. A future
fetcher must also call `openchange.sources.require_approved(source_id, split)` before
each source access. Use `pilot` for the initial slice, not an invented split enum.
The existing schema does not independently enforce acquisition revisions or geography:
those are additional manifest/curation requirements. Do not weaken the source gate.

## 1. Implementation layout and interfaces (all proposed)

Keep `scripts/train.py`, the current benchmark config and dry-run semantics intact.
Add training functionality behind an explicit new command only when it exists and
is tested. Do not rename a dry-run to look like training.

| Proposed location | Responsibility | Interface / invariants |
|---|---|---|
| `src/openchange/jepa/data.py` | Read approved local shards/manifests only | `Observation` contains sample/location IDs, source revision, S1/S2 dates, footprint/CRS, arrays and masks; no network I/O in a training worker |
| `src/openchange/jepa/views.py` | Apply declared interventions, crop, normalize, bucket | `make_views(observations, seed, step)` returns pixels, positions, sensor IDs, stratum and real/synthetic flags; no forbidden sensor reintroduced |
| `src/openchange/jepa/model.py` | Per-sensor stems, shared ViT, projector | Backbone features and projected embeddings kept separately; variable sequence lengths supported without inventing image values |
| `src/openchange/jepa/losses.py` | Live-centre invariance, pooled/stratified EP, T_mu | Input `[V,N_local,K]` plus sample strata; per-view reductions, correct global counts; no silent detach/autocast |
| `src/openchange/jepa/distributed.py` | Direction seeding, count/sum reductions | Standard mean-reduced DDP only initially; fail on unequal quotas or unsupported accumulation |
| `src/openchange/jepa/checkpoints.py` | Exact restart state | Model/projector, optimizer/scheduler, seed/RNG states, sampler step, actual tokens, BN state and manifests |
| `scripts/train_jepa.py` | Future local/offline training entry point | Require explicit config and provenance; `--dry-run` validates shape/config without data fetching or fitting |
| `scripts/evaluate_jepa.py` | Future Q1/Q2/Q3 entry point | Accepted task/split manifests required; never fit calibration on evaluation locations |
| `configs/jepa/` | Future executable config schema | Schema version, loss reduction, strata/quotas, corpus/split hashes, bands/units/normalization, views/token drop, optimizer and schedule |
| `tests/test_jepa_*.py` | Algebra, shape, restart, distributed and leakage checks | Synthetic/offline tests default; real-data tests require explicit approved fixtures |

Introduce PyTorch/timm only in an optional training extra with tested, pinned compatible
versions. Keep base checks lightweight. On Isambard use the documented aarch64 NGC
PyTorch/Apptainer route; do not transplant this VM's x86 wheels. Record upstream commit
IDs before porting any loss code, plus attribution/licence; local overrides must be explicit.

Every future config records:

```yaml
# Illustrative design record, NOT consumable by the current trainer.
status: planned
loss_reduction: mean_n_v_k
projector_dim: 128
global_batch: 1024
strata: [joint_clear, joint_occluded, s1_only, s2_only]
global_quotas: [256, 256, 256, 256]
stratum_weights: [0.25, 0.25, 0.25, 0.25]
global_views: 2
local_views: 6
global_crop_pixels: 224
local_crop_pixels: 96
global_token_drop: 0.9
local_token_drop: 0.5
sigreg_slices: 1024
sigreg_knots: 17
sigreg_t_max: 3.0
lambda: 0.05
lambda_mu: 0.0
gaussian_target: identity
drop_path: 0.0
gradient_accumulation: 1
```

This is not complete: executable configs must additionally supply accepted source/split
records and numerical normalization/packing values. Missing values are errors, not
fallbacks. In particular do not infer data roots, licence approval or a test cutoff.

## 2. P0 — Approve and curate at most 500 locations

**Owner:** human approves source/licence and holdout policy; implementer builds the local manifest.

1. Check the exact SSL4EO-S12 v1.1 endpoint/card and licence. Record full immutable
   revision, provider attribution, intended modalities, expected bytes and redistribution
   terms. Resolve date-order metadata issues; do not assume channel index i pairs dates.
2. Human adds a valid source entry in `configs/sources.yaml`, with approver and date.
   Preserve exact licence text separately. `require_approved` precedes even a small fetch.
3. Select candidates from pinned metadata **before** downloading imagery. Exclude the
   conservative UK box 49–61.5 N, -11–2.5 E and an approved footprint buffer; no patch
   footprint may cross an exclusion. Audit evaluation footprints and acquisition dates
   for the other task families. UK exclusion alone is not global benchmark decontamination.
4. Dedupe geographic footprints/cell IDs across corpora. A 2.64 km centre-distance
   filter is a useful pilot check, not proof of no footprint overlap. Use actual footprints.
5. Across six latitude bands consider up to 84 candidates each (504), then trim to 500
   deterministically by SHA-256 sample ID. If trying to enrich cloudy observations
   (e.g. 25% above mask fraction 0.05), record feasibility; never fabricate a real stratum
   to satisfy a quota. Synthetic occlusion is separately labelled.
6. Assign entire geographic blocks to pilot-train and pilot-validation, with all dates
   of a location kept together. Target 400/100 locations where feasible, but geography,
   buffer and time exclusions win over exact counts. Hash the manifest and record a
   pilot-only time policy. This does not accept/change OpenChange-UK's test specification.
7. Fetch only selected imagery using a bounded local staging process; checksum all
   source files. If the provider requires downloading a huge shard to reach 500 items,
   stop and price a different access route instead of downloading the full corpus.
8. Check units, shapes, band names, S1/S2 date gaps, mask polarity, nodata and georegistration.
   Visually inspect a fixed sample of clear, cloudy and missing-sensor examples.

**Exit:** committed manifest/checksum/approval record and inspection log; <=500 unique
locations; no forbidden overlap/date; no claim of full data readiness. No train/test
sampling by random image split. Selection of deterministic hash-ranked candidates is
not a substitute for geographic/time split assignment.

## 3. P1 — Algebra, model plumbing, tiny fit

Start without any dataset. Run synthetic offline tests, then a tiny fit only after
implementation is approved. The full N=1024 recipe is not a CPU smoke recipe.

### Acceptance tests before a real fit

- Invariance formula vs autograd and central finite differences in float64: atol 1e-7,
  rtol 1e-5, including different numbers of globals/locals and live-centre gradients.
- EP quadrature vs a direct complex-CF reference, same directions/knots/weights.
  Check shapes/per-view averaging and the factor N_m. Finite-difference gradients
  should pass the same tolerance. Do not require a Gaussian batch loss to equal zero.
- Monte Carlo standard Gaussian batches: compare the mean statistic with the
  **computed quadrature** null expectation using uncertainty across draws. Also check
  collapse, isotropic variance changes, one-coordinate gaps and independent rotations.
- Pooled vs stratified vs stratified+T_mu on synthetic mixtures; test that N_m and weights
  are applied once. A pooled rank-collapse counterexample is a test case, not proof of
  EO usefulness. Use K=64 and 128 (256 only a sensitivity if runtime permits).
- View shape/token counts (20/39 global, 18/36 local), retained positional IDs and CLS;
  actual sensor subsets respect sample availability. Reassembly preserves n,v indices.
- Projector normalization has no output BN/normalization. Test whether local/global BN
  semantics change values; make distributed BN choice explicit.
- One-rank vs two-rank Gloo: identical global batch, A, loss and **parameter gradients**
  and one optimizer update. First bypass/freeze BN to isolate loss algebra, then test
  the chosen normalization implementation. Scalar agreement is not enough. Compare
  float64 loss algebra to 1e-7/1e-5; set fp32 implementation tolerance from numerical tests.
- Checkpoint/resume vs continuous steps on fixed synthetic data: optimizer, RNG, BN,
  sampler and tokens agree in deterministic CPU mode. On GPU document nondeterministic
  kernels and numerical tolerances rather than promise bit-exact equivalence.
- Controlled latent generator: shared factors plus S2-private factors. Quantify whether
  S1-only output variance is supplied by nuisance factors and whether that damages shared
  factor recovery. Availability-probe chance is not the sole pass criterion.

### Tiny CPU fit and comparisons

Use ViT-Ti, <=50 optimizer steps and a deliberately small debug batch (e.g. N=32,
eight per stratum), with a fixed seed first. It verifies finite losses/gradients and a
complete checkpoint path; it cannot establish useful representation learning. Small
batch duplicates violate i.i.d. null calibration. Do not accumulate nonlinear EP across
separate microbatches and label the result N=1024. Limit CPU threads/runtime and log
the measured cost; expand only after this plumbing gate.

For a pilot comparison, use three seeds and pooled / per-actual-modality / sample-stratified
SIGReg (define actual-modality groups separately from sample strata). Include T_mu=0
and the specified ablation, a lambda sweep and the smaller K sensitivity. CPU debug
loss magnitudes and lambda choices are not directly comparable to the later N=1024
selection population. Recalibrate at the intended global batch before freezing Q1.

**Exit:** committed synthetic test logs and tiny-fit manifests; no NaN/Inf, correct
shapes/gradients/restart. A falling training loss is a plumbing check, not a paper result.

## 4. P2 — Map/reuse harness and schedule pilot

1. Synthetic known rotations: orthogonal map recovers orientation on separate held-out
   samples; nonorthogonal maps favour general linear/CCA as expected. Include shuffled
   pair and independent-junk controls. Do not score only on calibration samples.
2. Tiny independent seeds: extract backbone and projector outputs separately; fit
   calibration means/maps on calibration geographic blocks, evaluate on different blocks.
   Test orthogonal, linear (regularization on calibration-validation only), CCA and identity.
3. Transfer a frozen A-trained probe using the B-to-A feature map. Compare with a native-B
   probe fitted to the same labels. Report absolute scores and delta, not just a ratio;
   ratios are misleading with near-zero native performance.
4. Main-scale calibration sizes {500,1000,5000,20000} require that many **distinct**
   locations. The 500-location pilot validates the code; it cannot populate that grid.
5. Schedule: implement cosine first with an explicit floor below the peak (proposed
   lr/1000), linear warmup capped by the tiny run length. At main scale candidate warmup
   is 1500 steps. Compare WSD at matched data, steps, seeds and batch before adopting
   branched cooldowns. A branch starts from a saved stable checkpoint, including optimizer
   and sampler state; count its extra tokens/compute, and do not treat sibling checkpoints
   as independent statistical runs.

**Exit:** leakage-safe map tests/reports and a schedule decision. No claim of archive
compatibility from a rotation fit on training/calibration points alone.

## 5. P3 — Evaluation and selection protocol

Implement adapters only for approved benchmark sources; start with one S1/S2 paired
classification task and one dense-prediction task. Copernicus-Bench and PANGAEA are
candidate suites, not permission to fetch them. OpenChange-UK scores wait for acceptance
of its geographic/time contract. Preserve upstream official splits where needed for
comparability and document pretraining decontamination; label a result contaminated
if footprints cannot be ruled out. Sen1Floods11 requires its own baseline run, not a
borrowed Copernicus-FM score from another benchmark.

### Q1: two populations, kept separate

- **W (within setting):** candidate 64 independent Sobol configs per objective at a fixed
  size/data budget, later a second size. Sweep objective-appropriate hyperparameters;
  keep global batch, quotas, K and selection validation bank fixed. If sweeping number
  of training views, evaluate all configurations on the common fixed validation recipe.
  Keep a fixed-recipe cohort for the cleanest test and report view-changing sensitivity
  separately. Match data access and report both tokens and total compute.
- **A (across scales):** distinct models/data/token budgets; five or more model sizes
  are a candidate, not a fit guarantee. Multiple cooldown endpoints from one run are
  correlated. Estimate run count, uncertainty and feasible budgets after the smoke.
- Rank higher-is-better probes against **negative** loss (or explicitly reverse the
  rank). Otherwise a useful loss selector produces a negative rho by construction.
- Audit proxies: raw loss, loss/lambda^0.4, components, per-stratum EP, RankMe, LiDAR,
  a clearly defined spectrum metric and a small labelled probe as audit-only. Verify
  proxy citations/implementations; do not call labelled probes label-free selection.
- Report Spearman and held-out selection regret/top-k utility per task family; bootstrap
  by independent configuration/run, and by geographic block for task scores. Treat shared
  datasets/objective pairings appropriately for correlation-difference intervals. Do
  power simulations, multiple-comparison planning and finite-run uncertainty before
  declaring 64 sufficient; earlier numeric power claims are scenarios, not guarantees.

Candidate headline gate for preregistration: rho>=0.7 with lower 95% CI>=0.5 in at least
three of four declared task families, an advantage over tuned BT with an interval
excluding zero, and no collapse under OOD. These are **proposals** to finalize before
confirmatory data; they are not tests this repository has passed. Stop spending on the
loss-selection headline if the pilot upper CI is <0.5 in all declared ID families.
That can leave Q2/Q3 as useful negative-result studies, but needs a new approved scope.

### Q3 and the Gaussian-target gate

Report score(S2)-score(S1), joint-vs-single deltas, cloudy-vs-clear, per-region uncertainty,
and accuracy/IoU/F1 or regression metrics appropriate to the declared task. Missing
modalities at test time are withheld independently of task labels. Train availability
probes from backbone and projector features on train blocks, validate on new blocks.
Condition these diagnostics on geography/class where feasible: availability may
legitimately correlate with semantic content in real data.

Compare pooled, stratified and T_mu arms with the **same** examples/interventions, task
labels and compute ledger. Pilot warning margin: >1 absolute percentage point clean
classification/IoU degradation (or >2% relative error increase for regression) without
clear missing-sensor improvement. Treat this as a stop/review threshold, not statistical
equivalence from three seeds. Final non-inferiority margins and CIs must be frozen per
task before confirmatory runs. Require shared-factor preservation in the synthetic
test, task utility, cross-condition retrieval and absence of collapse together; chance
availability prediction by itself is not success.

If the full-Gaussian candidate fails: retain all results; do not choose a covariance
target from held-out task scores. Propose a separate shared-subspace/reduced-covariance
method, define its learning rule on training/calibration data, price it and freeze a
new protocol. No "NDVI is impossible from SAR" pass rule: evaluate legitimate shared
correlations and residual optical-private information against S1-only controls.

## 6. P4 — Packing and I/O measurements

Preserve original files/checksums. S2 integer bands may pack losslessly; S1 float dB
to uint16 is **quantization**, not lossless packing. Specify scale, offset, clipping,
nodata and reconstruction error per channel. Prefer source-precision S1 in the first
pilot; adopt quantization only after measured max/RMSE error and downstream sensitivity.

Write approximately 1–2 GB tar shards with local indices and sidecar acquisition metadata,
source revisions, geographic assignments and packing version. Read once per item and
produce all eight views. Do not store precomputed trainable stem tokens: they would
freeze the stem or become stale. Stable compressed pixels are the reusable cache.

Measure compressed bytes/pair, decoded bytes, CPU time, random/sequential access,
workers/prefetch memory, cache hit rate and projected per-GPU bandwidth. Benchmark
source-precision and proposed packing on the same slice. The old ~0.6 MB/pair estimate
is a scenario, not a universal compression result. Test shader/loss memory separately
from data loader memory. A node-local cache must fit the actual measured data and
document its lifecycle; do not assume NVMe capacity or speed.

**Exit:** committed byte/decode report and roundtrip tests; no unvalidated "pretokenization"
claim or throughput extrapolation from FLOPs alone.

## 7. P5 — Freeze the protocol and price one smoke

Freeze config/schema, sources/revisions, manifests/exclusions, held-out bank, seeds,
metrics, failure/non-inferiority margins, objective tuning budget, model/data/token
grid and statistical analysis. Sign/version this record before confirmatory training.
Pin all code/dependencies and record how deviations will be labelled. Data fractions
are nested geographic-cell subsets, not random patch pools; temporal grouping remains
intact. Splits are never changed because a result is disappointing.

Human confirms the approved award scope, end date and remaining units. Replace the
combined benchmark+JEPA resource ledger; do not rely on the stale 750/1100 NHR totals.
Have the human review a one-GPU <=30-minute smoke using a local approved dataset and
explicit config/git SHA. Verify current cluster instructions in the official docs;
do not invent a partition/QOS or install packages/download data inside the job.

The existing smoke checks hardware/environment, **not** JEPA throughput. A future JEPA
smoke must log warmup and steady-state time separately, actual token counts, memory,
data bandwidth, CPU decode, full step throughput, objective FLOPs (including projector,
SIGReg and MAE decoder) and, later, multi-GPU efficiency. Re-price the grid from that
log. Agent never submits the job and never raises the Slurm limit.

## 8. Provenance and honest handoff

Reuse `openchange.provenance.start_run` and extend its record deliberately for:

- config path/hash and resolved config; git SHA/dirty state; seed and all RNG states;
- environment/container identity, runtime/driver versions and DDP reduction convention;
- source revisions/checksums, split/curation/mask-donor hashes, packing/normalization versions;
- per-step loss components, stratum quotas, real/synthetic counts, rank diagnostics;
- realized encoder tokens, repeated/unique locations, FLOPs, step time, bytes, GPU-hours/NHR;
- checkpoint path/hash and resume history; maps/probes with calibration and evaluation IDs.

Commit lightweight logs/manifests and test reports before claiming a smoke/result;
keep imagery, large checkpoints and credentials out of git. Respect redistribution
terms. `make check` must stay offline and pass on the base environment; add a clearly
documented optional training-test command when the training extra exists. A dry-run
must refuse to present provisional splits or stub scores as real results.

First implementation handoff should contain only the synthetic loss/model tests and
their measured logs. P0 fetch approval and full pilot training are separate actions.
