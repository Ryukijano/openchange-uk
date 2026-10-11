# Clean-JEPA EO formulation

Authority: this document and [the selected design](README.md) override the archived
reviews where they disagree. All EO design choices below remain untrained.

## 1. Inputs, strata and views

One item n is a georeferenced location/acquisition pair with S1 VV/VH, S2 L1C bands,
dates, valid-pixel masks and a source record. A median acquisition gap is not a pairing
rule: approve a maximum date gap and reject violations before training. The paired
pilot uses SSL4EO-S12 v1.1; Copernicus-Pretrain is not presumed to supply same-date pairs.

Use a 10 m reference grid. Retain S2 B01, B02, B03, B04, B05, B06, B07, B08, B8A,
B09, B11, B12 (drop B10). Store native bands where provided; resample once to the
common reference grid before view generation, recording the kernel and nodata policy.
Do not invent high-resolution information for 20/60 m bands. S1 is in dB; check the
source representation before applying any conversion. Normalization constants are
versioned from the approved provider recipe or train-only statistics, never held-out data.

The four **sample-level** strata are:

| m | Input condition for this item | Pilot construction |
|---|---|---|
| 1 | Joint S1+S2 with clear optical data | Metadata/mask-based clear subset |
| 2 | Joint S1+S2 with degraded optical data | Real cloud observations when available; synthetic mask occlusion reported separately |
| 3 | S1 only, S2 withheld | Synthetic dropout on a paired observation |
| 4 | S2 only, S1 withheld | Synthetic dropout on a paired observation |

Define mask thresholds in the manifest before looking at task results. A pasted cloud
mask with constant fill is **occlusion**, not realistic cloudy radiometry. Donors must
come from the training split only. Record original condition, intervention, donor ID,
mask fraction and final sensor set. Availability strata are not semantic classes.

Two global views use all sensors currently available for that sample. Six local views
use subsets: for joint inputs choose S1/S2/both with probabilities 0.4/0.4/0.2; for
single-sensor inputs keep that sensor. Never resurrect a withheld sensor. Record the
actual per-view subset separately from m_n. A view has its own spatial crop, with the
same geometry across sensors within that view. Crop scales: global [0.4,1.0], local
[0.05,0.4]. Flips and 90-degree rotations are shared across sensors. S2 band-gain
jitter ±5% is a candidate; no RGB color jitter on SAR. S1 ±1 dB calibration jitter and
cross-date positives are separate ablations, not baseline invariances.

Let x_{n,v} be a view, b_{n,v}=f(x_{n,v}) its CLS backbone feature and
z_{n,v}=g(b_{n,v}) in R^K its projector output. Use separate linear 16×16 patch stems,
modality embeddings, shared 2-D positional semantics and shared ViT blocks. Missing
sensors are omitted rather than filled with a zero image. The shared projector is
d→2048→2048→K, BN+ReLU on hidden layers, no output BN or unit-vector normalization.
For DDP use synchronized BN (or a separately specified alternative); local BN must
not silently change the recipe. CPU algebra tests bypass BN or freeze its statistics.

Drop patch tokens **after** position/modality embedding, before ViT blocks. Keep CLS.
Use keep=max(1,round(T*(1-rho))), log the integer rounding convention, and preserve
original positional IDs. At 224 pixels: T=196/392 for one/two sensors, keep=20/39 at
rho_g=0.9. At 96 pixels: T=36/72, keep=18/36 at rho_l=0.5. Bucket views by
(resolution,sensor set), restore sample/view order before the projector and loss.

## 2. Live global centre and invariance

For G={1,2}, V=8:

```math
mu_n = (z_{n,1}+z_{n,2})/2,
L_inv = 1/(NVK) sum_n sum_v ||z_{n,v}-mu_n||_2^2.
```

The centre is live: no detach, teacher, EMA or predictor. This is per-coordinate MSE,
not a squared norm summed over K; switching conventions changes the relative loss
weight by K. K and crop/view semantics must be fixed inside a comparable Q1 cohort.

For V_g globals and V_l locals, let zbar_L be the mean local embedding:

```math
dL_inv/dz_{n,u} = 2/(NVK) * [
  z_{n,u}-mu_n + 1[u in G]*(V_l/V_g)*(mu_n-zbar_{L,n})
].
```

Globals feel the extra pull from locals (factor 3 at 2 globals/6 locals). The sum of
embedding gradients over all views is zero: common translation is unconstrained.
**This does not mean the global centre stays fixed**: the sum over global gradients
alone can be nonzero. Parameter sharing further couples updates between samples.
Invariance alone allows all views of all samples to collapse.

## 3. Per-view Epps–Pulley SIGReg

Draw M=1024 unit directions a_j in R^K from a generator shared across ranks and keyed
by (seed,global_step). Use t_l=3l/16 for l=0..16, dt=3/16, omega_l=2dt with endpoints
halved, phi_l=exp(-t_l²/2), q_l=omega_l*phi_l. The doubled positive half-axis quadrature
represents [-3,3]. q already includes the Gaussian window; do not multiply it twice.

For one view and a set S of N_S samples:

```math
C_{j,l} = (1/N_S) sum_{n in S} cos(t_l a_j^T z_n),
D_{j,l} = (1/N_S) sum_{n in S} sin(t_l a_j^T z_n),
EP(S) = (N_S/M) sum_j sum_l q_l [ (C_{j,l}-phi_l)^2 + D_{j,l}^2 ].
SIGReg_pooled = (1/V) sum_v EP({z_{n,v}}_n).
```

Compute projections, trigonometry, reductions and quadrature in fp32 **outside autocast**;
casting z to float inside an active bf16 autocast region alone is insufficient.
Chunk directions if needed, preserving the same A and the full M-normalized sum.
N=1024,V=8,M=1024,17 knots makes the naive x_t tensor ~570 MB in fp32 before saved
autograd intermediates. Profile loss memory, not just backbone memory.

For one slice with s_n=a_j^T z_n:

```math
dEP_j/dz_n = 2 a_j sum_l q_l t_l * [
  D_{j,l} cos(t_l s_n) - (C_{j,l}-phi_l) sin(t_l s_n)
].
```

The leading N_S cancels the 1/N_S derivative of the empirical CF. Systematic
per-sample EP gradients can be O(1) in N_S, while invariance gradients are O(1/N).
This is why batch size changes effective regularization; lambda is not batch-free.

For i.i.d. standard-normal samples, the **expected** statistic is about 1.0525 under
this quadrature. Over the entire real line the expectation is
sqrt(2*pi)-sqrt(2*pi/3)≈1.0594. This is a sampling null expectation, **not a positive
lower bound or guaranteed training plateau**. Optimized finite batches can go below it;
repeated locations violate the i.i.d. assumption. Population mismatch contributes
approximately N_S times a CF discrepancy plus a distribution-dependent sampling term.

## 4. Stratification, mean term and total loss

For four strata S_m, use w_m=1/4 independently of prevalence:

```math
SIGReg_strat = (1/V) sum_v sum_m w_m EP({z_{n,v}: m_n=m}),
T_mu = (1/V) sum_v sum_m w_m N_m ||zbar_{m,v}||_2^2 / K,
L = (1-lambda)*L_inv + lambda*SIGReg_strat + lambda_mu*T_mu.
```

Main candidate: lambda=0.05, lambda_mu=0. P1 includes lambda in {0.02,0.05,0.1}
and a fourfold multiplier as a sensitivity to pooled/stratified gradient scale.
Mean-term ablation: lambda_mu in {0,lambda,10lambda}. For independent Gaussian draws,
E[T_mu]=1; dT_mu/dz_{n,v}=2*w_m*zbar_{m,v}/(VK). T_mu buys a direct, slice-free
first-moment gradient. Its raw unnormalized form and this scaled form are different
objectives; do not reuse coefficients between them.

Uniform weighting deliberately prioritizes rare strata; frequency weighting is not
mathematically wrong, but answers a different risk objective. At N=1024 with four
quotas N_m>=256, all quotas must be exactly 256 and both weight choices coincide.
Fix actual quotas, not only weights, across the Q1 grid. Stratified EP has a different
effective signal scale from pooled EP, so compare both matched-lambda and tuned arms.

Ideal population Gaussianity conditional on m would imply independence from m. A finite
number of directions/knots and low empirical EP do not establish this. Moreover, each
stratum can have its own orthogonal orientation at the same penalty; only paired
cross-view alignment can constrain it. Report cross-condition retrieval, not just EP.

The information conflict is intentional to test: S1 cannot identify every optical
quantity. The squared-error optimum for an information-poor view is E[mu|S1]; forcing
unit variance in inaccessible directions may encourage nuisance features. This is
not proof that the model invents reliable missing optical information. Do not reject
all NDVI predictability from S1: shared geography/vegetation can legitimately correlate.
Test controlled synthetic shared/private factors, and compare held-out task utility
and residual optical targets against S1-only controls.

## 5. Distributed convention

One global batch is generated deterministically; rank r takes [r::world_size]. Each
rank has the same count per stratum. Generate identical directions on all ranks.
For each view/stratum all-reduce the **local cos/sin sums**, then divide by the
global stratum count. Counts use ordinary no-grad SUM; CF sums use an autograd-aware
collective. Every rank computes the same global EP scalar. L_inv is the local mean.

With a differentiable SUM collective, backward sums gradients from the replicated
losses: each local contribution receives world_size times its single-global-loss
gradient. Standard DDP's parameter-gradient average cancels that factor. AVG of
equal-sized local means follows the same cancellation. **SUM does not remove the
dependence on DDP reduction semantics**, contrary to the archived loss review.
Unequal local counts need weighted sums and correct invariance weighting; disallow
them in the first implementation. Assert quotas and the DDP reduction convention.

Do not claim two microbatches with separate EP calls equal one large EP batch: the
CF discrepancy is nonlinear. Disable gradient accumulation in the main recipe unless
the implementation explicitly constructs and tests the global statistic. Compare
single-process and two-rank parameter gradients and one optimizer update against the
same global batch. Scalar-loss agreement alone is not a sufficient distributed test.

## 6. Selection, alignment and scale accounting

Q1 validation loss uses a fixed geographic/time-held-out SSL bank, fixed availability
quotas, crop bank, directions, K and batch. Training views may differ across configs,
but selection is audited on this common validation recipe. Log raw loss, lambda-rescaled
loss (loss/lambda^0.4 as an upstream-inspired proxy, not universal normalization),
L_inv, EP by view/stratum, unscaled EP/N_m, T_mu, rank/spectrum and sensor/cloud probes.
A null-subtracted discrepancy may be negative; do not clip it to manufacture fit.
Use identical task manifests and probe protocols across objectives. MAE and BT losses
are not numerically commensurate with JEPA losses.

Q2: with centred calibration matrices H_A,H_B, SVD(H_A^T H_B)=U Sigma V^T gives
Q=UV^T for H_A Q≈H_B. Fit calibration means/maps **only** on calibration locations;
score orthogonal, linear and regularized CCA on separate geographic evaluation
locations. A >=5k calibration point is preferred when available; 500/1k are sensitivity
arms. The pilot cannot claim 5k distinct calibration locations from 500 observations.
The toy independent-junk model gives orthogonal residual 2(K-n)/K and linear residual
(K-n)/K at infinite calibration size, not a general theorem about trained EO encoders.
Fit/evaluate separation prevents falsely improving residual by calibration overfit.
The identifiability result has latent/transition/dimension assumptions; non-injective
sensor observations and partial crops violate a direct application here.

Count **actual post-drop patch tokens entering the encoder**, excluding CLS, summed
over every view/rank. Count repeated visits separately from unique locations/dates.
For joint items, expected tokens are 2*39+6*(.4*18+.4*18+.2*36)=207.6; for single-sensor
items 2*20+6*18=148. Equal quotas including two single-sensor strata yield ~177.8 tokens
per item, not 208 across the whole sampler. Record realized counts; 2B/8B/32B are
encoder-token budgets, not unique examples. Also record FLOPs, decode bytes, step time,
wall-clock/GPU-hours and NHR. MAE decoder compute belongs in the compute ledger even
though it is not an encoder token. Token matching alone is not compute matching.
