# Clean JEPA on Earth observation: the formulation

Date: 2026-10-10. Merges three independent desk reviews run today (formulation: subagent-formulation.md; data and I/O: subagent-data.md; evaluation, statistics and pricing: subagent-eval.md) plus two checks run on this VM (checks/sigreg_gap_result.md). Supersedes the formulation parts of CLEAN_JEPA_PLAN_V2.md; the three questions (Q1 label-free selection, Q2 version compatibility, Q3 sensor substitution) and the method row M stand, but most of the numbers under them change. Labels: **fact** (primary source linked), **measured** (run today, file named), **estimate** (arithmetic, assumptions stated), **hypothesis** (what we test), **UNVERIFIED**.

## 0. What the three reviews changed

| # | Plan v2 said | Now | Source |
|---|---|---|---|
| 1 | "prediction term", predictor implied | LeJEPA has no predictor and no masking: every view regresses onto the mean of the global-view embeddings (per-coordinate MSE). LeVJEPA keeps that and adds 95% uniform token dropping. | [MINIMAL.md](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md), [LeVJEPA code](https://github.com/MLO-lab/LeVJEPA) |
| 2 | Grid costs ~1,800 node-hours | At 2B/8B/32B processed tokens the LeJEPA S/B/L grid is ~390 GPU-h, the 1B run ~157 GPU-h; everything incl. matched BT/MAE grids and eval ~3,000 GPU-h = ~750 node-hours (estimate, 35% MFU UNVERIFIED). Eval is the largest line. | subagent-eval §6 |
| 3 | Allocation 15,000-20,000 NHR | OpenChange-UK's own SPEC.md and PLAN.md say u6xn is **5,000 NHR through 15 Mar 2027** (AI Builder, BriCS). 750 NHR is 15% of that. Still confirm against the award letter. | repos/openchange-uk/reports/SPEC.md:35, docs/PLAN.md:110 |
| 4 | Q1 at 3-4 shared cells | Underpowered: n=12 gives a 95% CI on rho=0.7 of [0.14, 0.92]. Need ~60 independent runs per objective per population; do it with 64-config Sobol sweeps at ViT-S/2B tokens (~0.2 GPU-h each). | subagent-eval §1c |
| 5 | LeJEPA 0.85 vs TESSERA v2 -0.18 is a contradiction | They measured different things: within-setting hyperparameter sweeps + in-domain linear probe vs 395 runs across four orders of compute vs a 15-task composite. Our design must produce both populations (W within-setting, A across-scale) for each objective; only then is there a contrast. | subagent-eval §1a |
| 6 | Q2 Procrustes on backbone features | The identifiability theorem constrains the projector output h, not the backbone f that probes read. Test the theorem on h; test the applied question on f with linear/CCA maps; report both. Fix projector dim K across sizes. | subagent-formulation §5 |
| 7 | "pooled SIGReg can be satisfied by a modality-gapped mixture" (asserted) | Measured twice. Two strata 1.8 sigma apart along one coordinate: pooled 0.99+-0.13 vs Gaussian 1.03+-0.20 (N=512, K=16); stratified 7.8 (K=16), 2.6 (K=64), +0.4 over floor at K=256. My check: in K=128 a perfect two-point gap along one coordinate scores 1.05 vs 1.07 Gaussian. Detection decays ~1/K. | checks/sigreg_gap_result.md; subagent-formulation §4.2 |
| 8 | ">=256 per stratum because gradient bounds scale 1/N" | Wrong reason (that bound is stability). Right reason is sensitivity: the N-scaled statistic has a floor ~1.05 and a deviation D is visible only when N*D clears the batch-to-batch spread. >=256 stands on the measured numbers. | subagent-formulation §4 |
| 9 | "1B samples per run", 1.1 MB/sample, <150 KB target | Copernicus-Pretrain S1/S2 = 1,067,267 patches in 247,723 cells, ~4 timestamps, 4.22M S2 images. 1B samples = ~240 passes. Measured bytes: 1.81 MB per S2 image (int16, 13 bands, 264^2), 0.56 MB per S1 image; packed+lossless ~0.6 MB per S1+S2 pair. 150 KB needs 128^2 tiles. | [HF card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain); subagent-data §0, §3 |
| 10 | S1/S2 "aligned", temporal pairs from Copernicus-Pretrain | Aligned in space only: S2 dates are the same season in 4 years (2019-2022), S1 dates differ (2021). Time-matched S1<->S2 pairs (median gap 3 days) and temporal positives come from SSL4EO-S12 v1.1. | subagent-data §0.5 |
| 11 | "real S2 cloud masks" for M | Corpora are cloud-filtered at collection: 80.9% of SSL4EO-S12 S2 images have cloud fraction 0, 5.2% > 0.3; Copernicus-Pretrain ships no per-pixel masks. Strata m2 (partly cloudy) and m3 (S1 only) must be synthesised (drop S2; paste a real mask from another sample). | subagent-data §6 |
| 12 | "DEM and ERA5 alignment already in Copernicus-Pretrain" | DEM yes; ERA5 is only the grid definition; no WorldCover. Theory-test variables need external joins. | HF card; subagent-eval §0.6 |
| 13 | Kill rule vs frozen Copernicus-FM on Sen1Floods11 | Sen1Floods11 is not in Copernicus-Bench; Copernicus-FM reports no number there. Rerun the baseline ourselves. | subagent-eval §0.7 |
| 14 | 3 sizes + 1B | Cannot fit a 3-parameter saturating law. Add ViT-Ti (5.7M) and a ~40M width; WSD cooldowns give 5+ token budgets per run. | subagent-eval §0.10, [Hägele et al.](https://arxiv.org/abs/2405.18392) |
| 15 | lambda-rescaled loss as a proxy | Only differs from raw loss if lambda varies across runs. lambda is a sweep axis in population W or the proxy is dropped. Also lambda's effective strength scales with batch N (SIGReg is multiplied by N); fix global batch across the grid and never compare loss totals across N. | subagent-eval §0.4; subagent-formulation §0.7 |

## 1. Model

**Input.** Sentinel-1 GRD IW, VV+VH in dB, 10 m. Sentinel-2 L1C (Copernicus-Pretrain is TOA, 13 bands, 20/60 m bands resampled to 10 m) [fact, Copernicus-FM Table 1]. Store 12 S2 bands (drop B10 cirrus; B1/B9 optional) and 2 S1 bands as uint16 at native resolution; crop 264 -> 256 at load. Per-band normalisation from the published SSL4EO/Copernicus-FM statistics (subagent-formulation §1.2 has the table and sources).

**Tokenisation (option b).** Per-sensor linear patch stems (S2: 12x16x16 -> d; S1: 2x16x16 -> d) into one ViT, with a learned modality embedding added to each token and shared 2-D position embeddings. 256 tokens per sensor per 256^2 crop; S1+S2 = 512 before dropping. The model accepts S1-only, S2-only or both at test time by construction. Rejected: one stacked 14-band stem (cannot take S1-only input without zero-filling), per-band hypernetwork stems as in Copernicus-FM (more flexible than needed, 13x tokens), per-band-group tokens (5x tokens for no byte saving). Tokenisation does not change bytes read; stored bands do.

**Projector.** MLP d -> 2048 -> 2048 -> K with BatchNorm, as in MINIMAL. K is fixed across all sizes (needed for Q2 on h). Default K=128; the M ablation adds K=64 because stratified-SIGReg sensitivity decays with K (§3).

**Sizes.** ViT-Ti 5.7M, S 22M, ~40M (width 512, 12 layers), B 86M, L 303M; one pre-registered ViT-g-class 1B run. Patch 16, 256^2 crops.

## 2. Views and loss

**Views.** Global: 2 crops at 224 of the 256 tile (scale 0.4-1.0). Local: 6 crops at 96 (scale 0.05-0.4). Flips and 90-degree rotations (EO has no canonical orientation). S2 photometric: per-band gain +-5%. S1: no photometric jitter; optional +-1 dB offset per view as a calibration nuisance. Token dropping rho=0.9 on all views (LeVJEPA-style). Centre = mean of the two global embeddings.

Arm A (shared encoder, S1<->S2): globals are full observations (S1+S2 tokens); locals draw sensor in {S1-only, S2-only, both} with probability (0.4, 0.4, 0.2), then crop. All views regress onto the full-observation centre. Samples without clear S2 use the S1 global as target and fall in stratum m3.

Temporal-positive row (ablation only): one extra global from another date, (t, t') sampled as an unordered pair with random role so p(z)=p(z') holds by construction (the stationarity assumption of [2605.26379](https://arxiv.org/abs/2605.26379)). Expect it to hurt change detection: invariance across dates removes the change signal. No world-model claim; SIGReg's gradient w.r.t. transition weights is zero ([2609.36227](https://arxiv.org/abs/2609.36227)).

**Loss.** For sample n, view v, z_{n,v} = g(f(x_{n,v})_cls) in R^K; mu_n = mean over global views.

L_inv = mean_{n,v,k} (mu_{n,k} - z_{n,v,k})^2 (per-coordinate MSE, no normalisation) [fact, MINIMAL].

SIGReg: draw M unit directions a_m (resampled each step, seeded by global step so DDP ranks agree). For view v and slice m, the empirical characteristic function phi(t) = (1/N) sum_n exp(i t a_m^T z_{n,v}); EP_{v,m} = N * sum_j w_j e^{-t_j^2/2} [(Re phi - e^{-t_j^2/2})^2 + (Im phi)^2](t_j) with t_j = linspace(0,3,17), trapezoid weights (symmetric integration over [-3,3]); SIGReg = mean over (v,m). Under DDP the ECF is all-reduced so N is the global batch [fact, code]. Paper recommends 1024 slices, [-5,5]; code uses 256 and [0,3]; Table 1a shows <0.5 points between them.

L = (1 - lambda) L_inv + lambda SIGReg, lambda = 0.05 (paper; MINIMAL run uses 0.02; LeVJEPA uses inv + 0.02 SIGReg). P1 sweeps lambda in {0.02, 0.05, 0.1} at the 5M model. Pre-register the convention; log L_inv and the unscaled EP separately; keep global batch fixed across the grid.

Other defaults to start from (sources in subagent-formulation §3): AdamW, lr warmup then cosine, wd 0.05, drop-path 0.1, bf16, batch 1024 global for S/B.

## 3. Per-modality and availability-stratified SIGReg (M)

Strata per (location, timestamp), c = SEnSeI cloud+shadow fraction in the crop: m1 S2 clear (c <= 0.01); m2 S2 partly cloudy, bins (0.01,0.1], (0.1,0.3], (0.3,0.7]; m3 S1 only (S2 absent or c > 0.7); m4 both (c <= 0.3 and S1 within +-7 days). Measured SSL4EO-S12 v1.1 frequencies: m1 ~83.6%, (0.01,0.1] 6.2%, (0.1,0.3] 5.0%, >0.3 5.2%, S1 within 7 days 72.5%. m2 upper bins and m3 are therefore synthesised: drop S2 for a sampled fraction of pairs; paste real SEnSeI masks from other samples onto clear scenes. State this in the paper.

SIGReg_strat = sum_m w_m SIGReg({z_{n,v} : m_n = m}), w_m = N_m / N, each stratum >= 256 samples per step (loader guarantees it by stratified sampling with replacement for rare strata). What it enforces: z Gaussian conditional on m, hence z independent of m. What it does not enforce: alignment of strata with each other; that comes only from L_inv on paired views (S1-only local onto a full-observation centre). Failure mode to watch: strata align by rotation rather than overlap (SIGReg_strat passes, cross-stratum retrieval of the same scene fails). Diagnostics: linear probe for m and for c from frozen f and from h (target: chance); cross-stratum kNN retrieval of the same scene; per-stratum EP on held-out data.

Sensitivity (measured): detection of a gap along a low-dimensional subspace decays ~1/K for both pooled and stratified SIGReg. Pre-registered variants: K=64, and an explicit stratum-mean term lambda_mu sum_m ||mean_m(z)||^2 on top of stratified SIGReg.

Ablation (B/8B tokens, 3 seeds): no SIGReg / pooled / stratified / stratified without the paired-view invariance / stratified + mean term. Pass only if the m-probe falls to chance, downstream drop when S2 is removed at test time shrinks, clean-input scores stay within seed noise.

Status of the theory link: the S1 and S2 views are different observation functions of the scene, which the identifiability theorem does not cover; "per-modality SIGReg is what stationarity needs" is a hypothesis motivated by the theorem, not implied by it.

## 4. What identifiability predicts and how Q2 tests it

Theorem ([2605.26379](https://arxiv.org/abs/2605.26379), fact): with stationary additive-noise latent transitions, same observation function for both views, encoder output dim = latent dim, and alignment + Gaussian regularisation at the optimum, the encoder recovers the latents up to an orthogonal map; an approximate version degrades with the optimisation gap. Prediction for two independently trained clean JEPAs at the optimum: h_B = Q h_A, Q in O(K).

Test (theory, on h): fit Q by orthogonal Procrustes (SVD of H_A^T H_B) on a calibration set of n_cal in {500, 1k, 5k, 20k} image embeddings (not pixels; a ViT/16 token covers 160 m); report normalised residual ||H_B - H_A Q||^2 / ||H_B||^2 against the general linear map and CCA, for seed pairs, size pairs, data-scale pairs, per objective. Predictions: LeJEPA residual < BT, MAE; seed pairs < data-scale pairs (Thm 3: error grows with the optimisation gap); residual floor (K - n)/K if K exceeds the true latent dim.

Test (applied, on f): retained fraction r = acc(probe_A o M applied to f_B) / acc(probe_B native) with M in {orthogonal, linear, CCA}, linear and kNN probes, in-distribution and on OpenChange-UK. External pairs: GeoTessera v1.0 / v1.1 (dclimate, cambridge) / v2 (two betas), all 128-d, README says variants do not interoperate; BT-family only, so a baseline point, not a theory test. AlphaEarth has one public model version (64-d), so no pair.

## 5. Q1 and Q3 in numbers (details: subagent-eval §1, §3)

Q1 populations. W (within-setting, the LeJEPA claim): ViT-S, 2B tokens, 10% data, 64 Sobol configurations per objective over lr (x10), wd (x10), lambda in [0.01, 0.2] (BT off-diagonal weight / MAE mask ratio for the baselines), number of local views, WSD cooldown length; 8 configurations x 3 seeds for seed noise; ~40 GPU-h for three objectives; repeat 24 configs at ViT-B/8B (~245 GPU-h). A (across-scale, the TESSERA claim): grid endpoints, 63 runs per objective. Unit of analysis = one independently trained run at its final checkpoint; intermediate checkpoints are not independent points. Percentile bootstrap, 10k resamples, cluster by configuration.

Proxies: raw loss, lambda-rescaled loss (alpha fixed on a held-out set of runs), SIGReg term, invariance term, RankMe, LiDAR, alpha-ReQ (reference UNVERIFIED in detail), 1k-label linear probe. Targets per task family: classification, segmentation, change, regression; in-distribution and OpenChange-UK.

Kill (pilot, population W at ViT-S): LeJEPA raw-loss Spearman upper CI bound < 0.5 on all four ID families. Success: rho >= 0.7 with lower bound >= 0.5 on >= 3 of 4 families at S and B; matched contrast rho_LeJEPA - rho_BT >= 0.3 with CI excluding 0; OOD lower bound >= 0.4 on families passed ID.

Q3: Delta = score(S2 input) - score(S1 input) on Copernicus-Bench EuroSAT-S1/S2, BigEarthNet-S1/S2, DFC2020-S1/S2, reported against compute; S1-only on Flood-S1 and PANGAEA Sen1Floods11 (rerun the Copernicus-FM ViT-B baseline ourselves). Theory test: DEM (in corpus), plus externally joined WorldCover water/built-up and ERA5 soil moisture, vs NDVI and S5P NO2; pre-stated threshold on MLP-minus-linear probe R^2.

## 6. Data, bytes, I/O (details: subagent-data)

Corpus: Copernicus-Pretrain S1+S2 raw GeoTIFFs (CC-BY-4.0, 6.3-6.9 TB raw) as the location pool, plus SSL4EO-S12 v1.1 (CC-BY-4.0, time-matched S1/S2, SEnSeI masks) for paired and temporal views; Major TOM (CC-BY-SA) only as a labelled extra arm if the 100% point is data-limited. Copernicus-Pretrain already contains the 251k SSL4EO-S12 locations: dedupe. Unique-location key: Major TOM 10 km grid cell of the patch centre; ERA5 0.25-degree cell for strata and holdout; footprints with IoU > 0.25 in one cell merge. Data axis counts cells, stratified by latitude band so 1% is not Europe-only. Known bias: city-centred sampling (10k largest cities, sigma 50 km) and cloud filtering at collection.

Bytes: uint16 native-resolution packing gives 0.77 MB per 10-band S2 image raw, ~0.38 MB after lossless compression; ~0.6 MB per S1+S2 pair. Tile size, not token dropping, sets bytes: read once, make all V views from one read. Shards: WebDataset tar, ~2 GB each, ~1,250 shards for 4.2M pairs, uint16 npy + json per sample, wids for indexed access; $PROJECTDIR 200 TiB / 50M inodes, scratch 5 TiB, node $LOCALDIR 48 GiB tmpfs [fact, Isambard docs]. Lustre bandwidth per node UNVERIFIED; measure in the first 1-GPU smoke. Compute-side demand at 0.35 MFU: ViT-S needs ~50k samples/s/node, B ~13k, L ~3.7k, so S and B are I/O-bound at any plausible bandwidth unless each read feeds many views; L and 1B are not.

Leakage: remove all UK land plus a buffer (49-61.5 N, 11 W-2.5 E) from pretraining, because OpenChange-UK regions and cutoff are still unset; date cutoff depends on corpus (Copernicus-Pretrain S2 runs to early 2023; SSL4EO-S12 ends 2021-10). EuroSAT, LCZ and BigEarthNet are city-centred in Europe like the corpus: spatial overlap is high and must be reported, with an exclusion of benchmark footprints from the pretraining set where footprints are published.

## 7. Compute (estimate; 35% MFU UNVERIFIED; GH200 per-GPU peak taken as H100)

FLOPs/token ~ 6N + 12 L T d. At T~102 visible tokens per view: S 1.4e8, B 5.3e8, L 1.9e9, 1B 6.1e9. GPU-hours per (data scale, seed), three budgets as separate runs: S 4.6, B 17.7, L 62.4; 1B at 32B tokens 157. LeJEPA grid ~390 GPU-h; BT/MAE at 4 shared cells ~65 each (full matched grids ~390 each); population W sweeps ~285; arms A/A'/S1/S2/M ~70; eval ~1.2-1.7 GPU-h per checkpoint on the fast suite, Copernicus-FM-matched UPerNet protocol only on ~30 final checkpoints. Total ~3,000 GPU-h = ~750 NHR; +50% contingency ~1,100 NHR = 22% of 5,000 NHR. Add the two extra sizes and WSD cooldowns before pricing again.

## 8. Harness

Copernicus-Bench {EuroSAT-S1/S2, BigEarthNet-S1/S2, DFC2020-S1/S2, Flood-S1, LCZ-S2, Cloud-S2} + PANGAEA {Sen1Floods11, BioMassters} + OpenChange-UK frozen splits; GEO-Bench m-bigearthnet and m-so2sat as an external comparability row only. Frozen linear + kNN (with and without L2 normalisation) + one fixed light decoder; fine-tuning on a subset. Reporting checklist from the GFM audit ([2605.12678](https://arxiv.org/abs/2605.12678)).

## 9. Local pilots, revised

| # | Pilot | Decides |
|---|---|---|
| P0 | SSL4EO-S12 v1.1 500-location slice via the OpenChange-UK source gate: UK box removed, 84 per latitude band, >=25% with cloud > 0.05, 2.64 km dedupe, selection by sha256(sample_id) at a pinned revision; record revision, licence, checksums | data legality, formats, mask availability |
| P1 | LeJEPA from MINIMAL with per-sensor stems, pooled / per-modality / stratified SIGReg, K in {64, 128, 256}; 5M and 20M models; 3 seeds; lambda sweep; log L_inv, EP per stratum, effective rank of f and h | code correctness; stratified sensitivity at real K |
| P2 | Procrustes / linear / CCA residuals on h and f for P1 seed pairs | Q2 harness and the h-vs-f question |
| P3 | Proxy harness and bootstrap code on P1 runs; two paired Copernicus-Bench loaders + one OpenChange-UK loader | Q1 harness |
| P4 | Pack the slice to uint16 shards; bytes/sample; CPU decode throughput; views-per-read | I/O plan |
| P5 | Pre-registration draft (template in subagent-eval §8) | protocol freeze |

Then: 30-minute 1-GPU smoke on Isambard (human-run, NGC image, Apptainer --nv), measured read bandwidth and MFU, re-priced grid, population-W pilot, M2-style report before the main grid.

## 10. Open items

- Confirm u6xn: 5,000 NHR through 15 Mar 2027 per OpenChange-UK docs; approved project wording; whether this study is in scope.
- MFU on GH200 for small ViTs at ~100 tokens; Lustre bandwidth per node.
- Whether the SSL4EO-S12 v1.1 zarr repo is per-sample (19,234 files) or chunked.
- alpha-ReQ reference and formula; Copernicus-Bench task footprints and dates for the exclusion list.
- Whether the identifiability assumptions hold approximately on EO at all; P2 is the first look.
