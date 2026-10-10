# Clean JEPA for Sentinel-1 + Sentinel-2: exact formulation for pilot P1

Date: 2026-10-10. Desk research only. Nothing trained. I read the LeJEPA and LeVJEPA code (cloned `galilai-group/lejepa` and `MLO-lab/LeVJEPA`) and the PDFs of the papers listed. The two numerical results in §4 come from a 60-line NumPy script I ran on the VM (`ep_mixture.py`, CPU, seconds). They are measurements of the estimator on synthetic Gaussians, not of any trained model. Labels: **[fact]** = read in a primary source (URL given); **[measured]** = my synthetic computation; **[estimate]**; **[hypothesis]**; **UNVERIFIED** = could not confirm.

## 0. What changes in plan v2 (summary)

1. **LeJEPA has no predictor and no masking.** The "prediction" term in the code is MSE from every view to the mean of the global-view embeddings. Plan v2's "pred" wording should say "invariance". LeVJEPA keeps this structure, uses one global view as the target, and adds 95% uniform token dropping. Neither paper uses a mask-token predictor.
2. **The "≥256 per stratum because gradient bounds scale as 1/N" argument is wrong as stated.** The 1/N bound (LeJEPA Thm 4) is a stability bound. It does not set a minimum batch. In the reference code the statistic is multiplied by N, so the per-sample gradient is O(1). The real constraint is sensitivity: under H0 the N-scaled statistic sits on a floor of about 1.05 for any N, and a population deviation D is only visible when N·D is well above the batch-to-batch spread. Keep ≥256, and justify it with the measured numbers in §4.
3. **"Pooled SIGReg allows a modality gap" now has a number.** Take a 50/50 two-stratum mixture separated by 1.8σ along one coordinate, with total covariance I. Pooled SIGReg at N=512 cannot tell it from a Gaussian: 0.99 ± 0.13 against 1.03 ± 0.20 under H0 at K=16. Stratified SIGReg at 2×256 reads 7.8 ± 0.5 at K=16 and 2.6 ± 0.1 at K=64 **[measured]**. The reason: a symmetric two-component mixture with matched variance departs from a Gaussian only at 4th order in the projected gap, while a mean-shifted stratum departs at 2nd order.
4. **The detection gets weaker as the projector dimension K grows**, for both pooled and stratified SIGReg. A random slice puts only about 1/K of its energy on the gap direction. At K=256 the stratified deviation is +0.39 above the 1.05 floor at N=256. Add an explicit stratum-mean term (or a small K) to M, as a pre-registered variant.
5. **Q2 has to be tested in projector space, not backbone space.** The identifiability theorem constrains h (the thing SIGReg sees), which is the projector output. The projector is a non-linear MLP that is thrown away after training, and downstream probes read backbone features. The theorem gives no prediction for backbone features. Change Q2: (i) Procrustes on projector outputs is the test of the theorem; (ii) backbone transfer is the applied question, using linear/CCA maps; report both.
6. **The theorem assumes encoder output dim = true latent dim and the same observation function for both views.** S1-vs-S2 views break the second assumption: the sensors are different functions of the scene and S2 sees things S1 cannot. So Q3's "per-modality SIGReg is what the stationarity assumption needs" is at best a heuristic. Restate it as a hypothesis motivated by the theorem, not as something the theorem implies.
7. **λ conventions differ.** LeJEPA uses (1−λ)·inv + λ·SIGReg with recommended λ=0.05 (paper) or 0.02 (MINIMAL run). LeVJEPA uses inv + 0.02·SIGReg. Because SIGReg is multiplied by the batch size, λ's effective strength changes with N (Shared Gaussianization: critical weight λ_c = 1/(1+B·g_S)). Pre-register the convention, keep the global batch fixed across the grid, and log both raw terms.
8. **Tokenisation choice does not change bytes read.** Bytes are set by which bands are stored. Recommend per-sensor linear stems into one ViT (option b), with modality embeddings and per-sensor token sets. Drop B10 (cirrus) from storage, and consider B1/B9 too: this saves 1–3 of 15 channels in bytes.
9. **Temporal-positive row:** sample (t, t′) symmetrically so stationarity holds by construction. Do not expect it to help change detection: invariance across dates removes the change signal. Keep it as an ablation row only.

## 1. Input specification

### 1.1 Sensors and bands
- **Sentinel-1 GRD**, IW mode, VV and VH, 10 m pixel spacing. Copernicus-Pretrain stores 264×264×2 S1 GRD patches at 10 m, about 4 timestamps per patch, 247,723 grid cells, 1,067,267 patches **[fact]** (Copernicus-FM paper Table 1, https://arxiv.org/abs/2503.11849). Values are backscatter in dB: the published means of −12.5 dB (VV) and −20.2 dB (VH) only make sense in dB **[fact, inferred from statistics]**. Speckle is multiplicative in linear power and roughly additive in dB. GRD products are multi-looked, so speckle is reduced but not removed **[fact about GRD; the residual-speckle level in these patches is UNVERIFIED]**.
- **Sentinel-2.** Copernicus-Pretrain S2 is **TOA (L1C), 13 bands**, 264×264×13 at 10 m (20 m and 60 m bands resampled to 10 m) **[fact]** (same table). SSL4EO-S12 ships both L1C (13 bands) and L2A (12 bands: no B10) **[fact]** (https://github.com/zhu-xlab/SSL4EO-S12, `ssl4eo_dataset.py` defines S2C 13-band and S2A 12-band statistics). Native resolutions: 10 m B2, B3, B4, B8; 20 m B5, B6, B7, B8A, B11, B12; 60 m B1, B9, B10 (standard ESA band table; https://sentiwiki.copernicus.eu/web/s2-mission).
- **Recommendation:** pretrain on **L1C** because it is what Copernicus-Pretrain has at scale. Evaluate on whatever the benchmark provides, and when an L2A task appears, report the L1C→L2A shift as its own row. Mixing L1C and L2A in pretraining is a hidden availability stratum. Do not do it in P1.

### 1.2 Normalisation (exact, sourced)
Per-channel z-score, x̂ = (x − μ_c)/σ_c, after S2 values are read as integer DN (reflectance × 10,000). S1 is already in dB. Statistics:

| Channel | μ | σ | Source |
|---|---|---|---|
| S1 VV (dB) | −12.54847 | 5.25698 | SSL4EO-S12 `ssl4eo_dataset.py` S1_MEAN/S1_STD |
| S1 VH (dB) | −20.19237 | 5.91151 | same |
| S2 L1C B1…B12 incl. B10 (13) | 1605.58, 1390.78, 1314.87, 1363.52, 1549.44, 2091.75, 2371.72, 2299.90, 2560.30, 830.07, 22.10, 2177.07, 1524.07 | 786.79, 850.35, 875.06, 1138.85, 1122.18, 1161.59, 1274.39, 1248.43, 1345.53, 577.32, 51.15, 1336.10, 1136.54 | SSL4EO-S12 S2C_MEAN/S2C_STD |
| S2 L2A (12, no B10) | 752.40, 884.30, 1144.16, 1297.47, 1624.91, 2194.64, 2422.21, 2517.76, 2581.65, 2645.52, 2368.51, 1805.07 | 1108.03, 1155.15, 1183.63, 1368.11, 1370.27, 1355.55, 1416.51, 1474.79, 1439.31, 1582.28, 1455.52, 1343.48 | SSL4EO-S12 S2A_MEAN/S2A_STD |

Source file: https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py. Copernicus-FM pretraining uses nearly the same numbers rounded (S1 −12.59/−20.26, 5.26/5.91; S2C 1612.9…, 791.0…), and also sets NaN/inf to 0 after normalising **[fact]** (https://github.com/zhu-xlab/Copernicus-FM, `Copernicus-FM/src/main_pretrain.py`, `RemoveOutlier` transform). The band order in the S2C list is B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12, which matches B10's tiny mean (22) at position 11 **[fact, inferred from values]**. P0 must check the order against the actual files. For P1, use the SSL4EO-S12 values and recompute them on the pilot slice as a check, not as a replacement: changing the statistics between pilot and grid changes the input distribution.

Clip S1 dB to [−50, +10] before normalising **[estimate; the threshold is not taken from a primary source]**. Very low-backscatter water pixels and no-data are otherwise heavy outliers. SIGReg is robust to embedding outliers, but the stem is not.

### 1.3 Tokenisation options
Notation: D = width (S 384, B 768, L 1024), p = patch size, C = channels (S2 L1C 13 + S1 2 = 15). At 224×224 and p=16 there are 196 spatial positions. At 264×264, p=16 does not divide 264. p=12 gives 484 positions, p=8 gives 1,089, p=22 gives 144. Copernicus-FM random-resize-crops 264 down to 224 **[fact]** (`main_pretrain.py`, `RandomResizedCrop((224,224), scale=(0.2,1.0))`). I use 224 / p=16 below.

| Option | Tokens, both sensors present | S1-only / S2-only | Stem params (ViT-B) | Notes |
|---|---|---|---|---|
| (a) one patch-embed over stacked 15-ch cube | 196 | needs zero-fill of missing channels → a distribution shift the model has to learn | 15·256·768 ≈ 2.95 M | The "missing" pattern is in the input itself, so it shows up as a stratum (§4). Cheapest. |
| (b) per-sensor linear stems, shared ViT | 392 (196 S1 + 196 S2, each + modality embedding); or 196 if summed | 196 | S1 0.39 M + S2 2.56 M = 2.95 M | A missing sensor just means no tokens. Concatenation doubles tokens; token drop cancels this. CR-JEPA and Le MuMo use modality stems **[fact]** (2606.00706, 2603.24327). |
| (c) spectral hypernetwork (Copernicus-FM/DOFA) | 196 per sensor image | 196 | generator cost does not depend on C; exact count UNVERIFIED | Takes any band set, given wavelength and bandwidth via Fourier features **[fact]** (2503.11849 §4.1, `dynamic_hypernetwork.py`). S1 has no wavelength in the same sense; Copernicus-FM has a separate path for it. More code and more ways to fail. |
| (d) per-band-group tokens | 5 groups (S2: 10 m VNIR; 20 m red-edge; 20 m SWIR; 60 m atmos; S1) → 980; per band → 2,940 | 196–784 | small per group | 5× (groups) or 15× (bands) more tokens; attention cost grows faster than linearly. Enables spectral-group dropping as a view. |

**Recommendation: (b), with concatenated per-sensor token sets, a learned modality embedding, and uniform token dropping.** Reasons. (1) S1-only, S2-only or both at test time is native: absent sensors contribute no tokens, rather than zero-filled channels. (2) The bytes read are the same for all four options because they depend on stored bands, not tokens. FLOPs are not the constraint (plan v2 §1.5). The FLOP penalty of 392 tokens is therefore cheap, and dropping at ρ=0.9 brings it to about 40 tokens. (3) The stem is a linear layer, so the "shared encoder" claim in Q3 stays clean: everything after the stem is shared. Keep (d) at 4 S2 groups as the spectral-dropping ablation (plan v2 M's cheaper second ablation). It does not need to be the default. Drop B10 from stored shards because it is cirrus-only and L1C-only. This cuts S2 bytes by 1/13 at no cost to land tasks **[hypothesis: no downstream loss]**.

## 2. What a "view" is

### 2.1 What the reference code actually does
- **LeJEPA (paper Algorithm 2, MINIMAL.md).** V = V_g + V_l views per sample. In the paper default V_g = 2 global views at 224² and V_l = 6 local views at 96², giving V = 8 ("eight views (V=8) containing two global views") **[fact]** (https://arxiv.org/abs/2511.08544 §6). The paper recommends "λ = 0.05, V_g = 2, V_l = 8, and batch size ≥128 as starting points". This conflicts with the V = 8 setup used elsewhere in the paper; both are quoted there. The invariance term is
  \[ \mathcal L_{\text{inv}} = \frac1{V}\sum_{v'=1}^{V}\big\|\mu_n - z_{n,v'}\big\|_2^2,\qquad \mu_n=\frac1{V_g}\sum_{v=1}^{V_g} z_{n,v}, \]
  averaged over n (paper Eq. 5–7). **There is no predictor network and no masking.** The "prediction" is all views regressed onto the centroid of the global-view embeddings, with gradient through both sides and no stop-gradient. In MINIMAL.md, `inv_loss = (proj.mean(0) - proj).square().mean()` uses the mean over all V views as the centre (MINIMAL draws V identically augmented views and no separate globals). `.mean()` averages over views, batch and K dimensions, so the term is per-coordinate MSE, not a sum over coordinates **[fact]** (https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).
- **LeVJEPA.** One global view (224², photometrically clean) and 10 local views (96², scale 0.02–0.4). Target is the global embedding z_0, `pred_loss = (global_emb - embeddings).pow(2).mean()`, with gradients through both. 95% uniform random token dropping per view during training only. SIGReg on the projected CLS of each view. Projector d→2048→BN→GELU→256. λ=0.02 used as **L = inv + 0.02·SIGReg**. AdamW lr 4e-4, betas (0.9, 0.95), wd 0.04 constant, 1,200-step warm-up then flat. Effective batch 3,072. SIGReg 1,024 slices, 17 knots on [0, 3]. EMA weights kept only as the evaluation checkpoint **[fact]** (https://github.com/MLO-lab/LeVJEPA `conf/config.yaml`, `main.py` `multiview_forward`, `paper.md` §3 and App. A/B). Structured (tube) masking hurt and uniform dropping helped (README: 50.7 → 39.6 with tube masking) **[fact]**.

So "view" in a clean JEPA means an input transformation whose embedding must agree with the target view's embedding. Every positive-view choice is an explicit invariance claim.

### 2.2 Candidate EO view generators

| Generator | Invariance imposed | Information thrown away | Role |
|---|---|---|---|
| Spatial global/local crops (same date, same sensor set) | Scale and position within the cell | Within-cell layout, context outside the crop | **Positive, default** |
| Uniform token dropping ρ ∈ {0.75, 0.9, 0.95} | Which subset of patches is observed | None in expectation; acts as augmentation (LeVJEPA) | **Positive, default** (cost lever) |
| Flips / 90° rotations | Orientation | Slope direction, shadow azimuth, SAR look geometry (S1 is **not** rotation-invariant: look direction and layover) | Positive for S2; S1 flips only along the azimuth axis **[hypothesis]**; ablate |
| Photometric jitter | Small radiometric shifts | Absolute reflectance, which matters for regression tasks (biomass, water quality) | Mild only: per-band gain ±5% **[estimate]**; never colour-jitter in RGB space |
| Spectral band subset (drop 1–2 S2 groups) | Which S2 groups are present | Information unique to the dropped group (e.g. SWIR moisture) | **Ablation row** (M, second ablation) |
| Sensor swap: S1-only view ↔ S2-only view, same cell and date (±days) | Sensor identity | Everything only one sensor sees: NDVI and spectral chemistry (S2); roughness, moisture, all-weather structure (S1) | **Positive for arm A**, with the full-observation view as target. This is Q3. |
| Temporal: same cell, different date | Season, phenology, **and real change** | Change signal and phenology | **Ablation row only.** Invariance across dates teaches the model to ignore change, which is the opposite of what OpenChange-UK tests. |
| Cloud-mask / cloud-token drop on S2 | Cloud occlusion | Content under the cloud | **Positive in M only**, with the real mask; target is the clear or full view |

### 2.3 Default view sets
- **Single-sensor S2 LeJEPA:** V_g = 2 global crops (224, scale 0.4–1.0), V_l = 6 local crops (96, scale 0.05–0.4), flips plus 90° rotations, per-band gain ±5%, token drop ρ = 0.9 on globals and locals. Centre = mean of the 2 globals (LeJEPA form, not LeVJEPA single-target), because two globals give a less noisy centre **[hypothesis]**.
- **Single-sensor S1 LeJEPA:** same crops, azimuth flips only, no photometric jitter. Instead, an optional additive dB offset of ±1 dB per view as a calibration nuisance **[estimate]**.
- **Arm A (S1↔S2 shared encoder):** for each paired sample, the global views are full observations (S1+S2 tokens). Local views draw sensor ∈ {S1-only, S2-only, both} with probabilities (0.4, 0.4, 0.2), then crop. All views regress onto the centre of the full-observation globals. When a sample lacks clear S2, the S1 global is the target and that sample goes to the S1-only stratum (§4).
- **Temporal-positive row:** arm A plus one extra global per sample from another date, with (t, t′) sampled symmetrically: unordered pair, random role. This makes p(z) = p(z′) hold by construction, which is the stationarity assumption of https://arxiv.org/abs/2605.26379. Log a linear k-step forecast probe as a diagnostic only. SIGReg's gradient with respect to transition weights is zero (https://arxiv.org/abs/2609.36227), so this row says nothing about "world models".

## 3. The loss, exactly

Embeddings: z_{n,v} = g_φ(f_θ(x_{n,v})_{[cls]}) ∈ ℝ^K, with g_φ the projector. Let Z_v ∈ ℝ^{N×K} be the batch of embeddings for view v.

**SIGReg (Epps–Pulley on random slices), as coded.** Draw M directions a_m ∼ Unif(S^{K−1}), resampled every step and seeded by global step so all ranks agree. For each view v, slice m and knot t_j:
\[ \hat\varphi_{v,m}(t)=\frac1N\sum_{n}e^{\,i t\, a_m^\top z_{n,v}},\qquad
\mathrm{EP}_{v,m}=N\sum_{j=1}^{J} w_j\,e^{-t_j^2/2}\Big[(\Re\hat\varphi-e^{-t_j^2/2})^2+(\Im\hat\varphi)^2\Big](t_j) \]
with t_j = linspace(0, 3, J=17), trapezoid weights w = 2Δt (endpoints Δt). These integrate over [−3, 3] by symmetry. Then SIGReg = mean over (v, m) of EP_{v,m} **[fact]** (MINIMAL.md `SIGReg`, `lejepa/univariate/epps_pulley.py`, LeVJEPA `module.py`). Under DDP the ECF is all-reduced first, so N becomes the global batch: `* N * world_size` **[fact]**. The paper's Algorithm 1 uses t ∈ linspace(−5, 5, 17) with trapz, and the paper recommends "17 integration points, an integration domain of [−5, 5], and 1024 slices" **[fact]** (2511.08544 §6). The library default and both codebases use [0, 3] with 17 knots. Table 1a shows [−3, 3] and [−5, 5] are within 0.5 points of each other.

**Invariance.** As in §2.1: L_inv = mean over n, v′, k of (μ_{n,k} − z_{n,v′,k})². This is per-coordinate MSE, not cosine. No normalisation of z.

**Total.** LeJEPA: L = (1−λ)·L_inv + λ·SIGReg. LeVJEPA: L = L_inv + λ·SIGReg. **For P1, use the LeJEPA convex form with λ = 0.05, plus a λ ∈ {0.02, 0.05, 0.1} sweep at the 5M model.** It matches the paper whose loss–accuracy claim Q1 tests, and LeJEPA's λ-by-views table shows a broad plateau for 8 views (about 83.0–84.3 top-1 over the middle of the λ sweep) **[fact; the exact column-to-λ mapping in my PDF text extraction is UNVERIFIED]**.

**The 1/N gradient bound and what it implies.** LeJEPA Thm 4: |∂EP/∂z_i| ≤ 4σ²/N and |∂²EP/∂z_i²| ≤ C√π σ³/(2N) **[fact]**. This is for the statistic before the ×N scaling. After scaling, each sample's gradient is O(1) whatever N is, which is why training is stable at small batch. **The bound does not say anything about a minimum batch.** What does: under H0, E[EP] = Σ_j w_j e^{−t_j²/2}(1 − e^{−t_j²}) ≈ **1.05** for every N **[measured, exact formula: Var cos + Var sin = 1 − e^{−t²}]**. A population deviation D(t) contributes about N·Σ w D on top of that floor. So the smallest detectable deviation scales as 1/N, and the batch-to-batch spread of the slice-averaged statistic is about 0.04–0.2 at M = 1024 (§4). Shared Gaussianization adds that the plug-in ECF estimator has a finite-batch bias that favours imperfect alignment, removable with an off-diagonal U-statistic **[fact]** (https://arxiv.org/abs/2610.10299 §5).

**Defaults with sources, and what I would change for EO:**

| Item | LeJEPA | LeVJEPA | P1 (EO) and why |
|---|---|---|---|
| λ | 0.05 recommended; MINIMAL run 0.02 | 0.02 (additive) | 0.05 convex; sweep; fixed global batch |
| Slices M | 1024 recommended; MINIMAL 256 | 1024 | 1024; cost is tiny compared with the encoder |
| Knots / domain | 17, [−5, 5] (paper); 17, [0, 3] (code) | 17, [0, 3] | 17, [0, 3] (match code) |
| Projector | MLP 2048-2048-K with BN (MINIMAL); Table 1d: no projector size collapses (which K is best is UNVERIFIED) | 2048 → 256 | 2048 → K=64 at P1 because of the 1/K dilution in §4; K = 256 as a row |
| Batch | ≥128; Table 1c 128–1024 within 2.5 points | 3,072 | P1 CPU: 512 (4 strata × 128, see §4); grid: ≥1,024 global |
| Views | V_g = 2, V_l = 6 (8 total) | 1 + 10 | 2 + 6 |
| Optimiser | AdamW, MINIMAL: wd 5e-2, lr 2e-3 in the example run, warm-up + cosine (paper sweep values UNVERIFIED) | AdamW 4e-4, betas (0.9, 0.95), wd 0.04, warm-up + flat; grad-accum 2 | AdamW 5e-4 (B), warm-up 5% + cosine, wd 0.05; flat schedule only if checkpoints along a run feed Q1 |
| Epochs | 100 (IN-1k ViT-L), 800 (MINIMAL inet10) | 26 (≈10k steps) | token-budget defined (plan v2) |
| Token drop | none | 0.95 | 0.9 (EO patches are less redundant than video frames **[hypothesis]**) |
| Embedding readout | CLS of backbone | CLS | CLS; mean-pool row for segmentation probes |

## 4. Per-modality and availability-stratified SIGReg

### 4.1 Definition
Availability mask m_n ∈ 𝓜 = {S2-clear, S2-cloudmasked, S1-only, both} (S2-clear/cloudmasked mean S2 without S1). Let N_m = |{n : m_n = m}|. Then
\[ \mathrm{SIGReg}_{\text{strat}}=\sum_{m\in\mathcal M} w_m\;\overline{\mathrm{EP}}\big(\{z_{n,v}: m_n=m\}\big),\qquad w_m=\frac{N_m}{N} \]
where EP̄ uses the **N_m-scaled** statistic. N_m-scaling with w_m = N_m/N means each stratum enters at the same gradient scale per sample as pooled SIGReg, so λ keeps its meaning. A stratum with N_m < N_min (default 64) is left out of SIGReg for that step and logged. It still enters L_inv.

### 4.2 Pooled SIGReg allows a modality gap: worked example
Two strata, 50/50. On coordinate e_1, stratum ± has mean ±δ and variance 1 − δ². Every other coordinate is N(0, 1). The pooled distribution has mean 0 and covariance exactly I. For a slice a with a_1 = ⟨a, e_1⟩, the projected characteristic functions are:
- pooled: φ_pool(t) = cos(δ a_1 t)·e^{−(1−δ²a_1²)t²/2}, so φ_pool − e^{−t²/2} = O((δ a_1 t)^4) (the moments match to order 3);
- one stratum: φ_+(t) = e^{iδa_1t} e^{−(1−δ²a_1²)t²/2}, so |φ_+ − e^{−t²/2}|² ≈ (δ a_1 t)² e^{−t²}, which is O((δa_1)²).

With a_1² ∼ Beta(½, (K−1)/2), E[a_1²] = 1/K and E[a_1⁴] = 3/(K(K+2)). So the pooled signal is about δ⁴/K² and the per-stratum signal about δ²/K.

Numbers, computed with the reference quadrature ([0, 3], 17 knots, Gaussian window) and M = 1024 random slices, 30 independent batches **[measured]**:

| K | δ (gap = 2δ σ) | Pooled, N = 512 | Stratified, 2 × 256 | H0 floor |
|---|---|---|---|---|
| 16 | 0 | 1.03 ± 0.20 | 1.05 ± 0.16 | 1.05 |
| 16 | 0.7 | 0.99 ± 0.14 | **5.16 ± 0.31** | |
| 16 | 0.9 | 0.99 ± 0.13 | **7.83 ± 0.46** | |
| 64 | 0 | 1.08 ± 0.06 | 1.05 ± 0.04 | |
| 64 | 0.7 | 1.04 ± 0.08 | **1.99 ± 0.10** | |
| 64 | 0.9 | 1.08 ± 0.09 | **2.63 ± 0.11** | |

Population deviation × N, no sampling noise: pooled at K=16, δ=0.9, N=512 gives 0.004. Per-stratum at N=256 gives 6.85 (K=16), 1.61 (K=64), 0.39 (K=256). So **pooled SIGReg cannot see a 1.8σ modality gap at all**: the mixture's value is inside the H0 spread and is in fact slightly below the H0 mean. Stratified SIGReg sees it clearly at K ≤ 64, but at K=256 only by +0.39 over the floor. The gap could also hide in a non-variance-matched form, but variance mismatch is 2nd-order and pooled SIGReg would catch it. The variance-matched mixture is the worst case, and an encoder can reach it.

### 4.3 What stratification enforces, and what it does not
At the stratified optimum, p(z | m) = N(0, I) for every m, so p(z | m) does not depend on m and **z ⟂ m in distribution**. A linear or non-linear probe cannot recover m from z. It does **not** enforce that the same scene gets the same z under different m. A model could send S1 samples to N(0, I) and S2 samples to N(0, I) through two unrelated maps (a rotation between strata) and pass SIGReg_strat perfectly. **Alignment comes only from L_inv across views with different m for the same scene.** Concretely: for a paired sample, the centre μ_n is computed from full-observation global views, and the local views include S1-only and S2-only crops. Then (μ_n − z_{n,S1-only})² ties the S1-only stratum to the same coordinates. Samples that have only one sensor (no paired views) get no alignment pressure. They should be ≤50% of any batch, otherwise their strata can rotate freely **[hypothesis]**.

**Mean term (proposed variant, cheap):** add β·Σ_m ‖mean(z | m)‖². It hits the gap at first order, does not dilute with K, and costs nothing. This is new and must be ablated.

### 4.4 Batching
- **Minimum per stratum:** 128 at K = 64 on CPU P1 (≈ 4σ detection of δ = 0.7 by the table above); 256 on the grid. A stratum with fewer than 64 samples in a step is skipped, as above.
- **Stratified sampler:** each global batch is built with fixed quotas, e.g. (both 40%, S2-clear 20%, S2-cloudmasked 20%, S1-only 20%). Quotas are per **global** batch, because SIGReg all-reduces the ECF across ranks: per-GPU quotas are not needed if the ECF is reduced per stratum. That costs one all-reduce of 4 × 2 × M × 17 floats per step, i.e. 4 × 34,816 values.
- **Rare stratum:** oversample to its quota from a stratum-specific shard pool. Log the effective number of epochs per stratum, since rare strata repeat.
- **Gradient accumulation:** SIGReg is not additive across micro-batches (ECF is non-linear), so with accumulation each micro-batch has a smaller N_m. Either keep the per-stratum batch large enough in each micro-batch, or accumulate ECFs and compute the statistic once. The second needs custom autograd and two passes. Avoid accumulation in M runs **[estimate]**. LeVJEPA's own recipe uses grad-accum 2, so its SIGReg is computed on per-micro-batch ECFs **[fact]** (README).
- **Compute overhead:** SIGReg costs O(N·K·M·J). Stratifying does not change the total sample count, only the grouping. Overhead is the extra all-reduce and masking: negligible compared with the encoder **[estimate]**.

### 4.5 Failure modes
1. **Stratum leakage through L_inv:** if S1-only locals are always smaller crops than S2 locals, crop scale becomes a proxy for m. Use identical crop distributions across sensor draws.
2. **Speckle as a per-view nuisance (Shared Gaussianization):** pure SG lowers its loss by adding per-view nuisance when the shared code is non-uniform. For LeJEPA, the nuisance-free solution is a strict local minimiser only when λ < λ_c = 1/(1 + B·g_S), where g_S is the channel gain and B the batch size **[fact]** (2610.10299 Prop. 6). Speckle is a ready-made isotropic-ish nuisance in S1. Bigger batches make it more dangerous at fixed λ. Pilot check: probe z for speckle realisation, using two independently speckled copies of the same scene made by re-multiplying by Gamma(L) noise (L = equivalent number of looks, UNVERIFIED for these patches). Report the measured gain g against (1−λ)/λ, as the paper does.
3. **Cloud fraction is continuous:** bin into {0, (0, 0.1], (0.1, 0.4], > 0.4} sub-strata of S2-cloudmasked. Alternative: a conditional variant that splits by the median cloud fraction in each batch. Binning is simpler and goes in the pre-registration.
4. **Cloud mask correlated with geography:** UK coast cloud fraction depends on region and season, so "z ⟂ m" also pushes z ⟂ (region, season) to the extent that m predicts them. This can remove useful signal. Diagnostic: report a probe for region from z within each stratum against a no-SIGReg baseline. If stratification lowers region-probe accuracy beyond seed noise, it is removing geography.
5. **K dilution (§4.2):** at K = 256 stratified detection is weak. Use K = 64 or add the mean term.

## 5. Identifiability and what it buys

### 5.1 Assumptions of https://arxiv.org/abs/2605.26379, in plain terms
Observations come from latents through x = g(z) with the same injective g for both views (paper §3, "shadows (x) of reality (z), projected (g)") **[fact, wording paraphrased]**. Assumptions 3.1:
- (i) **Independence:** latent coordinates are independent, and so are their transitions.
- (ii) **Stationarity:** both views have the same latent marginal, p(z) = p(z′).
- (iii) **Additive noise:** z′_i = m_i(z_i) + η_i, with η independent of z.

Gaussian world: z ∼ N(0, I_n), so the transition is OU, z′ = ρz + √(1−ρ²)η. Thm 1: any h with h(z) ∼ N(0, I_n) has L(h) ≥ 2(1−ρ)n, with equality iff h = Qz for some Q ∈ O(n). Thm 3 (approximate): if alignment is within δ of optimal and ‖Cov(h) − I‖_F ≤ ε, then there is a Q ∈ O(n) with E‖h(z) − Qz‖² ≤ D + (ε + D)², where D = δ/(2ρ(1−ρ)). The paper also says the theorem assumes encoder output dim m = n; for m < n the subspace is not determined, for m > n "extra dimensions must collapse or encode redundancy" **[fact]**. All results are population-level.

### 5.2 Assessment for EO

| Assumption | Spatial crops | S1 vs S2 views | Seasonal pairs |
|---|---|---|---|
| Same g for both views | Roughly (same sensor) | **Fails**: different physics; S2 observes variables S1 cannot | Roughly (same sensor) |
| Stationarity | Holds by symmetric sampling | Holds only for the sensor-shared part | Holds **only if (t, t′) are sampled symmetrically**; fails for ordered pairs |
| Independence | Unknown; land cover, soil and topography co-vary | Same | Same |
| Additive, state-independent noise | Crop "noise" depends on content | Speckle is multiplicative in linear units, close to additive in dB | **Fails** for land-cover change (jumps) and phenology (state-dependent) |
| m = n | Unknown n | Same | Same |

### 5.3 What it predicts for Q2, and how to test it
If both encoders reach the population optimum on the same data: h_A = Q_A z and h_B = Q_B z, so h_B = Q_B Q_Aᵀ h_A, which is orthogonal. Fit Q̂ = argmin_{Q∈O(K)} ‖H_B − H_A Q‖_F by SVD of H_Aᵀ H_B on a calibration set (5k samples, plan v2). Then a linear probe w_B can be replaced by Q̂ w_A.

What breaks this:
1. **Space.** The theorem is about h = projector output. Backbone features f(x) ∈ ℝ^D go through a non-linear MLP before h. So the prediction holds for h, not for f. Probes in plan v2 use f.
2. **Different data scale** changes the effective p(z) and the optimisation gap δ. Thm 3 says error grows with δ/(2ρ(1−ρ)), so data-scale pairs should align worse than seed pairs.
3. **Different size → different D** (384/768/1024) in backbone space. K is the same if the projector width is fixed across sizes, so for h the orthogonal map is still square. Fix K across the grid so this holds.
4. **m ≠ n:** if K > n, the extra coordinates are noise that SIGReg forces to be Gaussian. They will not align across seeds, and Procrustes residual has a floor of (K − n)/K of the variance. Diagnostic below.

**Diagnostic (pre-register):** on a calibration split, for each checkpoint pair, report the normalised residual r = ‖H_B − H_A M‖²_F / ‖H_B‖²_F for three maps:
- (i) orthogonal Procrustes;
- (ii) unconstrained least squares;
- (iii) CCA, as the canonical correlations sorted.
Do this in **both** projector space (h) and backbone space (f, with (i) replaced by orthogonal Procrustes after PCA-whitening to a common dim). Theory predicts, in h-space for LeJEPA pairs: r_orth ≈ r_lin, both small; CCA correlations near 1 for the first n̂ directions and then dropping. The index where they drop estimates n̂ (latent dimension). Baselines (Barlow Twins, MAE) should show r_orth ≫ r_lin. In f-space no prediction follows; only report. The headline metric (probe retained fraction) comes after these residuals and should be explained by them.

## 6. P1 pilot training step (PyTorch-style)

Lines marked `# NEW` differ from MINIMAL.md; `# LeV` follows LeVJEPA.

```python
STRATA = ["both", "s2_clear", "s2_cloud", "s1_only"]               # NEW
t = torch.linspace(0, 3, 17); dt = 3/16
w = torch.full((17,), 2*dt); w[[0, -1]] = dt
phi = torch.exp(-t**2/2); W = w*phi

def ep_stat(z, A):                     # z: (V, n, K) -> scalar, N-scaled (MINIMAL)
    x = (z @ A).unsqueeze(-1) * t       # (V, n, M, J)
    re, im = x.cos().mean(-3), x.sin().mean(-3)   # all_reduce here under DDP
    err = (re - phi)**2 + im**2
    return ((err @ W) * z.size(-2)).mean()        # mean over views and slices

def sigreg_strat(z, m, A, n_min=64):                     # NEW
    tot, logs, N = 0., {}, z.size(1)
    for s_id, s in enumerate(STRATA):
        idx = (m == s_id).nonzero().squeeze(1)
        if idx.numel() < n_min: logs[f"sig/{s}"] = float("nan"); continue
        ep = ep_stat(z[:, idx], A); logs[f"sig/{s}"] = ep.item()
        tot = tot + (idx.numel() / N) * ep                # w_m = N_m/N
    return tot, logs

def make_views(batch):                                    # NEW (EO views)
    G = [full_obs_global(batch, scale=(0.4, 1.0)) for _ in range(2)]  # S1+S2 tokens if available
    L = []
    for _ in range(6):
        sensor = sample_sensor(batch.avail, p=(0.4, 0.4, 0.2))  # S1-only / S2-only / both, per sample
        L.append(local_crop(batch, sensor, size=96, scale=(0.05, 0.4)))
    return G, L                                            # each: token sets + modality ids

def train_step(net, proj, batch, opt, lam=0.05, K=64, M=1024, beta=0.0, step=0):
    G, L = make_views(batch)
    views = G + L
    emb = [net(v.tokens, v.modality, drop=0.9) for v in views]       # LeV: uniform token drop
    z = torch.stack([proj(e) for e in emb])                          # (V, N, K)
    centre = z[:2].mean(0, keepdim=True)                             # LeJEPA: mean of globals
    inv = (centre - z).square().mean()                               # per-coord MSE, no stop-grad
    g = torch.Generator().manual_seed(step)                          # synced slices
    A = torch.randn(K, M, generator=g); A = A / A.norm(dim=0)
    sig_pool = ep_stat(z, A)                                         # logged always (pooled)
    sig_s, logs = sigreg_strat(z, batch.stratum, A)                  # NEW
    mean_pen = sum(z[:, batch.stratum == s].mean((0, 1)).square().sum()
                   for s in range(4) if (batch.stratum == s).sum() >= 64)  # NEW, optional
    loss = (1 - lam) * inv + lam * sig_s + beta * mean_pen           # NEW: sig_s replaces pooled
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                                            # NEW logging
        h = emb[0].float(); s = torch.linalg.svdvals(h - h.mean(0))
        p = s / s.sum(); erank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
        logs.update(inv=inv.item(), sig_pool=sig_pool.item(), sig_strat=float(sig_s),
                    erank_backbone=erank.item(), mean_pen=float(mean_pen))
    return logs
```

Arm A′ (pooled) replaces `sig_s` with `sig_pool`. The online linear probe from MINIMAL (detached features) stays and is logged. Effective rank is the RankMe form (exp of the entropy of normalised singular values) on the backbone CLS. Also log it on z. Unit test before any run: on synthetic Gaussians the per-stratum stat should be about 1.05, and on the §4.2 mixture it should match the table to within one SD.

## 7. Risks in the formulation (ranked)

| # | Risk | Hits | P1 can detect? |
|---|---|---|---|
| 1 | Q2 tested in backbone space, where the theorem says nothing (§5.3) | Q2 | Yes: compute residuals in h and f on P1 seed pairs |
| 2 | Strata separate by rotation, not by gap. SIGReg_strat passes; S1↔S2 alignment fails because paired samples are too few | Q3, M | Yes: cross-stratum kNN retrieval of the same scene; per-stratum probe of m |
| 3 | K dilution: at K = 256 the stratified signal is about 0.4 over the floor (§4.2) | M | Yes: P1 at K = 64 and 256 |
| 4 | λ interacts with batch size and speckle (λ_c = 1/(1+B·g_S)); grid batch ≠ pilot batch | Q1, Q3 | Partly: measure the speckle gain g at two batch sizes |
| 5 | Stratification removes geography/season through cloud–region correlation | M, OpenChange-UK OOD | Partly: region probe per stratum on the 500-sample slice is underpowered |
| 6 | Loss-ranking claim (Q1) depends on the λ convention and on N-scaling; comparing loss values across batch sizes is invalid | Q1 | Yes: log raw inv and un-scaled EP; never compare totals across N |
| 7 | Projector absorbs Gaussianity: z is Gaussian, f is anisotropic, so a linear probe on f gains nothing from SIGReg | Q1, Q2 | Yes: effective rank and Epps–Pulley on f against z |
| 8 | Temporal-positive row erases change signal | OpenChange-UK | Not in P1 (needs change labels) |
| 9 | Embedding dim and probe type: kNN depends on the norm and f is not normalised | Q1 | Yes: report linear and kNN with and without L2 normalisation |
| 10 | L1C/L2A or normalisation mismatch between pretrain and benchmarks | all | Yes, at P0 |

## 8. Closest prior work

| Title | URL | What it does | What it does not do |
|---|---|---|---|
| LeJEPA | https://arxiv.org/abs/2511.08544 | SIGReg + multi-view invariance; no predictor/EMA; loss–accuracy Spearman ≈ 0.85, to 0.99 with λ^α rescaling | EO; multi-sensor; strata; scaling across sizes on one dataset with OOD |
| LeVJEPA | https://arxiv.org/abs/2608.27395 | LeJEPA for video; 95% uniform token drop; 1 global + 10 locals; λ = 0.02 | EO; multiple sensors; stratified regularisation |
| When Does LeJEPA Learn a World Model? | https://arxiv.org/abs/2605.26379 | Linear (orthogonal) identifiability under independence, stationarity, additive noise; Gaussian uniqueness; approximate bound | Different sensors per view; m ≠ n; finite-sample; cross-checkpoint compatibility |
| Shared Gaussianization | https://arxiv.org/abs/2610.10299 | What SIGReg-type tests certify; nuisance channel; λ_c = 1/(1+B·g_S); U-statistic fix | Stratified or multi-sensor; SAR speckle |
| One-Step Next-Latent Prediction Is Not a World Model | https://arxiv.org/abs/2609.36227 | Isotropy penalty has zero gradient on transition; one-step ≠ kernel | Spatial/sensor views |
| CR-JEPA | https://arxiv.org/abs/2606.00706 | S1/S2 modality stems, shared trunk, masked latent prediction, SIGReg on retrieval projections | Clean (no-predictor) objective; stratified SIGReg; scaling; compatibility |
| HQ-JEPA | https://arxiv.org/abs/2605.31068 | S1/S2 masked latent prediction with EMA teacher, SIGReg on fused embedding, quantum similarity loss | Clean objective; per-modality or stratified SIGReg |
| Le MuMo JEPA | https://arxiv.org/abs/2603.24327 | LeJEPA multimodal (RGB + LiDAR/thermal) with fusion tokens; SIGReg on joint CLS | EO; per-modality SIGReg; missing-modality strata |
| Copernicus-FM / Pretrain / Bench | https://arxiv.org/abs/2503.11849 | 18.7M aligned images from all major Sentinel missions; spectral hypernetwork tokeniser; MIM + distillation | JEPA/SIGReg objective; loss-based selection |
| TESSERA v2 | https://arxiv.org/abs/2607.03949 | 395 Barlow Twins runs; loss vs score Pearson −0.18, Spearman −0.16 | SIGReg objective; image-level JEPA; cross-version alignment |
| SSL4EO-S12 | https://github.com/zhu-xlab/SSL4EO-S12 | S1 GRD + S2 L1C/L2A, 4 seasons, per-band statistics used above | SIGReg; availability strata |

I found nothing applying SIGReg per modality or per availability stratum. The three S1/S2 papers above apply it to one pooled or fused embedding **[fact for those three; absence elsewhere UNVERIFIED beyond arXiv search on 2026-10-10]**.

## 9. Plan v2 corrections, one line each
See §0. In short: say "invariance", not "prediction"; replace the 1/N justification with the sensitivity numbers; put §4.2's table in the paper as the motivating result for M; set K = 64 as the default and K = 256 as a row; add the mean-penalty variant; run Q2 in projector space with linear/CCA comparisons; call Q3's link to the theorem a hypothesis; fix one λ convention and one global batch; use per-sensor stems; drop B10; sample temporal pairs symmetrically and keep them as an ablation only.
