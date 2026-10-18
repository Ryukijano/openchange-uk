# Defensive Mathematical Formulation, Real-Data Pathology Audit, and Hybrid JEPA / Flow-Matching Engineering (18 October 2026)

This document records in full detail the mathematical derivations, empirical data-pathology measurements on the approved 480-pair `SSL4EO-S12 v1.1` pilot (`pilot_v1`), code-audit findings, and defensive loss/architecture implementations completed on **18 October 2026** across `src/openchange/jepa/` and `tests/`.

---

## 1. Strategic Context & `u6xn` AIRR Compute Award Framing

### 1.1 The `u6xn` Allocation and Its Scientific Mandate
- **Allocation**: BriCS project `u6xn` (`AI Builder: NV-Nemotron`) on **Isambard-AI Phase 2** (aarch64 NVIDIA GH200 Grace Hopper Superchips), awarded via the **DSIT × NVIDIA Open-Source AI Builder Fund** (20,000 GH200 GPU-hours = **5,000 Node-Hours (NHR)** through **15 March 2027**).
- **Engineering & Scientific Rule** ([`docs/PROGRESS.md`](../../PROGRESS.md)):
  $$\text{Useful task} \longrightarrow \text{Failure of existing methods} \longrightarrow \text{Independent target \& suitable data} \longrightarrow \text{Training objective} \longrightarrow \text{Architecture} \longrightarrow \text{Public demo}.$$
- **What Has Been Verified on Real 4×GH200 Hardware (`job 7527080`, 18 Oct 2026)**:
  - [`reports/jepa-pilot/2026-10-18/README.md`](../../../reports/jepa-pilot/2026-10-18/README.md) ran `scripts/train_jepa_distributed.py --local-fit` on 1 node / 4×GH200 (`pytorch_26.09-py3.sif`, `torch 2.14.0a0+nv26.09`, `bf16`) over 480 approved `SSL4EO-S12 v1.1` Sentinel-1 GRD + Sentinel-2 L2A pairs (`pilot_v1`).
  - Completed 50 optimizer steps in `00:01:35` (`exit 0`, ~`30.2k` encoder tokens/s, `471 MB` peak allocated memory per rank, checkpoints written at steps 25 and 50).
- **What Failed Earlier and Is Not Carried Forward**:
  - The CPU coastal shoreline-error metadata head (`Narrabeen + Duck`, [`reports/coastal/2026-10-14/README.md`](../../../reports/coastal/2026-10-14/README.md)) failed its predefined 15% retained-RMSE reduction gate (`+5.3%` on Fold A, `-14.3%` on Fold B; mean `-4.5%`).

### 1.2 Triage of Non-Benchmaxed Scientific Problems for `u6xn`
Before locking in the mathematical defensive overhaul for **Option A** and **Track A (WAM)**, we audited why standard Earth Observation (EO) representation papers feel "benchmaxed" (optimizing soft linear-probe mIoU or RankMe scores on homogeneous nadir splits) versus why **`AstroLoc` (ICCV 2025, `arXiv:2502.07003`) / `EarthMatch` (CVPRW 2024, `arXiv:2405.05422`)** achieved operational adoption in the **NASA Artemis II Lunar Science Data User Guide (Oct 2026, PDS DOI `10.17189/dtpt-2857`)**, and compared four candidate problem formulations:

| Candidate Track | Core Scientific Conflict / Failure Wall | Hard Physical / Operational Verification (Anti-Benchmax) | Code & Compute Status |
|---|---|---|---|
| **Option A (Primary EO Track): Defensive `Clean-JEPA` vs. `TESSERA v2` (`Barlow Twins`) + All-Weather `S1 ↔ S2` Continuity & Zero-Re-Embedding Upgrades** | **TESSERA v2** (`arXiv:2607.03949`, July 2026, 395 runs on Isambard-AI GH200) found Barlow Twins pretraining loss has near-zero correlation with downstream EO task rank ($|r| < 0.2$). Meanwhile **LeJEPA** (`arXiv:2511.08544`, `arXiv:2605.26379`) proves `SIGReg` yields orthogonal identifiability ($\rho \approx 0.85\text{–}0.99$), **but** naive full-space `Clean-JEPA` suffers a **multi-sensor Bayes-risk conflict** on single-sensor views and **multimodal mean collapse** under 100% cloud blackout. | **Q1**: Label-free checkpoint selection Spearman $\rho$ & top-1 regret against matched `Barlow Twins` / `VICReg`.<br>**Q2**: Unlabeled Subspace Orthogonal Procrustes version upgrade ($v_a \to v_b$) with $<2\%$ decision churn.<br>**Q3**: 100% cloud-blackout `S1 → S2` substitution and Conditional Flow Matching (CFM) mode preservation. | **Implemented & verified in `src/openchange/jepa/`**; 4×GH200 pilot (`job 7527080`) passed. |
| **Option B (Track A WAM): Hybrid Causal-JEPA + Conditional Flow Matching World-Action Models vs. 30k-GPU-Hour Pixel Video Generators** | **Long-WAM** (`arXiv:2610.10528`, Oct 2026) and **OpenWAM** (`arXiv:2610.07922`, Oct 2026) show long history ($0 \to 19.2\text{ s}$) boosts manipulation (`68.4% → 97.8%`), but claim bidirectional encoders fail and require 30,720 H100 GPU-hours of pixel video generation. Pure deterministic $L_2$ JEPA (`LeWM` / `LeWAM`) plans $48\times$ faster but suffers **multimodal mean collapse** on bifurcating futures and **action blindness** ($\|\partial \hat{z}_{t+1}/\partial a_t\|_F \to 0$) on 20 fps video. | Closed-loop manipulation/control rollouts, **16-way counterfactual action-outcome identification**, history-shuffle negative controls, and the **$2\times 2$ Oracle Factorial** (`learned vs. true dynamics` $\times$ `latent vs. true cost`). | **Implemented & verified in `src/openchange/jepa/wam.py`** (all 7 audit bugs fixed + CFM & Action-InfoNCE heads). |
| **Option C (Active Observation): Forcing-Conditioned Value-of-Information (VoI) Observation Selection & Conformal Hazard Nowcasting** | Passive EO forecasting dumps all historical cubes into a model and outputs uncalibrated heatmaps. Cloud cover blocks UK optical passes ~70% of the time. | **Strict Archive-Reveal Game**: pixels of candidate passes are concealed at decision time $t_0$; policies select passes under byte budget $B$ and are scored on withheld future truth via CRPS, Brier score, and Mondrian conformal coverage ($1-\alpha$). | **Implemented in `science_wm.py` & `guarantees.py`**. |
| **Option D (`Agentic-SfM 2.0` / `TerraLoc-3D`): Diagnostic-Guided 2D $\to$ 3D Specialist Routing & 3D DEM-Conditioned Geometric Lock** | Fast `LoFTR + RANSAC` (`0.062s/pair`) collapses to `0.240 AUC@10°` on low-overlap ScanNet pairs (`job 8235219`) where `VGGT` achieves `0.631 AUC@10°` (`+163%`, `1.28s/pair`). Similarly, `EarthMatch` (`arXiv:2405.05422`) had to exclude **55 of 323 (17%) high-tilt AIMS astronaut photos** because 2D homography ($H \in \mathbb{R}^{3\times 3}$) fails under 3D terrain relief parallax, SAR/illumination inversion, and repetitive-terrain false locks. | Exact 6-DoF camera pose registration (`AUC@5°/10°` under `pycolmap` bundle adjustment or 3D DEM ray-casting) and calibrated abstention (`FDR <= 2%`). | **Implemented in `/home/ubuntu/repos/agentic-sfm`** (`main` @ `2105f55`). |

---

## 2. Mathematical Formulation: Where Pure JEPA Fails on Multi-Sensor EO (`Option A`) & How We Resolved It

### 2.1 The Multi-Sensor Bayes-Risk & Law-of-Total-Variance Conflict
Consider paired multi-sensor Earth Observation data where each physical location/acquisition has latent state
$$z = \big[z_{\text{sh}},\, z_{s1},\, z_{s2}\big] \in \mathbb{R}^{K}, \qquad K = K_{\text{sh}} + K_{s1} + K_{s2},$$
decomposed into:
1. **Shared physical factors** $z_{\text{sh}} \in \mathbb{R}^{K_{\text{sh}}}$: geometry, coastline/field boundaries, canopy structure, inundation extent, and surface topography observable by both radar and optical sensors.
2. **SAR-private factors** $z_{s1} \in \mathbb{R}^{K_{s1}}$: C-band VV/VH dielectric roughness, Bragg scattering, and double-bounce phase/polarimetric structure not determined by optical reflectance.
3. **Optical-private factors** $z_{s2} \in \mathbb{R}^{K_{s2}}$: pigment/chlorophyll absorption, red-edge slope, and shortwave-infrared spectral albedo in Sentinel-2 bands `B01–B12` that are unidentifiable from C-band SAR backscatter alone:
   $$p(z_{s2} \mid S_1) \approx p(z_{s2}) \quad \text{(up to weak shared-habitat correlation }\rho \ll 1\text{)}.$$

#### How the Baseline Full-Space `jepa_loss` Creates a Mathematical Contradiction
In the baseline `jepa_loss` ([`src/openchange/jepa/losses.py`](../../../src/openchange/jepa/losses.py)):
1. **Full-$K$ Cross-View Invariance**:
   $$\mathcal{L}_{\text{inv}} = \frac{1}{NVK}\sum_{n=1}^{N}\sum_{v=1}^{V} \|z_{n,v} - \mu_n\|_2^2, \qquad \mu_n = \frac{1}{V_g}\sum_{v \in G} z_{n,v},$$
   penalizes squared error across **all $K$ coordinates** regardless of which sensors view $(n, v)$ observes.
2. **Full-$K$ Availability-Stratified Epps–Pulley `SIGReg`**:
   $$\text{EP}(S_m) = \frac{N_m}{M}\sum_{j=1}^{M}\sum_{l=0}^{16} q_l \Big[\big(C_{j,l}(S_m) - \phi_l\big)^2 + D_{j,l}(S_m)^2\Big], \qquad \phi_l = e^{-t_l^2 / 2},$$
   pushes the empirical distribution of **every stratum $m \in \{0, 1, 2, 3\}$** toward the full-space isotropic Gaussian $\mathcal{N}(0, I_K)$ along $M=1024$ dense random unit directions $a_j \in \mathbb{S}^{K-1}$.

Now examine **Stratum 2 (`s1_only`, where S2 is withheld/100% clouded)** on the optical-private subspace $z_{s2} \in \mathbb{R}^{K_{s2}}$:
- Under squared-error invariance $\mathcal{L}_{\text{inv}}$, the Bayes-optimal (MMSE) predictor of the optical-private coordinates from radar $S_1$ alone is the conditional expectation:
  $$\hat{z}_{s2}^*(S_1) = \mathbb{E}[z_{s2} \mid S_1].$$
- By the **Law of Total Variance**:
  $$\mathrm{Cov}\big(\mathbb{E}[z_{s2} \mid S_1]\big) = \mathrm{Cov}(z_{s2}) - \mathbb{E}\big[\mathrm{Cov}(z_{s2} \mid S_1)\big] = \rho^2 I_{K_{s2}} \prec I_{K_{s2}},$$
  where $\rho \in [0, 1)$ is the cross-sensor predictability of optical-private factors from SAR ($\rho = 0$ for strictly private spectral factors).
- Consequently, along any direction $a_j$ supported on the $S_2$-private subspace, the empirical characteristic function of the MMSE-optimal predictor in Stratum 2 is:
  $$\hat{\varphi}_{m=2}(t) \approx \exp\!\left(-\frac{1}{2} t^2 \rho^2\right) \neq \exp\!\left(-\frac{1}{2} t^2\right) = \phi(t).$$
- **The Irreducible Conflict**:
  - $\nabla_{z_{s2}(S_1)} \mathcal{L}_{\text{inv}}$ pulls $z_{s2}(S_1) \to \mathbb{E}[z_{s2} \mid S_1] \approx 0$ (variance $\rho^2 \ll 1$).
  - Simultaneously, $\nabla_{z_{s2}(S_1)} \text{EP}(S_2)$ actively pushes $z_{s2}(S_1)$ outward to unit variance ($\mathrm{Var} = 1$).
  - The encoder is forced into one of two destructive equilibria:
    1. **Hallucination**: The $S_1$ stem injects unaligned unit-variance nuisance/speckle noise into $z_{s2}$ to satisfy `SIGReg` on Stratum 2, corrupting cross-sensor alignment.
    2. **Spectral Suppression**: The $S_2$ stem discards optical-private spectral information (collapsing chlorophyll/albedo features in $z_{s2}$) so that $\mu_n$ carries nothing $S_1$ cannot predict.

### 2.2 Resolution: Factorized Split-Subspace Clean-JEPA (`split_subspace_jepa_loss`)
Implemented in [`src/openchange/jepa/losses.py`](../../../src/openchange/jepa/losses.py) (`split_subspace_jepa_loss`), we partition the projector output into explicit shared and sensor-private subspaces:
$$z = \big[z_{\text{sh}},\, z_{s1},\, z_{s2}\big] \in \mathbb{R}^{K_{\text{sh}} + K_{s1} + K_{s2}},$$
with dedicated unit direction matrices $A_{\text{sh}} \in \mathbb{R}^{K_{\text{sh}} \times M}$, $A_{s1} \in \mathbb{R}^{K_{s1} \times M}$, and $A_{s2} \in \mathbb{R}^{K_{s2} \times M}$:

1. **Shared Subspace ($z_{\text{sh}}$, active on all strata $m \in \{0, 1, 2, 3\}$)**:
   - Cross-view invariance with acquisition time-gap attenuation $w_n = \exp(-\Delta t_n / \tau_{\Delta t})$:
     $$\mathcal{L}_{\text{inv}}^{\text{sh}} = \frac{1}{NV K_{\text{sh}}} \sum_{n=1}^{N} w_n \sum_{v=1}^{V} \big\|z_{n,v}^{\text{sh}} - \mu_n^{\text{sh}}\big\|_2^2.$$
   - Stratified Epps–Pulley Gaussian regularization across all present strata $m \in \{0, 1, 2, 3\}$:
     $$\mathcal{L}_{\text{SIGReg}}^{\text{sh}} = \frac{1}{V |\mathcal{M}_{\text{sh}}|} \sum_{v=1}^{V} \sum_{m \in \mathcal{M}_{\text{sh}}} \text{EP}\big(\{z_{n,v}^{\text{sh}} : m_n = m\};\, A_{\text{sh}}\big).$$

2. **SAR-Private Subspace ($z_{s1}$, active only on S1-observing strata $\mathcal{M}_{s1} = \{0, 1, 2\}$)**:
   - Invariance and `SIGReg` are evaluated **exclusively on samples where S1 is present** ($m_n \in \{0, 1, 2\}$):
     $$\mathcal{L}_{\text{SIGReg}}^{s1} = \frac{1}{V |\mathcal{M}_{s1}|} \sum_{v=1}^{V} \sum_{m \in \mathcal{M}_{s1}} \text{EP}\big(\{z_{n,v}^{s1} : m_n = m\};\, A_{s1}\big).$$
   - On S2-only samples ($m_n = 3$), $z_{s1}$ receives **zero `SIGReg` gradient** ($\nabla_{z_{n,v}^{s1}} \mathcal{L}_{\text{SIGReg}}^{s1} \equiv 0$ for $m_n = 3$) and is regularized toward the prior mean (zero energy):
     $$\mathcal{L}_{\text{inactive}}^{s1} = \frac{1}{V |S_3| K_{s1}} \sum_{v=1}^{V} \sum_{n \in S_3} \big\|z_{n,v}^{s1}\big\|_2^2.$$

3. **Optical-Private Subspace ($z_{s2}$, active only on S2-observing strata $\mathcal{M}_{s2} = \{0, 1, 3\}$)**:
   - Invariance and `SIGReg` are evaluated **exclusively on samples where S2 is present** ($m_n \in \{0, 1, 3\}$):
     $$\mathcal{L}_{\text{SIGReg}}^{s2} = \frac{1}{V |\mathcal{M}_{s2}|} \sum_{v=1}^{V} \sum_{m \in \mathcal{M}_{s2}} \text{EP}\big(\{z_{n,v}^{s2} : m_n = m\};\, A_{s2}\big).$$
   - On S1-only samples ($m_n = 2$), $z_{s2}$ receives **identically zero `SIGReg` gradient** ($\nabla_{z_{n,v}^{s2}} \mathcal{L}_{\text{SIGReg}}^{s2} \equiv 0$ for $m_n = 2$) and is penalized toward zero energy:
     $$\mathcal{L}_{\text{inactive}}^{s2} = \frac{1}{V |S_2| K_{s2}} \sum_{v=1}^{V} \sum_{n \in S_2} \big\|z_{n,v}^{s2}\big\|_2^2.$$

4. **Combined Split-Subspace Objective**:
   $$\mathcal{L}_{\text{split}} = (1 - \lambda)\,\frac{\mathcal{L}_{\text{inv}}^{\text{sh}} + \mathcal{L}_{\text{inv}}^{\text{priv}}}{2} + \lambda\,\frac{\mathcal{L}_{\text{SIGReg}}^{\text{sh}} + \mathcal{L}_{\text{SIGReg}}^{\text{priv}}}{2} + \lambda_{\text{inactive}}\,\mathcal{L}_{\text{inactive}}.$$

**Empirical Verification (`test_split_subspace_resolves_bayes_risk_conflict_and_gradcheck`)**:
- When Stratum 0, 1, 3 embeddings have true $\mathcal{N}(0, I)$ optical-private features $z_{s2}$ while Stratum 2 (`s1_only`) outputs the Bayes-optimal conditional expectation $\hat{z}_{s2}(S_1) = 0$:
  - Full-space `jepa_loss` suffers a massive Epps–Pulley penalty (`sigreg = 12.69` vs. `1.05` null) and drives nonzero hallucination gradients into Stratum 2's $z_{s2}$ block.
  - `split_subspace_jepa_loss` achieves near-null `sigreg_private` (`0.72`), zero `inactive_private_penalty`, and **identically zero `SIGReg` gradient (`0.0`) on Stratum 2's $z_{s2}$ coordinates**, while passing `torch.autograd.gradcheck` in `float64`.

---

### 2.3 The Subspace Procrustes Corollary (`Q2`) & Matryoshka Prefix Evaluation
Under `LeJEPA`'s orthogonal identifiability theorem, two independently trained models $A$ and $B$ identify the shared latent factors up to an orthogonal transformation $Q_{\text{sh}} \in O(K_{\text{sh}})$. However, when sensor-private or run-specific nuisance dimensions of total width $K_{\text{priv}} = K - K_{\text{sh}}$ are unaligned across versions/modalities ($Z_{\text{priv}}^{(A)} \perp Z_{\text{priv}}^{(B)}$, each with unit variance):
- Fitting a **single full-$K$ Orthogonal Procrustes matrix** $Q \in O(K)$ on $Z = [Z_{\text{sh}}, Z_{\text{priv}}]$ incurs an asymptotic relative squared Frobenius residual floor of:
  $$\lim_{N \to \infty} \frac{\|(Z^{(A)} - \mu_A) Q^* - (Z^{(B)} - \mu_B)\|_F^2}{\|Z^{(B)} - \mu_B\|_F^2} = \frac{2(K - K_{\text{sh}})}{K}.$$
  For $K = 16, K_{\text{sh}} = 8$, this theoretical noise floor is $\frac{2(16 - 8)}{16} = 1.0$ (`100%` relative residual on the private half, dragging full-space residual to $\approx 0.56$ at $N=96$ even when the shared block is perfectly orthogonal).
- **Implemented Solution ([`src/openchange/jepa/guarantees.py`](../../../src/openchange/jepa/guarantees.py))**:
  - `fit_subspace_procrustes` and `apply_subspace_procrustes`: Fits block-wise Orthogonal Procrustes transformations $(Q_{\text{sh}}, Q_{s1}, Q_{s2})$ separately on $z_{\text{sh}} \in \mathbb{R}^{K_{\text{sh}}}$, $z_{s1} \in \mathbb{R}^{K_{s1}}$, and $z_{s2} \in \mathbb{R}^{K_{s2}}$, reporting `shared_relative_residual` (`< 1e-12` on exact shared rotations), `full_relative_residual`, and `theoretical_full_floor`, with optional `zero_inactive_private in ("none", "s1", "s2", "both")` for cross-sensor transfer.
  - `evaluate_matryoshka_prefixes`: Evaluates parameter-free nearest-class-centroid classification accuracy and retention across nested Matryoshka prefixes $z_{:, :k}$ for $k \in \{16, 32, 64, 128\}$ (directly testing `TESSERA v2`'s prefix compression question).

---

### 2.4 Distributed-Exact Comparator Losses (`Barlow Twins` & `VICReg`) for `Q1`
To test **Q1** (`Clean-JEPA` vs. `TESSERA v2` Barlow Twins checkpoint selection correlation) without confounding the training objective with distributed batch-normalization differences, we implemented distributed-exact `barlow_twins_loss` and `vicreg_loss` in [`src/openchange/jepa/losses.py`](../../../src/openchange/jepa/losses.py):
- **`barlow_twins_loss(z, lam_bt=5e-3, eps=1e-5, dist=None)`**:
  - Uses `all_reduce_sum_nograd` for global sample count $N_{\text{global}}$ and `all_reduce_sum_autograd` for global first moments $\sum_n z_{A,n}, \sum_n z_{B,n}$, global centered second moments $\sum_n (z_{A,n} - \mu_A)^2$, and the global cross-correlation matrix:
    $$C = \frac{1}{N_{\text{global}}} \sum_{r=0}^{W-1} \big(\bar{Z}_A^{(r)}\big)^\top \bar{Z}_B^{(r)} \in \mathbb{R}^{K \times K}, \qquad \mathcal{L}_{\text{BT}} = \frac{1}{K}\left[\sum_{i=1}^{K} (1 - C_{ii})^2 + \lambda_{\text{BT}} \sum_{i \neq j} C_{ij}^2\right].$$
- **`vicreg_loss(z, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0, dist=None)`**:
  - Computes invariance MSE across global views plus distributed-exact global standard-deviation hinge $\frac{1}{K}\sum_j \max(0, \gamma - \mathrm{Std}(z_j))$ and off-diagonal covariance penalty $\frac{1}{K}\sum_{i \neq j} \mathrm{Cov}(Z)_{ij}^2$.

---

## 3. Why Pure $L_2$ JEPA Fails Under Multimodal Conditionals & Dense Video (`Option A` Imputation & `Track A` WAM) vs. Conditional Flow Matching

### 3.1 The Multimodal Mean-Collapse Theorem
Whenever a predictor $f_\theta(c)$ is trained with deterministic $L_2$ regression $\mathcal{L}_{L_2} = \mathbb{E}[\|x_1 - f_\theta(c)\|_2^2]$, its unique Bayes-optimal solution is the conditional mean:
$$\hat{x}_1^*(c) = \mathbb{E}[x_1 \mid c].$$
When the true conditional distribution $p(x_1 \mid c)$ is **multimodal**—for example:
1. **In EO (`Option A`, `S1 → S2` under 100% cloud blackout)**: A given C-band SAR backscatter signature $c = z_{\text{sh}}(S_1)$ corresponds to two distinct optical land-cover or phenological modes $z^{(1)}, z^{(2)} \in \mathbb{R}^{K_{s2}}$ with probabilities $\pi$ and $1-\pi$, or
2. **In World-Action Models (`Track A`, `z_{<=t}, a_t → z_{t+1}`)**: A manipulation trajectory approaches an obstacle or contact bifurcation where going left ($z^{(1)}$) and going right ($z^{(2)}$) are both valid futures—

the $L_2$ JEPA predictor converges to the chord midpoint:
$$\hat{x}_1^*(c) = \pi z^{(1)} + (1 - \pi) z^{(2)},$$
which lies **off the data manifold** (in the zero-density interior between modes) and incurs an irreducible MSE floor $\pi(1-\pi)\|z^{(1)} - z^{(2)}\|_2^2 + \mathrm{Var}_{\text{within}}$.

#### Resolution: Optimal-Transport Conditional Flow Matching (`OT-CFM`)
Implemented in [`src/openchange/jepa/losses.py`](../../../src/openchange/jepa/losses.py) (`conditional_flow_matching_loss`, `sample_conditional_flow`, `ConditionalFlowBridge`) and [`src/openchange/jepa/wam.py`](../../../src/openchange/jepa/wam.py) (`FlowMatchingWAMHead`):
- Given source noise $x_0 \sim \mathcal{N}(0, I_D)$, target latent $x_1 \sim p(x_1 \mid c)$, and flow time $\tau \sim \mathcal{U}[0, 1]$, we define the Optimal-Transport linear probability path and conditional velocity:
  $$x_\tau = \big(1 - (1 - \sigma_{\min})\tau\big) x_0 + \tau x_1, \qquad u_\tau(x_\tau \mid x_0, x_1) = x_1 - (1 - \sigma_{\min}) x_0,$$
  and minimize the Conditional Flow Matching objective:
  $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{\tau, x_0, x_1}\Big[\big\|v_\theta(x_\tau, \tau \mid c) - u_\tau\big\|_2^2\Big].$$
- At inference time, integrating the ODE $\frac{dx_\tau}{d\tau} = v_\theta(x_\tau, \tau \mid c)$ from $\tau = 0 \to 1$ via **Euler** or **2nd-order Heun (predictor-corrector)** from distinct draws $x_0 \sim \mathcal{N}(0, I_D)$ transports mass directly onto the distinct modes $\{z^{(1)}, z^{(2)}\}$ without collapsing to the off-manifold mean.
- **Empirical Verification (`test_conditional_flow_matching_beats_l2_mean_collapse_on_bimodal_target`)**:
  - On a symmetric bimodal target $x_1 \in \{-\mu, +\mu\}$ ($\mu = [3, 3, 3, 3]$), the Bayes-optimal $L_2$ predictor outputs $0$, landing at distance $6.0$ from the true modes. `ConditionalFlowBridge` trained with OT-CFM and integrated with 2nd-order Heun lands within distance `0.41` of the two true modes (`>14×` closer to the manifold than $L_2$ regression), with approximately equal mode coverage across $+1$ and $-1$ branches.

---

### 3.2 Action Blindness in Dense-Video WAMs & Counterfactual Action-InfoNCE (`FLEX-WAM` / `OpenWAM`)
In high-frame-rate robotic manipulation video (e.g., $20\text{ fps}$, $\Delta t = 0.05\text{ s}$), adjacent frame latents $z_{\le t}$ are strongly auto-correlated with $z_{t+1:t+K}$. A deterministic predictor $\hat{z}_{t+1:t+K} = g_\theta(z_{\le t}, a_{t:t+K-1})$ can achieve low MSE by copying the visual context trend while ignoring the action sequence:
$$\hat{z}(z_{\le t}, a) \approx \mathbb{E}[z_{t+1:t+K} \mid z_{\le t}], \qquad \left\|\frac{\partial \hat{z}_{t+1:t+K}}{\partial a_{t:t+K-1}}\right\|_F \longrightarrow 0.$$
When the action Jacobian collapses, trajectory optimizers (`diffusion_steered_plan` and CEM) evaluate candidate action sequences as having identical rollout cost and degenerate to random walk.

#### Resolution Implemented in [`src/openchange/jepa/wam.py`](../../../src/openchange/jepa/wam.py)
1. **Counterfactual Action-InfoNCE & Action Elasticity (`action_contrastive_loss`)**:
   - Evaluates every context $b \in \{1, \dots, B\}$ under **all $B$ action sequences** $\{a^{(b')}\}_{b'=1}^{B}$ in the batch, forming the $B \times B$ pairwise rollout distance matrix:
     $$D_{b, b'} = \frac{1}{K G^2 D}\big\|\hat{z}\big(z_{\le t}^{(b)}, a^{(b')}, p_{\le t}^{(b)}\big) - z_{\text{future}}^{(b)}\big\|_2^2,$$
     and optimizes the cross-entropy InfoNCE loss over logits $-D_{b, b'} / \tau_T$ with target $b' = b$:
     $$\mathcal{L}_{\text{actNCE}} = -\frac{1}{B}\sum_{b=1}^{B} \log \frac{\exp(-D_{b,b} / \tau_T)}{\sum_{b'=1}^{B} \exp(-D_{b,b'} / \tau_T)}.$$
   - Explicitly computes and returns:
     - `action_elasticity`: $\mathcal{E}_{\text{act}} = \frac{1}{B(B-1)}\sum_{b \neq b'} \text{RMS}\big(\hat{z}(z_{\le t}^{(b)}, a^{(b')}) - \hat{z}(z_{\le t}^{(b)}, a^{(b)})\big)$, and
     - `action_identification_accuracy`: fraction of contexts where the true action sequence $b' = b$ achieves the minimum prediction distance to $z_{\text{future}}^{(b)}$.
2. **Dual-Head Conditional Flow Matching WAM (`FlowMatchingWAMHead`)**:
   - `velocity_dynamics`: $v_\theta^{\text{dyn}}(z_\tau, \tau \mid h_{\le t}, a_{t:t+K-1})$ models multimodal future latent chunks (`sample_future`).
   - `velocity_actions`: $v_\theta^{\text{act}}(a_\tau, \tau \mid h_{\le t}, g)$ models multimodal goal-conditioned action chunks (`sample_actions`) and provides `make_denoise_fn(context_summary, goal_summary)` directly consumable by `diffusion_steered_plan`.
3. **Robometer Hybrid Preference & Failure-Margin Loss (`robometer_preference_loss`, `arXiv:2603.02115`)**:
   - Replaces naive elapsed-time MSE (which assigns `progress = 0.8` to a robot that stalls for 80% of an episode) with:
     - Calibration MSE on valid trajectories,
     - **Bradley–Terry pairwise temporal ordering preference** $\text{softplus}\big((s_{t_1} - s_{t_2}) / \tau\big)$ for all step pairs $t_2 > t_1$ with $y_{t_2} > y_{t_1}$, and
     - **Failed/stalled trajectory margin penalty** $\max\big(0,\, s_{\text{fail}, K} - \bar{s}_{\text{valid}, K} + m\big)$.

---

## 4. Empirical Data Pathologies Measured on Real `SSL4EO-S12 v1.1` (`pilot_v1`) & WAM Audit Fixes

### 4.1 Quantitative Audit of the 480 Approved `SSL4EO-S12 v1.1` Pairs (`/home/ubuntu/pilot-data/pilot_v1`)
Running `audit_observation_batch_pathologies` and inspecting `/home/ubuntu/pilot-data/pilot_v1/{manifest,split,normalization}.json` across all 480 approved training pairs established four concrete data properties that directly affect training objectives:

| Pathology / Property | Measured Value on `pilot_v1` (480 Pairs) | Impact on Naive Objective | Implemented Remedy in `src/openchange/jepa/` |
|---|---|---|---|
| **1. Cross-Sensor Acquisition Time-Gap Drift ($\Delta t = |t_{S1} - t_{S2}|$)** | `min = 0.185 d` (`16,017 s`)<br>`median = 1.34 d`<br>`mean = 2.56 d`<br>`P90 = 5.84 d`<br>`P95 = 9.81 d`<br>`max = 27.28 d` (`2,356,875 s`)<br>**`8.1%` (`39/480`) $> 7\text{ d}$**<br>**`2.7%` (`13/480`) $> 14\text{ d}$** | Unweighted $\mathcal{L}_{\text{inv}}$ forces 27-day pairs to have zero latent difference, teaching the encoder to **erase real phenological, crop-harvest, and soil-moisture change**. | Added `pair_gap_seconds` to `Observation` ([`views.py`](../../../src/openchange/jepa/views.py)) and `ViewBatch`, populated from `record["pair_gap_seconds"]` in `LocalObservationDataset.__getitem__` ([`data.py`](../../../src/openchange/jepa/data.py)), and attenuated in `weighted_invariance_loss` / `split_subspace_jepa_loss` via $w_n = \exp(-\Delta t_n / \tau_{\Delta t})$. |
| **2. Heavy-Tailed SAR Speckle & Optical Reflectance Outliers** | Standardized pixels with $|z| > 3.5\sigma$:<br>- **S1 (VV/VH dB)**: **`1.74%`** (`37.4×` Gaussian expectation `0.0465%`; range `[-6.81σ, +6.68σ]`)<br>- **S2 valid bands**: **`1.95%`** (`41.9×` Gaussian expectation; range `[-2.39σ, +16.43σ]`) | Corner reflectors, water specular dark patches, and bright cloud/snow tails dominate unweighted squared errors. | `audit_observation_batch_pathologies` ([`views.py`](../../../src/openchange/jepa/views.py)) quantifies `s1_speckle_outlier_fraction` and `s2_valid_outlier_fraction` per batch before fitting. |
| **3. Cloud-Fraction Pool Partition & Cloud-Edge Halo Leakage** | Out of 480 pairs:<br>- **Clear ($\le 5\%$ cloud)**: `344` (`71.7%`, Stratum 0 pool)<br>- **Degraded ($\ge 50\%$ cloud)**: `87` (`18.1%`, Stratum 1 pool)<br>- **Grey-zone ($5\%\text{–}50\%$ cloud)**: `49` (`10.2%`, eligible only for single-sensor withheld Strata 2 & 3) | Coarse binary cloud masks leave unmasked cloud-edge halos and shadows (`valid=True` pixels adjacent to `valid=False` cloud blobs) with shifted radiometry. | `audit_observation_batch_pathologies` computes 8-connected morphological dilation around `~s2_valid` to report `cloud_halo_boundary_fraction`, and `weighted_invariance_loss` accepts per-view validity weights `view_weights`. |
| **4. Stratum Indexing Convention (`0..3` in Code vs. `1..4` in `FORMULATION.md`)** | Code (`views.py`, `data.py`, `losses.py`):<br>`0`: Joint S1+S2 clear<br>`1`: Joint S1+S2 degraded<br>`2`: S1-only (S2 withheld)<br>`3`: S2-only (S1 withheld) | Off-by-one confusion between prose tables (`m = 1..4`) and tensor indices (`0..3`). | Explicitly documented in [`FORMULATION.md`](FORMULATION.md) §1 and enforced by range checks in `Observation.__post_init__` and `split_subspace_jepa_loss`. |

---

### 4.2 Exact Resolution of All 7 `wam.py` Audit Issues (`topic_dossier.md`)
Our independent audit of [`src/openchange/jepa/wam.py`](../../../src/openchange/jepa/wam.py) identified 7 subtle implementation bugs that would have invalidated GPU runs or confounded the history-scaling dose–response study. Every issue was fixed and pinned with unit tests in [`tests/test_jepa_wam.py`](../../../tests/test_jepa_wam.py):

| # | Audit Issue in `wam.py` | Root Cause Before Fix | Exact Fix Implemented in `src/openchange/jepa/wam.py` |
|---:|---|---|---|
| **1** | **`HistoryStratifiedSIGReg.strata_index` device mismatch crash on CUDA** | Line 159 computed `(history.to(torch.float64)[:, None] - self.boundaries.cpu()[None, :]).abs()`, raising `RuntimeError` whenever `history_seconds` is on a CUDA device. | Moved `boundaries` to `history.device`: `boundaries = self.boundaries.to(device=history.device, dtype=torch.float64)` and returned indices on `history.device`. |
| **2** | **Prediction anchor shifted with requested history $H$ (`LeRobotWindowDataset`)** | `anchor = min(max_anchor, max(context_frames, span_frames))` depended on `target_hist_s`, so $H=0\text{ s}$ and $H=19.2\text{ s}$ predicted futures at **different episode timestamps**, confounding history length with episode stage. | Fixed `anchor = max_anchor = total_t - horizon_frames` (or explicit `fixed_anchor`) identically across all history strata for a given episode (`test_unconfounded_anchor_zero_history_and_action_alignment`). |
| **3** | **$H = 0.0\text{ s}$ stratum had nonzero temporal span** | `span_frames = max(context_frames, ...)` caused $H=0.0\text{ s}$ to sample `context_frames` distinct frames (`~0.15 s` at 20 fps) instead of a zero-history Markov state. | When `target_hist_s == 0.0`, `ctx_indices` repeats `anchor - 1` across `context_frames`, giving a true `0.0 s` history span (`realized_hist_s == 0.0`) with identical tensor shape `[T_ctx, C, H, W]`. |
| **4** | **False `history_seconds` stratum reporting on short episodes** | `max(target_hist_s, realized_hist_s)` labeled a `0.75 s` truncated window on a 20-frame episode as `38.4 s`, corrupting `HistoryStratifiedSIGReg` stratum assignment. | `history_seconds` now reports the true `realized_hist_s`, alongside `requested_history_seconds` and a boolean `history_truncated` flag. |
| **5** | **Action-to-observation off-by-one shift (`actions[fut_indices]`)** | Context ended at `anchor - 1`, so the action driving the transition from `anchor - 1` to `anchor` is `actions[anchor - 1]`, not `actions[anchor]`. | Changed action slice to `actions[anchor - 1 : anchor - 1 + horizon_frames]` so `a_fut[0]` is the exact action applied at the last context frame `anchor - 1`. |
| **6** | **Patch-flattened `SIGReg` satisfied by positional embeddings alone** | `wam_loss` only ran `SIGReg` on `[B*K*G*G, D]`, where spatial/temporal positional encodings can make collapsed representations appear Gaussian. | Added `sigreg_pool in ("patch_flat", "frame_mean", "context_and_pred")` plus detached SVD `effective_rank` monitoring on frame-pooled latents in `CausalPatchJEPAWorldModel.wam_loss`. |
| **7** | **Progress head regressed monotonic frame clock (`(t+1)/T`) with MSE** | A stalled or failed robot trajectory still reached `(t+1)/T -> 1.0`. | Added `robometer_preference_loss` (Bradley–Terry temporal ordering preference + failed-rollout margin penalty) and wired `EndToEndPixelWAM.pixel_wam_loss` to use it while keeping `progress_mse` in diagnostics. |

---

## 5. Summary of Modified Files & Verification Status

| File | Summary of Additions & Fixes |
|---|---|
| [`src/openchange/jepa/losses.py`](../../../src/openchange/jepa/losses.py) | Added `weighted_invariance_loss`, `split_subspace_jepa_loss`, `barlow_twins_loss`, `vicreg_loss`, `conditional_flow_matching_loss`, `sample_conditional_flow`, and `ConditionalFlowBridge`. |
| [`src/openchange/jepa/views.py`](../../../src/openchange/jepa/views.py) | Added `pair_gap_seconds` to `Observation` and `ViewBatch` (with `pin_memory` support) and `audit_observation_batch_pathologies`. |
| [`src/openchange/jepa/data.py`](../../../src/openchange/jepa/data.py) | Wired `pair_gap_seconds=float(record["pair_gap_seconds"])` into `LocalObservationDataset.__getitem__`. |
| [`src/openchange/jepa/guarantees.py`](../../../src/openchange/jepa/guarantees.py) | Added `fit_subspace_procrustes`, `apply_subspace_procrustes`, and `evaluate_matryoshka_prefixes`. |
| [`src/openchange/jepa/wam.py`](../../../src/openchange/jepa/wam.py) | Fixed all 7 `topic_dossier.md` issues; added `robometer_preference_loss`, `action_contrastive_loss`, and `FlowMatchingWAMHead`. |
| [`tests/test_jepa_losses.py`](../../../tests/test_jepa_losses.py) | Added unit & `gradcheck` tests for `weighted_invariance_loss`, `split_subspace_jepa_loss`, `barlow_twins_loss`, `vicreg_loss`, and `ConditionalFlowBridge` vs. $L_2$ mean collapse. |
| [`tests/test_jepa_views.py`](../../../tests/test_jepa_views.py) | Added tests for `pair_gap_seconds` propagation and `audit_observation_batch_pathologies`. |
| [`tests/test_jepa_guarantees.py`](../../../tests/test_jepa_guarantees.py) | Added tests for `fit_subspace_procrustes` vs. full-space Procrustes under unaligned private blocks and `evaluate_matryoshka_prefixes`. |
| [`tests/test_jepa_wam.py`](../../../tests/test_jepa_wam.py) | Added tests for `robometer_preference_loss`, `action_contrastive_loss`, `FlowMatchingWAMHead`, `sigreg_pool` modes, and unconfounded `LeRobotWindowDataset` anchors/actions. |
| [`tests/test_sources.py`](../../../tests/test_sources.py) | Updated `test_repo_metadata_approval_does_not_approve_imagery_or_training` to reflect that `ssl4eo_s12_v1_1_s1_pilot` and `ssl4eo_s12_v1_1_s2_pilot` are approved for the `pilot` split while `train`/`val`/`test` remain blocked. |

All **191 unit tests** (`make check PYTHON=.venv-jepa/bin/python`: `ruff`, `compileall`, `unittest`, `check_sbatch.py`, `validate_sources.py`, `train.py --dry-run`, `evaluate.py`, and `train_jepa.py --dry-run`) pass cleanly.
