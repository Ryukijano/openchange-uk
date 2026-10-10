# Clean-JEPA objective for S1/S2 EO: loss, gradients, stratified SIGReg, identifiability

Labels: [fact, source] / [derivation] / [measured, my code] / [estimate] / [hypothesis]. All measured numbers come from CPU PyTorch scripts (float64, seconds each) in the attached `checks.tgz`; nothing was trained. Full report also attached as `report.md` (212 lines).

Sources read: LeJEPA paper HTML (https://arxiv.org/abs/2511.08544), official code (https://github.com/rbalestr-lab/lejepa: `MINIMAL.md`, `lejepa/univariate/epps_pulley.py`, `lejepa/multivariate/slicing.py`, `scripts/*.md`), LeVJEPA paper (https://arxiv.org/abs/2608.27395; code advertised at https://levjepa.github.io/, not inspected), identifiability paper (https://arxiv.org/abs/2605.26379), one-step-prediction note (https://arxiv.org/abs/2609.36227, abstract only), `torch.distributed.nn.functional._AllReduce` source in torch 2.14.

## 0. What the official code actually does

| Item | MINIMAL.md | Paper 2511.08544 | `lejepa` package | LeVJEPA 2608.27395 |
|---|---|---|---|---|
| Knots / interval | 17 knots on $[0,3]$, symmetric doubling | code listing: `linspace(-5,5,17)` + `trapz`; text recommends 17 points, $[-5,5]$, 1024 slices; Table 1a: $[-3,3]$ vs $[-5,5]$ differ by $\le 0.5$ pt | `EppsPulley(t_max=3, n_points=17)`, same doubling | 17 knots on $[0,3]$ (App. A) |
| Quadrature weights | $\omega_j=2\Delta t$, halved at both ends, then $\tilde\omega_j=\omega_j e^{-t_j^2/2}$ (window folded in) | trapezoid on $[-5,5]$, window $e^{-t^2/2}$ | identical to MINIMAL | "trapezoidal quadrature" |
| Slices $M$ | 256, `torch.randn` **unseeded** (single GPU) | 256 in listing (seeded by `global_step`), 1024 recommended | generator seeded by `global_step`, synced with MAX all-reduce | $M=1024$ |
| Per view or concatenated | per view: `proj` is $(V,N,K)$, `.mean(-3)` over $N$ only, then `.mean()` over views and slices | eq. (9): $\frac{\lambda}{V}\sum_v \mathrm{SIGReg}(\{z_{n,v}\}_n)$ | per leading batch dim | "computed separately for the embeddings of each view" |
| $\times N$ | `* proj.size(-2)` | `* N`, $N=$ local $\times$ `world_size` | `* N * self.world_size` | $N$ not shown in eq. (3) |
| Projector | torchvision `MLP(512,[2048,2048,K], norm_layer=BatchNorm1d)`: BN+ReLU after hidden layers, none after output | ablates proj dim 64–1024 | n/a | $d\to2048$, BN, GELU, $2048\to K{=}256$ |
| Invariance | `(proj.mean(0) - proj).square().mean()`: mean over all $V$ views (no locals in MINIMAL), over $N,V,K$; **no detach** | eq. (8): $\frac1V\sum_{v'}\|\mu_n - z_{n,v'}\|_2^2$, $\mu_n$ = mean of globals; listing `(centers - a_emb).square().mean()`, no detach | n/a | $\frac{1}{V+1}\sum_v\|z_0-z_v\|^2$, "gradients propagate through both variables" |
| Total | $\lambda\,\mathrm{SIGReg} + (1-\lambda)\,\mathrm{inv}$; run: $\lambda=0.02$, $V=4$, $K=16$, bs 256, bf16 autocast | $(1-\lambda)\mathrm{inv}+\lambda\mathrm{SIGReg}$, $\lambda=0.05$, $V_g=2$, $V_l=8$, bs $\ge128$ | n/a | $\mathrm{inv}+\lambda\,\mathrm{SIGReg}$, $\lambda=0.02$, token drop $\rho=0.95$ |
| DDP | none | all-reduce of ECF, `op="AVG"` | `torch.distributed.nn.all_reduce` (autograd-aware) | one all-reduce of a $2\times M\times17$ tensor |

[fact: MINIMAL.md; paper §4.2.3, §5.1, Table 1a; epps_pulley.py; slicing.py; LeVJEPA §3.1, App. A, B]

Two things easy to miss:

- **$K$-convention of the invariance term.** Paper eq. (8) and LeVJEPA eq. (2) write $\|\cdot\|_2^2$ (sum over $K$); both code listings (MINIMAL, paper §5.1) use `.square().mean()` (mean over $K$). Effective SIGReg:inv ratio is $\lambda K/(1-\lambda)$ under "mean" vs $\lambda/(1-\lambda)$ under "sum". MINIMAL ($\lambda=0.02$, $K=16$): $0.33$; paper ($\lambda=0.05$): $0.053$ if sum, $0.05K$ if mean; LeVJEPA ($\lambda=0.02$, $K=256$): $0.02$ if sum, $5.1$ if mean. "$\lambda=0.05$ from the paper" is not transferable to a per-coordinate MSE unless the paper's runs used the mean (the listing suggests so; Table 1 was produced by `scripts/je.py` from `stable_pretraining`, not in the repo). [fact for equations/listings; hypothesis for which convention produced the published numbers]
- **The repo's Table-1 launch files** (`scripts/launch_*_ablation.md`) pass `++teacher_student=true ++patch_mask_ratio=0.3` (also `++bstat_lambda=0.05 ++bstat_num_slices=1000`) to a script not in the repo. I cannot tell what `teacher_student=true` did in those runs; MINIMAL.md has no teacher/student. Do not cite Table 1 as evidence about the *clean* variant without checking `stable_pretraining`. [fact, code]

## 1. $L_{inv}$: definition, gradient, fixed point

Sample $n$, views $v\in\{1..V\}$, globals $G$ ($|G|=V_g$), locals $L$ ($|L|=V_l$), $z_{n,v}=g(f(x_{n,v})_{cls})\in\mathbb R^K$, $\mu_n=\frac{1}{V_g}\sum_{v\in G}z_{n,v}$:

$$L_{inv}=\frac{1}{NVK}\sum_{n}\sum_{v=1}^{V}\|\mu_n-z_{n,v}\|_2^2 .$$

No stop-gradient anywhere: MINIMAL (`proj.mean(0)` is live), paper listing (`centers` live), LeVJEPA text. [fact]

Gradient [derivation; checked to $7\times10^{-18}$ against autograd, `check_a.py`], with $\bar z_{L,n}=\frac{1}{V_l}\sum_{v\in L}z_{n,v}$:

$$\frac{\partial L_{inv}}{\partial z_{n,u}}=\frac{2}{NVK}\Big[(z_{n,u}-\mu_n)+\mathbb 1[u\in G]\,\frac{V_l}{V_g}\,(\mu_n-\bar z_{L,n})\Big].$$

Each view is pulled toward the centre with weight $2/(NVK)$; each global is additionally pulled toward the *mean of the locals* with $V_l/V_g$ times that weight ($3\times$ for 2/6). The gradient summed over the views of one sample is exactly zero (joint translation is flat), so $L_{inv}$ cannot move $\mu_n$; only SIGReg places it. Per-sample magnitude $\approx 2\|z-\mu\|/(NVK)\sim 2/(NV\sqrt K)$, i.e. $\propto 1/N$.

Fixed points of $L_{inv}$ alone: $z_{n,v}=\mu_n$ for all $v$, any $\mu_n$ (including collapse). With SIGReg: $z_{n,v}=\mu_n$ and $\{\mu_n\}\sim\mathcal N(0,I_K)$. $L_{inv}=0$ is unreachable when a view carries less information than the centre (S1-only local vs S1+S2 centre), §5.

## 2. SIGReg as coded

**Statistic.** One view, one unit direction $a_m$, $s_n=a_m^\top z_n$, ECF $\hat\varphi(t)=\frac1N\sum_n e^{its_n}=C(t)+iS(t)$, target $\varphi(t)=e^{-t^2/2}$, window $w(t)=e^{-t^2/2}$:

$$EP_m=N\!\int_{-\infty}^{\infty}\!|\hat\varphi-\varphi|^2w\,dt\;\approx\;N\sum_{j=0}^{16}\tilde\omega_j\Big[(C(t_j)-\varphi(t_j))^2+S(t_j)^2\Big],\quad t_j=\tfrac{3j}{16},\ \tilde\omega_j=\omega_j e^{-t_j^2/2},$$

$\omega_j=2\Delta t$ except $\Delta t$ at the ends (integrand even, $\int_{-3}^{3}=2\int_0^3$). $\mathrm{SIGReg}=\frac1M\sum_m EP_m$, then averaged over views. [fact, MINIMAL/epps_pulley.py; derivation for the symmetry]

**Why $\times N$.** Under $H_0$, $\mathbb E|\hat\varphi(t)-\varphi(t)|^2=(1-|\varphi(t)|^2)/N$ (paper Thm 6 / App. B.13: $|\varphi_P-\varphi|^2+(1-|\varphi_P|^2)/N$ in general). Without the factor the statistic vanishes as $1/N$; with it, $EP$ has a non-degenerate null law (weighted $\chi^2_1$ sum). So **the null expectation is $N$-free**:

$$\mathbb E_{H_0}[EP]=\int w(1-|\varphi|^2)=\sqrt{2\pi}-\sqrt{2\pi/3}=1.0594\ \text{(exact)},\qquad \sum_j\tilde\omega_j(1-\varphi_j^2)=1.0525\ \text{(17 knots on }[-3,3]).$$

[derivation + measured: $z\sim\mathcal N(0,I_{128})$ gives $1.02\pm0.11$ ($N{=}256$), $1.08\pm0.08$ ($N{=}1024$), $1.06\pm0.07$ ($N{=}4096$); `check_a.py`]. A healthy run plateaus near $1.05$, not $0$. Away from the null, $\mathrm{SIGReg}\approx 1.05 + N\cdot D(P,\mathcal N)$, $D=\int w|\varphi_P-\varphi|^2$: **the informative part of the value scales linearly with $N$**.

**Gradient for one slice** [derivation; autograd error $4\times10^{-17}$, central finite differences $5.5\times10^{-10}$ on entries $\sim3\times10^{-2}$; `check_a.py`]:

$$\frac{\partial EP_m}{\partial z_n}=2\,a_m\sum_j\tilde\omega_j\,t_j\Big[S(t_j)\cos(t_js_n)-\big(C(t_j)-\varphi(t_j)\big)\sin(t_js_n)\Big].$$

The leading $N$ cancels the $1/N$ in $\partial\hat\varphi/\partial s_n$: **the per-sample gradient of the coded SIGReg is $O(1)$ in $N$**. The paper's Theorem 4 bound $4\sigma^2/N$ is proved (B.12) for $\hat D_V$ *without* the $N$ factor; for the coded statistic the bound is $4\sum_j\tilde\omega_jt_j\approx 7.9$, $N$-independent. [fact for B.12; derivation]

**Scaling with $N$, $M$, $K$** [measured, `check_c.py`, $K=128$, $M=256$, per-sample $\|\partial\mathrm{SIGReg}/\partial z_n\|$, 5 draws]:

| $N$ | null $\mathcal N(0,I)$ | all variances $=0.5$ | one coordinate dead | $L_{inv}$ per sample ($V{=}8$, $\|z-\mu\|\sim\sqrt K$) |
|---|---|---|---|---|
| 256 | $8.7\times10^{-3}$ | $4.04\times10^{-2}$ | $8.8\times10^{-3}$ | $8.6\times10^{-5}$ |
| 512 | $6.1\times10^{-3}$ | $4.02\times10^{-2}$ | $6.5\times10^{-3}$ | $4.3\times10^{-5}$ |
| 1024 | $4.8\times10^{-3}$ | $4.02\times10^{-2}$ | $4.6\times10^{-3}$ | $2.2\times10^{-5}$ |
| 2048 | $3.1\times10^{-3}$ | $4.01\times10^{-2}$ | $3.1\times10^{-3}$ | $1.1\times10^{-5}$ |
| 4096 | $2.2\times10^{-3}$ | $4.01\times10^{-2}$ | $2.3\times10^{-3}$ | $5.4\times10^{-6}$ |

Loss values for the same draws: null $1.00$–$1.15$; variance-0.5: $13.1, 25.7, 51.0, 101.4, 201.5$ (exactly $\propto N$); one dead coordinate of 128: $1.01$–$1.15$, **indistinguishable from the null**.

- Systematic deviation: per-sample gradient constant in $N$ (summed gradient $\propto N$); loss $\propto N$.
- At the null: per-sample gradient $\propto N^{-1/2}$ ($8.7/2.2=3.9\approx\sqrt{16}$), ECF sampling noise.
- $L_{inv}$ per-sample gradient $\propto 1/N$; at $N=1024$ the SIGReg/inv per-sample ratio for variance-0.5 is $\approx1.9\times10^3$ before $\lambda$, $\approx100$ after $\lambda=0.05$.
- $M$: mean gradient unchanged; slice-resampling noise falls as $M^{-1/2}$ ($4.4, 2.2, 1.1\times10^{-2}$ for $M=64,256,1024$, $N=1024$). At $M=256$ noise $\approx$ signal ($2.2$ vs $3.3\times10^{-2}$); 1024 slices is the better default.
- $K$: isotropic deviations give per-sample gradient $\propto K^{-1/2}$ ($4.6, 3.3, 2.5\times10^{-2}$ for $K=64,128,256$). One-coordinate deviations: loss excess $\propto1/K$ for a mean shift, $\propto1/K^2$ for a variance deficit ($a_1^2\sim1/K$; EP excess is quadratic in projected shift, quartic in projected variance change; §4 confirms both exponents).

**DDP correctness** [fact for code; derivation for argument]. `EppsPulley.forward` computes local $\bar C_r,\bar S_r$, calls `torch.distributed.nn.all_reduce(·, AVG)` (differentiable: `_AllReduce.backward` applies the *same* reduce op to the incoming gradient), forms the error from the global ECF, multiplies by $N_{loc}\cdot W$. Backward on rank $r$: the true $\partial L/\partial\bar C_r$ is $g/W$, but AVG-backward delivers $g$ — $W\times$ too large; DDP then averages parameter gradients over ranks ($1/W$), so the result equals the single-process gradient on the concatenated global batch. This only holds with standard DDP mean-reduction; with sum-reduction or micro-batch accumulation it is off by $W$. $A$ is identical on all ranks via the `global_step`-seeded generator (synced by MAX all-reduce). MINIMAL.md has neither (unseeded `randn`, no all-reduce): single-GPU only. Also: MINIMAL computes the ECF inside bf16 autocast (`proj @ A` in bf16, cos/sin inherit); for a statistic whose informative part is $O(N\cdot10^{-3})$ this is a precision risk — cast `proj.float()` before SIGReg.

## 3. $\lambda$–batch coupling and Q1

With $L=(1-\lambda)L_{inv}+\lambda\,\mathrm{SIGReg}$ and $\mathrm{SIGReg}\approx 1.05+N\bar D$: [derivation from §2]

- Gradient balance: $\|\nabla_\theta(\lambda\mathrm{SIGReg})\|$ grows like $\lambda N$ (per-sample $O(1)$ times $N$ samples), $\|\nabla_\theta((1-\lambda)L_{inv})\|\propto(1-\lambda)$. **$\lambda_{eff}\propto\lambda N$.** Doubling global batch at fixed $\lambda$ doubles the Gaussianity pull relative to invariance. Paper Table 1c (bs 128–1024, $\pm1.3$ pt) shows this is tolerated on ImageNet, not absent.
- Loss comparability: SIGReg values comparable only at equal $N_{glob}$; $L_{inv}$ only at equal $V_g,V_l$, crop recipe and S1/S2 view mixture (an S1-only local has an irreducible floor, §5); the mixture only at equal $\lambda$. $K$ rescales neither term but changes sensitivity. The paper's Spearman (85%, 99% after dividing the loss by $\lambda^{0.4}$, Fig. 11/eq. 10) was over lr/wd/epochs/$\lambda$ [fact], with no batch variation reported, so for Q1 fix $N$ or pre-register a correction; log $L_{inv}$, per-view unscaled $EP$, and $\bar D=(\mathrm{SIGReg}-1.05)/N$.
- Stratified: each $EP_m$ carries its own $N_m$, so the loss value depends on per-step stratum counts; freeze the sampler's $N_m$ across the Q1 grid. [derivation]

Pre-register: per-coordinate mean $L_{inv}$, $(1-\lambda)/\lambda$ mixing, $N_{glob}$ fixed per model size, fp32 SIGReg, $M=1024$, 17 knots on $[0,3]$ doubled; state the $K$-convention caveat of §0.

## 4. Stratified SIGReg, weights, optimum, mean term

$$\mathrm{SIGReg}_{strat}=\sum_m w_m\,\mathrm{SIGReg}(\{z_n:m_n=m\}),\qquad \frac{\partial\mathrm{SIGReg}_{strat}}{\partial z_n}=w_{m_n}\frac{\partial\mathrm{SIGReg}_{m_n}}{\partial z_n},$$

with the §2 per-slice gradient on stratum statistics $C_m,S_m$ ($N_m$ cancels as before). [derivation]

**Optimum.** Floor reached iff $a^\top z|m\sim\mathcal N(0,1)$ for all $a$, i.e. $z|m\sim\mathcal N(0,I_K)$ for every $m$ (Cramér–Wold); then $z\perp m$ and any $m$/$c$ probe is at chance. [derivation] Not fixed: any per-stratum orthogonal $Q_m$ leaves $Q_mz|m\sim\mathcal N(0,I)$, so strata can be mutually rotated/reflected at zero cost; the stratum-mean term is zero there too. Only $L_{inv}$ on paired views across strata penalises $Q_{m_3}\neq Q_{m_1}$: under $z^{(m)}=Q_m\zeta$ the alignment cost is $2K-2\,\mathrm{tr}(Q_m^\top Q_{m'})$, minimised at $Q_m=Q_{m'}$ [derivation]. The cross-stratum kNN diagnostic is the right test.

**Weights** [measured, `check_w.py`: $K=128$, $N=2048$, rare stratum $N_r=256$ (12.5%), gap $\delta=2$ along $e_0$, pooled mean zero]:

| weighting | per-sample grad, rare | per-sample grad, majority | loss |
|---|---|---|---|
| $w_m=N_m/N$ | $2.4\times10^{-3}$ | $3.6\times10^{-3}$ | 1.74 |
| $w_m=1/S$ | $9.8\times10^{-3}$ | $2.2\times10^{-3}$ | 2.75 |
| pooled | $3.2\times10^{-3}$ | $3.2\times10^{-3}$ | 1.07 (null 1.05) |

Since the per-sample gradient of $EP_m$ is $O(1)$ in $N_m$, $w_m=N_m/N$ multiplies the push on each rare-stratum sample by its frequency: the rare stratum (mean $7\times$ further off) gets *less* correction than the majority. **Default $w_m=1/S$**: frequency-independent per-sample pull, floor stays $\approx1.05$ for any $S$. Per-sample pull under $1/S$ is $1/S$ of pooled at equal deviation, so include $S\lambda$ in the sweep. [derivation + measured]

**Modality-gap blindness** [measured, `check_b.py`: $K=128$, $N=2048$ (two strata of 1024), $M=256$, 10 draws; mean term $\sum_m\|\bar z_m\|^2$ unnormalised]:

| $\delta$ | pooled | stratified | mean term |
|---|---|---|---|
| 0 | $1.047\pm0.109$ | $1.033\pm0.087$ | $0.249\pm0.026$ (floor $2K/N_m=0.25$) |
| 1 | $1.066\pm0.107$ | $2.030\pm0.183$ | $0.742\pm0.043$ |
| 2 | $1.131\pm0.124$ | $4.870\pm0.404$ | $2.290\pm0.105$ |
| 3 | $1.290\pm0.079$ | $9.488\pm0.953$ | $4.734\pm0.090$ |

Pooled SIGReg does not see a two-sigma gap ($+0.08$, below one null SD). Excess over $\delta=0$ vs $K$ ($\delta=2$):

| $K$ | pooled | stratified | mean term | $K\cdot$strat | $K^2\cdot$pooled |
|---|---|---|---|---|---|
| 16 | 2.433 | 31.73 | 2.08 | 508 | 623 |
| 32 | 0.645 | 15.85 | 2.05 | 507 | 661 |
| 64 | 0.201 | 7.93 | 2.08 | 507 | 824 |
| 128 | 0.049 | 3.89 | 1.99 | 498 | 808 |
| 256 | 0.013 | 1.90 | 2.02 | 486 | 852 |

Stratified excess $\approx N\delta^2c/K$ ($c\approx0.24$): the $1/K$ decay; pooled $\propto1/K^2$ (a between-stratum mean gap only changes pooled *variance* along $e_0$); mean term $K$-free ($=\delta^2/2$ exactly plus the $2K/N_m$ floor). [measured + derivation]

Gradient geometry at $\delta=2$, $K=128$ (mean per-sample gradient on the "+" stratum): pooled: along $e_0$ $-1.8\times10^{-4}$, orthogonal noise $2.1\times10^{-3}$ (no usable signal); stratified: $+3.8\times10^{-3}$ vs noise $3.2\times10^{-3}$ (SNR $\approx1.2$ at $M=256$, grows as $\sqrt M$); mean term: $+1.9\times10^{-3}$ ($=2(\delta/2)/N_m$) vs noise $6.5\times10^{-4}$ (SNR $\approx3$; noise is the $\sqrt{K/N_m}$ null mean). [measured]

**Is the mean term redundant?** At the optimum yes ($z|m\sim\mathcal N(0,I)\Rightarrow\bar z_m\to0$). During training no: stratified SIGReg sees a one-coordinate gap at $1/K$ strength with a random-slice gradient direction (relative noise $\sim\sqrt{K/M}$); the mean term is $K$-free, exact in direction, and costs one $K$-vector per stratum. It is a first-moment penalty added to the CF test, not a replacement, so the paper's moment-shortcut warning (§5.2, B.14) does not apply in the same way. [derivation]

**Proposed form and defaults** [estimate]. Put it in $EP$ units ($N$-free floor, $O(1)$ per-sample gradient):

$$T_\mu=\sum_m w_m\,\frac{N_m\|\bar z_m\|^2}{K},\qquad \mathbb E_{H_0}T_\mu=\sum_m w_m=1,\quad \frac{\partial T_\mu}{\partial z_n}=\frac{2w_{m_n}}{K}\bar z_{m_n}.$$

Excess under gap $\delta_m$ is $w_mN_m\delta_m^2/K$ — the same $1/K$ as stratified SIGReg in loss units; what it buys is a deterministic, slice-free gradient (the SNR gain above), not a $K$ gain. To get the $K$ back, drop the $1/K$ and subtract the known floor $K$ (unbiased for $N_m\|\mu_m\|^2$), at the cost of a term $K\times$ larger than SIGReg at equal mis-specification. Defaults: $w_m=1/S$; $\lambda_\mu=\lambda$, ablate $\{0,\lambda,10\lambda\}$; $K\in\{64,128\}$ (single-coordinate effects twice as visible at 64; paper Table 1d shows $K=64$ not worse on ImageNet); minimum $N_m=256$ — null SD of $EP$ is $0.106$ at 256 vs $0.072$ at 4096 (mild), ECF bias $O(1/N)$ (paper: fine to 16), mean-term floor $K/N_m=0.5$ at $K=128$ sets the smallest per-step detectable gap at $\delta\approx1.4$ (step averaging lowers it). [measured SDs; estimate for defaults]

## 5. Identifiability: assumptions, what S1/S2 breaks, Procrustes floor

Theorem 1 of 2605.26379 [fact]: (i) independent latents and per-coordinate transitions, (ii) stationarity $p(z)=p(z')$, (iii) additive noise $z_i'=m_i(z_i)+\eta_i$; Gaussian latents force OU $z'=\rho z+\sqrt{1-\rho^2}\eta$. Learner $h=f\circ g:\mathbb R^n\to\mathbb R^n$ with the **same** $g$ and $h$ on both views, minimising $\mathbb E\|h(z')-h(z)\|^2$ s.t. $h(z)\sim\mathcal N(0,I_n)$. Then $L(h)\ge2(1-\rho)n$, equality iff $h(z)=Qz$, $Q\in O(n)$. Theorem 3: alignment gap $\delta$, whitening error $\varepsilon$ $\Rightarrow\exists Q$: $\mathbb E\|h(z)-Qz\|^2\le D+(\varepsilon+D)^2$, $D=\delta/(2\rho(1-\rho))$. Limitations: output dim must equal latent dim; extra dims "must collapse or encode redundancy".

What the pairs break [derivation / hypothesis]:
- *Two observation functions.* S1-only vs S1+S2 are $g_1(z)$, $g_{12}(z)$ of one latent: two maps $h_1,h_{12}$, not one $h$. With both $g$ invertible the Hermite argument extends: $\mathbb E[h_1(z)h_{12}(z')]=\sum_k\rho^k\langle c^{(1)}_k,c^{(12)}_k\rangle\le\rho$ by Cauchy–Schwarz, equality iff both are the same linear map, so alignment pins $Q_1=Q_{12}$ [derivation, not in the paper]. The real break is informational: $g_1$ is not injective (cloud/phenology/spectral state invisible to SAR), so coordinates of $z$ unobservable from S1 are not identifiable from S1 views by any loss. $Q$-identifiability holds at best on the subspace observable from both sensors.
- *Stationarity is fine* for the two-date ablation (unordered pair, random roles). The within-date S1/S2 pair has $\rho=1$ between latents; the whole gap is in the observation functions, which the theorem does not model.
- *Conflict between stratified SIGReg and $L_{inv}$ in information-poor strata* [derivation; hypothesis for its practical weight]. Split $\mu_n=(u_n,w_n)$ into S1-predictable ($u$) and not ($w$). The $L_{inv}$-optimal S1-only embedding is $\mathbb E[\mu|S1]=(u,0)$, which is **not** $\mathcal N(0,I_K)$ (variance 0 along $w$). An $\mathcal N(0,I_K)$ target for $m_3$ forces unit variance along $w$, which S1 cannot supply; the compromise is S1-driven "hallucinated" coordinates along $w$ (free for SIGReg, cost $\ge1$ per coordinate for $L_{inv}$). "$m$-probe at chance" and "smaller drop when S2 is removed" can both be met by filling S2-specific directions with S1-correlated noise. Add a pass criterion: S1-only embeddings must not predict S2-only quantities (NDVI, cloud-free reflectance) beyond what S1 physically supports; report per-stratum $L_{inv}$, whose $m_3$ floor measures S2-only information. A target $z|m_3\sim\mathcal N(0,\Sigma_3)$, $\Sigma_3\preceq I$, or SIGReg on the shared subspace, removes the conflict but is a different method; decision needed.

**Procrustes test on $h$** [derivation]. Centred $H_A,H_B\in\mathbb R^{n_{cal}\times K}$. Orthogonal: $U\Sigma V^\top=\mathrm{SVD}(H_A^\top H_B)$, $Q=UV^\top$, $r_{orth}=\|H_B-H_AQ\|_F^2/\|H_B\|_F^2=(\|H_A\|^2+\|H_B\|^2-2\|H_A^\top H_B\|_*)/\|H_B\|^2$. Linear: $W=H_A^+H_B$. CCA: same SVD on whitened $H$'s.

**Residual floor when $K>n_{true}$** [measured, `check_proc.py`; $n$ shared coordinates rotated by random $Q_n$ plus $K-n$ independent unit-variance junk coordinates per model, random global rotations, $K=128$]:

| $n$ | $n_{cal}$ | orthogonal | linear | $2(K-n)/K$ | $(K-n)/K$ |
|---|---|---|---|---|---|
| 128 | all | 0.000 | 0.000 | 0 | 0 |
| 96 | 500 / 1k / 5k / 20k | 0.326 / 0.388 / 0.459 / 0.481 | 0.186 / 0.221 / 0.245 / 0.249 | 0.500 | 0.250 |
| 64 | 500 / 1k / 5k / 20k | 0.623 / 0.748 / 0.899 / 0.949 | 0.374 / 0.437 / 0.488 / 0.497 | 1.000 | 0.500 |
| 32 | 500 / 1k / 5k / 20k | 0.900 / 1.074 / 1.314 / 1.410 | 0.560 / 0.656 / 0.729 / 0.745 | 1.500 | 0.750 |

"$(K-n)/K$" is the **linear** floor; the **orthogonal** floor is $2(K-n)/K$ (a rotation must map unit-variance junk onto unit-variance junk) and exceeds 1 when $n<K/2$ — worse than predicting zero. Finite $n_{cal}$ biases both downward (nuclear-norm overfitting, $\sim\sqrt{K/n_{cal}}$); use $n_{cal}\ge5\mathrm k$ at $K=128$ and report orthogonal minus linear, which under this model estimates $(K-n)/K$. [measured + derivation] Prediction to add: if SIGReg succeeds, junk coordinates cannot collapse, so a LeJEPA pair with $K>n_{true}$ sits *at* the $2(K-n)/K$ orthogonal / $(K-n)/K$ linear floors; a BT/MAE pair has no such clean floor. [hypothesis]

## 6. Complete loss and one training step

$z_{n,v}=g(f(x_{n,v}))$, $\mu_n=\frac1{V_g}\sum_{v\in G}z_{n,v}$, $N_m=|\{n:m_n=m\}|$ (global counts), slices $a_1..a_M$ from a `global_step`-seeded generator, $t_j=3j/16$, $\tilde\omega_j=\omega_je^{-t_j^2/2}$, $\bar z_{m,v}=\frac1{N_m}\sum_{n:m_n=m}z_{n,v}$:

$$L=(1-\lambda)\underbrace{\frac{1}{NVK}\sum_{n,v}\|\mu_n-z_{n,v}\|^2}_{L_{inv}}+\lambda\underbrace{\frac1V\sum_{v}\sum_m\frac1S\,\frac1M\sum_{a}\,N_m\sum_j\tilde\omega_j\Big|\tfrac{1}{N_m}\!\!\sum_{n:m_n=m}\!\!e^{it_ja^\top z_{n,v}}-e^{-t_j^2/2}\Big|^2}_{\mathrm{SIGReg}_{strat}}+\lambda_\mu\underbrace{\frac1V\sum_v\sum_m\frac1S\,\frac{N_m\|\bar z_{m,v}\|^2}{K}}_{T_\mu}$$

Defaults: $\lambda=0.05$ (sweep $\{0.02,0.05,0.1\}\times\{1,S\}$), $\lambda_\mu\in\{0,\lambda,10\lambda\}$, $M=1024$, $K\in\{64,128\}$, $N_m\ge256$.

```python
def train_step(batch, net, proj, sigreg_t, sigreg_phi, sigreg_w, lam, lam_mu, step, S, M, K):
    views, m = batch                      # V entries of (tokens, modality_ids); first Vg are full S1+S2 globals; m: (N,) strata
    N, V, Vg = m.numel(), len(views), 2
    with autocast(dtype=bfloat16):
        emb = [net(tok, mod) for tok, mod in views]          # (N, d) cls features; token-drop inside net
        z = torch.stack([proj(e) for e in emb]).float()      # (V, N, K); projector has BN; fp32 from here
    mu = z[:Vg].mean(0)                                      # (N, K) live tensor: no detach (MINIMAL/paper/LeVJEPA)
    L_inv = (mu.unsqueeze(0) - z).square().mean()            # per-coordinate MSE over N, V, K
    g = torch.Generator(device=z.device); g.manual_seed(int(all_reduce(step, MAX)))   # same slices on all ranks
    A = torch.randn(K, M, generator=g, device=z.device); A /= A.norm(dim=0)
    sig, tmu = 0.0, 0.0
    for s in range(S):                                       # loader guarantees local N_s >= N_min per stratum
        zs = z[:, m == s]                                    # (V, N_s, K)
        N_s = all_reduce(torch.tensor(zs.shape[1]), SUM)     # global stratum count
        xt = (zs @ A).unsqueeze(-1) * sigreg_t               # (V, N_s, M, 17)
        C = all_reduce_sum(xt.cos().sum(1)) / N_s            # global ECF: SUM of local sums over global N_s ...
        Sn = all_reduce_sum(xt.sin().sum(1)) / N_s           # ... via torch.distributed.nn (autograd-aware)
        err = (C - sigreg_phi).square() + Sn.square()        # (V, M, 17)
        sig = sig + ((err @ sigreg_w) * N_s).mean() / S      # N_s * EP, mean over views and slices, w_m = 1/S
        zbar = all_reduce_sum(zs.sum(1)) / N_s               # (V, K) global stratum mean per view
        tmu = tmu + (N_s * zbar.square().sum(-1) / K).mean() / S
    loss = (1 - lam) * L_inv + lam * sig + lam_mu * tmu
    loss.backward(); opt.step(); opt.zero_grad(); sched.step()
    log(L_inv=L_inv, sigreg=sig, sigreg_excess=sig - 1.05, tmu=tmu, per_stratum_EP=..., per_stratum_Linv=...)
```

SUM of local sums over the global $N_s$ gives the exact global ECF and correct gradient without relying on the AVG-backward/DDP-mean cancellation of the official code (valid only under rank-averaged parameter gradients). Keep SIGReg in fp32. The sampler must deliver fixed $N_s$ per step for the Q1 grid.

## What this changes in the formulation

1. §2 Loss: pre-register the per-coordinate-mean convention and state that paper/LeVJEPA $\lambda$ values do not transfer across the sum/mean convention (factor $K$). Use $M=1024$, fp32 SIGReg.
2. §2/§3: SIGReg $\approx1.05+N\bar D$; floor $N$-free, signal $\propto N$; per-sample gradient $O(1)$ in $N$ vs $O(1/N)$ for $L_{inv}$, so $\lambda_{eff}\propto\lambda N$. For Q1 fix $N_{glob}$ and per-step $N_m$, log $(\mathrm{SIGReg}-1.05)/N$. Do not quote Theorem 4's $4\sigma^2/N$ for the coded loss.
3. §3: change $w_m=N_m/N$ to $1/S$ ($N_m/N$ down-weights exactly the rare strata the method targets). Define the mean term as $T_\mu=\sum_m w_mN_m\|\bar z_m\|^2/K$; redundant at the optimum, useful for its slice-free, $K$-independent gradient direction, not for loss scale. Blindness: pooled $1/K^2$, stratified $1/K$, mean term $K^0$ (measured).
4. §3 status / §5: add the information conflict — an $\mathcal N(0,I_K)$ target for $m_3$ contradicts $L_{inv}$ along S2-only directions and invites hallucinated coordinates; add the "no better than physics" pass criterion and report per-stratum $L_{inv}$ floors. Decide between a shrunken-covariance target for $m_3$ or SIGReg on the shared subspace.
5. §4: $(K-n)/K$ is the linear floor; the orthogonal floor is $2(K-n)/K$ and can exceed 1; use $n_{cal}\ge5\mathrm k$ and report orthogonal minus linear. The theorem's unhandled ingredient is non-injective $g_1$, not stationarity; the two-map extension is a short Cauchy–Schwarz lemma, not a citation.
6. §0 caveat: the repo's Table-1 configs pass `teacher_student=true`, `patch_mask_ratio=0.3`; cite MINIMAL.md, not Table 1, for "no EMA/predictor/stop-gradient".

ATTACHMENT:{"url":"https://app.devin.ai/attachments/055393f1-0200-4a95-835f-2f1d621e093b/report.md","fileSize":32994}
ATTACHMENT:{"url":"https://app.devin.ai/attachments/34a318a4-57ea-4a3d-a742-adb87ab8df6a/checks.tgz","fileSize":4238}