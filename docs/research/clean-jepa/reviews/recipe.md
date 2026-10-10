# Training recipe for clean JEPA on Sentinel-1/2 — verified against the official code

Sources read on this VM: LeJEPA paper (arXiv HTML) https://arxiv.org/abs/2511.08544; LeJEPA code https://github.com/rbalestr-lab/lejepa (`MINIMAL.md`, `lejepa/univariate/epps_pulley.py`, `lejepa/multivariate/slicing.py`, `scripts/launch_*`); `stable-pretraining` https://github.com/rbalestr-lab/stable-pretraining; LeVJEPA paper https://arxiv.org/abs/2608.27395 and code https://github.com/MLO-lab/LeVJEPA (`conf/config.yaml`, `main.py`, `module.py`, `paper.md`); identifiability paper https://arxiv.org/abs/2605.26379 ("When Does LeJEPA Learn a World Model?", abstract only, 2026-05-25); https://arxiv.org/abs/2609.36227 ("One-Step Next-Latent Prediction Is Not a World Model", abstract only). CPU checks: `checks.py` (NumPy, seconds; attached). Labels: [fact, src] / [derivation] / [measured] = `checks.py` / [estimate] / [hypothesis].

## 1. Paper says / code does / we adopt

| Item | Paper says | Code does | We adopt (Sentinel study) |
|---|---|---|---|
| Optimizer | AdamW, no WD schedule [fact, LeJEPA §Exp. details] | MINIMAL: `AdamW`, two param groups (net wd 5e-2, projector group) [fact, MINIMAL.md]. LeVJEPA: AdamW, betas (0.9, 0.95), wd 0.04 [fact, config.yaml] | AdamW, betas (0.9, 0.95), constant wd [derivation: LeVJEPA defaults for ViT at scale] |
| LR | grid $\{5\cdot10^{-3},5\cdot10^{-4}\}$ [fact, LeJEPA]; README start lr $5\cdot10^{-4}$ [fact, README] | LeVJEPA peak `lr: 4e-4` [fact, config.yaml] | $5\cdot10^{-4}$ at batch 1024 default; swept in §5 [derivation] |
| Schedule | linear warmup + cosine; final lr = lr/1000 [fact, README] | MINIMAL: `LinearLR(start_factor=0.01, total_iters=1 epoch)` then `CosineAnnealingLR(eta_min=1e-3)` — `eta_min` is absolute, not lr/1000 [fact, MINIMAL.md; derivation]. LeVJEPA: `LinearWarmupCosineAnnealing`, `start_lr 1e-4`, `peak_step 1200`, `end_lr 4e-4` = peak, i.e. warmup then flat [fact, config.yaml; derivation] | WSD (§4): warmup → constant → linear cooldown to lr/1000. Neither official recipe uses WSD; this is a study-design choice [derivation] |
| Warmup | "standard linear warm-up" [fact, LeJEPA] | MINIMAL: 1 epoch; LeVJEPA: 1200 optimizer steps at effective batch 3072 [fact] | 1500 steps at batch 1024 (≈1.5M samples ≈ 160M tokens) [estimate]; swept in §5 |
| Weight decay | grid $\{10^{-1},10^{-2},10^{-5}\}$ [fact, LeJEPA] | MINIMAL 5e-2; `launch_inet10.py` $\{3\cdot10^{-2},10^{-5}\}$; LeVJEPA 0.04 [fact] | 0.05 default, swept log-uniform $[10^{-5},10^{-1}]$ [derivation] |
| Batch size | not fixed in paper text I read | `launch_inet10.py` 512; LeVJEPA $96\times8\times2\times2=3072$ clips [fact, config.yaml; derivation] | 1024 global, fixed across all sizes and the Q1 sweep [derivation, needed so SIGReg's $N$-scaling is comparable] |
| Epochs | IN-100: 400 epochs (Table 4) [fact, LeJEPA] | LeVJEPA matched retraining: 240 epochs [fact, paper.md] | token budgets, not epochs (§4) [derivation] |
| Drop-path | not stated | `launch_inet10.py` `drop_path_rate=0` [fact]; LeVJEPA config: no drop-path key found [fact, as read] | 0 default (the formulation's 0.1 has no official backing); swept $\{0,0.05,0.1,0.2\}$ [derivation] |
| Precision | not stated | MINIMAL `autocast(bfloat16)` plus a `GradScaler` (redundant for bf16) [fact; derivation]; LeVJEPA `bf16-mixed` [fact] | bf16 autocast, fp32 master weights, no GradScaler; SIGReg ECF in fp32 [derivation] |
| Projector | 1/2/3-layer ablation, 3-layer best for ViTs (Table 4) [fact, LeJEPA] | MINIMAL `MLP(d,[2048,2048,K], BatchNorm1d)` [fact]; LeVJEPA $d\to2048\to$BN$\to$GELU$\to256$, discarded after pretraining [fact, main.py, paper.md] | MINIMAL 3-layer, $K=128$ fixed across sizes (Q2) [derivation] |
| Multi-crop | $V=8$, $V_g=2$, 224 global / 96 local [fact, LeJEPA]; README says local 98 [fact, README] | LeVJEPA: 1 global (scale 0.8–1.0) + 10 locals at 96 (scale 0.02–0.4) [fact, config.yaml] | 2 globals 224 (scale 0.4–1.0) + 6 locals 96 (0.05–0.4) as in the formulation [derivation] |
| Centre | $\mu$ = mean of global embeddings; optional SWA on the encoder producing $\mu$ (small gain for ViTs) [fact, LeJEPA Table 4] | LeVJEPA: `global_emb = embeddings[:, :1]`, one global, no SWA in loss; EMA only for eval checkpoints [fact, main.py, paper.md] | no SWA/EMA anywhere (clean-JEPA constraint); accept the Table 4 gap [derivation] |
| Token dropping | LeJEPA: none in MINIMAL [fact]; LeVJEPA $\rho=0.95$, 90 vs 95 within variability, uniform > tube [fact, paper.md] | after `patch_embed` and after adding abs. pos-embed; `keep=max(1,round(N(1-\rho)))`; independent `rand` per sample per forward ⇒ per view; CLS not dropped; with RoPE the kept `token_ids` are passed so positions stay correct [fact, module.py] | same mechanism; $\rho_g=0.9$ on globals, $\rho_l$ lower on locals (§2) [derivation] |
| SIGReg integral | 17 points, $[-5,5]$, 1024 slices [fact, LeJEPA] | MINIMAL: $t\in[0,3]$, 17 knots, 256 slices, symmetric via weights [fact]; LeVJEPA: 17 knots, 1024 proj [fact] | $[0,3]$ symmetric, 17 knots, 1024 slices [derivation] |
| SIGReg under DDP | not in paper text | `epps_pulley.py`: all-reduce cos/sin means, statistic $\times N_{\rm local}\times$`world_size`; `slicing.py` all-reduces the global step before drawing $A$ [fact]. LeVJEPA: `all_reduce(ecf, AVG)`, $\times$`proj.size(-2)`$\times$`world_size` [fact, module.py]. MINIMAL: unseeded `randn` per call, no DDP [fact] | all-reduce ECF, scale by global $N$, $A$ seeded by (seed, global step) [derivation]; equivalence to a single-process global-batch statistic verified to $5\cdot10^{-14}$ [measured] |
| Loss weighting | $(1-\lambda)L_{\rm inv}+\lambda\,{\rm SIGReg}$; $\lambda$ grid $\{0.01,0.02,0.05,0.1\}$ [fact, LeJEPA; launch_inet10.py] | LeVJEPA `loss = pred + 0.02*sigreg`, `normalize_by_n: false` [fact] | paper convention, $\lambda=0.05$; log $L_{\rm inv}$ and unscaled EP separately [derivation] |
| Seeds | none stated in either paper text I read | no seed policy found in `MINIMAL.md` or `config.yaml` as read [fact, as read] | §5 seed policy [derivation] |

Two corrections to the formulation's "LeVJEPA-style" phrase: (i) token dropping is after patch embedding and position embedding, not on pixels [fact, module.py]; (ii) LeVJEPA uses one global view and 10 locals, and its loss is unweighted invariance + 0.02 SIGReg [fact, config.yaml]. Both are compatible with our 2-global, $(1-\lambda)/\lambda$ choice but should be cited as differences, not as "LeVJEPA defaults".

## 2. Mixed-sensor sequences

Pre-drop and post-drop token counts with the LeVJEPA rule at $\rho=0.9$ [measured]:

| View | sensors | pre-drop | kept |
|---|---|---|---|
| global 224 | one | 196 | 20 |
| global 224 | both | 392 | 39 |
| local 96 | one | 36 | 4 |
| local 96 | both | 72 | 7 |
| 256 crop | one / both | 256 / 512 | 26 / 51 |

Four tokens for a one-sensor local is far below LeVJEPA's regime (158 kept of 3136) [fact, README] and probably destroys the local view [hypothesis]. Adopt $\rho_l=0.5$ on locals (18 / 36 kept) and $\rho_g=0.9$ on globals (20 / 39) [derivation]; put $\rho_l$ in the §5 sweep.

Options. (a) Pad to 512 pre-drop with a key-padding mask: wastes up to 50% of attention/MLP compute before dropping and forces the masked SDPA path [derivation]. (b) Bucket by (resolution, sensor set): every bucket is a static shape, one encoder forward per bucket, no mask [derivation]. (c) Drop to a fixed count across sensor sets: S1-only globals would need $\rho=0.8$ to reach 39 tokens, i.e. the drop rate would depend on sensor availability and leak $m$ into compute/statistics [derivation]. Recommend (b): at most 6 buckets per step (globals both / globals S1-only (m3) / locals S1 / S2 / both at the two drop counts); the bucket sizes are fixed by the stratum quotas (§3) and sensor-mix probabilities, so shapes are static up to rounding, which keeps `torch.compile`/cudnn graphs stable. Position and modality semantics are preserved by construction: per-sensor stem → add modality embedding → add shared 2-D pos-embed from the 16×16 or 6×6 grid (interpolated as in standard ViT) → gather the kept `token_ids` → prepend CLS. Dropping never removes the CLS or the modality tag, and "both" views keep independent random subsets of the two sensors' tokens (one `argsort` over the concatenated 392) [derivation].

## 3. Stratum guarantee at global batch 1024, $R$ ranks

Arithmetic first: four strata with $q_m\ge256$ and $\sum_m q_m=1024$ force $q_m=256$ exactly, so $w_m=N_m/N=0.25$ for all $m$ [derivation; measured by `quotas()`]. The formulation's data-proportional $w_m$ is therefore unreachable at batch 1024; either accept uniform strata (recommended: it is also what makes per-stratum ECFs equally precise) or raise the batch. Per-stratum ECFs at $N_m=256$ have the same null floor as pooled at 1024 (EP mean 1.04 vs 1.13 vs 1.00 for $N=256/1024/4096$, $K=128$) [measured], so 256 is enough for the statistic not to be noise-dominated; collapsed ($\approx412$) and rank-8 ($\approx32$) embeddings are separated by $>30\times$ [measured].

Design: one global stratified sampler, deterministic per step, sharded by slicing. Every rank runs the same generator and takes its own slice, so there is no communication and no duplicate sample within a step.

```
# strata pools built once from the manifest (location, date, c, s1_dt)
pools = {m1: ids_clear, m2: ids_partly, m3: ids_s1only_real, m4: ids_both}
donors_m2 = ids_partly                      # real SEnSeI masks to paste
q = {m: 256 for m in strata}                # forced by B=1024, qmin=256
cursor = {m: shuffled_cycle(pools[m], seed) }   # per-stratum epoch iterators

def global_batch(step):                     # identical on every rank
    g = Generator(seed ^ hash(step))
    items = []
    for m in strata:
        for i in range(q[m]):
            sid = next(cursor[m])            # with replacement across steps when pool < q[m]*steps
            spec = None
            if m == m3 and g.rand() < p_synth_m3:        # synthetic S1-only
                sid = next(cursor[m1]); spec = ("drop_s2",)
            if m == m2 and g.rand() < p_synth_m2:        # synthetic partly-cloudy
                sid = next(cursor[m1]); d = g.choice(donors_m2)
                while d == sid: d = g.choice(donors_m2)
                spec = ("paste_mask", d, g.choice(bins_upper))
            items.append((sid, m, spec))
    assert len({(sid, spec) for sid, m, spec in items}) == 1024    # no duplicate within step
    return items

def rank_batch(step, r, R):                 # interleave so each rank gets q[m]/R per stratum
    return global_batch(step)[r::R]         # 128 items, 32 per stratum when R=8
```
Per-rank quotas $q_m/R$ with remainder spread over the first ranks sum exactly to $q_m$ [measured]. Equal per-rank stratum counts are not required for correctness (the stratified ECF is all-reduced, so the global $N_m$ is what matters) but keep the bucket shapes of §2 identical on all ranks [derivation]. Synthetic items carry `spec`; the loader applies it after reading the pair: `drop_s2` deletes the S2 tokens from globals and locals and labels $m_3$; `paste_mask` loads the donor's SEnSeI mask, pastes it (mask bit sets those S2 pixels to a cloud-like value or masks the tokens), recomputes $c$ from the pasted mask and asserts it lands in the requested bin, otherwise redraws. Donor masks are read from a small pre-extracted mask store, so synthesis costs one extra small read, not a second 0.6 MB pair [derivation]. Log the real/synthetic flag per item; report per-stratum EP separately for real and synthetic members [derivation].

## 4. Scale grid and WSD schedule

Definition (used everywhere below): **tokens processed = post-drop patch tokens entering the encoder, summed over all views of all samples, CLS excluded, each view forwarded once.** Note LeJEPA's paper pseudocode forwards the globals twice (`g_emb` and inside `a_emb`) [fact]; we forward once and use the global rows as $\mu$, as MINIMAL does [fact, MINIMAL.md].

Tokens per sample with $\rho_g=0.9$, $\rho_l=0.5$, local sensor mix (0.4, 0.4, 0.2): $2\cdot39+6\,(0.8\cdot18+0.2\cdot36)=78+129.6\approx208$ for full-observation samples and $2\cdot20+6\cdot18=148$ for m3 [derivation]. With the formulation's original $\rho=0.9$ everywhere it is $\approx106$, which is where the "≈100 tokens" figure comes from — but that is per *sample* (8 views), not per view [derivation]. Steps at batch 1024 for 2B / 8B / 32B tokens: $\approx9.4$k / 37.5k / 150k at 208 tok/sample (or 18.4k / 74k / 294k at 106) [derivation].

WSD: lr rises linearly from $0.01\,\eta$ to $\eta$ over 1500 steps, stays at $\eta$, then for each cooldown budget $T_b$ a branch decays linearly to $\eta/1000$ over the last 20% of $T_b$ [derivation; the lr/1000 floor is the LeJEPA README value, fact]. Branching: the stable run saves a full checkpoint (weights, AdamW moments, RNG, loader cursors) at $0.8\,T_b$ for every $T_b$ in the run's list; a child job loads it, switches the scheduler to the linear decay, consumes the *next* $0.2\,T_b$ tokens from the same loader cursor (so no data is replayed), evaluates and stops. The parent continues unchanged. Cooldown budgets (≥5 per run): $T\in\{32B\}\to\{2,4,8,16,32\}$B; $\{8B\}\to\{0.5,1,2,4,8\}$B; $\{2B\}\to\{0.125,0.25,0.5,1,2\}$B [derivation]. Because the stable phase is shared, the 32B run also yields 2B and 8B cooled checkpoints; separate 2B and 8B runs are still needed for seed/variance estimates and for the Q1 populations (§5). Whether WSD cooldowns match a full cosine run at equal tokens is a published LLM result, not verified for LeJEPA; treat the WSD-vs-cosine gap as something P2 measures on ViT-S/2B [hypothesis].

| Size | depth × width (heads) | params | budgets | warmup | stable | cooldowns | batch | views | $\rho_g/\rho_l$ | precision |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-Ti | 12×192 (3) | 5.7M [fact, timm] | 2B, 8B | 1500 steps | to $0.8\,T_{\max}$ | 5 per run | 1024 | 2g+6l | 0.9/0.5 | bf16 |
| ViT-S | 12×384 (6) | 22M [fact, timm] | 2B, 8B, 32B | 1500 | same | 5 | 1024 | 2g+6l | 0.9/0.5 | bf16 |
| ~40M | 12×512 (8) | ≈40M [derivation] | 2B, 8B, 32B | 1500 | same | 5 | 1024 | 2g+6l | 0.9/0.5 | bf16 |
| ViT-B | 12×768 (12) | 86M [fact, timm] | 2B, 8B, 32B | 1500 | same | 5 | 1024 | 2g+6l | 0.9/0.5 | bf16 |
| ViT-L | 24×1024 (16) | 303M [fact, timm] | 8B, 32B | 3000 (LeVJEPA used 12k steps for ViT-L [fact, config]) | same | 5 | 1024 | 2g+6l | 0.9/0.5 | bf16 |
| ~1B (ViT-g class) | 40×1408 (16) | ≈1.0B [estimate] | 32B only, pre-registered | 3000 | same | 5 | 1024 | 2g+6l | 0.9/0.5 | bf16 |

All lr/warmup values are starting points [estimate]; P1/P2 fix them before the grid runs.

## 5. Q1 populations: 64-config Sobol sweep per objective at ViT-S / 2B tokens

Swept (scrambled Sobol, 64 points, same points for LeJEPA / BT / MAE where the dimension exists) [derivation]:
lr log-uniform $[10^{-4},3\cdot10^{-3}]$ (covers both LeJEPA grid values); wd log-uniform $[10^{-5},10^{-1}]$ (the paper's grid span); $\lambda$ log-uniform $[0.01,0.2]$; warmup steps log-uniform $[300,5000]$; drop-path $\{0,0.05,0.1,0.2\}$ (quantised); local-crop count $\{4,6,8,10\}$; $\rho_l\in\{0.3,0.5,0.7,0.9\}$ with $\rho_g$ fixed at 0.9 (varying $\rho$ changes tokens/sample, so the step count is recomputed per config to hit 2B tokens exactly); projector $K$ stays 128 (not swept).
Fixed: global batch 1024; $K=128$ (Q2 needs one $h$-space); loss convention $(1-\lambda)L_{\rm inv}+\lambda\,{\rm SIGReg}$ with SIGReg scaled by the global $N=1024$, 1024 slices, $[0,3]$; data manifest and stratum quotas; view geometry other than the swept counts; WSD with cooldown at 2B; bf16. Seed policy: config $i$ uses seed $i$ for init and for the data order (`seed ^ hash(step)` in §3); three anchor configs (defaults, lowest-lr corner, highest-$\lambda$ corner) are repeated with seeds $\{101,102,103\}$ to measure seed noise [derivation].
Because $\lambda$ varies, the raw training loss is not comparable across configs; LeJEPA's own fix is $C^{(\alpha)}=\rho_s({\rm train\_loss}/\lambda^{\alpha},{\rm acc})$ with $\alpha\approx0.4$ giving ≈99% correlation on their data (Eq. 10) [fact, LeJEPA]. Q1 pre-registers $\alpha\in\{0,0.4\}$ and also ranks by $L_{\rm inv}$ alone and by unscaled EP alone [derivation].
Logged per checkpoint (5 cooldowns + every 10% of the stable phase), on a fixed held-out set of 8192 samples with fixed slice seeds: $L_{\rm inv}$; unscaled EP (mean over views and 1024 slices); SIGReg$_{\rm strat}$ per stratum, split real/synthetic; RankMe (https://arxiv.org/abs/2210.02885) [fact, arXiv search]; LiDAR (https://arxiv.org/abs/2312.04000) [fact, arXiv search]; $\alpha$-ReQ — I could not retrieve its arXiv/OpenReview record from this VM; I recall it as Agrawal et al., NeurIPS 2022 (eigenspectrum decay exponent) but that citation is unverified, so log the spectrum decay exponent under a descriptive name until the reference is pinned; $m$-probe and $c$-probe on $f$ and $h$; 1k-label linear probe on the eval task from subagent-eval (audit only, not used for ranking) [derivation].

## 6. Throughput and I/O (all [estimate], unverified on hardware)

Assumptions: GH200 bf16 dense peak taken as $\approx989$ TFLOP/s (H100-class; not re-checked against the datasheet from this VM); training FLOPs $\approx 6\,P$ per token (attention and projector ignored, which is fair at 20–40 tokens per sequence); MFU 0.3 and 0.4; 0.6 MB per packed S1+S2 pair read once for all 8 views. tokens/s $=\mathrm{MFU}\cdot 989\mathrm{T}/(6P)$; GPU-h $=T/{\rm tok/s}/3600$ [derivation]. Values from `checks.py` [measured arithmetic on estimated inputs]:

| Size | tok/s (MFU .3 / .4) | GPU-h 2B | 8B | 32B | MB/s per GPU needed at 106 tok/sample, MFU .4 |
|---|---|---|---|---|---|
| Ti | 8.7M / 11.6M | 0.1 / 0.05 | 0.3 / 0.2 | 1.0 / 0.8 | 8,700 |
| S | 2.25M / 3.0M | 0.2 / 0.2 | 1.0 / 0.7 | 4.0 / 3.0 | 2,250 |
| 40M | 1.24M / 1.65M | 0.4 / 0.3 | 1.8 / 1.3 | 7.2 / 5.4 | 1,240 |
| B | 0.57M / 0.77M | 1.0 / 0.7 | 3.9 / 2.9 | 15.5 / 11.6 | 580 |
| L | 0.16M / 0.22M | 3.4 / 2.6 | 13.6 / 10.2 | 54.5 / 40.8 | 165 |
| 1B | 0.05M / 0.07M | 11.2 / 8.4 | 44.9 / 33.7 | 180 / 135 | 53 |

Reading: at $\rho=0.9$ the compute per sample is so small that every size up to B would need 0.5–9 GB/s of input per GPU to stay compute-bound; that is not a realistic Lustre/NVMe-per-GPU figure, so Ti–B are I/O/CPU-bound and the FLOP numbers above are lower bounds [derivation]. An I/O floor at 300 MB/s per GPU gives 500 samples/s ≈ 53k tok/s (106 tok/sample) → 2B tokens ≈ 10.5 GPU-h and 32B ≈ 170 GPU-h *for any size up to B* [estimate]. Three mitigations, in order: (1) lower $\rho_l$ (the §2 recommendation already doubles tokens per byte); (2) cache the uint16 tile store in local NVMe/RAM once per node (SSL4EO-S12-scale: 250k locations × 4 dates × 0.6 MB ≈ 600 GB [estimate]); (3) draw two independent 8-view sets from each read in consecutive steps. The MFU numbers must be measured in P2 on one GH200 before any GPU-hour claim goes into the paper.

## 7. Matched baselines

**Barlow Twins.** Same backbone, per-sensor stems, modality/pos embeddings, token dropping, data manifest, stratified loader, WSD schedule, batch 1024, bf16 and the same token budget under the §4 definition. BT's objective is the cross-correlation loss between two views, $\sum_i(1-C_{ii})^2+\lambda_{BT}\sum_{i\ne j}C_{ij}^2$, computed on batch-standardised projector outputs; the original uses a 3-layer 8192-wide projector and $\lambda_{BT}=5\cdot10^{-3}$ (Zbontar et al. 2021, https://arxiv.org/abs/2103.03230; I did not re-read the paper on this VM, so treat the exact projector width as a value to check) [fact, recalled; unverified here]. To spend the same tokens, BT consumes the same 8 views: the loss is the mean over all pairs (global $g$, view $v\ne g$), which keeps the "regress every view onto the globals" structure. For Q2 the Procrustes test runs on the BT projector output restricted to $K=128$ by a final linear layer so $h$-dimensions match; the BT sweep varies $\lambda_{BT}$ in place of $\lambda$ [derivation]. What differs is therefore only the loss and the projector width/normalisation.

**MAE.** Same backbone, stems, embeddings, data, loader, schedule, batch and token budget. MAE's 75% random masking *is* the token drop (He et al. 2021, https://arxiv.org/abs/2111.06377, recalled) so set $\rho=0.75$ on a single 224 view per sample and count encoder tokens exactly as in §4 (98 of 392); the lightweight decoder (8 blocks, width 512 in the original, recalled) reconstructs per-band-normalised pixels of *both* sensors at the masked positions, with per-sensor output heads; decoder FLOPs are excluded from the token count but included in reported GPU-hours. MAE has no projector; for Q2 and for the loss-ranking question, $h$ is a 128-d linear map fitted on the CLS token of the frozen encoder with the same calibration set, which is a weaker test and is stated as such. Q1 for MAE ranks by reconstruction loss [derivation]. Fairness rests on equal encoder tokens, data order (same seeds) and identical evaluation; it does not equalise wall-clock (MAE pays for the decoder, LeJEPA for 8 views) and that is reported, not hidden [derivation].

## 8. Training-loop pseudocode

```
init_distributed(); R, r = world_size(), rank()
seed = cfg.seed; torch.manual_seed(seed + r)             # rank-offset for dropout/token-drop noise
sampler_seed = seed                                      # shared: identical stratified batches on all ranks
model = MultiSensorViT(stems={s1: Linear(2*16*16,d), s2: Linear(12*16*16,d)},
                       mod_emb=Embedding(2,d), pos_emb=Param(16*16,d), cls=Param(d), depth, width)
proj  = MLP(d,[2048,2048,K], BatchNorm1d)
opt   = AdamW([{params: model, wd: cfg.wd}, {params: proj, wd: cfg.wd}], lr=cfg.lr, betas=(0.9,0.95))
sched = WSD(warmup=1500, peak=cfg.lr, floor=cfg.lr/1000, cooldowns=cfg.cooldown_tokens)
sigreg = EppsPulley(knots=17, t=linspace(0,3), slices=1024)      # weights = trapezoid * exp(-t^2/2)
tokens_seen = 0
for step in count(start=ckpt.step):
    items = global_batch(step)[r::R]                            # §3: (sid, stratum, synth_spec) x 128
    pairs = [read_pair(sid, spec) for sid,_,spec in items]      # one 0.6 MB read each, spec applied
    views = make_views(pairs)                                   # 2 globals (both; S1-only if m3), 6 locals, sensor mix (.4,.4,.2)
    buckets = group_by(views, key=(res, sensor_set))            # <= 6 static shapes
    cls_out = {}
    with autocast(bf16):
        for key, vs in buckets.items():
            x = cat([stem_s(v.pixels_s) + mod_emb[s] + pos_emb(grid_s) for s in key.sensors])  # (b, N_pre, d)
            keep = max(1, round(N_pre * (1 - rho[key.res])))
            ids = rand(b, N_pre).argsort(1)[:, :keep]           # LeVJEPA rule, independent per view
            x = gather(x, 1, ids); x = cat([cls.expand(b,1,d), x], 1)
            tokens_seen_local += b * keep
            cls_out[key] = model.blocks(x)[:, 0]                # (b, d)
        z = proj(reassemble(cls_out, order=(n, v)))             # (128, 8, K)
    z = z.float()
    mu = z[:, :n_global].mean(1, keepdim=True)                  # centre = mean of the two globals
    L_inv = (mu - z).square().mean()
    A = unit_slices(K, 1024, seed=hash(sampler_seed, step))     # same on all ranks (step is identical by construction)
    xt = (z @ A)[..., None] * sigreg.t                          # (128, 8, 1024, 17)
    if cfg.stratified:
        ecf = {m: stack([cos(xt[idx_m]).mean(0), sin(xt[idx_m]).mean(0)]) for m in strata}   # per-view per-slice
        for m in ecf: all_reduce(ecf[m], AVG)                   # global N_m = 256
        EP = {m: ((ecf[m][0]-phi)**2 + ecf[m][1]**2) @ w * 256 for m in strata}
        SIG = sum(0.25 * EP[m].mean() for m in strata)          # w_m forced to 1/4 at B=1024
    else:
        ecf = stack([cos(xt).mean(0), sin(xt).mean(0)]); all_reduce(ecf, AVG)
        EP = ((ecf[0]-phi)**2 + ecf[1]**2) @ w * 1024; SIG = EP.mean()
    loss = (1 - lam) * L_inv + lam * SIG
    loss.backward(); clip_grad_norm_(all_params, 3.0); opt.step(); opt.zero_grad()
    tokens_seen = all_reduce(tokens_seen_local, SUM)            # §4 definition
    sched.step(tokens_seen)
    if tokens_seen >= 0.8 * next(cfg.cooldown_tokens):
        save_ckpt(step, model, proj, opt, rng_states, loader_cursors, tokens_seen, tag=f"branch_{T_b}")
        launch_cooldown(tag)                                    # child: linear lr -> floor over the next 0.2 T_b tokens
    if step % log_every == 0 and r == 0:
        log(step, tokens_seen, lr, L_inv, EP.mean() (unscaled: /1024), {m: EP[m].mean()}, grad_norm,
            tokens_per_bucket, frac_synthetic, loss / lam**0.4)
    if step % eval_every == 0: eval_heldout(model, proj)        # RankMe, LiDAR, m/c-probes, 1k-label probe
```
Grad clipping at 3.0 and the per-bucket token accounting are our additions [derivation]; neither official recipe states clipping in the text I read.

## 9. P1 local CPU pilot: ViT-Ti, 500 samples, 50 steps

Setup: 500 real pairs (or synthetic uint16 arrays with the real band statistics if the data are not local), strata forced to 125 each, batch 64 (16 per stratum) with `R=2` simulated by `torchrun --nproc_per_node=2` on CPU (gloo) [derivation]. Numbers that must come out, all [derivation] unless marked:
1. Shapes: globals both → (b, 1+39, 192); globals S1-only → (b, 1+20, 192); locals S1/S2 → (b, 1+18, 192); locals both → (b, 1+36, 192); `keep` equals the §2 table exactly [measured for the counts].
2. Modality embeddings: zeroing `mod_emb` changes the S1-only and S2-only CLS outputs on the same pixels by a non-zero amount, and swapping the two embeddings changes them as well (guards against the tag being ignored or ordered wrongly).
3. Position semantics: for a fixed `ids`, permuting the kept tokens leaves the CLS output unchanged to $10^{-5}$ (permutation invariance after pos-embed is added), while shifting `ids` by one changes it.
4. Loss sanity at init: $L_{\rm inv}$ finite and positive; unscaled EP per view in $[0.5,3]$ for a Gaussian-ish init and far above it ($>10$) if the projector output is replaced by a rank-8 embedding (the measured null floor is $\approx1.0$–$1.1$ and rank-8 gives $\approx32$ [measured]); no NaN/Inf in any step; grad norm finite.
5. DDP correctness: the two-rank stratified SIGReg equals the single-process value on the concatenated batch to $<10^{-6}$ relative (measured $5\cdot10^{-14}$ in NumPy fp64 [measured]; bf16/fp32 will be looser); both ranks draw identical `A` (measured true for the seeding rule [measured]).
6. Quotas: every step has exactly 16 items per stratum per rank, no duplicate `(sid, spec)` within a step, and over 50 steps every id in the smallest pool appears at least once (replacement engaged and logged).
7. Synthetic strata: `drop_s2` items produce no S2 tokens anywhere; `paste_mask` items have recomputed $c$ in the requested bin for 100% of accepted items; donor ≠ self in 100%.
8. Trend: over 50 steps at lr $10^{-3}$ the training loss falls monotonically in a 10-step moving average and $L_{\rm inv}$ drops by $>20\%$ [estimate; a threshold to confirm, not a claim].
9. Checkpoint: save at step 25, reload into a fresh process, run step 26 and compare loss and parameters to the uninterrupted run — identical to bf16 round-off ($<10^{-3}$ relative) [estimate].
10. Determinism: two runs with the same seed give identical `items` for all 50 steps and identical `A`; a different seed gives different `items` at step 0.
11. Walltime budget: $<15$ min on 8 CPU cores for the 50 steps [estimate].
Pass means all of 1–7, 9, 10 hold and 8 is observed; otherwise the pipeline is not ready for GPU time.

## What this changes in the formulation below

1. **Tokens per view is far smaller than written.** With $\rho=0.9$ on 224/96 crops a view keeps 20–39 (global) or 4–7 (local) tokens; "≈100 tokens" is per sample (8 views) [measured]. Set $\rho_l=0.5$ for locals, keep $\rho_g=0.9$, and recompute budgets with the §4 definition (tokens = post-drop encoder patch tokens, all views, CLS excluded) [derivation].
2. **Uniform strata are forced.** $q_m\ge256$ at batch 1024 with four strata means $w_m=1/4$; drop "$w_m=N_m/N$" or raise the batch [derivation].
3. **Drop-path 0.1 and wd 0.05 are not LeJEPA defaults.** Official code uses drop-path 0 and wd in $\{5\cdot10^{-2},3\cdot10^{-2},10^{-5}\}$ (LeJEPA) / 0.04 (LeVJEPA); make drop-path 0 the default and sweep it [fact; derivation].
4. **SWA/EMA statement.** LeJEPA's paper applies optional SWA to the encoder producing $\mu$ and reports a small ViT gain (Table 4); LeVJEPA's EMA is evaluation-only. Our "no EMA" is a deliberate departure from the best LeJEPA configuration, not a copy of it; say so [fact].
5. **SIGReg conventions.** Paper: 17 points, $[-5,5]$, 1024 slices; MINIMAL: $[0,3]$ symmetric, 256 slices; LeVJEPA: 1024 slices. Under DDP both codebases all-reduce the ECF and multiply by the global batch; MINIMAL's `A` is unseeded and single-process, so the seeded `(seed, step)` rule is ours, and LeJEPA's package syncs the step via all-reduce [fact]. Verified numerically that the all-reduced statistic equals the global-batch statistic [measured].
6. **LeVJEPA differences to cite correctly.** One global view, 10 locals, loss $L_{\rm inv}+0.02\,{\rm SIGReg}$, warmup-then-flat lr, batch 3072, dropping after patch+pos embedding with CLS kept [fact].
7. **Schedule.** WSD with branched cooldowns is a study design; neither official recipe uses it, and MINIMAL's cosine floor is an absolute $10^{-3}$, not lr/1000 [fact; derivation].
8. **I/O, not FLOPs, bounds Ti–B at $\rho=0.9$** (0.5–9 GB/s per GPU needed to stay compute-bound); plan a node-local tile cache and measure MFU in P2 before quoting GPU-hours [estimate].
9. **$\alpha$-ReQ reference unresolved** from this VM; RankMe (2210.02885) and LiDAR (2312.04000) are confirmed [fact].
10. **Q1 ranking statistic.** Pre-register LeJEPA's Eq. 10 normalisation ${\rm loss}/\lambda^{0.4}$ alongside the raw loss, $L_{\rm inv}$ and unscaled EP, since $\lambda$ is swept [fact; derivation].

Not done: no repos, PRs or commits were created; no training was run beyond the NumPy checks in the attached `checks.py`.

ATTACHMENT:{"url":"https://app.devin.ai/attachments/30ae8478-24ca-4b78-a531-fb88f18cf879/REPORT.md","fileSize":30317}
ATTACHMENT:{"url":"https://app.devin.ai/attachments/dd06c297-6aa2-41ba-a509-8e85de81b5b0/checks.py","fileSize":3361}