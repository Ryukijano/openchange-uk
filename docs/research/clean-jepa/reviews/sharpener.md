 # Sharpening T1/T2/T3 for a clean-JEPA Earth-observation study

Date 2026-10-09. Web research only. Nothing was trained or run. Compute figures are my own arithmetic, not measurements.

## 0. Assumptions, and prior work that changes the plan

Assumptions: "clean JEPA" means LeJEPA as published: one encoder, an invariance/prediction MSE, and SIGReg (an Epps-Pulley test applied to random 1-D projections, 256 slices by default, λ = 0.05 recommended) ([LeJEPA](https://arxiv.org/abs/2511.08544)). Budget is 20,000 node-hours = 80,000 GH200 GPU-hours. The primary data is the S1/S2 part of Copernicus-Pretrain, which the HF card lists as CC-BY-4.0, 18.7M images in ~310k ERA5-aligned 0.25° grids ([card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain)).

Things the current plan does not yet account for:

- **SIGReg has already been used across S1/S2.** CR-JEPA applies SIGReg to cross-modal retrieval projections on BEN-14K ([2606.00706](https://arxiv.org/abs/2606.00706)). HQ-JEPA uses SIGReg in a fused S1/S2 latent and evaluates on GEO-Bench ([2605.31068](https://arxiv.org/abs/2605.31068)). Le MuMo JEPA extends LeJEPA to RGB+LiDAR/thermal with fusion tokens ([2603.24327](https://arxiv.org/abs/2603.24327)). "Cross-sensor LeJEPA" is therefore not a new method by itself. The new parts have to be scale, the substitution question, and a test of the theory.
- **There is a temporal EO JEPA from 5 Oct 2026.** T-JEPA learns time-gap-conditioned latent transitions. It uses an EMA target encoder with stop-gradient, adds a masked-pixel branch, and trains HiViT-S on CACo Sentinel-2 RGB, 200k locations ([2610.05731](https://arxiv.org/abs/2610.05731)). It is the closest prior work for T2.
- **"One-Step Next-Latent Prediction Is Not a World Model"** ([2609.36227](https://arxiv.org/abs/2609.36227)) proves two things. One-step regression identifies only a conditional mean, so open-loop error still grows with horizon. And an isotropy penalty depends only on the embedding marginal, so its gradient with respect to transition weights is zero. Any T2 design has to answer this paper directly.
- **Shared Gaussianization** ([2610.10299](https://arxiv.org/abs/2610.10299)) shows that a pure Gaussianity test can lower its loss by adding per-view nuisance unless the alignment weight exceeds the nuisance channel's gain. SAR speckle is exactly such a per-view nuisance, so λ is not a free choice in T1.
- Other EO JEPA-style work to cite: AnySat ([2412.14123](https://arxiv.org/abs/2412.14123)), SpectralEarth-FM (JEPA-style global/single-sensor views, 40 TB, [2605.21075](https://arxiv.org/abs/2605.21075)), Mini-JEPA fleet (22M per-sensor models, [2605.14120](https://arxiv.org/abs/2605.14120)), SAR-JEPA ([2311.15153](https://arxiv.org/abs/2311.15153)). I found no LeJEPA scaling study on EO, and no heuristics-free temporal EO JEPA (searches of arxiv.org on 2026-10-09: "LeJEPA", "JEPA remote sensing", "SIGReg multimodal").

## 1. Shared protocol (all three directions)

- Backbone: ViT with fixed per-sensor linear patch stems (S1: 2 bands, S2: 12–13 bands). Do not use a wavelength hypernetwork. It would confound the objective comparison with Copernicus-FM's tokenizer ([2503.11849](https://arxiv.org/abs/2503.11849)). LeVJEPA-style uniform token dropping (they use ρ = 0.95 and report accuracy rising with ρ; [2608.27395](https://arxiv.org/abs/2608.27395)). Pick ρ for EO in the pilot.
- Evaluation: frozen encoder. Linear probe plus one fixed light segmentation head, with hyperparameters frozen before pretraining. 3 probe seeds, variance reported. The audit found 46 cross-paper disagreements of 10+ points ([2605.12678](https://arxiv.org/abs/2605.12678)), so every baseline is re-run in our harness, never copied from papers.
- Benchmarks: [Copernicus-Bench](https://arxiv.org/abs/2503.11849) (15 tasks; it has *paired* S1/S2 versions of EuroSAT and BigEarthNet, plus Flood-S1, LCZ-S2 and Cloud-S2), a PANGAEA subset with S1+S2 inputs (Sen1Floods11, PASTIS-R, BioMassters, CropTypeMapping-SS, plus DynamicEarthNet and SpaceNet 7 for change; [README](https://github.com/VMarsocci/pangaea-bench)), and OpenChange-UK (coastal shoreline/land-cover change, geographically and temporally held out; [repo](https://github.com/Ryukijano/openchange-uk)). Skip GEO-Bench v1, which is saturated and mostly S2. [GEO-Bench-2](https://arxiv.org/abs/2511.15658) (19 datasets, capability groups) is optional if reviewers ask.
- Frozen external baselines: Copernicus-FM ViT-B (same corpus, MAE + distillation; the cleanest controlled comparison), TerraMind-B ([2504.11171](https://arxiv.org/abs/2504.11171)), CROMA, Panopticon ([2503.10845](https://arxiv.org/abs/2503.10845)), DINOv3-SAT-493M (RGB, 0.6 m; report the band and resolution mismatch), and the TESSERA v2 student where tasks are pixel-wise.
- Compute arithmetic: training FLOPs ≈ 6 × params × visible tokens. At an assumed 300 TFLOP/s sustained per GH200 (UNVERIFIED; the pilot measures it), ViT-H (630M) on 32B visible tokens costs 1.2e20 FLOPs ≈ 110 GPU-h. A full 4-size × 3-budget grid is about 2.6e20 FLOPs ≈ 250 GPU-h per arm before overheads. With a 10× allowance for multi-view passes, I/O stalls and evaluation, that is roughly 2.5k GPU-h (≈ 600 node-hours) per arm. **Compute is not the binding constraint. Data staging (S1+S2 subset of 20.7 TB) and one person's time are.**

## 2. T1: cross-sensor latent translation (S1 → S2)

**(a) Method novelty.** One shared encoder with per-sensor stems. Local views are S1 crops (and optionally S1 + 10% of S2 tokens). Each local view predicts the [cls] embedding of the global S2 view of the same grid cell and date window. Gradients flow through both branches: no EMA, no stop-gradient, no pixel decoder. Three things separate this from prior work:
1. **Per-modality SIGReg, not pooled.** A mixture of S1 and S2 embeddings can be N(0, I) overall even when each modality occupies its own region of the space (for example, split by the sign of one coordinate). So pooled SIGReg does not remove a modality gap. Applying SIGReg separately to the S1 and S2 embedding sets enforces equal marginals, p(z_S1) = p(z_S2) = N(0, I). That is the stationarity assumption the identifiability theorem needs (assumptions: independence, stationarity p(z) = p(z′), additive noise; [2605.26379](https://arxiv.org/abs/2605.26379)). CR-JEPA and HQ-JEPA apply SIGReg to retrieval or fused projections and give no argument of this kind.
2. **Against Copernicus-FM.** Copernicus-FM solves *input* heterogeneity with a wavelength/bandwidth hypernetwork in the patch embedding, trained with MAE plus distillation. T1 solves *target* heterogeneity: SAR has to predict an optical latent. The two are orthogonal, and T1 deliberately keeps the tokenizer simple.
3. **A testable prediction from the theory (my derivation, not stated in the paper).** S1 and S2 do not see the same world variables. SAR sees structure, roughness and moisture. Optical sees reflectance and chlorophyll. If the theorem applies to the shared part, the cross-sensor embedding should *linearly* recover variables both sensors observe (DEM elevation, open water fraction, built-up fraction), but not optical-only variables (NDVI, S5P NO₂). The approximate-identifiability result bounds nonlinear energy by the alignment gap. That predicts a second test: per variable, (MLP-probe R² − linear-probe R²) should increase with the residual cross-sensor alignment loss. Ground truth is free here: DEM is in Copernicus-Pretrain, the grids are ERA5-aligned (soil moisture, temperature), and WorldCover gives water and built-up fractions.

**(b) Primary hypothesis.** If an S1→S2 LeJEPA with per-modality SIGReg is trained at matched compute, then the gap between SAR-only and optical-only frozen-probe scores on paired tasks shrinks as a power law in compute, with a steeper exponent than an S1-only LeJEPA. Measured by Δ = score(S2 input) − score(S1 input) on EuroSAT-S1/S2 and BigEarthNet-S1/S2 (Copernicus-Bench), and by S1-only mIoU on Sen1Floods11 and OpenChange-UK cloudy dates.

**(c) Minimal grid.**
- Sizes: ViT-S (22M), B (86M), L (300M), H (630M).
- Visible-token budgets: 2B, 8B, 32B (fixed after the pilot).
- Data: 1%, 10%, 100% of S1/S2 grid cells, run at B and L at the 8B budget only.
- Seeds: 3 at S and B, 1 elsewhere.
- Arms: (A) S1→S2 per-modality SIGReg; (A′) pooled SIGReg, at B only; (B) single-sensor LeJEPA on S1 and on S2, the controls; (C) MAE at B and L, same tokens; (D) DINOv2-style EMA teacher at B; (E) supervised ViT-B per task (PANGAEA already includes UNet/ViT supervised baselines).
- About 45 pretraining runs in total. Arms A and B cover the full size × budget grid; the rest are single points.

**(d) Metrics.** Paired S1/S2 tasks in Copernicus-Bench are the only standard benchmark where the substitution gap is defined at a fixed label set. Sen1Floods11 and PASTIS-R check dense transfer. OpenChange-UK gives the cloud-cover use case. Report latent-translation cosine error on held-out grid cells as the scaling-law y-axis, and the linear/MLP probe R² table for the identifiability test.

**(e) Kill criterion.** At the pilot point (ViT-B, 8B tokens, 3 seeds), stop the T1 framing if either holds:
- the S1-only probe of arm A does not beat arm B-S1 by more than 2 seed standard deviations on both EuroSAT-S1 and Sen1Floods11, or
- it is below frozen Copernicus-FM ViT-B on both.

Kill the scaling claim if Δ at L is within seed noise of Δ at S.

## 3. T2: temporal latent forecasting (an EO world model without generation)

**(a) Method novelty.** Against T-JEPA (EMA + stop-gradient + MAE branch, RGB, HiViT-S), Sat-JEPA-Diff (predicts AlphaEarth embeddings to steer a diffusion decoder, [2603.13943](https://arxiv.org/abs/2603.13943)) and RS-WorldModel (text-guided generation, [2603.14941](https://arxiv.org/abs/2603.14941)):
- a single block-causal encoder (LeVJEPA showed block-causal attention at no measured cost);
- a Δt-conditioned predictor trained with **direct multi-horizon targets**. T-JEPA reports direct multi-scale supervision beats recursive rollout, and 2609.36227 explains why one-step fitting is insufficient;
- SIGReg on the encoder output only.

Because the SIGReg gradient into the predictor is zero (2609.36227), the predictor is trained by the prediction MSE alone. That is consistent, but it has to be stated.

The theory gives one sharp test. Under its assumptions, stationary Gaussian latents with additive noise must follow an Ornstein–Uhlenbeck transition z′ = ρz + √(1−ρ²)η ([2605.26379](https://arxiv.org/abs/2605.26379)). So per-dimension latent autocorrelation should decay as ρ^Δt once day-of-year is conditioned out. Fit that decay on stable pixels. Where held-out data departs from it, that departure is a change signal, which links T2 to OpenChange-UK.

**(b) Hypothesis.** If the predictor is Δt-conditioned and trained with direct targets at horizons of 1–3 seasons under SIGReg, then (i) its open-loop latent error on a held-out future year beats persistence (z_{t+Δ} = z_t) and seasonal climatology at every model size, and (ii) its forecast residual detects change better than plain embedding differencing. Measured by relative latent MSE versus persistence, and by AUROC/F1 on OpenChange-UK coastal change, DynamicEarthNet and SpaceNet 7.

**(c) Grid.**
- Sizes S, B, L; 3 token budgets; 2 seeds at S.
- Baselines: a T-JEPA-recipe re-implementation at the same budget (EMA + MAE); linear AR on frozen Copernicus-FM and TESSERA embeddings; persistence; climatology.

**(d) Metrics.** As above, plus static Copernicus-Bench tasks to show the encoder did not lose quality.

**(e) Kill criterion.** At ViT-B, stop if the forecaster is not more than 10% better than persistence at a one-season horizon on held-out years, or if residual AUROC does not beat embedding differencing.

**Main weakness:** time-series depth. SSL4EO-S12 has 4 seasonal snapshots per location ([HF](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1)). Copernicus-Pretrain's per-modality timestamp counts are in its card table, but I did not extract them. UNVERIFIED that they support more than about 4 steps. A dense S2 time series would need a new download pipeline (Planetary Computer or Major TOM time slices), which is new data-approval work under OpenChange-UK's source rules.

## 4. T3: geographic and temporal transfer to held-out UK coasts

**(a) Method novelty.** As a protocol this is not new. The method question it can carry is: **does the LeJEPA loss, computed on unlabelled target-region imagery, select models for that region?** LeJEPA reports Spearman of about 0.85 between training loss and ImageNet linear-probe accuracy ([2511.08544](https://arxiv.org/abs/2511.08544)). TESSERA v2 reports |r| < 0.2 for Barlow Twins pretraining loss on EO ([2607.03949](https://arxiv.org/abs/2607.03949)). Label-free proxies such as RankMe ([2210.02885](https://arxiv.org/abs/2210.02885)) predate this, so the claim is narrower: I found no test of whether a SIGReg-based loss on held-out *target-region* EO data closes the gap, compared head-to-head with RankMe on the same checkpoints. A second question: does the per-region SIGReg statistic flag distribution shift before labels exist?

**(b) Hypothesis.** If the LeJEPA loss is evaluated on unlabelled UK target-region and future-date tiles, then across all T1 checkpoints (~45) its Spearman correlation with OpenChange-UK target-region scores is at least 0.7, and higher than that of the source-region loss. Measured by Spearman with bootstrap CIs over checkpoints.

**(c) Grid.** Reuse the T1 (and T2) checkpoints. Extra pretraining compute is about zero. Baselines: RankMe computed on the same target tiles, the same correlation for the MAE and DINO arms, and for TESSERA-style Barlow Twins at one size if budget allows.

**(d) Metrics.** OpenChange-UK primary (held-out regions and dates, splits fixed before any pretraining), plus Copernicus-Bench as an in-distribution reference.

**(e) Kill criterion.** None for the paper. If ρ < 0.4 and the CI covers 0, report it as an independent replication of TESSERA's negative result for a different objective. It stays a section, not a standalone paper.

## 5. Ranking

| | Method novelty | Feasibility (≤20k NHR, 1 person, 6 months) | Product |
|---|---|---|---|
| **T1** cross-sensor | Medium. SIGReg on S1/S2 exists; per-modality SIGReg, substitution-gap scaling and the identifiability test are new | High. Aligned pairs already exist; compute ≈ 600 NHR per arm | **Highest** |
| **T2** temporal | Medium-high, but T-JEPA (Oct 2026) narrows it and 2609.36227 raises the bar | Medium-low. Time-series depth unverified; new data pipeline likely | Middle |
| **T3** transfer | Low as a method, high as a protocol | Highest. Reuses checkpoints | Lowest alone, essential as a component |

**Write T1 first, with T3 inside it as the evaluation backbone and second headline.** Working claim: "a heuristics-free JEPA learns SAR-to-optical latents whose substitution gap shrinks predictably with compute, and its loss on unlabelled target data selects models for held-out regions." T1 has the least data risk, a closed-form prediction from the theory that can fail, and a direct controlled comparison with Copernicus-FM on the same corpus. Keep T2 as the follow-up paper once T1 has given us the data pipeline and the scaling harness.

## 6. A LeJEPA improvement that EO motivates: availability-stratified SIGReg

**Problem.** EO batches mix availability patterns: S2 clear, S2 cloudy (masked tokens), S1 only, both sensors. SIGReg on the pooled batch is satisfied by a Gaussian *mixture* whose components separate by availability pattern. The embedding can then encode "which sensors were present and how cloudy it was" as a nuisance. Shared Gaussianization (2610.10299) shows Gaussianity tests can even be lowered by adding per-view nuisance. LeJEPA bounds the Epps-Pulley gradient and curvature by terms that scale as 1/N in batch size ([2511.08544](https://arxiv.org/abs/2511.08544), Section B.12), so per-pattern tests on small sub-batches will be noisy.

**Proposal.** Build stratified batches with at least 256 samples per availability stratum. Apply SIGReg *within* each stratum, and align every stratum to the full-observation global view through the invariance term. The target condition is that the embedding is N(0, I) conditional on the availability mask, which implies the embedding is independent of the mask. Concretely, a few lines change inside the reference SIGReg loop: group the batch by mask ID and average the per-group statistics.

**Ablation that would prove it** (ViT-B, 8B tokens, 3 seeds, cloud and missing-S2 rates taken from real S2 cloud masks):

1. No SIGReg (collapse control), pooled SIGReg, stratified SIGReg, and stratified SIGReg without the alignment term.
2. Measure:
   - (i) a linear probe predicting availability pattern and cloud fraction from the frozen embedding: should drop to chance under stratified SIGReg;
   - (ii) the downstream drop when S2 is removed or cloud-masked at test time (Copernicus-Bench S1 tasks, Sen1Floods11, OpenChange-UK cloudy dates): should be smaller;
   - (iii) per-stratum Epps-Pulley statistics on held-out data;
   - (iv) clean-input scores: must not fall by more than seed noise.
3. The claim holds only if (i) and (ii) improve and (iv) does not regress.

A second, cheaper ablation for the same paper: stratified dropping (spectral-band dropping on top of spatial token dropping) against LeVJEPA's uniform spatial dropping, at ρ ∈ {0.75, 0.9, 0.95}.

## 7. Weaknesses to state in the paper

- DINOv3-SAT comparisons are RGB at 0.6 m against 10 m multispectral. Present them as context, not a win.
- Copernicus-FM shares our corpus, which makes it a clean comparison but means we cannot claim a data advantage.
- Copernicus-Bench's S1/S2 pairs are small (EuroSAT-S1 16,200 and BigEarthNet-S1 11,894 training images; [2503.11849](https://arxiv.org/abs/2503.11849)), so seed variance may swamp scaling effects at the top end. That is the reason for 3 probe seeds and bootstrap CIs.
- The identifiability prediction in §2(a)(3) is my extrapolation of the theorem to non-shared latents. A null result might mean the assumptions do not hold for EO, not that the theory is wrong.
- UNVERIFIED: GH200 sustained throughput for ViT-H with 90–95% token dropping; Copernicus-Pretrain S1/S2 timestamp depth; X-JEPA's exact setup (I took its numbers only as quoted in CR-JEPA).

    verdict_one_line: Write T1 (S1->S2 clean JEPA with per-modality SIGReg) first, with T3 loss-as-model-selector on held-out UK coasts inside it; T2 is narrowed by T-JEPA (Oct 2026) and limited by time-series depth.