# Evaluation, statistics and pre-registration protocol for the clean-JEPA EO scaling study

Desk research, 2026-10-10. Nothing trained, no cluster access, no repository touched. Labels: **Fact** (primary source cited), **Estimate** (my arithmetic, assumptions stated), **Hypothesis** (what we test), **UNVERIFIED** (could not confirm from a primary source).

## 0. What changes in plan v2 (read this first)

1. **The compute budget in plan v2 is about 10x too high for the token budgets it states, and that money should go into Q1's sample size.** At 2B/8B/32B processed tokens, the whole LeJEPA S/B/L grid plus the 1B run is roughly 550 GPU-h (about 140 node-hours, under 1% of 20,000) at 35% MFU (Estimate, §6). The 1,800 node-hour figure assumes ~1B samples at full token counts, which is a different experiment. Evaluation, not pretraining, is now the largest compute line (Estimate, §5).
2. **Q1 as designed is underpowered.** BT and MAE at "3-4 shared cells" give at most ~12 checkpoints per family; at n = 12 the 95% CI on a Spearman of 0.7 is roughly [0.14, 0.92] (Estimate, §1c). You need about 60 independent runs per objective family for a ±0.15 half-width. Do this with cheap hyperparameter sweeps at ViT-S/2B tokens (0.2 GPU-h each), as LeJEPA itself did, not with more grid cells.
3. **LeJEPA and TESSERA v2 did not measure the same thing.** LeJEPA's 0.85 is a within-setting correlation across hyperparameter sweeps with an in-domain linear probe; TESSERA v2's -0.16 is across 395 runs spanning four orders of compute, against a 15-task composite (Fact, §1a). There is no direct contradiction yet. The matched comparison has to be designed so that it can produce one.
4. **The λ-rescaled loss only differs from the raw loss if λ varies across runs.** With λ fixed, dividing by λ^α is monotone and gives identical Spearman. λ must be one of the sweep axes, or the proxy should be dropped (§1b).
5. **S2 in Copernicus-Pretrain has 13 bands, not 10**, and 264 is not divisible by 16 (Fact, [HF card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain)). Crop to 256 (16×16 = 256 patches per band group) and state the band-group count before pricing anything.
6. **ERA5 values and WorldCover are not in Copernicus-Pretrain.** The card lists S1, S2, S3, S5P, DEM; ERA5 is only the grid definition (Fact, [HF card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain)). Plan v2's "DEM and ERA5 alignment are already in Copernicus-Pretrain" is half wrong.
7. **Sen1Floods11 is not in Copernicus-Bench**, and Copernicus-FM reports no Sen1Floods11 number. The Q3 kill rule against "frozen Copernicus-FM ViT-B on Sen1Floods11" needs a RERUN, not a copied number (Fact, [Copernicus-FM](https://arxiv.org/abs/2503.11849)).
8. **External pair for Q2: GeoTessera has more pairs than plan v2 lists, but all are Barlow-Twins-family.** v1.0, v1.1 (two variants: `dclimate`, `cambridge`) and v2 (two variants: `2B-L~beta1`, `beta2`), all 128-d; the README says variants "do not interoperate, even within one version" (Fact, [GeoTessera README](https://github.com/ucam-eo/geotessera)). These are a BT baseline data point, not a test of LeJEPA theory. AlphaEarth has one public model (V1, 64-d); its annual layers are not version pairs (Fact, [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)).
9. **Q2 calibration is in "pixels" in plan v2, but a ViT/16 on 10 m S2 emits one token per 160 m patch.** Calibrate on patch tokens or image embeddings; state which.
10. **3 model sizes cannot fit a 3-parameter saturating law.** Add at least 2 sizes (e.g. ViT-Ti 5.7M and a ~40M width) and use warmup-stable-decay cooldowns to get 5+ token budgets from one run ([Hägele et al.](https://arxiv.org/abs/2405.18392)). A held-out larger-run check is not new on EO: TESSERA v2 fit on ≤ 278M and validated on 0.5/1/2B models ([2607.03949](https://arxiv.org/abs/2607.03949)). Ours differs only in being pre-registered and on downstream families; say so.
11. **The α-ReQ link in the brief is wrong.** arXiv 2206.02256 is an unrelated paper. α-ReQ is NeurIPS 2022, [doi:10.52202/068431-1281](https://doi.org/10.52202/068431-1281) (title confirmed via Crossref; contents not read here, so the formula below is UNVERIFIED in detail).

## 1. Q1 protocol: label-free model selection

### 1a. How the two correlations were computed

**LeJEPA** (Fact, [2511.08544](https://arxiv.org/abs/2511.08544)). Training loss, meaning the full LeJEPA objective at the end of training, is correlated with frozen-backbone linear-probe top-1 accuracy on the same dataset's test set. Each point is one pretraining run from a hyperparameter sweep (learning rate, weight decay, number of epochs, λ) within one (architecture, dataset) setting. The headline is Spearman about 0.85 averaged across settings; per-setting values vary. It also defines

$$C(\alpha)=\rho_s\!\left(\frac{\mathcal{L}_{\text{train}}}{\lambda^{\alpha}},\ \text{acc}_{\text{test}}\right),$$

and reports ρ close to 0.99 at α ≈ 0.4. α is chosen on the same points it is then scored on, so the 0.99 is an in-sample number.

**TESSERA v2** (Fact, [2607.03949](https://arxiv.org/abs/2607.03949)). Barlow Twins pretraining; 395 runs varying encoder size, data and projector width over nine iso-FLOP buckets. Loss is the converged loss on held-out pixels, normalised across projector widths. Downstream metric is a composite over 15 tasks. Over all 395 runs pooled, Pearson r = −0.18 and Spearman ρ = −0.16. Their decision-level result: selecting bucket peaks by loss instead of by score needs roughly 2–5× the compute for the same downstream score. They also fit the encoder law on runs ≤ 278M and checked it out of sample on 0.5/1/2B models.

**Comparable?** No, on four counts. (i) Population: within-setting hyperparameter sweeps vs pooled across size, data and projector width. A pooled correlation across sizes can be near zero even if loss ranks runs well within each size. (ii) Target: one in-domain linear probe vs a 15-task heterogeneous composite. (iii) Objective: SIGReg + invariance vs Barlow Twins. (iv) Domain: ImageNet-scale natural images vs pixel-time-series EO. A clean test must hold (ii) and (iv) fixed and vary (iii), at both populations from (i).

### 1b. Proxies and targets

Let \(z = g(f(x)) \in \mathbb{R}^{K}\) be the embedding SIGReg acts on, computed on a fixed held-out, unlabelled proxy set \(\mathcal{P}\) (20,000 images drawn from deduplicated grid cells not used for training; frozen before pretraining).

| Proxy | Definition | Source |
|---|---|---|
| P1 raw loss | \(\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{inv}} + \lambda\,\mathcal{L}_{\text{SIGReg}}\) evaluated on \(\mathcal{P}\) with fixed view sampling and fixed random projections (not the logged train loss) | [LeJEPA](https://arxiv.org/abs/2511.08544) |
| P2 rescaled loss | \(\mathcal{L}/\lambda^{\alpha}\); α fixed on a separate *selection split* (see below) | LeJEPA |
| P3 SIGReg alone | Epps–Pulley statistic averaged over M = 1,024 fixed random unit directions, N = 4,096 samples | LeJEPA |
| P4 invariance alone | mean squared distance of view embeddings to their centre, same views as P1 | LeJEPA |
| P5 RankMe | \(\exp(-\sum_k p_k\log p_k)\), \(p_k = \sigma_k(Z)/\lVert\sigma(Z)\rVert_1+\epsilon\), Z = 25,600 × K | [RankMe](https://arxiv.org/abs/2210.02885) |
| P6 LiDAR | entropy effective rank of \(\Sigma_w^{-1/2}\Sigma_b\Sigma_w^{-1/2}\), n = 1,000 source images, q = 50 views each, δ = 1e-4 | [LiDAR](https://arxiv.org/abs/2312.04000) |
| P7 α-ReQ | slope α of a least-squares fit of \(\log\lambda_k\) vs \(\log k\) for the covariance eigenvalues of Z; score is \(-\lvert\alpha-1\rvert\) or α itself, whichever the paper uses (UNVERIFIED detail) | [doi:10.52202/068431-1281](https://doi.org/10.52202/068431-1281) |
| P8 1k-label probe | logistic regression on 1,000 labelled samples from the *selection split* of one task per family, no tuning beyond a fixed 5-value L2 grid | ours |

P2's α: on the selection split (one classification and one segmentation task, separate from all evaluation tasks, labels used only here), choose α ∈ {0, 0.1, …, 1.0} maximising mean Spearman over the LeJEPA sweep. Freeze α. Report Spearman on the evaluation targets with that α, and the LeJEPA-style in-sample \(C(\alpha)\) curve separately and labelled as in-sample. If λ is held fixed in a population, P2 is identical to P1 there and is not reported as a separate proxy.

RankMe, LiDAR and α-ReQ are computed on the backbone output (pooled) and on the projector output; pre-register the projector output as primary for LeJEPA (that is where SIGReg acts) and backbone as primary for BT/MAE, and report both.

**Targets** (frozen encoder, one score per task, then per family as the mean of chance-adjusted scores \(\tilde s = (s - s_{\text{chance}})/(s_{\text{max}} - s_{\text{chance}})\)):

- Classification: EuroSAT-S1, EuroSAT-S2, BigEarthNet-S1, BigEarthNet-S2 (multilabel mAP), LCZ-S2.
- Segmentation: DFC2020-S1, DFC2020-S2, Sen1Floods11, Cloud-S2.
- Regression: BioMassters (RMSE, sign-flipped and normalised by target SD) and the OpenChange-UK regression target if one exists (UNVERIFIED).
- Change: Flood-S1 (Kuro Siwo) and OpenChange-UK change.
- OOD: the same families on the frozen OpenChange-UK UK regions/future dates, reported separately and never averaged into ID.

The composite (mean of four families) is secondary only.

### 1c. Statistics

**Unit of analysis.** One point = one independently trained run evaluated at its final checkpoint. Intermediate checkpoints of a run are not independent points.

**CI.** Percentile bootstrap, 10,000 resamples of runs (cluster bootstrap if seeds share configs: resample configurations, keep all seeds of a resampled configuration).

**How many runs.** Using Fisher's z with the Bonett–Wright variance for Spearman, \(\mathrm{Var}(z)\approx(1+\rho^2/2)/(n-3)\):

$$z=\operatorname{artanh}(0.7)=0.867,\quad \text{half-width in }\rho \approx 1.96\,(1-\rho^2)\sqrt{\frac{1.245}{n-3}} = 0.15 \Rightarrow n-3\approx 55,\ n\approx 58\text{–}60.$$

Exact Fisher-z intervals (asymmetric) give mean half-width 0.165 at n = 50 and 0.150 at n = 60 (CI [0.52, 0.82]). A Gaussian-copula simulation with percentile bootstrap gave median half-width 0.29 / 0.18 / 0.15 / 0.12 at n = 20 / 40 / 60 / 80 (Estimate; my simulation, 150 replicates × 500 resamples). At n = 12, the planned BT/MAE size, the interval is [0.14, 0.92].

For the comparison LeJEPA (ρ = 0.7) vs BT (ρ = 0.2), independent samples, two-sided α = 0.05: power is 0.26 at n = 12, 0.63 at n = 30, 0.77 at n = 40, 0.92 at n = 60 (Estimate, Fisher z).

**Therefore: 64 runs per objective family per population.** Concretely:

- *Population W (within-setting, the LeJEPA claim):* ViT-S, 2B tokens, 10% data, 64 configurations per objective from a Sobol design over learning rate (×10 range), weight decay (×10), λ ∈ [0.01, 0.2] (LeJEPA) or the BT off-diagonal weight (BT) or mask ratio (MAE), number of local views, and training length via WSD cooldown at 1B/2B. One seed per configuration; 8 configurations repeated with 3 seeds to estimate seed noise. 64 × 3 objectives = 192 runs at ~0.2 GPU-h = ~40 GPU-h (Estimate). Repeat a 24-configuration version at ViT-B/8B (~3.4 GPU-h each, ~245 GPU-h for three objectives) to check the result holds at a second size.
- *Population A (across-scale, the TESSERA claim):* LeJEPA grid endpoints, S and B × 3 seeds, L × 1 seed, × 3 data scales × 3 budgets = 63 runs. BT and MAE need the same population to be comparable; at 63 runs each it costs ~390 GPU-h per objective (Estimate, §6), which the corrected budget allows. If BT/MAE stay at shared cells only, report their across-scale ρ as descriptive with its (wide) CI and make no claim.

**Within-run vs across-run.** The decision Q1 informs is "which run to keep", so the claim is across runs. Within a single run, loss falls and probe accuracy rises together almost by construction (both track training progress), so within-run Spearman is high for any objective and says nothing about selection. Pooling intermediate checkpoints with endpoints inflates n with dependent points and mixes the time trend into the across-run comparison, which biases ρ upward and the CI downward. Within-run curves are used only for mechanism (§1e) and for one secondary question: does the proxy pick the best checkpoint *within* a run (reported as regret, the gap between the chosen and the oracle checkpoint's score).

**Selection regret** (the decision metric, reported next to ρ): for each proxy, pick the top run by proxy in each bootstrap resample; regret = best achievable family score − score of the picked run, in units of seed SD.

### 1d. Kill and success criteria (numbers pre-registered)

Pilot (population W at ViT-S, 64 runs per objective):

- **Kill H1** if LeJEPA P1 has Spearman upper CI bound < 0.5 on all four ID families. The paper becomes "loss-based selection fails on EO for every SSL family tested", reported with all proxies.
- **Continue** if LeJEPA P1 point ρ ≥ 0.5 on at least two families.

Full study:

- **H1 success:** LeJEPA P1 (or P2 with α frozen) ρ ≥ 0.7 with lower CI bound ≥ 0.5 on ≥ 3 of 4 ID families in population W, at both ViT-S and ViT-B.
- **H2 success (matched contrast):** ρ_LeJEPA − ρ_BT ≥ 0.3 with the bootstrap CI of the difference excluding 0, on ≥ 3 families. "BT |ρ| < 0.3" alone is not a contrast and is not claimed.
- **H3 OOD:** on OpenChange-UK, LeJEPA ρ lower CI bound ≥ 0.4 on the families it passed ID. Fail = ID-only result, stated as such.
- **H4 proxy ranking:** report the full proxy × family matrix. Claim "loss is as good as a 1k-label probe" only if ρ_P1 ≥ ρ_P8 − 0.1 with CI of the difference including 0.
- **Across-scale (population A):** reported, no success threshold claimed unless pre-registered later on pilot data; prediction is ρ_A < ρ_W for all objectives, because loss scale changes with model size.

### 1e. Mechanism analysis

Saved per run at 10 log-spaced checkpoints (only for the 8 seed-repeated configurations × 3 objectives + all grid runs): (i) RankMe and entropy effective rank of backbone and projector covariance; (ii) the Epps–Pulley statistic on \(\mathcal{P}\), median over 1,024 directions, and the fraction of directions rejected at 5% against a Monte-Carlo null for the same N and K; (iii) kNN (k = 20) and linear probe on EuroSAT-S2 and DFC2020-S2 patch features. Pre-registered prediction (Hypothesis): for LeJEPA, final-checkpoint ρ between P1 and downstream is explained by SIGReg pinning the embedding scale, so the partial correlation of P4 (invariance) with downstream given P3 is positive and BT's loss–downstream ρ rises if BT loss is normalised by its projector covariance trace. If both fail, the mechanism claim is dropped and only the empirical correlation is reported.

## 2. Q2 protocol: version compatibility

### 2a. Maps

Calibration set: n pairs \((x_i, y_i)\), \(x_i = f_B(u_i)\), \(y_i = f_A(u_i)\) for the same inputs \(u_i\), centred with calibration means.

- **M1 orthogonal Procrustes:** \(Q^* = \arg\min_{Q^\top Q = I}\lVert XQ - Y\rVert_F\). With \(X^\top Y = U\Sigma V^\top\), \(Q^* = UV^\top\) (Fact, [2510.13406](https://arxiv.org/abs/2510.13406)).
- **M2 Procrustes + isotropic scale:** \(s^* = \operatorname{tr}\Sigma/\lVert X\rVert_F^2\), map \(sXQ\).
- **M3 whitened Procrustes:** apply M1 to \(X\Sigma_X^{-1/2}\), \(Y\Sigma_Y^{-1/2}\), then un-whiten into A's space. For a perfectly SIGReg-ed embedding \(\Sigma\approx I\), so M3 ≈ M1; a large M3−M1 gap measures how far the embedding is from isotropic.
- **M4 general linear (ridge):** \(W = (X^\top X+\gamma I)^{-1}X^\top Y\), γ by 5-fold CV on calibration data. Upper bound for any linear map.
- **M5 CCA (top-k components)** as a symmetric control.
- **Mismatched dimensions** (\(d_B \ne d_A\), e.g. 384 vs 768 backbone): semi-orthogonal Procrustes with \(Q\in\mathbb{R}^{d_B\times d_A}\), \(Q^\top Q=I\) or \(QQ^\top = I\) on the smaller side, solution again \(UV^\top\) from the thin SVD; and Procrustes after projecting both to their top-k PCA subspace (k = 128). **Recommendation:** keep the SIGReg projector output dimension K identical across all sizes (e.g. 512) so the object the theory is about has equal dimension, and run Q2 primarily on it. Backbone results are secondary.

The theory ([2605.26379](https://arxiv.org/abs/2605.26379)) predicts identifiability up to an orthogonal map under stationarity, independence and additive noise, with an approximate bound under near-optimal training (Fact). It says nothing about how the error scales with model size; "r increases with size" is our hypothesis, not a prediction from the theorem.

### 2b. Retained fraction

$$r = \frac{\mathrm{acc}\big(\mathrm{probe}_A \circ Q\ \text{applied to}\ f_B\big)}{\mathrm{acc}\big(\mathrm{probe}_B\ \text{native}\big)},\qquad \tilde r = \frac{\mathrm{acc}_{\text{mapped}} - \mathrm{chance}}{\mathrm{acc}_{\text{native}} - \mathrm{chance}}.$$

Report both; \(\tilde r\) is primary because a raw ratio flatters easy tasks. The calibration set is unlabelled, drawn from the pretraining distribution (not from the task), so the test is genuinely label-free for B.

- Calibration sizes: 500, 1k, 5k, 20k **patch tokens** (and, separately, image embeddings), 5 random draws each.
- Probes: linear (logistic / per-patch linear for segmentation), kNN (k = 20, cosine), and one light decoder (2-layer conv head on frozen patch tokens, fixed hyperparameters). Probe_A is trained once on A and never retrained.
- Tasks: EuroSAT-S2, BigEarthNet-S2, DFC2020-S2 (seg), Sen1Floods11 (seg), plus OpenChange-UK for OOD.
- Pairs per objective (LeJEPA, BT, MAE): seed–seed at S and B (3 pairs per cell), size–size (S↔B, B↔L, S↔L) at fixed data/budget, data-scale pairs (1%↔10%↔100%) at fixed size. The pairs reuse Q1 checkpoints; Q2 costs inference only.

**Pre-registered predictions (Hypothesis H5–H7):** H5: at 5k calibration tokens, linear probe, LeJEPA seed–seed pairs have median \(\tilde r_{M1}\) ≥ 0.90 at ViT-B. H6: \(\tilde r_{M1}\)(LeJEPA) − \(\tilde r_{M1}\)(BT and MAE) ≥ 0.05 at matched size, CI over pairs × draws excluding 0. H7: M4−M1 gap for LeJEPA ≤ 0.03 (the map is close to orthogonal), larger for BT/MAE. Size trend (H8): \(\tilde r\) for seed pairs is non-decreasing S → B → L; with one seed at L, H8 is tested with S/B seed pairs plus S↔B and B↔L size pairs and reported descriptively. **Kill:** LeJEPA not better than both baselines (H6 fails) at every size.

### 2c. External pairs

| Pair | Downloadable | Dim | Licence | Use |
|---|---|---|---|---|
| GeoTessera v1.0 vs v1.1 | Yes, Zarr/NPY via `geotessera` | 128 / 128 | code MIT; embedding licence UNVERIFIED | BT-family version pair |
| v1.1 `dclimate` vs `cambridge` | Yes | 128 | UNVERIFIED | same version, separate inference runs; README says they do not interoperate |
| v2 `2B-L~beta1` vs `beta2` | Yes, coverage partial / on request (UNVERIFIED for UK) | 128 (Matryoshka prefixes 16/32/64/128) | UNVERIFIED | v2 variant pair |
| AlphaEarth V1 annual 2017–2024 | Earth Engine | 64, unit length | CC-BY-4.0 | year-to-year stability only, same model |

Sources: [GeoTessera README](https://github.com/ucam-eo/geotessera), [AlphaEarth catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL). Plan v2's quote "never mix embeddings from different versions" was not found verbatim; the README's wording is the variants sentence above. AlphaEarth embeddings lie on the unit sphere, so they cannot be N(0, I) and are not a test of the theory. Use the GeoTessera pairs as a baseline row ("how compatible are BT products in practice"); check embedding licence and UK coverage before using them.

## 3. Q3 protocol: S1 for S2 under cloud

### 3a. Paired tasks

From the Copernicus-FM paper (Fact, [2503.11849](https://arxiv.org/abs/2503.11849), Table 3/4; [Copernicus-Bench card](https://huggingface.co/datasets/wangyi111/Copernicus-Bench)). Copernicus-FM ViT-B/16, frozen encoder; classification: linear probe, SGD, batch 64, 50 epochs; segmentation: UPerNet, AdamW, batch 16, 50 epochs; mean of 3 runs.

| Task | Input | Metric | Images (approx.) | Licence | Copernicus-FM ViT-B (COPIED) | Δ = S2 − S1 |
|---|---|---|---|---|---|---|
| EuroSAT-S1 / -S2 | 2 / 13 bands, 64×64 | OA | 27k each, one-to-one paired | MIT | 87.2 / 97.9 | +10.7 |
| BigEarthNet-S1 / -S2 | 2 / 12 bands, 120×120 | mAP (multilabel) | ~24k each subset, paired | CDLA-Permissive-1.0 | 77.9 / 79.0 | +1.1 |
| DFC2020-S1 / -S2 | 2 / 13 bands, 256×256 | mIoU | ~5.1k, paired | CC-BY-4.0 | 52.4 / 64.5 | +12.1 |
| Flood-S1 (Kuro Siwo) | S1 only | mIoU (change) | see card | see card | not paired | — |
| Sen1Floods11 (PANGAEA) | S1 and S2 available | mIoU | 446 hand-labelled chips | [repo](https://github.com/cloudtostreet/Sen1Floods11) licence UNVERIFIED here | not reported; RERUN | RERUN |

Image counts and licences are from the Copernicus-Bench table as read; per-split counts should be re-checked from the card before freeze. The README lists 12/13 channels for BigEarthNet-S1/DFC2020-S1, which conflicts with the paper's 2-band S1; I take the paper (UNVERIFIED which is the typo). Note the processing-level mismatch: Copernicus-Pretrain S2 is L1C TOA, BigEarthNet-S2 is L2A (Fact, cards) — report it as a known shift.

**Δ reporting.** For higher-is-better metrics \(\Delta = s(S2) - s(S1)\) on chance-adjusted scores; for RMSE use \(\Delta = \mathrm{RMSE}(S1) - \mathrm{RMSE}(S2)\) so positive always means S2 better. Plot Δ against training FLOPs for each arm (A per-modality SIGReg, A′ pooled, single-sensor S1, single-sensor S2) with seed CIs. Hypothesis H9: for arm A, Δ decreases with compute and falls below Copernicus-FM's Δ on ≥ 2 of 3 paired tasks at ViT-L. Hypothesis H10 (kill rule from plan v2, made numeric): at B/8B/3 seeds, arm A's S1-only score exceeds single-sensor S1 LeJEPA by ≥ 2 pooled seed SDs on EuroSAT-S1 and Sen1Floods11, and is ≥ Copernicus-FM ViT-B (COPIED for EuroSAT-S1, RERUN under our light-decoder protocol for Sen1Floods11) on at least one. Compare protocol-matched numbers only: our light-decoder number is never compared with Copernicus-FM's UPerNet number.

### 3b. Theory-test variables

Available inside Copernicus-Pretrain (Fact, [card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain)): Copernicus DEM, S5P (CO, NO2, SO2, O3 products), S3 OLCI. Not inside: ERA5 values (only the 0.25° grid is shared), ESA WorldCover. Both must be fetched; their licences are UNVERIFIED here (I believe ERA5 is under the Copernicus licence and WorldCover is CC-BY-4.0; confirm). S5P at ~5.5 × 3.5 km and ERA5 at ~28 km are coarser than a 2.64 km image, so those targets are grid-cell level: many images share one value, and cross-validation must be grouped by 0.25° cell.

Variables: shared-physics (elevation, slope from DEM; water fraction and built-up fraction from WorldCover; ERA5 soil moisture) and optical-only (NDVI residual after regressing out WorldCover class fractions, S5P NO2). Raw NDVI is not "optical-only": SAR backscatter tracks vegetation structure, so raw NDVI would fail the test for the wrong reason.

**Gap test.** For variable v and embedding e: \(G_v = R^2_{\text{MLP}} - R^2_{\text{lin}}\), both from 5-fold spatially grouped CV, MLP = 2 hidden layers of 256, fixed. Pre-registered (Hypothesis H11): for shared variables on the S1 embedding of arm A, \(R^2_{\text{lin}} \ge 0.8\,R^2_{\text{lin}}(\text{S2 embedding})\) and \(G_v \le 0.05\); for optical-only variables on S1, \(R^2_{\text{lin}} \le 0.5\,R^2_{\text{lin}}(\text{S2})\). H12: across checkpoints of arm A, Spearman(G_v, residual cross-sensor alignment loss) ≥ 0.5 for shared variables. The 0.05 and 0.8 thresholds are my choices, set before data; change them only before freeze.

## 4. M ablation protocol

Strata: s ∈ {S2 clear, S2 cloud-masked, S1 only, S1+S2} (4 strata). Arms: no SIGReg, pooled, stratified, stratified without alignment. B, 8B tokens, 3 seeds each (12 runs; ~41 GPU-h, Estimate).

- **(i) Availability leakage.** Linear probe (logistic, balanced classes, 10k held-out samples, 5-fold) predicting stratum from the frozen embedding of the input as observed. Chance = 0.25 balanced accuracy. Cloud fraction: linear R², chance 0. Pre-registered pass: balanced accuracy ≤ 0.30 and cloud-fraction R² ≤ 0.10 for stratified; permutation test (1,000 label shuffles) gives the empirical chance band. Pooled is expected to be far above (Hypothesis). Note that "chance" is not reachable if stratum is visible in content (e.g. a cloud-masked image has masked tokens); the comparison that matters is stratified vs pooled, with a CI over seeds.
- **(ii) Missing-S2 robustness.** Drop in family score when S2 is removed or masked at test time, relative to full input. Pass: stratified drop ≤ pooled drop − 2 pooled seed SDs on ≥ 2 families.
- **(iii) Per-stratum Gaussianity.** Epps–Pulley on 2,048 held-out samples per stratum, 1,024 random directions. Threshold from a Monte-Carlo null (N(0, I) with the same N, K): pass if the median statistic is below the null's 95th percentile and ≤ 10% of directions reject at 5% (null expectation 5%). Also report the between-stratum mean distance \(\lVert\mu_s - \mu_{s'}\rVert\) in units of \(\sqrt{K/N}\).
- **(iv) Clean-input regression.** Seed SD σ pooled over the 3 seeds and the 4 arms per task. Pass: stratified clean-input score ≥ pooled − 1σ on every family mean and ≥ pooled − 2σ on every single task. With 3 seeds σ is poorly estimated; bootstrap the test set too and report the CI.

Claim M holds only if (i), (ii) pass and (iv) holds; (iii) is mechanism evidence.

## 5. Harness decision

| | Copernicus-Bench | PANGAEA | GEO-Bench | OpenChange-UK |
|---|---|---|---|---|
| Families | cls, seg, change, regression (S3/S5P only) | seg, change, regression, multi-temporal | 6 cls, 6 seg | change (+ others, UNVERIFIED) |
| Paired S1/S2 single-sensor tasks | EuroSAT, BigEarthNet, DFC2020 | multi-modal inputs (Sen1Floods11, PASTIS-R, CropTypeMapping, BioMassters), not split into S1-only/S2-only tasks | none; m-so2sat stacks S1+S2 | S1/S2 dates (UNVERIFIED) |
| Licences | per-dataset, permissive (MIT, CDLA, CC-BY) | per-dataset | mostly permissive, one CC-BY-SA | ours |
| Out-of-scope sensors | S3, S5P | many high-res RGB tasks | RGB high-res tasks | — |
| Decoder protocol | linear / UPerNet | UPerNet for all | linear / fine-tune, ≥ 10 seeds recommended | ours |

Sources: [Copernicus-FM](https://arxiv.org/abs/2503.11849), [PANGAEA](https://arxiv.org/abs/2412.04204), [GEO-Bench](https://arxiv.org/abs/2306.03831).

**Recommendation.** Primary suite: Copernicus-Bench {EuroSAT-S1/S2, BigEarthNet-S1/S2, DFC2020-S1/S2, Flood-S1, LCZ-S2, Cloud-S2} + PANGAEA {Sen1Floods11, BioMassters} + OpenChange-UK frozen splits. GEO-Bench is not the primary harness: it has no paired single-sensor tasks and much of it is RGB high-resolution. Add GEO-Bench m-bigearthnet and m-so2sat as an external comparability row, because the GFM audit asks for widely shared benchmarks.

**Eval cost per checkpoint (Estimate).** Feature extraction is cheap: ~20M patch tokens across the suite × 2N FLOPs ≈ 3.5e15 at ViT-B, under 1 minute on one H100. Cost is probe fitting: linear + kNN on cached features ≈ 0.2 GPU-h; light decoder on cached patch tokens, 4 seg/change tasks × 3 LRs × 1 seed ≈ 1 GPU-h at B, ~1.5 at L. Fast suite ≈ 1.2–1.7 GPU-h per checkpoint. Full Copernicus-FM-matched protocol (UPerNet 50 epochs, 3 runs) for comparison with copied numbers: UNVERIFIED cost, my guess 10–20 GPU-h per checkpoint; run it only on ~30 final checkpoints.

Checkpoint count: population W 192 + 72 (ViT-B check) = 264; LeJEPA grid 63 + 1B; BT/MAE grid 2 × 63 if run in full (else ~24); DINOv2-EMA 3; arms A/A′/S1/S2/M ≈ 30; within-run curves 10 × ~90 runs on the cheap kNN+linear suite only. Fast suite on ~520 checkpoints ≈ 700 GPU-h; within-run curves ≈ 180 GPU-h; full protocol on 30 ≈ 450 GPU-h. **Total eval ≈ 1,300 GPU-h ≈ 330 node-hours (Estimate)**, more than pretraining.

**GFM audit checklist** ([2605.12678](https://arxiv.org/abs/2605.12678)), applied:
- [ ] Every baseline number marked COPIED (with table/page) or RERUN (with config).
- [ ] Weights, code and evaluation harness released with a version tag and licence; checkpoint licence compatible with CC-BY-4.0 data.
- [ ] Core shared benchmarks included with their standard protocol (EuroSAT, BigEarthNet, So2Sat via GEO-Bench row).
- [ ] Uncertainty: mean ± SD over seeds, bootstrap CI over test samples; number of seeds stated per row.
- [ ] Data controls: pretraining/test geographic overlap checked and reported; deduplicated grid cells listed.
- [ ] Hyperparameter search budget per method stated and equal between our model and rerun baselines.
- [ ] Frozen-probe and fine-tuning results kept in separate tables.

## 6. Compute pricing

**Formula.** Per processed token, training ≈ \(6N + 12\,L\,T\,d\) FLOPs (6N for the parameter matmuls forward+backward, \(12LTd\) for attention scores and value mixing; Kaplan-style accounting, [2001.08361](https://arxiv.org/abs/2001.08361)), N = non-embedding parameters, L layers, d width, T tokens per sequence.

| Model | d, L | N | FLOPs/token, T = 102 | T = 256 | T = 1024 |
|---|---|---|---|---|---|
| ViT-S | 384, 12 | 21.7M | 1.36e8 | 1.44e8 | 1.87e8 |
| ViT-B | 768, 12 | 86M | 5.26e8 | 5.43e8 | 6.28e8 |
| ViT-L | 1024, 24 | 303M | 1.85e9 | 1.89e9 | 2.12e9 |
| ViT-g (1B) | 1408, 40 | 1.0B | 6.13e9 | 6.23e9 | 6.75e9 |

**Tokens per sample (Fact + Estimate).** S2 is 264 × 264 × 13, S1 264 × 264 × 2 ([card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain)). Crop to 256 × 256 → 16 × 16 = 256 patches. One joint S2 patch embedding: 256 tokens. Per-band-group (e.g. 4 S2 groups + 1 S1 group): 1,280 tokens. Per band (13 + 2): 3,840. At 0.9 token drop, the band-group case leaves ~128 visible per view. I price with T = 102 visible tokens per view (1,024 × 0.1); attention is a small share at these lengths, so the result moves less than 15% between T = 102 and T = 1,024.

**Budgets are processed encoder tokens, all views included.** At H100 dense BF16 peak 989 TFLOP/s ([NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)) and **35% MFU (UNVERIFIED; I found no primary measured ViT MFU on GH200; small ViTs at ~100 tokens per sequence will likely run lower, so scale every number by 0.35/MFU):**

| Size | 2B | 8B | 32B | GPU-h per (data scale, seed) with all 3 budgets as separate runs |
|---|---|---|---|---|
| S | 0.2 | 0.9 | 3.5 | 4.6 |
| B | 0.8 | 3.4 | 13.5 | 17.7 |
| L | 3.0 | 11.9 | 47.5 | 62.4 |
| 1B | 9.8 | 39.3 | 157 | — |

- LeJEPA grid (S, B × 3 seeds, L × 1, × 3 data × 3 budgets): 41 + 159 + 187 ≈ **390 GPU-h ≈ 98 node-hours**; with WSD cooldowns instead of separate runs, ~0.45×.
- BT/MAE at 4 shared cells (S, B × 8B, 32B × 3 seeds): ~65 GPU-h per objective; full matched grid ~390 each.
- Population W sweeps: ~285 GPU-h. Arms A/A′/S1/S2/M at B/8B × 3 seeds: ~70 GPU-h.
- 1B pre-registered run at 32B: **157 GPU-h ≈ 39 node-hours** (~41 h on 4 GPUs, one 2-job chain).
- Total pretraining with matched BT/MAE grids: ~1,700 GPU-h; with eval (§5) ~3,000 GPU-h ≈ **750 node-hours = 3.8% of 20,000** (allocation UNVERIFIED; fraction = 750/A for allocation A). Add 50% for failures and restarts: ~5.6%.

**I/O bound.** Samples/s required per node = \(4 \cdot 0.35 \cdot 989\text{e}12 / (V\,T\,F_{\text{tok}})\), V = 2 views. Bandwidth needed = that × X MB. A node supplying Y GB/s caps throughput at \(1000\,Y/X\) samples/s.

| Size | samples/s/node (compute) | GB/s at X = 0.15 MB | at X = 1.1 MB |
|---|---|---|---|
| S | 50,000 | 7.5 | 55 |
| B | 12,900 | 1.9 | 14 |
| L | 3,700 | 0.55 | 4.0 |
| 1B | 1,100 | 0.17 | 1.2 |

So with token dropping, ViT-S and ViT-B are I/O-bound at any plausible Y unless shards are band-packed (X ≈ 0.15 MB) and Y ≥ 2–8 GB/s; L and 1B are not. Isambard's sustained Lustre bandwidth per node is UNVERIFIED ([storage docs](https://docs.isambard.ac.uk/user-documentation/information/system-storage/) give capacities only: 200 TiB project, 5 TiB scratch, 48 GiB local). 32B tokens at ~204 tokens per sample = 157M sample reads = 23 TB at 0.15 MB or 172 TB at 1.1 MB. Raw Copernicus-Pretrain S1+S2 at ~1.1 MB × 18.7M ≈ 20 TB fits project storage. Mitigation: multiple crops/views per read (cuts reads by the crops-per-read factor) and measure Y in the 1-GPU smoke.

## 7. Scaling-law fitting

**Target.** Chance-adjusted family error \(e = 1 - \tilde s\), per family, ID and OOD separately.

**Forms (pick by rule below, before the 1B run):**
1. Saturating power law in compute: \(e(C) = e_\infty + A\,C^{-a}\).
2. Two-variable Chinchilla form: \(e(N,D) = E + A N^{-\alpha} + B D^{-\beta}\) ([2203.15556](https://arxiv.org/abs/2203.15556)), fit by Huber loss (δ = 1e-3) on log e with L-BFGS from a grid of initialisations, as in Chinchilla.
3. Broken power law with one break, as a check for a regime change.
4. Logit-linear: \(\operatorname{logit}\tilde s = c_0 + c_1\log C\), for bounded scores with no plateau yet.

**Points.** ≥ 5 sizes (Ti 5.7M, S 22M, ~40M, B 86M, L 303M), ≥ 5 budgets via WSD cooldowns (1, 2, 4, 8, 16, 32B; [Hägele et al.](https://arxiv.org/abs/2405.18392)) at 100% data, plus the 3 data scales at B. Iso-FLOP slices: 3 slices (3e17, 1e18, 3e18 FLOPs) with ≥ 4 sizes each. The cheap small-model points cost little (Ti/S/~40M at all budgets < 30 GPU-h, Estimate).

**Selection rule (pre-registered).** Fit each form on all points except the largest size (L); choose the form with the lowest error predicting the L points; refit on all points ≤ L. This uses L as a validation set and keeps 1B as the test set. RoboJEPA ([2610.10515](https://arxiv.org/abs/2610.10515)) holds out larger runs, but its candidate forms were compared on the held-out runs themselves; we do not.

**Uncertainty.** Bootstrap 2,000 resamples of runs (configurations with their seeds), refit each; report percentile CIs of exponents and of the 1B prediction; leave-one-iso-FLOP-slice-out refits as a stability check (TESSERA v2 uses leave-one-bucket-out, [2607.03949](https://arxiv.org/abs/2607.03949)).

**1B pre-registration.** Before the 1B run starts, post (timestamped OSF) the predicted \(\tilde s\) per family and the 95% bootstrap prediction interval. Acceptance band = PI widened by seed noise: \(\lvert\tilde s_{\text{obs}} - \hat s\rvert \le \sqrt{h_{\text{PI}}^2 + (2\sigma_{\text{seed},L})^2}\). If \(h_{\text{PI}} > 0.05\) on a family, declare the extrapolation uninformative for that family before training. If downstream error is flat (fitted a CI including 0), report that.

## 8. Pre-registration template (OSF-style)

**Title / authors / date / version.** Code commit hash of the analysis repository at freeze.

**Data.** Copernicus-Pretrain (CC-BY-4.0) deduplicated by 0.25° grid cell; 1/10/100% subsets as nested cell lists (files hashed). Proxy set \(\mathcal{P}\), selection split, ID test sets, OpenChange-UK frozen splits: cell/date lists hashed and posted. Geographic overlap between pretraining cells and every test set computed and posted.

**Grid.** As §6 tables: sizes, budgets, data scales, seeds, objectives, arms, population-W Sobol design (seed of the Sobol generator posted).

**Harness.** §5 suite, probe hyperparameter grids, light decoder spec, seeds, metric definitions, chance levels; evaluation harness version tag.

**Hypotheses (directional, with thresholds).**
- H1 LeJEPA loss ranks runs: ρ ≥ 0.7, lower CI ≥ 0.5, ≥ 3/4 families, W population, S and B.
- H2 Matched contrast: ρ_LeJEPA − ρ_BT ≥ 0.3, CI excludes 0, ≥ 3/4 families.
- H3 OOD: lower CI ≥ 0.4 on OpenChange-UK for families passing H1.
- H4 Loss vs 1k-label probe: ρ_P1 ≥ ρ_P8 − 0.1.
- H5–H8 Q2: §2b thresholds.
- H9–H12 Q3: §3 thresholds.
- H13 M: §4 (i), (ii), (iv) pass thresholds.
- H14 1B prediction inside the acceptance band per family.

**Kill rules.** Q1 pilot kill (§1d); Q3 pilot kill (H10); M dropped if (iv) fails; scaling claim dropped if the selection rule yields no form with L-validation error below a constant-in-log baseline.

**Analysis freeze.** All analysis scripts (proxy computation, bootstrap, fits) frozen and hashed before the first population-W result is opened; any change after freeze is logged as a deviation with date and reason.

**Reported regardless of outcome.** Every run, including failed and diverged runs (with cause); every seed; every proxy × family × split cell; in-sample LeJEPA-style \(C(\alpha)\) next to the held-out α result; COPIED/RERUN status; GPU-hours used per component; all deviations.

## 9. Closest prior work

| Title | URL | What it does | What it does not do |
|---|---|---|---|
| LeJEPA | https://arxiv.org/abs/2511.08544 | SIGReg objective; loss vs linear-probe Spearman ~0.85 within hyperparameter sweeps; λ^α rescaling | No EO; α fit in-sample; no matched objective comparison; no OOD |
| TESSERA v2 | https://arxiv.org/abs/2607.03949 | 395 BT runs on EO; loss vs 15-task composite ρ = −0.16; iso-FLOP scaling fits; out-of-sample check on 0.5/1/2B | Not SIGReg; correlation pooled across sizes; no within-setting sweep; not pre-registered |
| RankMe | https://arxiv.org/abs/2210.02885 | Label-free effective-rank proxy for SSL selection | No EO; no matched-loss comparison |
| LiDAR | https://arxiv.org/abs/2312.04000 | LDA-based effective rank proxy for JE models | No EO; no SIGReg |
| α-ReQ | https://doi.org/10.52202/068431-1281 | Eigenspectrum-decay proxy | Contents not read here (UNVERIFIED) |
| Copernicus-FM / Copernicus-Bench | https://arxiv.org/abs/2503.11849 | Multi-sensor FM; paired S1/S2 benchmark tasks with ViT-B numbers | No S1-for-S2 substitution study; no loss–downstream analysis |
| PANGAEA | https://arxiv.org/abs/2412.04204 | GFM benchmark, UPerNet protocol, multi-modal tasks | No paired single-sensor tasks; heavy decoder per checkpoint |
| GEO-Bench | https://arxiv.org/abs/2306.03831 | 12-task EO benchmark, seed protocol | No S1/S2 pairing; much RGB high-res |
| GFM evaluation audit | https://arxiv.org/abs/2605.12678 | 152-paper audit; reporting recommendations | Not a benchmark; no proxy analysis |
| Backward-compatible representations | https://arxiv.org/abs/2003.11942 | Trains new model against old for compatibility | Requires access to old model at training; no post-hoc map from objective |
| Procrustes alignment bounds | https://arxiv.org/abs/2510.13406 | General bounds for post-hoc orthogonal alignment | No link to an SSL objective; no EO |
| LeJEPA identifiability | https://arxiv.org/abs/2605.26379 | Recovery up to orthogonal map under assumptions | No empirical cross-checkpoint test; no size dependence |
| RoboJEPA | https://arxiv.org/abs/2610.10515 | Held-out larger-run scaling prediction for JEPA | Robotics; form compared on held-out runs |
| WSD / beyond fixed durations | https://arxiv.org/abs/2405.18392 | Cooldown branches give multiple budgets from one run | Language models; not SSL |
| CR-JEPA, HQ-JEPA, Le MuMo JEPA | https://arxiv.org/abs/2606.00706, https://arxiv.org/abs/2605.31068, https://arxiv.org/abs/2603.24327 | SIGReg across sensors (from plan v2; not re-read here) | Per plan v2: no scaling, no selection/compatibility tests (UNVERIFIED by me) |

**UNVERIFIED list:** GH200 ViT MFU; Isambard per-node Lustre bandwidth; α-ReQ formula details; GeoTessera embedding licence and v2 UK coverage; ERA5 and WorldCover licences; Sen1Floods11 licence; BigEarthNet-S1/DFC2020-S1 channel count conflict between README and paper; OpenChange-UK task families; full-protocol eval cost; u6xn allocation size.
