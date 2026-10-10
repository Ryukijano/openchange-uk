# Clean-JEPA on Earth observation, plan v2

Date: 2026-10-10. Supersedes CLEAN_JEPA_PLAN.md. Built from three independent reviews run today (desk research only, nothing trained):
sharpener (methods and experiments), disruptor (applied ideas), red team (novelty, claims, feasibility). Their full reports: subagent-sharpener.md, subagent-disruptor.md, subagent-redteam.md. Numbers below are estimates unless marked measured. Nothing here has been run.

## 1. What changed since v1

The three reviews agree on these points, so I treat them as settled for planning:

1. **No LeJEPA/SIGReg scaling study on EO exists, and nobody has tested LeJEPA's loss-predicts-downstream claim on EO.** The gap is real but narrow.
2. **T1's mechanics are taken.** CR-JEPA ([2606.00706](https://arxiv.org/abs/2606.00706)), HQ-JEPA ([2605.31068](https://arxiv.org/abs/2605.31068)) and Le MuMo JEPA ([2603.24327](https://arxiv.org/abs/2603.24327)) already apply SIGReg across S1/S2 or RGB+other sensors. "Cross-sensor LeJEPA" is not a method contribution.
3. **T2's mechanics are taken and its framing is unsafe.** T-JEPA ([2610.05731](https://arxiv.org/abs/2610.05731), 5 Oct), Earth-JEPA (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=CTH0mXU2lj)) and a SIGReg latent world model on EarthNet2021 S2 ([2607.13651](https://arxiv.org/abs/2607.13651)) cover temporal latent prediction on satellite data. *One-Step Next-Latent Prediction Is Not a World Model* ([2609.36227](https://arxiv.org/abs/2609.36227)) shows one-step latent regression identifies a conditional mean, not a rollable kernel, and that an isotropy penalty has zero gradient on transition weights. We do not claim an "EO world model".
4. **T3 is an evaluation axis, not a contribution.** Held-out UK regions and future dates (OpenChange-UK, frozen splits) are where every claim below is tested out of distribution.
5. **FLOPs are not the constraint; bytes are.** The 4 sizes x 3 data scales grid costs about 1,800 node-hours without token dropping (about 12% of 15,000), up to 35-50% with per-band-group tokenisation. Reading pixels (about 1.1 MB per Copernicus-Pretrain sample, about 1 PB per 1B-sample run) is the biggest cost. Scratch is 5 TiB; data must live in $PROJECTDIR (200 TiB). Source: red-team section 3, [Isambard storage docs](https://docs.isambard.ac.uk/user-documentation/information/system-storage/).
6. **Licences.** Copernicus-Pretrain is CC-BY-4.0 (checked on the HF card today). Major TOM Core-S2L2A is CC-BY-SA-4.0 (ShareAlike; fine for a paper, matters for derived products). SSL4EO-S12 v1.1 is CC-BY-4.0.

Two corrections to v1: LeJEPA's loss claim is not "one plot": it reports Spearman about 0.85 across datasets and architectures (0.99 after a lambda rescaling), within-family sweeps, linear probe only. And TESSERA v2's figure is r = -0.18 with Barlow Twins over 395 runs.

## 2. The sharpened paper

**Working title.** *What a clean JEPA guarantees on Earth observation: label-free selection, version compatibility and sensor substitution from 20M to 1B parameters.*

**One sentence.** LeJEPA's theory (Gaussian latents, linear identifiability) makes three promises that would change how EO embedding products are built and used; we test all three on one controlled grid of checkpoints, against matched Barlow Twins and MAE grids, in and out of distribution.

The three promises are the applied part. Each is a decision somebody makes today without evidence:

### Q1. Label-free model selection (headline; resolves a contradiction)
*Decision changed:* which checkpoint, model size or data mix to keep, without labels. TESSERA v2 says choosing by loss "wastes a large share of the compute". LeJEPA says loss ranks models. Nobody has tested which holds on EO or why.
*Hypothesis:* on matched data and evaluation, the SIGReg-regularised LeJEPA loss ranks checkpoints by downstream utility (Spearman > 0.7, bootstrap CI) where the Barlow Twins loss does not (|r| < 0.3), and the ranking survives the OOD UK split.
*Design:* LeJEPA vs Barlow Twins vs MAE at 3-4 shared grid cells. Label-free proxies compared head to head: raw loss, lambda-rescaled loss with alpha fixed on a held-out split in advance, SIGReg statistic alone, prediction term alone, RankMe, LiDAR, and a 1k-label probe as the paid alternative. Correlations reported per task family (classification / segmentation / regression / change) with bootstrap CIs, in-distribution and on OpenChange-UK.
*Why it is more than a proxy comparison:* the matched-objective design isolates whether the Gaussian constraint is what makes the loss informative. Mechanism analysis (effective rank, Epps-Pulley statistic, probe accuracy over training) is part of the result.
*Kill:* if no proxy, including LeJEPA loss, reaches Spearman 0.5 on the pilot grid (S, B at three token budgets, 3 seeds), the headline becomes "label-free selection fails on EO for all SSL families", which is still publishable and still useful.

### Q2. Version compatibility (new; from the identifiability theorem)
*Decision changed:* whether a new embedding version forces every downstream classifier to retrain and every archive to be re-embedded. GeoTessera docs: "never mix embeddings from different versions". AlphaEarth's global mosaic is about 4 PB.
*Theory:* [2605.26379](https://arxiv.org/abs/2605.26379) says LeJEPA recovers true latents up to an orthogonal map under stationarity, independence and additive noise. If that holds even approximately on EO, two independently trained clean JEPAs (different seed, size or data scale) differ by a rotation.
*Hypothesis:* a Procrustes map fitted on 5k calibration pixels transfers a frozen probe from checkpoint A to checkpoint B with > 90% of B's native accuracy for LeJEPA pairs, and the retained fraction is higher than for MAE and Barlow Twins pairs at matched size. Prediction from theory: retained fraction increases with model size and is highest for same-size pairs.
*Design:* every pair in the grid (all three objectives), plus TESSERA v1.1 vs v2 as an external pair. Report retained fraction vs size, seed and data scale, in and out of distribution.
*Cost:* inference and small probes only; uses checkpoints Q1 already produces.
*Kill:* LeJEPA pairs no more alignable than baseline pairs at any size.
*Closest prior:* backward-compatible training needs the new model trained against the old ([Shen et al. CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)); Procrustes post-hoc alignment has general bounds but no claim tied to an SSL objective ([2510.13406](https://arxiv.org/html/2510.13406v1)). The reviews found nothing testing cross-version compatibility as a property of the training objective.

### Q3. Sensor substitution under cloud (the surviving part of T1)
*Decision changed:* whether an optical-trained product can be run from SAR when Sentinel-2 is cloudy, and how much is lost. The UK coast is cloudy most of the year.
*Hypothesis:* for a shared-encoder LeJEPA with **per-modality SIGReg** (S1 and S2 embeddings each constrained to N(0, I), not the pooled batch), the gap Delta = score(S2 input) - score(S1 input) on paired tasks shrinks as a power law in compute, and the S1-only score at ViT-L beats frozen Copernicus-FM ViT-B. Pooled SIGReg can be satisfied by a mixture with a modality gap (split by the sign of one coordinate), so per-modality SIGReg is the condition the theorem's stationarity assumption actually needs. This argument is the sharpener's and is not in prior work; it must be stated as ours and tested.
*Theory test (free ground truth):* variables both sensors observe (DEM elevation, open-water and built-up fraction from WorldCover, ERA5 soil moisture) should be *linearly* recoverable from the S1 embedding; optical-only variables (NDVI, S5P NO2) should not. Per variable, MLP-probe R2 minus linear-probe R2 should grow with the residual cross-sensor alignment loss. DEM and ERA5 alignment are already in Copernicus-Pretrain.
*Design:* arm A (S1->S2, per-modality SIGReg) vs A' (pooled, B only) vs single-sensor S1 and S2 LeJEPAs; Delta on EuroSAT-S1/S2 and BigEarthNet-S1/S2 (Copernicus-Bench), S1-only mIoU on Sen1Floods11, OpenChange-UK cloudy dates. Drop S3 and S5P as translation targets (footprint mismatch; S5P grid UNVERIFIED).
*Kill:* at the pilot (B, 8B tokens, 3 seeds) arm A's S1-only probe does not beat the S1-only LeJEPA by 2 seed SDs on both EuroSAT-S1 and Sen1Floods11, or is below frozen Copernicus-FM ViT-B on both.

### M. Method contribution: availability-stratified SIGReg
EO batches mix availability patterns: S2 clear, S2 cloud-masked, S1 only, both. Pooled SIGReg is satisfied by a Gaussian mixture whose components separate by pattern, so the embedding can encode "which sensors were present and how cloudy it was". Shared Gaussianization ([2610.10299](https://arxiv.org/abs/2610.10299)) shows a Gaussianity test can even be lowered by adding per-view nuisance; SAR speckle is such a nuisance, so lambda is not free.
*Proposal:* stratify batches by availability mask (>= 256 samples per stratum, because LeJEPA's Epps-Pulley gradient bounds scale as 1/N), apply SIGReg within each stratum, align every stratum to the full-observation global view through the invariance term. Target condition: embedding is N(0, I) conditional on the mask, hence independent of it. A few lines inside the reference SIGReg loop ([galilai-group/lejepa MINIMAL](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)).
*Ablation (B, 8B tokens, 3 seeds, real S2 cloud masks):* no SIGReg / pooled / stratified / stratified without alignment. Measures: (i) linear probe for availability pattern and cloud fraction from the frozen embedding, should fall to chance; (ii) downstream drop when S2 is removed or masked at test time, should shrink; (iii) per-stratum Epps-Pulley on held-out data; (iv) clean-input scores must not regress beyond seed noise. Claim holds only if (i) and (ii) improve and (iv) holds.
*Cheaper second ablation:* spectral-band dropping on top of spatial token dropping vs LeVJEPA's uniform spatial dropping at rho in {0.75, 0.9, 0.95}.

### Demoted, not dropped
- **Temporal (old T2):** temporal pairs enter as one *positive-view* ablation row (this is the design the identifiability theorem is actually about: stationary transitions), plus a linear k-step embedding-forecast probe. No world-model claim. Note Copernicus-Pretrain's per-location time depth is UNVERIFIED; SSL4EO-S12 has 4 seasonal snapshots.
- **Calibrated change alerts from latent-forecast residuals, kept running under cloud by SAR** (disruptor idea 2): strongest applied idea for a *second* paper; needs the Q3 encoder first and a decision-metric protocol (lead time, false-alarm rate per season and land cover) against a RADD-style radar baseline.
- **Kalman filtering in the Gaussian latent space** (disruptor idea 5): high novelty, likely clean negative on seasonal data; keep as a probe, not a claim.
- **Continuity across Sentinel-2A/2B/2C** (disruptor idea 4): tandem-phase acquisitions give a free, sharp test of embedding stability across satellite units; cheap inference-only appendix if the data checks out.

## 3. Grid and budget (estimates)

| Axis | Values | Note |
|---|---|---|
| Sizes | ViT-S 22M, B 86M, L 300M, then one pre-registered 1B run | 1B run's downstream score is predicted from the fit before it is trained (RoboJEPA-style) |
| Visible-token budgets | 2B, 8B, 32B | fixed after the pilot |
| Data | 1%, 10%, 100% of **deduplicated grid cells** (unique locations); epochs is a separate axis | answers the "data scale = repetition" objection |
| Seeds | 3 at S and B, 1 at L and 1B | |
| Objectives | LeJEPA (full grid), Barlow Twins and MAE at 3-4 shared cells, DINOv2-style EMA at B | matched data, matched eval |
| Arms | A per-modality SIGReg S1->S2, A' pooled, single-sensor S1 and S2, M stratified, temporal-positive row | |
| Eval | Copernicus-Bench paired S1/S2 tasks + a PANGAEA subset (frozen linear + kNN + one fixed light decoder) + OpenChange-UK frozen UK splits; fine-tuning on a subset | pre-registered before any pretraining result |

About 45-60 pretraining runs. FLOPs: about 1,800 node-hours without token dropping; 35-50% of 15,000 with per-band tokenisation; token dropping at 0.9 cuts encoder FLOPs 10-20x but not bytes read. Largest cell (1B x 1B samples) about 3,200 GPU-h, about 100 h on 32 GPUs, i.e. 5 chained 24 h jobs. All numbers are planning arithmetic from published FLOP counts at an assumed 35-45% MFU, not measurements.

**I/O plan (the real risk):** pre-tokenise once into uint16 band-packed shards (target < 150 KB per sample) in $PROJECTDIR; dedupe by grid cell; measure read throughput in the first 1-GPU smoke before pricing anything; if shared-storage bandwidth cannot feed about 5 GB/s per node, cap the data axis rather than the model axis.

## 4. Reviewer objections and the answers built in

- *"TESSERA v2 with a different loss; T1/T2 already done."* Answered by matched objectives on identical data and eval at shared grid cells, mechanism analysis, and Q2/Q3 being tests of the theory nobody has run.
- *"Your downstream is a choice."* One pre-registered harness, per-task-family correlations with bootstrap CIs, fine-tuning subset, every checkpoint and the eval code released. The 152-paper GFM audit ([2605.12678](https://arxiv.org/abs/2605.12678)) is the reason.
- *"Three points cannot support a scaling law."* Deduplicated data axis, >= 4-5 points on two iso-FLOP slices, bootstrapped CIs, and one pre-registered extrapolation. If downstream is flat over 20M-1B, that is the finding.
- *"Identifiability assumptions do not hold on EO."* Agreed; Q2/Q3 are tests of whether the guarantees survive approximately, with baselines that decide the paper either way.

## 5. Local pilots before any Isambard hours (this VM, CPU, no cluster)

| # | Pilot | Decides | Size |
|---|---|---|---|
| P0 | Source approval + 500-sample SSL4EO-S12 v1.1 slice (S1/S2 aligned, 4 seasons) through the OpenChange-UK source gate; verify alignment and cloud-mask availability | data legality and pipeline | <= 500 samples |
| P1 | LeJEPA from the MINIMAL reference + per-modality SIGReg + stratified SIGReg; two tiny models (5M, 20M), 3 seeds | code correctness; stratified loop works | hours of CPU |
| P2 | Procrustes compatibility harness on the P1 seed pairs; Q2 metric end to end | Q2 harness | minutes |
| P3 | Proxy harness: raw loss, rescaled loss, SIGReg stat, RankMe, LiDAR; two paired Copernicus-Bench tasks + one OpenChange-UK task loader | Q1 harness | hours |
| P4 | Bytes-per-sample after band-packing; CPU decode throughput | I/O plan | minutes |
| P5 | Pre-registration draft: hypotheses, grid, harness, kill rules, frozen UK splits | protocol freeze | writing |

Only after P0-P5: a 30-minute 1-GPU smoke on Isambard (human-run, documented NGC image, Apptainer --nv), then the S/B pilot grid, then the M2-style priced report before the main grid.

## 6. Open items

- Meaning of "translation": now operationalised as **sensor substitution (Q3)** and **version compatibility (Q2)**. Confirm.
- u6xn award wording: does it name Earth observation? Remaining balance, units, end date, storage quota.
- Primary harness: Copernicus-Bench paired tasks + PANGAEA subset + OpenChange-UK (recommended), vs GEO-Bench.
- UNVERIFIED: throughput and MFU figures; Copernicus-Pretrain per-location time depth; S5P grid; Isambard Lustre bandwidth; Earth-JEPA (under review, may change); whether identifiability assumptions hold approximately on EO.
- Disagreement between reviews, noted: the disruptor scored label-free selection low on novelty (RankMe/LiDAR exist); the red team made it the headline. Resolution above: the headline is the matched-objective contradiction plus OOD, and the novelty sits in Q2/Q3/M, not in the proxy comparison alone.
