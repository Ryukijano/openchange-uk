 # Applied, method-novel ideas for a clean-JEPA EO study

Date: 2026-10-09. Desk research only; nothing was trained or run. "Clean JEPA" means the LeJEPA family as in the plan ([LeJEPA](https://arxiv.org/abs/2511.08544), [LeVJEPA](https://arxiv.org/abs/2608.27395), [identifiability](https://arxiv.org/abs/2605.26379)). Scores below are my judgement on a 1-5 scale, not measurements.

## Two new papers that change the plan

- **T-JEPA (5 Oct 2026)** is a temporal JEPA for remote sensing. It learns latent transitions conditioned on the time gap, adds masked pixel reconstruction, and reports the best transfer at matched data and throughput ([arXiv:2610.05731](https://arxiv.org/abs/2610.05731)). As a representation-learning method, T2 ("predict the next acquisition's embedding") is now taken. T-JEPA is not heuristics-free (it has a reconstruction branch), and its abstract mentions no scaling study and no decision-scored forecasting. What remains open in T2 is what the forecasts are used for (idea 2), not the training objective.
- **S2S-JEPA (2 Oct 2026)** applies JEPA to subseasonal weather forecasting and reports skill comparable to the ECMWF ensemble ([arXiv:2610.03106](https://arxiv.org/abs/2610.03106)). "JEPA forecasts only the predictable part" is now a published argument in Earth-system forecasting.

Two facts that set the bar for idea 3: LeJEPA reports a Spearman correlation of about 0.85 between training loss and linear-probe accuracy "across datasets and architectures" ([LeJEPA HTML, sec. 6.2](https://arxiv.org/html/2511.08544)). TESSERA v2 found |Pearson r| < 0.2 for its Barlow Twins loss and says that "selecting models by loss wastes a large share of the compute" ([arXiv:2607.03949](https://arxiv.org/html/2607.03949)).

## Ranking

| # | Idea | Benefit | Novelty | Feasibility | Product |
|---|---|---|---|---|---|
| 1 | Embedding products that survive model upgrades (linear identifiability as a compatibility guarantee) | 4 | 4 | 5 | 80 |
| 2 | Calibrated change alerts from latent-forecast residuals, kept running under cloud by SAR | 5 | 4 | 3 | 60 |
| 3 | Label-free checkpoint and region selection, compared head to head | 5 | 2 | 5 | 50 |
| 4 | Continuity across satellite units (S2A to S2C, S1B loss to S1C/S1D) | 4 | 3 | 4 | 48 |
| 5 | Kalman filtering in a Gaussian latent space for multi-sensor fusion | 4 | 5 | 2 | 40 |
| 6 | SAR-conditioned optical latents scored by decisions, with abstention | 4 | 2 | 4 | 32 |
| - | Flagged: compute-optimal small-lab recipe, counterfactual queries, pretraining-data curation | see below | | | |

Ideas 1, 3 and 4 need only inference and small probes on the checkpoints the planned scaling grid already produces. They add almost no compute and turn one scaling paper into a paper about properties of an embedding product that users can rely on.

---

## 1. Embedding products that survive model upgrades

**User and decision.** Users of embedding products such as [TESSERA](https://geotessera.readthedocs.io/en/stable/index.html) and [Google Satellite Embedding / AlphaEarth](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme), and agencies that re-run national maps each year. UKCEH builds each new Land Cover Map partly from the previous one ("bootstrap training") ([UKCEH](https://www.ceh.ac.uk/our-science/projects/uk-scape-land-cover-map)). The decision is whether a new embedding version forces every downstream classifier to be retrained and every archive re-embedded. Today it does: the GeoTessera docs say "Never mix embeddings from different versions... the 128-channel feature spaces are independently learned and not interchangeable" ([GeoTessera docs](https://geotessera.readthedocs.io/en/stable/index.html)). The AlphaEarth global int8 mosaic is listed at about 4 PB ([Wherobots](https://wherobots.com/blog/alphaearth-foundations-change-detection-part-2/)), so re-embedding is expensive.

**Method novelty.** The identifiability result says LeJEPA recovers the true latents up to a linear (orthogonal) map in a class of worlds ([arXiv:2605.26379](https://arxiv.org/abs/2605.26379)). If that holds even approximately on EO data, two independently trained clean JEPAs (different size, seed or data scale) should differ by an orthogonal rotation. One Procrustes fit on a few thousand calibration pixels would then carry every downstream probe from version k to version k+1. This turns a theorem into a testable promise for an embedding product.

**Closest prior work.** Backward-compatible training (BCT) needs the new model to be trained against the old one ([Shen et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)). Procrustes post-processing comes with general bounds but no claim that any SSL objective makes it work ([arXiv:2510.13406](https://arxiv.org/html/2510.13406v1)). TESSERA v2 measures Procrustes/CKA alignment between its own student and teacher, which is a different question from alignment between independent versions ([TESSERA v2](https://arxiv.org/html/2607.03949)). I found no work testing cross-version compatibility as a property of the training objective.

**Experiment that settles it.** For every pair of grid checkpoints, fit an orthogonal map on 5k pixels. Train a probe on version A, apply it to mapped version B, and report the retained fraction of native-B accuracy on PANGAEA-style tasks and OpenChange-UK. Run the same protocol on MAE and Barlow Twins pairs (the plan already has one MAE baseline) and on TESSERA v1.1 vs v2 betas. Kill: LeJEPA pairs are no more alignable than baseline pairs.

**Risks.** The theorem assumes stationary, additive-noise latent dynamics, and seasonal EO is neither. Alignment may hold for large models and fail across sizes. A positive result could be read as "SSL embeddings are generally alignable", so the baselines decide the paper.

## 2. Calibrated alerts from latent-forecast residuals

**User and decision.** The Environment Agency and coastal groups that track shoreline and vegetation-edge change from Sentinel-2 ([CoastSat](https://dl.acm.org/doi/10.1016/j.envsoft.2019.104528), [VedgeSat on a Scottish coast](https://eprints.gla.ac.uk/369870/1/369870.pdf)). Forest-alert operators such as RADD, which issues Sentinel-1 alerts and confirms them once disturbance probability passes 97.5% ([WUR RADD](https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert)). Insurers who need triggers with known false-alarm and lead-time behaviour ([index insurance validation criteria](https://ieeexplore.ieee.org/document/10042166)). The decision: send a survey team, issue a warning, or trigger a payout.

**Method novelty.** A predictor forecasts the next acquisition's embedding from the multi-sensor history and the time gap. The residual between the forecast and the actual new embedding is the change score. Split-conformal calibration, stratified by season and land cover, turns that score into an alert with a stated false-alarm rate. When optical is cloudy, Sentinel-1 acquisitions keep the series going. Alerts are scored with decision metrics: lead time at fixed false-alarm rate, and relative economic value under a cost-loss model as used in flood-warning evaluation ([forecast-based action cost-benefit](https://www.sciencedirect.com/science/article/pii/S2212420918314274)). Pixel error is not the score.

**Closest prior work.** RADD and BayTS use per-pixel Bayesian updating of backscatter, with no learned representation ([Reiche et al. 2021](https://iopscience.iop.org/article/10.1088/1748-9326/abd0a8), [bayts](https://github.com/jreiche/bayts/)). Change detection on AlphaEarth annual embeddings uses cosine distance or robust z-scores, and its authors say distances "do not live on a consistent scale across pixels or land-cover types" ([Wherobots](https://wherobots.com/blog/alphaearth-foundations-change-detection-part-2/), [yag.xyz](https://yag.xyz/en/post/alphaearth-embedding-change-detection/)). Conformal false-alarm control exists for despeckled SAR pairs ([unsupervised SAR CD](https://exa.ai/library/publication/187m51lg87k)) and as a wrapper around supervised change detectors ([conformal CD repo](https://github.com/HaloMind-Research/Conformal-Satellite-Change-Detection)). T-JEPA learns temporal transitions but its abstract evaluates transfer, not alerts ([arXiv:2610.05731](https://arxiv.org/abs/2610.05731)). I found nothing that combines a learned multi-sensor latent forecast with conformal false-alarm control and scores the result by lead time.

**Experiment that settles it.** Choose one region with dated events, for example RADD-confirmed disturbances in one tropical tile, or OpenChange-UK coastal events. At matched empirical false-alarm rate on held-out years, compare detection lead time against (a) a RADD-style backscatter baseline, (b) embedding differences on AlphaEarth/TESSERA, and (c) the residual method. Kill: (c) does not beat (a) at matched false-alarm rate.

**Risks.** Few dated events in the UK. Seasonality and autocorrelation break the exchangeability that conformal guarantees rely on, so calibration has to be stratified and checked empirically. Rare, large events dominate economic value. RADD covers humid tropics only, so a UK result needs its own labels.

## 3. Label-free checkpoint and region selection

**User and decision.** Labs and agencies choosing which checkpoint to release or deploy, and users choosing a model for a region with no labels (a held-out UK coast). TESSERA v2 says selecting by loss "wastes a large share of the compute", which makes the saving concrete ([TESSERA v2](https://arxiv.org/html/2607.03949)).

**Novelty: low to moderate, stated plainly.** Label-free selection is not new. RankMe (effective rank) ([ICML 2023](https://proceedings.mlr.press/v202/garrido23a.html)) and LiDAR ([arXiv:2312.04000](https://arxiv.science/abs/2312.04000)) already exist, and EO studies have applied intrinsic diagnostics ([arXiv:2607.05207](https://arxiv.org/html/2607.05207v1)) and intrinsic dimension ([arXiv:2511.02101](https://www.scholarfeed.org/paper/2511.02101v2)). The contribution is the first pre-registered EO comparison at scale of LeJEPA loss vs RankMe vs LiDAR vs a small labelled probe. It adds a per-region variant: does the loss computed only on unlabelled UK data rank checkpoints the way UK downstream labels do? In the TESSERA v2 HTML I found RankMe mentioned only in an ablation carried over from v1, not as a selector comparison (my text search may have missed it; UNVERIFIED).

**Experiment.** Rank correlation per task over all grid checkpoints, for each proxy, with bootstrap intervals. Kill: LeJEPA loss Spearman below RankMe, or below about 0.5.

**Risks.** Cloud and orbit variance may dominate the loss, as TESSERA v2 argues. Correlation across model sizes is easy to get; within-size correlation (hyperparameters, seeds) is the useful one and the harder one to get.

## 4. Continuity across satellite units

**User and decision.** Anyone using multi-year embedding time series. Google says its embedding space is "consistent across years" for change detection ([GEE catalog](https://developers.google.cn/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)). The hardware underneath changed: Sentinel-1B failed on 23 Dec 2021 ([ESA](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Mission_ends_for_Copernicus_Sentinel-1B_satellite)), Sentinel-1C launched 5 Dec 2024 ([Sentinel Online](https://sentinels.copernicus.eu/-/from-launch-to-images-the-roadmap-to-sentinel-1c-operations)), Sentinel-1D launched 4 Nov 2025 and will replace 1A ([ESA](https://www.esa.int/Newsroom/Press_Releases/Copernicus_Sentinel-1D_reaches_orbit_on_Ariane_6)), and Sentinel-2C replaced 2A in 2025 ([DLR](https://elib.dlr.de/221377/)). S2C is within 1-2% of S2A after vicarious adjustment, but spectral-response differences reach 15% in specific cases ([MDPI 2026](https://www.mdpi.com/2072-4292/18/9/1387)). The decision: is a jump in a pixel's embedding real change or a satellite swap?

**Method novelty.** Use the S2A/S2C tandem-phase acquisitions (same place, near-same time) as a natural control. Measure embedding distance between units against same-unit revisit distance. Then train with satellite unit as an explicit nuisance (cross-unit pairs as positives) and check that real change survives. Any-sensor models (Panopticon ([CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Waldmann_Panopticon_Advancing_Any-Sensor_Foundation_Models_for_Earth_Observation_CVPRW_2025_paper.html)), Copernicus-FM ([arXiv:2503.11849](https://arxiv.org/html/2503.11849v2))) solve input flexibility, not this break-detection question. A NeurIPS ML4PS 2025 workshop paper shows that representations shift across instruments ([The View From Space](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_215.pdf)), but it does not handle within-mission unit swaps or downstream time-series breaks.

**Experiment.** On tandem-phase tiles, compare the cross-unit/same-unit distance ratio across models (clean JEPA, AlphaEarth, TESSERA). Kill: the ratio is about 1 for all models. In that case the problem does not exist, which is still worth reporting in a short note.

**Risks.** Whether Copernicus-Pretrain or Major TOM keep the satellite unit ID per sample is UNVERIFIED; it may be recoverable from product names. Tandem-phase data has to be pulled separately from CDSE. This idea is half evaluation; it is not enough for a main-track paper on its own.

## 5. Kalman filtering in a Gaussian latent space

**User and decision.** Producers of gap-free products (land cover, crop condition) who fuse S1, S2 and S3 with uncertainty. The decision: whether a map pixel on date t is trustworthy.

**Method novelty: high.** SIGReg forces latents towards an isotropic Gaussian, and the identifiability theorem assumes additive-noise linear dynamics, which is exactly the setting where a Kalman filter is optimal ([arXiv:2605.26379](https://arxiv.org/abs/2605.26379)). Fit a linear transition and per-sensor linear observation maps on frozen embeddings, then filter. The output is a posterior mean and covariance per pixel, with each sensor's contribution explicit. Prior latent assimilation (LAE-EnKF ([arXiv:2603.06752](https://arxiv.org/html/2603.06752v2)), LAINR ([arXiv:2309.09574](https://arxiv.org/pdf/2309.09574)), [arXiv:2104.00430](https://ar5iv.labs.arxiv.org/html/2104.00430)) uses autoencoders on physics toy or weather systems, not SSL embeddings of satellite data. AlphaEarth "assimilates" sources inside its network but does not expose a filter or a posterior ([arXiv:2507.22291](https://arxiv.org/html/2507.22291)).

**Experiment.** Use synthetic cloud masks on held-out S2 dates. Compare filtered embeddings against the learned nonlinear predictor and against TESSERA-style invariance, on downstream accuracy and on the calibration of the predicted covariance. Kill: the filter loses to the learned predictor and its covariance is miscalibrated.

**Risks.** Seasonal, non-stationary dynamics and cloud effects that are not additive noise probably break the assumptions; this has the highest chance of a clean negative result. A negative result is still publishable as a test of the theory, but only at workshop level.

## 6. SAR-conditioned optical latents scored by decisions

**User.** Flood mappers (Copernicus GFM ([GloFAS](https://global-flood.emergency.copernicus.eu/technical-information/glofas-gfm/)); Met Office/EA FloodMap, which improves on GFM over the UK using EA LiDAR ([ICASP report](https://icasp.org.uk/wp-content/uploads/sites/13/2025/05/flood_report_metoffice.pdf))) and crop monitors on cloudy days.

**Why this ranks low on novelty.** It is crowded. TerraMind's Thinking-in-Modalities generates missing modalities as tokens and gains about 2 mIoU on Sen1Floods11 ([IBM](https://research.ibm.com/blog/thinking-in-modalities-terramind)). CloudLULC-Net fuses SAR with cloudy optical using 40k labelled triplets ([arXiv:2606.17713](https://arxiv.org/abs/2606.17713)). STARS handles missing optical ([arXiv:2601.17342](https://arxiv.org/html/2601.17342)). GeoSET, X-JEPA and CR-JEPA are already in the plan. The remaining new piece is narrow: the scaling law of the SAR-vs-optical downstream gap, plus learned abstention. The sharpest test is the GFM exclusion mask (urban areas, dense vegetation), where SAR is physically insensitive to flooding ([GFM PUM](https://extwiki.eodc.eu/gfm_assets/gfm4.0_pum_2025.pdf)). A latent that "predicts optical" there is extrapolating, and should say so.

**Experiment.** Gap vs compute on Sen1Floods11 and the OpenChange-UK tasks, split inside and outside the exclusion mask. Kill: the gap does not shrink with scale outside the mask, or the model is confident inside it.

**Risks.** Results may only confirm that SAR carries less information. Hallucinated confidence is the main safety risk for flood users.

---

## Sounds disruptive but is already done, mostly engineering, or not supportable

- **Compute-optimal recipe for a small lab.** Mostly done. TESSERA v2 gives a compute allocation rule and a 44M distilled student that beats every product they test ([TESSERA v2](https://arxiv.org/html/2607.03949)), and PhilEO covers data and model scale ([arXiv:2506.14765](https://arxiv.org/html/2506.14765v5)). Checking whether LeVJEPA's token dropping carries over to multispectral tokens is a useful ablation, not a paper.
- **Counterfactual or intervention queries on land-surface state.** Not supportable. Linear identifiability recovers latents up to a linear map; it gives no causal semantics, and EO archives are observational with no interventions. Existing "counterfactual" EO work explains classifiers ([CFE4SITS](https://mlanthology.org/ecmlpkdd/2023/dantas2023ecmlpkdd-counterfactual/)). Pursue it only through natural experiments with external data, which is a separate project.
- **Pretraining-data curation by geography.** Done. "Pretrain Where?" (CVPRW 2026) finds that spectral diversity, not geographic spread, correlates with downstream performance ([paper](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/papers/Kaur_Pretrain_Where_Investigating_How_Pretraining_Data_Diversity_Impacts_Geospatial_Foundation_CVPRW_2026_paper.pdf)); see also [arXiv:2501.12535](https://arxiv.org/html/2501.12535).
- **"Temporal JEPA for EO" as the contribution.** Taken by T-JEPA ([arXiv:2610.05731](https://arxiv.org/abs/2610.05731)).
- **Unified embedding access library.** Engineering, and rs-embed exists ([arXiv:2602.23678](https://arxiv.org/pdf/2602.23678)).

## Recommendation

Keep the scaling grid as the backbone and replace "does it scale" as the headline with three tested properties of the resulting embedding product: selectable without labels (3), stable across versions (1), and continuous across satellite units (4). All three can be read off grid checkpoints with probes only. Add idea 2 as the single applied demonstration, on one region with dated events, because it is the one a user like the Environment Agency, RADD or an insurer would adopt. Drop T2 as a training objective, and cite T-JEPA and S2S-JEPA as close prior work.

## Assumptions and limits of this report

- Claims about T-JEPA, S2S-JEPA, RoboJEPA and several other papers come from abstracts only; I did not read the full texts.
- No compute figures are given; the plan's rule (none before a one-GPU pilot) stands. "Feasibility" scores reflect relative cost given the planned grid.
- UNVERIFIED: per-sample satellite unit IDs in Copernicus-Pretrain/Major TOM; whether TESSERA v2 compares RankMe-style proxies as selectors; access route for S2A/S2C tandem-phase data.
- Searches were via web search and arXiv pages on 2026-10-09; a missed 2026 preprint on ideas 1, 2 or 5 is possible. Run a final arXiv check on "backward compatible embeddings remote sensing", "conformal change detection embeddings", and "latent Kalman satellite" before committing.

    verdict_one_line: Six ranked ideas. Best: version-stable embedding products via LeJEPA linear identifiability, and conformal latent-residual change alerts. Temporal JEPA for EO is already taken (T-JEPA, arXiv:2610.05731); the compute-recipe and counterfactual angles are already done or unsupportable.