# Overlap check: two papers that could undercut the plan

Date: 2026-10-13. Nothing was downloaded, trained or run apart from reading the two papers.

## 1. Coastal: Aladejana et al., RSMA 2026

"When is satellite-derived shoreline change interpretable? A signal-to-noise framework for shoreline-change detectability." Regional Studies in Marine Science, October 2026, doi:10.1016/j.rsma.2026.105374. Authors: Aladejana, Davis, Ramlal, Maharaj, Hosein.

**What I read:** the abstract and keywords on ScienceDirect. The paper is closed access and I could not get the full text, so anything below that goes beyond the abstract is marked as inference.

**What they did (from the abstract):**
- Five microtidal beaches in Trinidad and Tobago. Decadal Landsat 7/8/9 and Sentinel-2 shorelines (keywords list CoastSat), 454 beach-profile surveys, and DSAS 6.0 change rates.
- They compare the geomorphic signal against a noise floor made of positional, tidal and wave-runup error, and in one variant add registration bias in quadrature.
- Sites are graded by SNR: robust (> 1.5), marginal (0.5 to 1.5), low (< 0.5).
- 344 profile-matched observations: pooled RMSE 15.1 m, 10.5 m at three well-calibrated sites.
- Error splits into an elevation-dependent part, which contour calibration can fix, and a registration-limited part, which it cannot.
- Only 1 of 5 sites was clearly signal-dominated (Irois, SNR 2.74). At Turtle Beach, 82.7% of transects flipped sign between the EPR and LRR rates.
- Their conclusion: "no site was simultaneously robust, well calibrated, and directionally coherent."

**Overlap with our plan: high on the question, low on the method.**
- Our headline question ("when do satellite views support a signed change claim?") is now published, at the level of a site and a decadal trend.
- From the abstract, their method is a physical noise budget per site, not a learned model. It is not tested on sites without surveys, and it gives no per-image error or date-pair intervals. I can't rule out that the full text does some of this.

**What is still open:**
- Predicting the error of each image from metadata (tide, waves, cloud, geometry, algorithm spread), so bad images can be dropped at beaches with no survey.
- Change intervals between two dates, rather than a decadal trend grade.
- Calibrated coverage on held-out sites.

**Changes to the coastal plan:**
1. Add their noise floor as baseline 9: positional, tidal, runup and registration error in quadrature, applied per site. The learned model has to beat it at the same false-direction rate. It's the strongest non-learned baseline we have.
2. Narrow the novelty claim to "per-image error prediction that transfers to beaches with no surveys." The paper must be cited as the source of the detectability framing.
3. Their registration-limited error component is a warning: if registration error dominates at our held-out sites, metadata features may not be able to predict it. Add co-registration features to Stage 0 and expect a ceiling.
4. Their result that only 1 of 5 sites gave robust claims makes kill criterion K3 (fewer than 25% of date pairs get a claim) more likely to trigger. That would still be a valid finding, not a failed project.
5. Their Trinidad surveys are probably not public. The plan does not rely on them.

**Verdict: still go, with a narrower claim and one extra baseline.**

## 2. Forecasting: Defonte et al., arXiv 2605.04239

"Densification and forecasting of Sentinel-2 time series from multimodal SAR and Optical satellite data using deep generative models." Thales Services Numériques and CNES, May 2026, arXiv preprint, CC BY 4.0.

**What I read:** the full arXiv HTML.

**What they did:**
- A generative model that produces a Sentinel-2 image (RGB + NIR) for any past or future date. It takes up to 8 Sentinel-1 (VV/VH, descending orbit only, S1Tiling) and Sentinel-2 images, uses cross-attention over time, and has a Laplace probabilistic decoder.
- Data: 25,813 patches of 96x96 pixels from GEODES (CNES's platform), mostly farmland, water excluded. They say their training and evaluation tiles are separate; I did not check where the tiles are.
- Interpolation: beats linear interpolation and U-TILISE (cropland RMSE 0.025 vs 0.030 and 0.035).
- Extrapolation (all inputs before the target date): RMSE 0.034 on valid pixels and 0.040 on cropland. **This table has no baseline at all**: no persistence, climatology or linear extrapolation.
- S1 ablation (same setup, S2 only): RMSE 0.028 vs 0.030 on all pixels, 0.033 vs 0.036 on crops. **Interpolation and extrapolation are pooled**, the values are reflectance rather than NDVI, there are no confidence intervals, and there is no breakdown by horizon.
- Uncertainty: calibration curves close to the diagonal (shown as a figure, interpolation mode).
- No weather inputs. I found no code or data link in the paper.

**Overlap with our plan: moderate.**
- "Adding S1 improves S2 forecasts" is already published as a small pooled gain on European farmland. We can't claim it as new in general.
- Still not done: horizon-by-horizon (5 to 30 day) NDVI forecasts in African shrub and grassland, forecasting only (no interpolation), with no future weather, compared against persistence and climatology, with confidence intervals.

**Changes to the forecasting plan:**
1. Cite it. Treat the general "S1 helps" claim as prior art.
2. Report forecasting only, per horizon, against persistence and climatology. Those are the checks the paper leaves out.
3. Their S1 gain is small (about 0.002 reflectance RMSE, pooled). That lowers my estimate that we will reach K1 (NDVI RMSE down by at least 0.005). NDVI and reflectance errors are on different scales, so the numbers can't be compared directly.
4. Their architecture is a possible alternative to Contextformer, but there are no public weights.

**Verdict: narrower novelty, lower chance of passing K1. Forecasting stays second.**

## Effect on the ranking

- Coastal stays first. Stage 0 now includes baseline 9 (the RSMA noise floor).
- Forecasting stays second, with a lower prior. It still needs a licence decision before Step A.
