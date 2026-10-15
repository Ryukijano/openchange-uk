# Task-first training plan for the EO Hugging Face demo

Date: 2026-10-13. This is research only. Nothing was downloaded beyond small metadata, trained, or deployed.

Rule used: useful job, then where existing methods fail, then a target with independent labels, then data, then objective, then architecture, then demo. JEPA, world models and training from scratch were all optional. None of them won on this evidence.

## Verdicts

| Problem | Verdict | What would be trained | Labels independent of inputs? | First check (cost) |
|---|---|---|---|---|
| Coastal: is the shoreline change real? | LIGHT_HEAD_ON_FROZEN | Small quantile and abstain model (GBM, then optionally an MLP on frozen EO embeddings) on top of CoastSat/CoastSeg outputs. Predicts each satellite shoreline's error against field surveys | Yes. Field surveys (GNSS, lidar, jetski) | Stage 0. CPU only, no imagery, uses the five teams' existing SDS_Benchmark submissions |
| Forecasting greenness 5-30 days ahead | FINETUNE_EXISTING, gated | Contextformer-6M (MIT weights) fine-tuned on EarthNet2023-Africa with an added Sentinel-1 pathway. Quantile NDVI output | In time only. The target is future S2 NDVI, so it is a proxy and not ecological truth | Step A. CPU LightGBM test of whether S1 helps, on 400 held-out instances |
| Burn-scar analogue search | NO_TRAINING_NEEDED | Nothing. Shape and severity descriptors with kNN and a calibrated no-match threshold | No. MTBS-derived labels would be circular. Only blind human grades are independent | Two raters grade 60 pairs to test whether they agree (kappa) |

Full reports from the three sessions:
- coastal_change_training_plan.md (https://app.devin.ai/sessions/1568d45c7bae431fbf1c3421923c55da)
- forecast_observe_training_plan.md (https://app.devin.ai/sessions/1ea0712ce8194693b47828db76780da2)
- event_search_training_plan.md (https://app.devin.ai/sessions/ddf88c8b70924aed9a360ba2a318d38a)

Claims I checked myself:
- SDS_Benchmark is GPL-3.0, was last pushed 2026-03-31, and contains `submissions/`.
- CoastSeg is GPL-3.0.
- The Contextformer weights (Zenodo 10793870) are MIT and 2.45 GB.
- HLS Burn Scars is CC-BY-4.0.
- The Prithvi-EO-2.0-300M-BurnScars card reports test IoU 87.52 and is Apache-2.0.
- Both arXiv IDs cited (2605.04239 and 2409.15311) exist with the stated titles.

I did not re-check the EarthNet2023 cube headers or the dataset licence.

## Ranking

1. **Coastal, first.** It has the cleanest science and the cheapest falsifier.
   - The labels are physically independent of the inputs, and the survey data are CC0 (Narrabeen, San Diego, Truc Vert).
   - The existing literature identifies screening and uncertainty as the gap, not segmentation. In LICS, an NDWI baseline beat a U-Net.
   - Stage 0 needs no imagery and no GPU.
   - The model is small. That is the honest answer: the thing worth learning is per-image error, not a new backbone.
   - Demo: upload a CoastSat/CoastSeg transect CSV. Get keep or abstain flags per row, and for two dates either "retreat X +/- Y m", "advance" or "cannot tell". Runs on CPU.
   - Risks:
     - CoastSat/CoastSeg are GPL-3.0, so the Space code must be GPL-compatible.
     - FES tide grids cannot be redistributed.
     - UKHO SWED is excluded because its licence is R&D only.
     - The Duck licence is unverified.
     - Aladejana et al. (RSMA 2026) already publish a site-level signal-to-noise detectability framework (abstract read, full text closed). Our claim narrows to per-image error prediction that transfers to unsurveyed beaches, and their noise floor becomes baseline 9. See OVERLAP_PAPERS_REVIEW.md.
2. **Forecasting, second, in parallel.** This is the only candidate where a real neural model is justified, and only if Step A passes.
   - EarthNet2023-Africa is the only checked public dataset with native S1 and S2 acquisition days plus future S2 targets: 44,141 train and 4,534 test cubes.
   - Problems:
     - The licence is CC BY-NC-SA 4.0, so ShareAlike may cover weights and Space artefacts.
     - Sampled train cubes cover only 2016-01 to 2017-03, while test cubes cover 2016-2021.
     - Contextformer's published margin over climatology in Europe is small (R2 0.62 vs 0.58, source's numbers), and it uses future weather.
     - Defonte et al. (arXiv 2605.04239) already report a small S1 gain for S2 generation on European farmland (pooled interpolation + extrapolation, no forecasting baselines). Our version must be forecast-only, per horizon, against persistence and climatology. See OVERLAP_PAPERS_REVIEW.md.
   - Observation selection waits until S1 is shown to help.
3. **Event search: no training.** It could still be a non-learned side demo if raters agree (kappa >= 0.4). A different target, 1-5 year post-fire recovery trajectories, would have independent labels and needs its own audit.

## Hypotheses and kill criteria

**Coastal.** If a quantile model is trained on survey-matched residuals from Duck, Narrabeen and Torrey Pines, then on held-out Truc Vert and Waikiki it will cut retained-shoreline RMSE by at least 15% at 70% coverage. The comparison is the best of: CoastSat QC, Landsat RMSE / S2 geometry filters, tide cut-off plus setup, CoastSeg's learned QC, co-registration, inter-algorithm spread, and (baseline 9) the Aladejana et al. 2026 per-site noise floor (positional, tidal, runup and registration error in quadrature).
- K1: below 15% on 2 or more held-out sites, ship a non-learned uncertainty tool instead.
- K2: 90% interval coverage below 85%, the tool only abstains.
- K3: false-direction rate above 10% of claims, or fewer than 25% of pairs claimed, no "real change" verdicts.
- K4: embeddings add less than 5% over metadata alone, drop the encoder.
- K5: fewer than 100 matches within ±48 h per test site, widen the window with the time gap as a feature, or stop.

**Forecasting.** If S1 history is added for shrub/grass patches with sparse S2, then without future weather the 5-30 day NDVI RMSE will fall by at least 0.005, with a cube-bootstrap 95% CI excluding 0.
- K1 failure drops S1 and observation selection.
- K2: if the best model beats climatology plus anomaly on fewer than 55% of held-out cubes, NO_TRAINING_NEEDED.
- K3: if fine-tuned Contextformer does not beat LightGBM, ship LightGBM.
- K4: if more than 30% of target windows lack a usable S2 date, redesign the horizon.
- K5: any skill that needs future weather is reported as an oracle, not as forecast skill.

**Event search.**
- K1: kappa below 0.4 on 60 pairs, NO_GO.
- K2: descriptors within 0.05 nDCG@5 of frozen embeddings, and P@5 >= 0.6, ship without training.

## Compute (estimates, not measured)

- Coastal Stage 0: CPU minutes.
- Coastal Stage 1: 1-3 GPU-h.
- Coastal full run: 20-40 GPU-h. The real cost is GEE imagery retrieval and survey preprocessing.
- Forecasting Step A: CPU.
- Forecasting pilot fine-tune: 3-8 GPU-h on about 90 GB.
- Forecasting full run: 60-150 GPU-h on 0.8 TB, likely I/O bound.
- Event search: CPU.

## Next steps (each needs your approval)

1. Coastal Stage 0, local CPU.
   - Clone SDS_Benchmark (GPL-3.0).
   - Compute residuals of the five submissions against surveys.
   - Count survey matches within ±48 h per site.
   - Run leave-one-site-out GBM quantile models against baselines 1-9.
   - Report pass or fail on K1/K2/K5 as measured.
2. Forecasting Step A, local CPU. Before this, you decide whether CC BY-NC-SA is acceptable for a public Space.
   - Range-read the time axis of about 300 cubes to confirm the train window.
   - Extract patch features for about 2,000 train and 200 test cubes (tens of GB, estimated).
   - Run the LightGBM S1 ablation.
3. Done: read the two possible overlap papers (OVERLAP_PAPERS_REVIEW.md). Neither kills its candidate. Both narrow the claim.
4. Only after a pass: a hand-checked pilot of at most 500 samples, then GPU training through the existing approved Isambard process (you run the jobs), then a CPU Space.
