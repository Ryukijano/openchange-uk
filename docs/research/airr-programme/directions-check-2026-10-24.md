# Which ideas hold up: agentic SfM vs the world-model ideas (24 Oct 2026)

Checked against arXiv/project pages and your own repos and HF account. "Verified" means I read the source; "Inference" is my judgement.

## Short answer

You are right to be unsure about the world-model ideas. Both blind world-model directions in the pasted notes are now crowded. Agentic SfM is the one with measured headroom in your own data, and the pasted notes misquote that number. Make agentic SfM the main track. Gate it with a cheap oracle test on HardMatch before any more RL.

## 1. Agentic SfM: the headroom claim in the notes is wrong

Verified, from `Ryukijano/agentic-sfm` README.md:81, on 1,170 frozen ScanNet pairs (job 8235219):

| | AUC@10 (actually success rate, see README:82) |
|---|---|
| Best single tool (VGGT head) | 0.340 |
| Per-scene oracle | 0.343 |
| **Per-pair oracle** | **0.416** |

The notes say routing has "almost no headroom (0.340 vs 0.343)". That is the per-scene number. Per pair, the ceiling is +7.6 points, about 22% relative. Your README calls it "a loose, noise-selecting ceiling", which is fair: a max over noisy arms is biased upward. So the honest reading is that the per-scene router is dead and the per-pair router is unproven, not dead.

The real weaknesses (verified, README:31, 45-47, 81):
- The pair agent loses to plain LoFTR on hard validation (62% vs 88%, n=50).
- Scene GRPO collapsed onto LoFTR (0 calls to MASt3R/VGGT).
- The specialist pool is out of date. LoMa (ECCV 2026 oral, https://arxiv.org/abs/2604.04931) beats ALIKED+LightGlue by +18.6 mAA on HardMatch and +29.5 on WxBS. RoMa-Ω (https://arxiv.org/abs/2609.09507) is +8.1 mAA over RoMa v2 on WxBS. Neither is in your pool.
- ScanNet is indoor, possibly in VGGT's training data, and easy. Your own README flags this (line 111).

Prior art check (verified, search 24 Oct): I found no paper that trains a VLM to pick matchers and crops per pair for SfM. The nearest are AstraLOD3 (https://arxiv.org/abs/2609.28061: an agentic LOD3 building pipeline that starts from SfM, not matching), SPAgent (general spatial tool calling) and SpaceTools (CVPR 2026: RL for multi-tool spatial QA). The idea is Gabriele Berton's, posted publicly, and has no paper yet, so credit and timing both matter.

**Decisive experiment (about 1 GPU-day, inference only):** run HardMatch's 100 validation pairs (MIT code; pairs from Wikimedia Commons, each with its own licence; `pip install hardmatch`) through {LoMa-G, RoMa-Ω, MASt3R, VGGT, LoFTR} × {full frame, 2-3 fixed crops}. Report mAA for the best single tool, the per-pair oracle and the per-category oracle, with a pair-bootstrap CI.
- Kill rule: if the per-pair oracle gain over LoMa-G is under about 5 mAA, or its CI touches 0, drop pair routing and keep only the scene-level question.
- Baseline the agent must beat: a small MLP or GBM router over cheap features (inlier counts, ratios, DINO similarity per matcher). If that router matches the VLM, the VLM is not the contribution.

Why it is worth it: the paper writes itself as "a learned router recovers X% of the oracle gap over the best single matcher on HardMatch/WxBS at Y× cost". The demo is AstroLoc-shaped: upload two hard photos, see which tool the agent picked and why, and get a pose.

## 2. World-action model on robot data: crowded, including the "robustness" version

Verified, all from Oct 2026 unless noted:
- Long-WAM (2610.10528): history helps only with AR pretraining; GR-1 63.3% → 78.7%.
- OneWM (2610.21095): unified causal world+action training; 36.68% on RoboCasa365.
- GreenWAM (2610.18774): 20 robot families; top of RoboTwin 2.0.
- RM-WAM (2610.15305): learns when history is useful, via counterfactual memory.
- Mem4Act (2610.22323): latent memory for long-horizon control.
- ReCAP (2610.14583): context scheduling for long rollouts.
- **What Matters in Designing WAMs (2609.24048, Sept 2026):** compares 8 latent spaces, including V-JEPA, on RoboCasa-GR1 and LIBERO-Plus. It finds the ranking flips under sensor noise and camera shifts.

That last paper already runs the experiment Grok suggests ("frozen encoder + one robustness axis"), on the same benchmarks. History in WAMs is now a field with three papers from the same week. What's left, a causal latent WAM measured against Long-WAM checkpoints, is a narrow workshop-sized delta, and you would have to build the closed-loop harness first. Inference: not worth main-track time unless the history question excites you specifically.

## 3. Weather-conditioned UK EO anomaly feed: the applied gap is smaller than claimed

Verified:
- Weather-guided Sentinel-2 forecasting exists already: EarthNet2021 and its follow-ups (Benson et al., meteo-guided video prediction, residual anomalies beyond the seasonal cycle).
- CloudFerro's Anomaly Watch (https://research.cloudferro.com/anomalywatch-demo) already runs a near-real-time anomaly demo on TerraMind embeddings, plus Tessera pixel change.
- Google Earth "detect change" ships AlphaEarth year-over-year change.
- EA Recorded Flood Outlines are OGL and updated quarterly, so labels exist.

Inference: a calibrated UK flood and vegetation anomaly score checked against EA outlines could be a useful applied tool or a workshop paper. It is not a top-venue method paper, and someone already hosts a similar demo. Use the one-day gate from the notes (do annual embedding differences already catch the EA outlines?) only if you want an applied side project.

## 4. What your record says to extend

Verified from your HF account (97 repos) and portfolio:
- 3D: Mast3r, Flash3d, LGM, Splat_to_mesh and Multiview_diffusion_3d Spaces; B3tt3r (MASt3R+Spann3r); Dalton Mills VR reconstruction.
- RL: about 20 deep-RL course agents, DeepRacer PPO, ML-Agents.
- World models: the Causal-JEPA repro Spaces, marl-world-model, Cosmos_Sentinel.
- CUDA: Blackwell labs.

Agentic SfM is where these meet (3D + RL + tool use), and it is the only track with a measured, unclaimed gap in your own numbers. The world-model work becomes a component, not the headline. Example: a learned reconstruction verifier (does this two-view geometry look right?) that the agent queries before committing an edge. That reuses your JEPA/latent-prediction skills inside a problem that has a metric.

## Suggested order

1. Finish scene-oracle job 230690 and write its result into the README. You already run it.
2. Run the HardMatch per-pair oracle with LoMa and RoMa-Ω added (section 1). Stop or go on the 5-mAA rule.
3. If go: fit the cheap feature router first, then the VLM agent. The agent has to beat both LoMa-G alone and the cheap router.
4. Park the WAM. Treat the EO anomaly feed as optional applied work.
