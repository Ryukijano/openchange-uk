# Scaling-law experiments for u6xn: explorer report (2026-10-07)

Scope: web research only (arXiv abstracts via the arXiv and Hugging Face papers APIs, GitHub licence files, HF dataset cards, docs.isambard.ac.uk, gov.uk). I read abstracts, not full papers, unless stated. Anything I couldn't check is marked **UNVERIFIED**. No Isambard access, no downloads, no compute beyond arithmetic.

## 0. Shared assumptions and facts

**Hardware (checked):** the Isambard spec page says each node has 4 GH200, 4 × 72 Grace cores, 4 × 96 GB H100 memory, Slingshot 11 with 4 × 200 Gbps NICs per node, and **each GH200 is power-capped at 660 W shared between CPU and GPU** (https://docs.isambard.ac.uk/specs/). The paragraph I read is labelled Phase 1; that Phase 2 nodes are identical is **UNVERIFIED** from that page. The power cap means sustained throughput may sit below spec-sheet numbers; measure it.

**FLOP arithmetic (assumptions, not measured):**
- Training FLOPs C ≈ 6·N·D (N = parameters, D = training tokens). This ignores attention cost, which matters for long sequences (e.g. QEC at large distance, video).
- H100 dense BF16 peak taken as 989 TFLOPS (NVIDIA spec sheet value; **UNVERIFIED** in this session).
- Large models at 35% MFU → 989e12 × 0.35 × 3600 ≈ **1.25e18 FLOPs per GPU-hour**. 80k GPU-hours ≈ **1.0e23 FLOPs**.
- Small models (< 50M params) at 15% MFU → **5.3e17 FLOPs per GPU-hour**. This is a guess; small models are usually input-pipeline or launch-bound.
- GPU-hours ÷ 4 = node-hours (NHR). 1% of budget = 800 GPU-hours = 200 NHR.

**UK priorities (checked):** the UK AI for Science Strategy names five priority areas: engineering biology, fusion energy, materials science, medical research, quantum technologies (https://www.gov.uk/government/publications/ai-for-science-strategy/ai-for-science-strategy). The NHS / safer streets / clean energy missions are from the government missions programme (https://www.gov.uk/missions — **UNVERIFIED**, not fetched today).

**Two findings that shape every idea below:**
- TESSERA v2 ran 395 EO pretraining runs on 1,024 GH200s and found pretraining loss barely predicts downstream performance (|r| < 0.2) (https://arxiv.org/abs/2607.03949).
- A 2026 LunarLander study found world-model "validation loss and multi-step prediction RMSE keep improving long after closed-loop performance has collapsed" (https://arxiv.org/abs/2607.01736). The opposite was reported for driving: Waymo-style motion forecasting found "a strong correlation between model training loss and model evaluation metrics", including closed-loop (https://arxiv.org/abs/2506.08228). So whether loss predicts decisions is genuinely open and domain-dependent.
- An independent reproduction of LeWorldModel showed the evaluation protocol alone moved the same checkpoint from 14% to 84% success (https://arxiv.org/abs/2608.10145). Any decision-quality scaling law must freeze its evaluation protocol before fitting.

**Method references for every idea:** fit on intermediate checkpoints, train several small seeds rather than one big run (https://arxiv.org/abs/2410.11840); loss-to-loss transfer laws (https://arxiv.org/abs/2411.12925); broken/non-monotone laws (https://arxiv.org/abs/2210.14891); data-constrained repetition law (https://arxiv.org/abs/2305.16264).

---

## 1. Latent (JEPA) vs generative world models: does lower loss mean better plans?

- **Question:** For action-conditioned world models trained on the same simulator data, do JEPA-style (latent prediction) and generative (pixel/token prediction) families follow different scaling exponents in (a) held-out loss, (b) loss under shift, (c) planning success? Does loss predict planning success within and across families?
- **Closest prior work:** Pearce et al., power laws for world models, coefficients depend on tokenizer/task/architecture, offline games only (https://arxiv.org/abs/2411.04434). LeWorldModel, ~15M-param JEPA trained end to end from pixels on one GPU (https://arxiv.org/abs/2603.19312). JEPA-WMs design study (architecture, objective, planner) (https://arxiv.org/abs/2512.24497). Loss vs closed-loop divergence (https://arxiv.org/abs/2607.01736). Video diffusion scaling laws, loss only (https://arxiv.org/abs/2411.17470). Video generation fails OOD while scaling ID (https://arxiv.org/abs/2411.02385).
- **What would be new:** I found no paper fitting scaling laws for JEPA world models, or comparing JEPA vs generative exponents on a decision metric. Pearce et al. use generative losses only (from the abstract).
- **Testbed and licence:** ManiSkill3 GPU-parallel simulator, code Apache-2.0 (https://github.com/haosulab/ManiSkill, LICENSE checked; asset licences **UNVERIFIED**), reported up to 30,000+ FPS with rendering (https://arxiv.org/abs/2410.00425). Optionally the LeWM environments (PushT, TwoRoom; LeWM repo licence **UNVERIFIED**). Note: facebookresearch/jepa-wms is **CC BY-NC 4.0** (LICENSE checked), so reuse its code only for non-commercial research. V-JEPA 2 code is MIT (https://github.com/facebookresearch/vjepa2).
- **Scaling axes:** parameters 1M–300M; data 1e4–1e7 trajectories (generated); compute 1e17–1e20 FLOPs; shift = unseen object shapes/masses/lighting held out by seed.
- **Compute (6ND):** IsoFLOP grid, budgets {1e17, 1e18, 1e19, 1e20}, 6 sizes each, 2 families: 2 × 6 × 1.111e20 = **1.33e21 FLOPs ≈ 1,070 GPU-h (1.3%)**. Extra 2 seeds on the two smallest budgets ≈ 50 GPU-h. Planning eval, one 100M checkpoint: 2N × 196 tokens/frame × 300 CEM samples × 30 iterations × horizon 5 × 50 replans × 100 episodes ≈ 8.8e18 FLOPs ≈ **7 GPU-h per checkpoint**; ~100 checkpoints ≈ 700 GPU-h. Total ≈ **2k GPU-h (2.5%)**. Simulator time not included (**UNVERIFIED**).
- **Multi-node?** No for the grid (all ≤300M fits one GPU). Use Slurm arrays. Multi-node only if you add a 1B+ hold-out run.
- **First week:** On one GPU, train LeWM-size JEPA and a same-size pixel decoder model on one ManiSkill3 task at 3 sizes (1M, 5M, 20M). Record loss curves, success rate with a fixed CEM planner, and wall-clock split between simulator, training and planning.

## 2. Test-time compute scaling for world-model planning

- **Question:** At fixed planning success, what is the exchange rate between training compute (model size) and planning compute (CEM samples, iterations, horizon, latent depth)? Is there a crossover like in LLMs?
- **Closest prior work:** Motion forecasting: sampling + clustering lets small models match larger ones up to a crossover (https://arxiv.org/abs/2506.08228). SWIFT: test-time scaling for Cosmos world foundation models (https://arxiv.org/abs/2503.24320). DeepJEPA: adaptive per-transition depth as an inner test-time axis; uniform extra depth can hurt planning (https://arxiv.org/abs/2610.00368). Gradient-based planning train-test gap (https://arxiv.org/abs/2512.09929).
- **What would be new:** A joint law Success(N, C_plan) for action-conditioned control, with an explicit iso-success frontier. Prior work reports the crossover for driving forecasting only and test-time scaling for video WFMs only (from abstracts).
- **Testbed and licence:** same as idea 1 (ManiSkill3, Apache-2.0); reuses idea 1 checkpoints.
- **Scaling axes:** 4 model sizes (10M–300M) × CEM samples {30, 100, 300, 1000, 3000} × horizon {3, 5, 10, 20}.
- **Compute:** sum over the grid of 2N × 196 × S × 30 × H × 50 × 100 ≈ **4.4e21 FLOPs ≈ 3,500 GPU-h (4.4%)**. This is inference-heavy; FP8 inference could cut it (**UNVERIFIED** speedup).
- **Multi-node?** No. Planning rollouts are embarrassingly parallel.
- **First week:** Take the 3 models from idea 1 week 1, sweep samples × horizon on 100 fixed episodes, plot success vs total FLOPs (train + plan).

## 3. SUMO junction world model: ID loss, shifted loss, and control quality (the draft's main plan)

- **Question:** As in the draft: do ID loss, loss under shift (unseen layouts, demand surges, sensor dropout) and control quality (delay, spillback when planning with the model) follow power laws with the same exponents?
- **Closest prior work:** Driving motion forecasting scaling, 500k hours proprietary (https://arxiv.org/abs/2506.08228). Imitation-learning driving data scaling (https://arxiv.org/abs/2412.02689, abstract not fetched). GPUDrive, GPU driving sim at 1M steps/s on Waymo data (https://arxiv.org/abs/2408.01584). I found no scaling-law paper on SUMO world models (searched arXiv/HF papers for "traffic world model SUMO"; nothing relevant).
- **What would be new:** First open, reproducible scaling study of a traffic world model with simulator-scored control and controlled layout shift. Fits safer-streets.
- **Testbed and licence:** SUMO, EPL-2.0 (https://github.com/eclipse-sumo/sumo, LICENSE checked). Road networks from OpenStreetMap (ODbL, **UNVERIFIED** here).
- **Scaling axes:** params 1M–1B; data 1e8–1e11 tokens (vehicle-state tokens); number of distinct junction layouts 10–10,000 (diversity axis, motivated by the robot finding that diversity beats demo count: https://arxiv.org/abs/2410.18647).
- **Compute:** IsoFLOP {1e18, 1e19, 1e20, 1e21} × 6 sizes = 6.7e21 FLOPs ≈ **5,350 GPU-h (6.7%)**. Example: 1B × 20B tokens = 1.2e20 FLOPs ≈ 96 GPU-h.
- **Multi-node?** Only for the 1e21 runs at 1B params (FSDP over 2–8 nodes). Everything else single GPU.
- **Main risk:** SUMO runs on CPU; its throughput on Grace (aarch64) is **UNVERIFIED**. If it is slow, data generation, not GPUs, sets the limit. GPUDrive is a GPU alternative but uses Waymo Open data whose licence is non-commercial (**UNVERIFIED**).
- **First week:** Build one 4-arm junction, run libsumo on CPU, measure vehicle-steps per second per core. If < ~1e5 steps/s per node (my threshold, not a sourced number), switch the main testbed.

## 4. Neural QEC decoders: compute-optimal scaling with code distance

- **Question:** For a transformer decoder trained on Stim circuit-level noise, what is the law LER(N, D, d, p)? How should parameters and training shots scale with code distance d to stay below MWPM? Does a decoder trained at small d transfer, and how does that change the law?
- **Closest prior work:** AlphaQubit 1, recurrent transformer, d up to 11 (https://arxiv.org/abs/2310.05900). AlphaQubit 2, near-optimal surface and colour codes, real time to d=11 (https://arxiv.org/abs/2512.07737; abstract has no scaling-law fit, full paper **UNVERIFIED**). NTU foundation decoders: transfer across distances to d=25 (https://arxiv.org/abs/2606.27119). "Rethink neural decoders": near-term performance is driven more by data scale than architecture, d ≤ 9 (https://arxiv.org/abs/2605.12046). NVIDIA AI pre-decoders: a larger model further lowers LER, beats correlated PyMatching to d=13 (https://arxiv.org/abs/2604.12841).
- **What would be new:** An explicit, pre-registered compute-optimal law with distance as an axis (the abstracts above report larger-is-better or data-matters, not fitted exponents). Exact labels and unlimited data make this the cleanest scaling testbed on the list. Fits quantum priority and syndrome-net.
- **Testbed and licence:** Stim, Apache-2.0 (https://github.com/quantumlib/Stim, checked); Stim samples a d=100 surface code at about 1 kHz shots (https://arxiv.org/abs/2103.02202); rates at d ≤ 15 are **UNVERIFIED** but should be much higher. PyMatching sparse blossom baseline (https://arxiv.org/abs/2303.15933; licence **UNVERIFIED**).
- **Scaling axes:** d ∈ {3, 5, 7, 9, 11, 13, 15} (stretch 17–25); params 1M–100M (stretch 300M); shots 1e6–1e9; physical error rate p at 3 values; shift = unseen noise model (e.g. add leakage or biased noise).
- **Compute:** tokens per shot taken as (d²−1)·d (stabilizers × rounds): 120 at d=5, 1,320 at d=11, 4,896 at d=17. Full factorial (7 d × 5 sizes × 4 shot counts): **7.7e21 FLOPs ≈ 14,500 GPU-h at 15% MFU (18%)**. Too much; an IsoFLOP slice per distance would cut this to roughly a third (estimate). Single big runs: 100M × 1e9 shots at d=11 = 7.9e20 ≈ 640 GPU-h; 300M × 1e9 shots at d=17 = 8.8e21 ≈ 7,100 GPU-h. A recurrent per-round design (as in AlphaQubit) changes the attention cost; 6ND understates attention at 5k-token sequences.
- **Multi-node?** Yes for d ≥ 15 at 100M+ params (data parallel; long sequences). Not for d ≤ 11.
- **Data generation:** generate on the fly on the 72 Grace cores attached to each GPU (no storage, no download). Rate **UNVERIFIED**; measure first.
- **First week:** On the Devin VM or a laptop, time Stim + PyMatching shots/s for d=3–13 (cheap, CPU). Then one 30-minute GPU job: a 5M transformer at d=5, three shot budgets, LER vs MWPM.

## 5. Physics surrogates on The Well: simulator diversity vs size

- **Question:** For a single transformer recipe trained on The Well, how do model size, tokens and the number of distinct physics datasets (1→16) trade off, for ID one-step loss, long-rollout error, and shifted regimes?
- **Closest prior work:** The Well, 15 TB over 16 datasets (https://arxiv.org/abs/2412.00568). Walrus, cross-domain continuum foundation model, 19 scenarios (https://arxiv.org/abs/2511.15684). Poseidon "scales with model and data size" (https://arxiv.org/abs/2405.19101). DPOT up to 0.5B params (https://arxiv.org/abs/2403.03542). Physics foundation models are "conditional rather than universal generalists" and scaling does not remove biases (https://arxiv.org/abs/2605.29283). Weather: pooled scaling looks good while many channels degrade at late leads (https://arxiv.org/abs/2604.05068); weather models prefer width over depth (https://arxiv.org/abs/2602.22962).
- **What would be new:** A diversity axis fitted as a scaling variable (like environments in https://arxiv.org/abs/2410.18647), with rollout-horizon-aware fits.
- **Testbed and licence:** The Well code BSD-3-Clause (LICENSE checked); HF dataset cards for shear_flow and active_matter say CC-BY-4.0 (https://huggingface.co/datasets/polymathic-ai/shear_flow); the other 14 cards **UNVERIFIED**. Needs a large download, which requires your approval.
- **Scaling axes:** datasets {1, 2, 4, 8, 16}; params 10M–1B; budgets {1e18, 1e19, 1e20}.
- **Compute:** 5 diversity levels × 4 sizes × (1e18+1e19+1e20) = **2.2e21 FLOPs ≈ 1,800 GPU-h (2.2%)**. 1B × 1e11 tokens = 6e20 ≈ 480 GPU-h.
- **Multi-node?** Yes for 3D datasets and 1B models (Walrus describes load-balanced distributed 2D/3D training). 2D-only grid fits single GPU/node.
- **First week:** Pick two small 2D datasets (e.g. shear_flow, active_matter), 3 model sizes, measure one-step and 20-step rollout error vs FLOPs.

## 6. Fusion: data-constrained scaling on real MAST data plus simulator pretraining

- **Question:** With fixed real tokamak data, how many epochs help before returns vanish (Muennighoff-style law), and how much does simulator pretraining (TORAX) shift that curve?
- **Closest prior work:** TokaMark, MAST benchmark, 14 tasks (https://arxiv.org/abs/2602.10132). TokaMind foundation model on MAST (https://arxiv.org/abs/2602.15084). TORAX differentiable transport simulator in JAX (https://arxiv.org/abs/2406.06718). GyroSwin 5D gyrokinetic surrogates (https://arxiv.org/abs/2510.07314). Data-constrained LM law (https://arxiv.org/abs/2305.16264).
- **What would be new:** First data-constrained scaling law for fusion plasma models, plus a "sim tokens are worth X real tokens" exchange rate.
- **Testbed and licence:** FAIR-MAST / TokaMark data licence **UNVERIFIED**; TORAX licence **UNVERIFIED**. UK-relevant (MAST is a UKAEA machine — **UNVERIFIED** in this session).
- **Scaling axes:** params 1M–100M; unique tokens fixed (assumed 5e9, **UNVERIFIED**, depends on MAST size); epochs 1–64; sim:real ratio 0–10×.
- **Compute:** Σ 6·N·U·epochs over 5 sizes × 7 epoch counts = 5.5e20 FLOPs ≈ **1,000 GPU-h at 15% MFU (1.3%)**.
- **Multi-node?** No.
- **First week:** Read TokaMark data terms and size; count tokens per task. Stop if licence or size doesn't fit.

## 7. Materials: MLIP scaling vs downstream physics (does force MAE predict MD stability?)

- **Question:** As MLIPs scale on OMat24, does energy/force error keep predicting downstream properties (MD stability, phonons), or does it decouple like TESSERA v2?
- **Closest prior work:** UMA, empirical scaling laws on ~0.5B structures (https://arxiv.org/abs/2506.23971). Equivariant models have better exponents (https://arxiv.org/abs/2510.09768). Universal force fields evaluated against experiment (https://arxiv.org/abs/2508.05762, abstract not fetched).
- **What would be new:** A loss-to-downstream scaling relation for MLIPs with pre-registered downstream tests.
- **Testbed and licence:** OMat24 CC-BY-4.0 (https://huggingface.co/datasets/facebook/OMAT24, tag checked); fairchem code MIT (LICENSE checked).
- **Compute:** 6ND does not apply directly to GNNs (cost scales with edges, not tokens). Must be measured. Suggest a cap of 5k GPU-h (6%) until measured.
- **Multi-node?** Probably not below ~100M params (**UNVERIFIED**).
- **First week:** Train 3 small fairchem models on a 1% OMat24 slice; measure throughput and run a 10 ps MD stability test.
- **Caveat:** outside the user's background; Meta FAIR already covers the loss side at much larger scale.

## 8. Distillation scaling into small local models

- **Question:** For world models (idea 1) or decoders (idea 4), what is the distillation law for a student of size N_s given teacher size and distillation tokens, measured on planning success / LER rather than loss? What student size runs at control rate on a GB10?
- **Closest prior work:** Distillation scaling laws for LMs (https://arxiv.org/abs/2502.08606). TESSERA v2 distilled a 21M student that beat larger models (https://arxiv.org/abs/2607.03949). Distilling world-model representations into compact robot policies (https://arxiv.org/abs/2609.24682, abstract not fetched). Mobile world models (https://arxiv.org/abs/2603.07799, abstract not fetched).
- **What would be new:** Distillation law on a decision metric, tied to a real deployment target.
- **Compute:** teacher 300M × 30B tokens = 5.4e19 ≈ 43 GPU-h; 16 students (1M–30M × 1e9–3e10 tokens), counting teacher forward 2·N_t per token: 1.2e20 ≈ **220 GPU-h**. Cheap (< 0.5%).
- **Multi-node?** No.
- **First week:** Reuse idea 1 or 4 teacher; distill 3 student sizes; measure success/LER and GB10 latency (GB10 local, no Isambard needed).

## 9. Precision scaling for deployment (FP8/INT4) on decision metrics

- **Question:** Does post-training quantization hurt more as models see more data (as found for LMs), when measured by LER or planning success?
- **Closest prior work:** Precision-aware scaling laws: PTQ degradation grows with training data (https://arxiv.org/abs/2411.04330). INT4 needed for µs-latency QEC decoding on FPGA (https://arxiv.org/abs/2605.12046).
- **What would be new:** Precision law for control/decoding, where small errors can flip a decision.
- **Compute:** 5 sizes (1M–100M) at 20 tokens/param, 2 training precisions: 2.6e18 FLOPs ≈ **5 GPU-h**. PTQ eval is cheap. H100 has FP8; FP4 needs Blackwell (GB10) (**UNVERIFIED** for GB10 FP4 kernels in your stack).
- **Multi-node?** No.
- **First week:** Quantize idea 4 checkpoints to FP8/INT8/INT4, plot LER penalty vs training shots.

## 10. Cross-domain loss-to-loss and loss-to-decision laws (analysis layer)

- **Question:** Across ideas 1, 3, 4, 5, does shifted loss follow a shifted power law of ID loss (as in https://arxiv.org/abs/2411.12925), and does the decision metric follow loss? Where does "accuracy on the line" (Miller et al., https://arxiv.org/abs/2107.04649: OOD accuracy strongly correlates with ID accuracy across many shifts, weaker on some, e.g. Camelyon17-WILDS) break?
- **Closest prior work:** time-series foundation models: OOD loss scales like ID loss, but architecture changes OOD scalability (https://arxiv.org/abs/2410.12360). Observational scaling laws (https://arxiv.org/abs/2405.10938).
- **What would be new:** The same analysis repeated in 2–4 simulator domains with exact ground truth. This is what turns separate grids into one paper.
- **Compute:** ~0 extra; uses checkpoints from other ideas. Requires that every grid saves ID loss, shifted loss and decision metric at every checkpoint.
- **First week:** Write the pre-registration: functional forms, shift splits, decision metric, held-out largest run.

## 11. EO: label-scaling and geographic-shift scaling on frozen TESSERA embeddings (UK)

- **Question:** On frozen TESSERA embeddings, how does UK downstream accuracy scale with number of labels and embedding dimension (Matryoshka prefix), and how fast does the gap to held-out regions close?
- **Closest prior work:** TESSERA (https://arxiv.org/abs/2506.20380); TESSERA v2, Matryoshka 16-dim prefix keeps 92% of performance (https://arxiv.org/abs/2607.03949); PhilEO scaling (https://arxiv.org/abs/2506.14765).
- **What would be new:** Downstream-data scaling under geographic shift, using your OpenChange-UK split rules.
- **Testbed and licence:** tessera and geotessera code MIT (https://github.com/ucam-eo/tessera, https://github.com/ucam-eo/geotessera, checked); embedding data licence **UNVERIFIED**. Needs approved sources per OpenChange-UK rules.
- **Compute:** small heads on frozen embeddings, under ~100 GPU-h (estimate). Mostly CPU and I/O.
- **Multi-node?** No. It doesn't use the allocation's strengths.
- **First week:** Check embedding licence and UK coverage; no download until approved.

## 12. Medical: data-constrained SSL scaling on surgical video (optional)

- **Question:** You rejected surgical world models for lack of data. A data-constrained law asks the opposite: with fixed surgical frames, how many epochs and what model size are optimal, and does in-domain repetition beat adding general video?
- **Closest prior work:** SurgeNetXL, 4.7M frames, insights on scaling data and duration (https://arxiv.org/abs/2501.09436). CheXficient: 22.7% of data matches full-data CXR model (https://arxiv.org/abs/2602.22843). Data-constrained law (https://arxiv.org/abs/2305.16264).
- **Testbed and licence:** public surgical datasets; licences vary and many are non-commercial (**UNVERIFIED**).
- **Compute:** 4.7M frames × 196 tokens = 9.2e8 unique tokens; 4 sizes (5M–300M) × epochs {1, 4, 16, 64}: 1.9e20 FLOPs ≈ **155 GPU-h**.
- **Multi-node?** No.
- **Caveat:** downstream labels are scarce, so the decision-side metric is weak. Lowest priority.

---

## Budget view (estimates from the arithmetic above)

| Idea | GPU-h | % of 80k | Multi-node needed |
|---|---|---|---|
| 1 JEPA vs generative WM | ~2,000 | 2.5% | No (unless 1B hold-out) |
| 2 Test-time compute | ~3,500 | 4.4% | No |
| 3 SUMO | ~5,350 | 6.7% | Only 1e21 runs |
| 4 QEC (factorial) | ~14,500 (prune to ~5k) | 18% (~6%) | Yes for d ≥ 15 |
| 5 The Well | ~1,800 | 2.2% | For 3D / 1B |
| 6 Fusion | ~1,000 | 1.3% | No |
| 7 MLIP | cap 5,000 (unmeasured) | 6% | Probably not |
| 8 Distillation | ~260 | 0.3% | No |
| 9 Precision | ~5 | <0.1% | No |
| 12 Surgical | ~155 | 0.2% | No |

GPU FLOPs are not the bottleneck for most ideas. Simulator throughput (SUMO, ManiSkill rendering, Stim at large d) and the user's time are. Only ideas 3, 4 and 5 give an honest reason for multi-node runs.

## Top 3

1. **Idea 1 + 2 together: world-model scaling on loss, shift and planning success, with test-time compute as a second axis.** It is the question the user wants, it builds on his LeWM / jepa-wms / C-JEPA PushT work, and the evidence on whether loss predicts decisions conflicts across papers (https://arxiv.org/abs/2607.01736 vs https://arxiv.org/abs/2506.08228). About 5.5k GPU-h (7%). Risk: protocol sensitivity (https://arxiv.org/abs/2608.10145), so pre-register the planner and goals.
2. **Idea 4: QEC decoder scaling with code distance.** Cleanest testbed (exact labels, on-the-fly data, Apache-2.0), quantum is a named UK priority, it matches syndrome-net, and recent work hints at but does not fit the law (https://arxiv.org/abs/2605.12046, https://arxiv.org/abs/2604.12841, https://arxiv.org/abs/2606.27119). It has a real reason for multi-node runs at d ≥ 15. It is also the best second domain for idea 10.
3. **Idea 5: The Well with a diversity axis** (or idea 3 SUMO if SUMO throughput on Grace is good). The Well has existing open data (CC-BY-4.0 on the cards checked), touches fusion-adjacent MHD and engineering, and multi-node is justified for 3D. SUMO is better aligned with safer streets and his traffic work, but its data-generation rate is unmeasured; decide after the week-1 throughput test.

Suggested order: week 1 measures Stim and SUMO/ManiSkill throughput on CPU (no Isambard needed), plus a 30-minute 1-GPU smoke run. Then pick.
