# World models on public video + agentic planning: what's open, what fits 80k GPU-hours (2026-10-07)

Sources are linked. "Unverified" means I didn't find or read the primary number. Nothing here has been run.

## 1. What's already known (prior work you'd be measured against)

| Work | What it showed | What it leaves open |
|---|---|---|
| Pearce et al., ICML 2025, https://arxiv.org/abs/2411.04434 | LLM-style power laws hold for world models and imitation learning on game (Bleeding Edge) and robot (RT-1) offline data; coefficients depend on tokenizer, task, architecture | Loss only; no rollouts of their own models; no shift axis |
| World-in-World, ICLR 2026, https://arxiv.org/abs/2510.18135 | Closed-loop benchmark for world models; "first data scaling law for world models in embodied settings"; controllability matters more than visual quality; scaling action-observation post-training data beats upgrading the video generator; more inference compute helps | Post-training data axis only, on off-the-shelf generators; no parameter/compute law; no latent (JEPA) models |
| Scaling Video Generation for Reasoning, Sept 2026, https://arxiv.org/abs/2609.36599 | Rubik's-cube video task with exact ground truth: validation MSE follows a power law, but lower MSE does not predict state accuracy; small models win at low compute, 1B wins at 3 PF-days; symbolic state supervision helps a lot | Single synthetic task; generative models only |
| V-JEPA 2, https://arxiv.org/abs/2506.09985 | Data scaling 2M -> 22M videos and model scaling 300M -> 1B (ViT-L to ViT-g) improve downstream; planning via V-JEPA 2-AC post-trained on under 62 h of DROID video | Ablation, not a fitted law; planning not measured across scales; internet-scale pretraining is far outside our budget |
| DINO-world, https://arxiv.org/abs/2507.19468 | Latent predictor on frozen DINOv2 features, trained on large uncurated video; action-conditioned fine-tune enables planning | Scaling not studied |
| Dreamer 4, https://arxiv.org/abs/2509.24527 | First agent to get Minecraft diamonds purely offline, from the VPT contractor dataset (about 2.5k hours per secondary sources; unverified); action conditioning learned from a small labelled fraction | Model/data scaling of the agent's success not reported as a law; compute unknown to me |
| Atari scaling probe, https://arxiv.org/abs/2605.08578 | Isolates model scale in a minimalist transformer world model on Atari 100k | Small data regime |
| Test-time scaling: SWIFT https://arxiv.org/abs/2503.24320, DeepJEPA https://arxiv.org/abs/2610.00368, World-in-World | Test-time compute helps world-model inference and planning; DeepJEPA allocates depth at decision-critical transitions | No study of the training-compute vs planning-compute trade-off at matched budgets |
| WebDreamer, https://arxiv.org/abs/2411.06559; agentic test-time scaling for web agents, https://arxiv.org/abs/2602.12276 | LLMs as world models for web agents; test-time scaling behaviour on multi-step tasks | Text/GUI, not video; depends on frontier LLMs |
| Physics-IQ, https://arxiv.org/abs/2501.09038 | Visual realism without physical understanding | Not a scaling study |

Takeaway: "do world models scale?" is answered (yes, for loss). "Does lower loss buy better decisions, and does that hold across model size, data, shift and planning budget?" has conflicting evidence (2609.36599 says no for state accuracy; Tuyls 2023 and Waymo 2506.08228 say yes for return/planning). That conflict is the opening.

## 2. Public video you can actually train on

| Source | Hours / size | Licence | Actions? | Notes |
|---|---|---|---|---|
| Open X-Embodiment, https://github.com/google-deepmind/open_x_embodiment | about 1M+ episodes, 22 embodiments | CC-BY 4.0 (data), Apache-2.0 (code) | yes | Best licence-clean action-labelled robot video |
| DROID, https://droid-dataset.github.io | 76k episodes, 564 scenes | CC-BY 4.0 (per site; re-check) | yes | What V-JEPA 2-AC used (under 62 h) |
| 1X World Model Challenge, https://www.1x.tech/discover/1x-world-model-sampling-challenge | about 100 h humanoid video + states | raw video CC-BY-NC-SA 4.0; tokenized data Apache-2.0 | states | Non-commercial for raw video |
| VPT contractor data (Minecraft), https://github.com/openai/Video-Pre-Training | about 2.5k h (unverified) | code MIT; data licence unverified | yes (keyboard/mouse) | What Dreamer 4 used; needs MineRL (Java) for evaluation |
| Ego4D / Ego-Exo4D, https://ego4d-data.org | 3.6k h / 1.4k h | Signed licence; research and model training allowed | no | Access agreement, not open |
| Something-Something v2, https://www.qualcomm.com/developer/software/something-something-v-2-dataset | 220k clips | Qualcomm research licence | no | Research only |
| Kinetics, HowTo100M | YouTube link lists | murky; videos decay | no | Avoid for a reproducible study |
| AgiBot World, https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta | 1M trajectories claimed | unverified | yes | Check licence before use |

Pretrained models usable as baselines or frozen encoders: V-JEPA 2 checkpoints (300M/600M/1B, https://github.com/facebookresearch/vjepa2), DINOv2/DINOv3, Cosmos-Predict2.5-2B under the NVIDIA Open Model License (commercial use allowed, https://huggingface.co/nvidia/Cosmos-Predict2.5-2B).

## 3. Candidate studies

### A. Latent world-model scaling on frozen video features, scored on planning (recommended core)
- **Question.** For latent predictors trained on public video, how do (i) prediction loss, (ii) loss on held-out scenes/embodiments, and (iii) closed-loop planning success scale with predictor size, pretraining hours, action-labelled hours, and planning compute? Do the exponents agree, and does loss predict success?
- **Why it fits you.** Direct line from LeWM / jepa-wms / V-JEPA work. Nobody has fitted joint laws for latent predictors with planning as the target (World-in-World did data-only on generative models).
- **Setup.** Frozen encoder (DINOv2 or V-JEPA 2 ViT-L). Encode the corpus once (about 10 GPU-hours per 10k video hours at 4 fps with a ViT-L; my estimate). Train predictors 10M to about 1B parameters on OXE + DROID video (action-free), post-train action conditioning on DROID / simulator data, evaluate planning in ManiSkill or LIBERO with one frozen planner (CEM/MPPI) and fixed budgets, following the World-in-World protocol so results are comparable.
- **Compute (estimate).** IsoFLOP grid 1e17-1e21 with 4 points per decade and 2-3 seeds: about 10-15k GPU-hours; hyperparameter sweeps about 15%; planning evaluation is the slow part (sim rollouts) and needs measuring. Fits in 25-35% of u6xn. Multi-node only for the top 2-3 runs.
- **Risks.** The headroom test from before still applies: planning with the true simulator must beat simple baselines in ManiSkill/LIBERO, or the decision axis is flat. Sim-only planning evaluation, so no real-robot claim. Frozen encoder caps what "scaling" means (the encoder is fixed); that's a design choice to state up front.

### B. Latent vs generative at matched compute, scored on decisions (architecture axis for A)
- Train a pixel/latent-diffusion world model family on the same data and compare with A's latent predictors at 3-4 matched FLOP budgets on planning success and on state-level accuracy (2609.36599 style). Cosmos-Predict2.5 as an off-the-shelf generative reference.
- New if measured on decisions across scales; LeWM and 2609.36599 give hints but no matched comparison. Adds about 30-50% to A's cost because generative models are more expensive per FLOP of useful prediction. Do it at 3 sizes, not the full grid.

### C. Training compute vs planning compute (test-time) frontier
- For each model in A, sweep planner samples, horizon and iterations; fit success = f(train FLOPs, plan FLOPs). This is the LLM "compute-optimal test-time scaling" question for world models. SWIFT and DeepJEPA show gains exist, World-in-World shows it in closed loop, but no one has mapped the frontier. Cheap: inference only. Strongest "agentic planning" angle that stays on video.

### D. Offline agents in Minecraft (Dreamer 4 setting)
- Scale world-model size and labelled fraction on the VPT contractor data; train imagination policies; measure milestone success (log, planks, pickaxe, ..., diamond) against world-model loss.
- Best match to "public video + agentic planning" literally, but highest risk: Dreamer 4's compute is unknown to me, MineRL evaluation is slow and Java-based, and one person building tokenizer + dynamics + RL-in-imagination + eval is a lot. Treat as a stretch after A/C.

### E. LLM world models for web/GUI agents
- Scaling law for simulated-transition accuracy vs task success (WebArena/OSWorld). Not video; data is LLM-synthesised; results depend on which frontier LLM you use. Weak fit for u6xn and for your trajectory. Not recommended as the core.

## 4. Recommendation
A + C as the core (one data pipeline, one planner, one simulator suite), B at 3 sizes as the architecture comparison, D only if time remains. Claim stays narrow: "for latent world models trained on public robot/ego video and evaluated in simulation, here is how loss, shifted loss and planning success scale with model, data and planning compute, and whether they agree."

## 5. Week-1 checks (CPU/1-GPU, no Isambard needed)
1. Licence confirmation for OXE, DROID, 1X, VPT data (read the actual licence files).
2. Headroom: true-dynamics CEM vs scripted/random baselines on 3 ManiSkill or LIBERO tasks, 50 seeds each. Kill the decision axis if headroom < 3x seed SD.
3. Encode 100 hours of DROID with DINOv2 ViT-L and V-JEPA 2 ViT-L; measure throughput and storage per hour.
4. Train a 10M and a 50M latent predictor for 1 GPU-hour each; measure MFU and loss curves to size the grid.
5. Confirm u6xn end date and usage rules.
