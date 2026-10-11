# World-model news check, 2026-10-07

Sources: web search and page fetches. X itself could not be read logged-out; I used mirror pages for @ylecun and @drfeifei (texxr.com) and ThreadReader. The @_akhaliq mirror was stale (Sept 2025), so X coverage is partial.

## What shipped, Sept–Oct 2026
Robotics world-action models (WAMs), the dominant line this autumn:
- OpenWAM (Stanford, Fei-Fei Li / Jiajun Wu, 6 Oct): Wan2.2-5B, causal robot-video pretraining on >10k h, Mixture-of-Transformers action expert; LIBERO-Long 68.4 -> 97.8%; open framework. https://arxiv.org/abs/2610.07922
- InternW0-Δ (Shanghai AI Lab, 25 Sep): WAM pretrained on 20k+ hours of *open* data; MoT with frozen VLM and 4D prior distillation. https://arxiv.org/abs/2609.31394
- FLEX-WAM (4 Oct, block-causal, KV-cacheable, "FD elasticity" to force action responsiveness) https://arxiv.org/abs/2610.05483; WALL-SS (Aug, next-scale AR long-horizon) https://arxiv.org/abs/2608.26239
- Kairos 3.1 (Jul): 4B open cross-embodiment WAM, Apache-2.0 weights incl. LIBERO-plus checkpoint. https://github.com/kairos-agi/kairos
- Cosmos 3 (NVIDIA, Jun): omnimodal MoT (language/image/video/audio/action); code, checkpoints, synthetic datasets and benchmark under OpenMDW-1.1. https://arxiv.org/abs/2606.02800
- Token-World (5 Oct): world model in VLM token space; closed-loop simulated policy performance correlates with real at r=0.79 vs 0.58 for Ctrl-World. https://arxiv.org/abs/2610.00575

Latent / JEPA line:
- V-JEPA 2.1 (16 Mar): dense features, +20 points real-robot grasping over V-JEPA 2-AC; HF transformers support. https://arxiv.org/abs/2603.14482
- LeJEPA (Nov 2025) and "When Does LeJEPA Learn a World Model?" (May 2026): linear identifiability of latents under Gaussian regulariser, enabling optimal latent-space planning; stable to 1.8B. https://arxiv.org/abs/2605.26379
- JEPA-Anything (5 Oct): one JEPA recipe (Orthogonal Predictive Factorization) across 7 domains incl. control, molecular dynamics, physical fields and weather. https://www.marktechpost.com/2026/10/05/beyond-domain-specific-world-models-jepa-anything-uses-1-recipe-for-7-fields/
- AMI Labs (LeCun, Paris): $1.03B seed, Mar 2026; still research phase; promises open code.

Planning compute:
- "How Much Planning Is Enough?" / SufficientPlan (6 Oct): sufficient planner budget varies per model–task pair; certifies reduced budgets from paired closed-loop evidence. https://arxiv.org/abs/2610.08350
- DeepJEPA (30 Sep): transition depth as test-time scaling axis. https://arxiv.org/abs/2610.00368

Scaling and evaluation:
- "Probing the Impact of Scale on ... World Models for Atari" (May): environments fall into distinct scaling regimes; joint multi-env training makes scaling monotonic; fidelity transfers to control (median 0.77). Nearest competitor to our question, in a toy domain. https://arxiv.org/abs/2605.08578
- "Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond" (Apr): levels x laws taxonomy (physical/digital/social/scientific), argues for decision-centric evaluation, 400+ works. https://arxiv.org/abs/2604.22748
- Benchmarks: WorldMark (action following, control-systems metrics) https://arxiv.org/abs/2604.21686; WorldRoamBench, HappyWorld-Bench, SANA-WM-Bench (long-horizon memory). Interactive-video evaluation is crowded; almost all score visuals and action following, not decisions.

Industry:
- AMD to acquire World Labs for $8.2B (28 Sep); World Labs Atlas (1 Sep). Odyssey-2 Max (Apr), Runway GWM Worlds 2 (3 Sep), Wayve GAIA-4 (3 Aug, closed-loop AV safety sim), Genie 3 via Project Genie (Jan), Gemini 4 Argon (30 Sep). Hafner left DeepMind (Nov 2025) for a humanoid world-model startup (Embo).
- A solo student trained a 960M real-time world model on 8xH100 for 3–4 weeks (~5–6k GPU-h). Scale reference for our 80k GPU-h.

## What this changes for us
1. Do not build a generative WAM. OpenWAM/InternW0-Δ/Cosmos 3 are 5B-class models on 10–20k hours of video with industrial teams. But they are open, so the generative side of a latent-vs-generative comparison (option B) can be an off-the-shelf model fine-tuned on our data, not trained from scratch. That makes B cheap.
2. The latent line is where a one-person, 80k GPU-h study can still own something. LeJEPA explicitly claims a training loss that tracks downstream quality and a latent geometry that supports optimal planning. Nobody has tested either claim as a function of scale and under distribution shift. That is a sharper version of our question than "do loss and decisions agree".
3. JEPA-Anything spans video, control, MD, physical fields and weather with one recipe. Our "one experiment, two substrates" plan (public video + The Well physics) is exactly what the field is now set up for, and there are no scaling laws for it.
4. Planning-compute (option C) is being populated at the inference-efficiency end (SufficientPlan, DeepJEPA). The open part is the joint law: success = f(train FLOPs, planner budget, shift). Keep C but state it that way.
5. Closed-loop evaluation is now standard (Token-World, World-in-World, GAIA-4). A decision metric is expected, not a contribution. The contribution has to be the pre-registered scaling law, the shift axis and the agreement test.
6. Data: InternW0-Δ's 20k+ h open corpus is a new candidate alongside OXE/DROID; licence to check. LIBERO (incl. LIBERO-plus, LIBERO-90 held-out) is the common simulator across OpenWAM and Kairos, so results are comparable.
7. Half-life risk: this field turns over every few months. Narrow claim, pre-registration, open frozen backbones (V-JEPA 2.1, Cosmos 3) to avoid being outdated by the next release.

## Revised recommendation
Core: latent predictors trained with the LeJEPA/V-JEPA 2.1 recipe on frozen video features, 10M–1B, on OXE/DROID (+InternW0-Δ data if licence allows); planning in LIBERO with one frozen planner; fit joint laws for in-distribution loss, shifted loss, planning success and planner budget. Second substrate: the same recipe on The Well (SCIENCE_TESTBEDS.md S1) with HydroGym or inverse design as the decision. Comparator: Cosmos 3 / OpenWAM fine-tuned at one or two sizes. Pre-register fits and the largest-run prediction.
