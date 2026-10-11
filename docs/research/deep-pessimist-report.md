# Red-team review: "do loss, shifted loss and decisions scale together?" (2026-10-07)

Role: pessimist. Web research and reasoning only. Nothing was cloned, downloaded, trained or run on Isambard. Every factual claim has a URL. **[V]** means I read it on the linked page. **[I]** means my inference. **[U]** means unverified.

## 0. Bottom line

- The question as worded ("does total predictive loss improve together with decision quality?") has mostly been answered already, and the answer is "not reliably". Several 2026 papers show this directly. A study whose headline is "loss scales, decisions don't" is predictable, so reviewers will call it confirmatory.
- The version that is still open is narrower: **does a planner-relevant error (error on the states the planner actually reaches) follow a predictable scaling law, and does it predict success at a held-out larger scale?** That is not just "measure DRPE instead". Nobody has shown DRPE or reachable-state error to be scale-predictable, and you need a simulator to measure it.
- The biggest single risk is that planning success barely moves as the predictor scales, because the planner's cost function is the bottleneck (2608.12959). In that case the scaling law measures the cost function, not the model.
- Two substrates in six months for one person is not realistic. The physics substrate as framed ("plasma control" on The Well MHD) does not hold up: The Well has no actions, MHD_256 is interstellar-medium turbulence, and it has 100 trajectories.

## 1. Is the question well-posed?

**What the new papers show [V]:**
- 2607.10362: the planner queries the model on the states its candidate actions reach, and those states are usually off the data manifold. Planner suboptimality is bounded by twice the predicted-vs-true plan-cost gap. Data-averaged prediction error "neither bounds nor tracks it". In their latent MPC experiments, "single-step validation error does not separate a fourteen-point spread in success across seeds". The theory assumes a linear-control premise. https://arxiv.org/abs/2607.10362
- 2609.32322 (DRPE): 55 models in a *factored gridworld with known state relevance*. Total error vs success has Spearman ρ = −0.25; DRPE has ρ = −0.84. Models 1% apart in total error differ by 60 points in success (97% vs 37%). Model rankings reverse across tasks at the same total error. https://arxiv.org/abs/2609.32322
- 2608.12959: on LeWorldModel/TwoRoom, the CEM objective (squared latent distance) tracks true distance at only r = 0.426 and stops being monotone beyond ~120 units. Replacing only the objective lifts success at offset 100 from 26.0% to 98.0%, with no retraining. Across four checkpoints, long-horizon success orders *inversely* with one-step accuracy. https://arxiv.org/abs/2608.12959
- 2608.10145 (the reproduction 2608.12959 builds on): changing only how the goal is constructed moves the authors' checkpoint from 84.0% to 8.0% on the same 50 episodes. One-step accuracy "fails to order long-horizon success at all" across a sevenfold range of prediction error. https://arxiv.org/abs/2608.10145
- 2609.36599 (2×2×2 Rubik's cube video): validation MSE follows an approximate power law, but "lower MSE loss does not reliably indicate downstream reasoning capabilities". At ~0.1 PF-days a 70M model gets 44.6% state accuracy against 0.3% for a 1B model. https://arxiv.org/abs/2609.36599
- VIScore 2608.11174: a metric covering encoder, predictor and planner reaches Spearman > 0.75 with success on seen and unseen models. https://arxiv.org/abs/2608.11174
- LLM evidence: only 39% of downstream scaling laws are predictable, and "seemingly benign changes to the experimental setting can completely change the scaling behavior" (https://aclanthology.org/2025.findings-emnlp.877/). Schaeffer et al. show that the chain of transformations from loss to a discrete downstream metric degrades predictability (https://proceedings.mlr.press/v267/schaeffer25b.html). Relative scaling laws show that gaps between test distributions can close, persist or widen with scale (https://arxiv.org/abs/2510.24626).

**What a scaling law of total loss can still tell you [I]:**
1. Whether the predictor is still improving on its own training distribution, i.e. whether you are compute-limited or capacity-limited. That is useful for budget allocation and nothing more.
2. Whether shifted loss keeps a constant ratio to in-distribution loss as scale grows (a relative scaling law). That question is still open for world models, but it is a statement about loss, not about decisions.
3. A null baseline: the study has to *show* that total loss fails to predict success at the held-out top run, so that the planner-relevant metric has something to beat. On its own this is not a contribution.

**Does the study collapse into "measure DRPE instead"? Partly [I].**
- DRPE as defined needs known decision-relevant state dimensions. You don't have those in a V-JEPA latent space. The 2607.10362 metric needs the true plan cost on off-manifold states, i.e. a simulator you can reset to arbitrary states. LIBERO (robosuite) and HydroGym allow that. Real-robot video (OXE, DROID) does not. So the planner-relevant metric is only measurable in the evaluation environment, not on the training data.
- The defensible reframing is: (a) total loss, (b) shifted loss, (c) **reachable-state cost gap** (2607.10362) or VIScore, and (d) success, all measured across scale, with a pre-registered prediction of (c) and (d) at the top run. The new claim is "(c) is scale-predictable and (d) follows (c)". If (c) is not scale-predictable either, the paper is a negative result about world-model scaling for control. That is publishable at a workshop and hard to place at a main venue.
- If success is limited by the cost function (2608.12959), no predictor metric will track success. The study then becomes a study of planning objectives. That would need an oracle-cost arm (the true simulator cost on predicted states) to separate the predictor from the objective.

## 2. Confounds in a scaling study with a frozen planner

| Confound | Evidence | Why it breaks the law | Mitigation |
|---|---|---|---|
| Planner objective | 2608.12959: 26%→98% from the objective alone [V] | Success may be flat in predictor scale; the slope reflects the cost function | Run three cost arms: latent L2, learned reachability cost, oracle simulator cost. Report the scaling law per arm. This roughly triples eval cost [I] |
| Planner budget | SufficientPlan: the sufficient budget "varies across model–task pairs"; 2,400 vs 9,000 sequences gave the same 96% on Push-T [V] https://arxiv.org/abs/2610.08350 | A fixed budget favours some model sizes. Larger predictors also cost more FLOPs per query, so "same budget" in samples is not the same in FLOPs | Sweep budget in FLOPs, not samples. Report success at matched planning FLOPs |
| Evaluation protocol | Goal construction alone: 84%→8% [V] (2608.10145) | Small protocol choices are larger than scale effects | Freeze the protocol and episode list before Stage 1. Publish them |
| Seed variance vs effect size | 14-point success spread across seeds (2607.10362) [V]; few-run RL point estimates mislead [V] https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html | BUDGET_DESIGN plans 1 seed at the top run. One seed at the point the headline depends on is not defensible [I] | Run a power analysis in Stage 0: measure seed SD at two sizes and size the grid from it. Use ≥3 seeds at any point in the fit and rliable intervals |
| Encoder quality (frozen V-JEPA 2.1) | V-JEPA 2.1 is a fixed ViT-G family [V] https://arxiv.org/abs/2603.14482 | Scaling becomes predictor-only. The encoder caps what the predictor can know. LeJEPA's identifiability result is about a jointly trained encoder with a Gaussian regulariser under stationary additive-noise latents [V] https://arxiv.org/abs/2605.26379, so it does not carry over to a frozen V-JEPA encoder [I] | Either state this as "predictor scaling on a fixed representation", or add an encoder-size axis (ViT-L/H/G), which costs another grid dimension |
| Tokenizer / architecture | Pearce et al.: scaling coefficients "are heavily influenced by the tokenizer, task & architecture" [V] https://proceedings.mlr.press/v267/pearce25a.html | Exponents fit for one pooling/token scheme won't transfer to another | Fix one token scheme. Report sensitivity at one compute level |
| Data entropy / mixture | OXE is a mixture of many contributed datasets [V] https://github.com/google-deepmind/open_x_embodiment; relative scaling laws show subpopulations diverge [V] | Growing D by adding sub-datasets changes the distribution, not just its size | Grow D by uniform subsampling of a fixed mixture. Treat mixture as a separate shift axis |
| Discrete success metric | Schaeffer et al. [V] | Success is binary per episode; near 0% or 100% the curve is flat regardless of model | Also report continuous decision metrics: final distance to goal and regret against an oracle planner |
| Matched compute, latent vs generative | Cosmos 3 / OpenWAM are 5B-class, pretrained on 10k+ hours (user notes; OpenWAM https://arxiv.org/abs/2610.07922) | Fine-tuning them leaves their pretraining FLOPs outside your FLOP axis. Any "matched-compute" claim is false [I] | Call it a reference point, not a matched comparison. Or count pretraining FLOPs and accept that the generative point sits far right |

## 3. Substrate-specific risks

### 3a. Video: latent predictors on frozen V-JEPA 2.1, planning in LIBERO
- **Licences.** OXE: software Apache-2.0, "all other materials" CC-BY 4.0, but it asks you to cite the contributed datasets, and I found no blanket licence that overrides upstream terms [V/U] https://github.com/google-deepmind/open_x_embodiment. DROID: CC-BY 4.0 [V] https://arxiv.org/html/2403.12945. InternW0-Δ: "Processed data will be shared where licenses permit"; no per-dataset licence list found [V/U] https://arxiv.org/html/2609.31394v1. LIBERO: code MIT, datasets CC-BY 4.0 [V] https://github.com/Lifelong-Robot-Learning/LIBERO. V-JEPA 2 card says apache-2.0, the repo code is mostly MIT, and I found **no checkpoint-specific licence for the V-JEPA 2.1 weights** [U] https://github.com/facebookresearch/vjepa2. Licence risk is low for OXE/DROID, unknown for InternW0-Δ data and V-JEPA 2.1 weights.
- **Training data and test environment don't match [I].** OXE/DROID are real robots with varied action spaces and cameras. LIBERO is simulated Franka in robosuite. A predictor trained on OXE/DROID cannot plan in LIBERO without action-space alignment and fine-tuning on LIBERO data. Then "data scaling" is really "pretraining data scaling, measured through a fine-tune". That adds a confound (fine-tune budget) and an axis reviewers will ask about.
- **LIBERO headroom works against you in both directions.** Policies already score about 95–99%: OpenVLA-OFT 97.1, π0 94.2 (LIBERO-Plus table [V] https://arxiv.org/pdf/2510.13626), LaWAM 98.6% [V] https://arxiv.org/html/2606.15768, VLA-JEPA 97.2% [V] https://www.emergentmind.com/papers/2602.10098. A CEM latent planner will very likely score far lower [I]. Reviewers will ask why anyone should care about scaling a method that is 50 points behind a policy. LIBERO tasks are language-specified, so goal-image planning means redefining the tasks, and results won't be comparable to the published numbers [I]. LIBERO-Plus shows drops from 95% to below 30% under perturbations [V] https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf. That is a ready-made shift axis, and also a crowded one.
- **Headroom test.** BUDGET_DESIGN's Stage-0 gate (true-state planner vs scripted policy) is the right idea, but LIBERO ships human demonstrations, not scripted policies [I; U on the exact demo count]. The relevant gate is: does an oracle-dynamics planner using the *same cost* succeed, and does a small learned predictor fail? If the oracle planner also fails, the cost function is the ceiling (see §2).
- **Frozen encoder.** With V-JEPA 2.1 frozen, the 10M–1B grid scales only the predictor on a fixed ~1B-class representation. JEPA-WMs (Meta FAIR) already varied encoder size (ViT-S/B/L) and predictor depth (3–12) against planning success [V] https://arxiv.org/html/2512.24497v4. A predictor-only law is a narrower result than the working question implies.

### 3b. Physics: The Well MHD, control in HydroGym
- **MHD_256 is not plasma control.** The dataset card says isothermal MHD without self-gravity, "such as found in the diffuse ISM", with Ms ∈ {0.5, 0.7, 1.5, 2.0, 7.0} × MA ∈ {0.7, 2.0}, 10 initial conditions × 10 parameter combinations = 100 trajectories, 100 timesteps of 256³, 4.58 TB [V] https://polymathic-ai.org/the_well/datasets/MHD_256/. It has no actuators, no tokamak geometry, no confinement and no wall. Calling it "plasma control" would be judged misleading by any fusion reviewer [I]. The honest framing is "compressible MHD turbulence surrogate under Mach-regime shift".
- **No action-conditioned data [I].** The Well consists of autonomous simulations. A decision metric there is limited to inverse design over the 10 parameter combinations or the initial conditions. HydroGym is a *different* system (cylinder, cavity, pinball, wall turbulence, airfoil, nozzle, shock [V] https://www.nature.com/articles/s41586-026-10917-6). So in this design the scaling law is fitted on one system and the decision is measured on another. That breaks the "do they improve together" logic unless you generate action-conditioned data in HydroGym yourself.
- **Data axis on MHD_256 [I].** 100 trajectories × 100 steps = 10,000 snapshots. Holding out a Mach regime for shift leaves roughly 80 trajectories. Subsampling gives about 1.5–2 decades of D at most, with very few independent trajectories at the low end. That is thin for a power-law fit and very thin for a pre-registered extrapolation.
- **HydroGym cost and headroom.** The supplement says evaluations "can require hours on high-performance computing clusters"; I found no per-episode figure [V/U] https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10917-6/MediaObjects/41586_2026_10917_MOESM1_ESM.pdf. The reported headline is RL policies beating opposition control and uniform blowing (11.6% relative drag reduction) [V] (MOESM2). I did not find model-based planning with a learned surrogate beating those baselines in HydroGym [U]. Backends are Firedrake, MAIA LBM/FV, NEK5000 and JAX-Fluids [V] https://dynamicslab.github.io/hydrogym/. The breakeven paper argues neural solvers only pay off once enough forward solves amortise data and training cost [V] https://arxiv.org/abs/2605.15399. A physics reviewer will ask for that accounting.

### 3c. Fusion: TORAX, FreeGSNKE, RL4F, TokaMark
- **TORAX** is a 1D transport simulator in JAX, Apache-2.0, needs jax ≥ 0.10 and Python ≥ 3.11 [V] https://pypi.org/project/torax/1.4.3/. Gym-TORAX currently ships one environment (ITER hybrid ramp-up) [V] https://arxiv.org/html/2510.11283v2. A simulator gives exact ground truth and unlimited data, so a data axis is possible. But it is reduced physics, and the scaling law would describe a surrogate of TORAX, not of a plasma [I].
- **FreeGSNKE**: LGPL v3, Python 3.10–3.14 [V] https://github.com/FusionComputingLab/freegsnke. It is a free-boundary equilibrium solver, i.e. slow magnetic control, not turbulent dynamics [I].
- **RL4F is circular as ground truth [V+I].** "We first train a reference dynamics model from historical DIII-D experimental discharges, and then use this model to generate trajectories ... evaluated in closed loop on the reference dynamics model" https://arxiv.org/html/2606.07550. Scaling a world model and scoring it in RL4F measures how well one learned model imitates another learned model. Out-of-distribution "shift" there is shift of the reference model, which has no physical ground truth. Seo et al. (Nature 2024) used the same pattern but then validated on DIII-D hardware [V] https://www.nature.com/articles/s41586-024-07024-9. You can't do that step.
- **TokaMark**: 14 forecasting/reconstruction tasks on MAST [V] https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark; GitHub licence "Other", exact terms not confirmed [U]. FAIR-MAST data is CC-BY-SA 4.0 [V] https://www.ukaea.org/service/fair-mast/. There are no actions and no closed loop, so no decision metric. Share-alike affects derived datasets [I].

## 4. Novelty: closest existing work

**Formulation A — video/latent world models, loss vs decisions vs scale**
1. Pearce et al., ICML 2025: power laws for world-model loss vs model size. Loss only, no decisions [V] https://proceedings.mlr.press/v267/pearce25a.html
2. World-in-World, ICLR 2026: closed-loop evaluation, "the first data scaling law for world models in embodied settings", and "allocating more inference-time compute ... substantially improve[s] closed-loop performance" [V] https://arxiv.org/html/2510.18135. **This covers data scaling against success plus inference compute, for generative WMs.** It is the closest prior work to the study.
3. JEPA-WMs (Terver et al.): model size, predictor depth, encoder and planner choices against planning success [V] https://arxiv.org/html/2512.24497v4
4. Atari WM scaling 2605.08578: distinct scaling regimes; joint training makes scaling monotonic; fidelity transfers to control (median 0.770) [V] https://arxiv.org/abs/2605.08578
5. 2609.36599 Rubik's cube video: MSE power law that does not predict state accuracy [V]. Also VIScore and 2607.10362 on the metric side.

Verdict [I]: nobody has fitted joint laws for in-distribution loss, shifted loss and planning success with a planner-budget axis and a pre-registered extrapolation, as far as I found. Every component has been done separately, and World-in-World already covers two of the three. Expect "incremental" reviews unless the reachable-state metric and the pre-registered test carry the paper.

**Formulation B — "planning compute vs training compute"**
1. Jones 2021 (Hex/AlphaZero): "for each extra order of magnitude of train-time compute, we can reduce test-time compute by a similar factor" [V] https://arxiv.org/pdf/2104.03113
2. World-in-World: inference compute improves closed-loop performance [V]
3. SufficientPlan 2610.08350: certified reduced planning budgets [V]
4. DeepJEPA 2610.00368: transition depth as a test-time axis (from user notes; not re-read) [U]
5. 2608.12959: a better cost beats more search and reaches 92% at a third of the budget [V]

Verdict [I]: the trade-off has been shown in board games and partly in WMs. A train/plan frontier for latent WMs under shift is new, but reviewers will cite Jones as the precedent.

**Formulation C — PDE/MHD surrogates, scale vs shift vs control**
1. Subramanian et al., NeurIPS 2023: scaling of pretrained SciML models with model size, downstream data and **physics parameters pushed out of distribution** [V] https://proceedings.neurips.cc/paper_files/paper/2023/file/e15790966a4a9d85d688635c88ee6d8a-Paper-Conference.pdf
2. Poseidon, NeurIPS 2024: "scales with respect to model and data size" [V] https://proceedings.neurips.cc/paper_files/paper/2024/hash/84e1b1ec17bb11c57234e96433022a9a-Abstract-Conference.html
3. Walrus (Polymathic): 1.3B, trained on 19 Well-style datasets on 96 H100s, MIT [V] https://huggingface.co/polymathic-ai/walrus. It is the obvious baseline and is far larger than any Well model you would train.
4. Breakeven complexity 2605.15399: scaling laws used for cost-aware evaluation [V]
5. DiffPhyCon (NeurIPS 2024) and HydroGym: control with learned models / RL control benchmarks [V] https://arxiv.org/html/2407.06494v4

Verdict [I]: scaling plus OOD shift for PDE surrogates is done (Subramanian). Linking surrogate scale to closed-loop control quality is open as far as I found [U]. But see §3b: The Well can't measure it.

**Formulation D — fusion**
1. Char et al. 2023: offline model-based RL for tokamak control, learned dynamics on DIII-D [V] https://proceedings.mlr.press/v211/char23a/char23a.pdf
2. Seo et al., Nature 2024: learned dynamics model as RL training environment, validated on DIII-D [V]
3. RL4F 2606.07550: offline RL benchmark; "offline model-based RL methods obtain the best average performance" [V]
4. Gym-TORAX 2510.11283 [V]; Degrave et al., Nature 2022 (TCV magnetic control, simulator-trained) [V] https://www.nature.com/articles/s41586-021-04301-9
5. QLKNN / ADEPT: surrogate data efficiency for transport (up to 20× less data via active learning) [V] https://iopscience.iop.org/article/10.1088/1741-4326/ad240d

Verdict [I]: no scaling-law study of plasma dynamics models against control quality found [U]. Real data is too small for laws, and the only closed-loop benchmark is circular (RL4F).

## 5. Isambard-specific risks
- **Walltime:** workq max 24 h [V] https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/. Every run over ~90 GPU-h on 4 GPUs needs checkpoint/resume. Chained jobs wait in the queue between chunks [I].
- **Storage:** $SCRATCHDIR 5 TiB, "Files ... not accessed for 60 days will be deleted"; $PROJECTDIR 20 TiB or 200 TiB depending on system column [V] https://docs.isambard.ac.uk/user-documentation/information/system-storage/ (check which column applies to u6xn). BUDGET_DESIGN estimates ~35 TB of features for 5k hours of video. That is 7× scratch and above a 20 TiB project quota. MHD_256 alone is 4.58 TB. **Storage, not GPU-hours, sets the data axis.**
- **Access Terms:** use only "for the purposes of the project ... for which you are given System access ('Research Project')" [V] https://docs.isambard.ac.uk/policies/access_terms/. If u6xn was awarded for OpenChange-UK/EO (as UK_FIT.md flags), a world-model scaling study may be out of scope without written agreement from the allocator. That is a hard stop and needs checking first.
- **Hours expire:** "Any node hours remaining at the end of a project will be lost" [V] https://docs.isambard.ac.uk/user-documentation/guides/accounting/. BriCS cannot extend hours or end dates [V] https://docs.isambard.ac.uk/access/. The project end date is still unknown (BUDGET_DESIGN §6.5).
- **aarch64:** JAX ships CUDA wheels for Linux aarch64 [V] https://docs.jax.dev/en/latest/installation.html. Stim 1.16.0 has no Linux aarch64 wheel [V] https://pypi.org/project/stim/1.16.0/#files. Firedrake, NEK5000 and MAIA on Grace: not checked [U]. JAX-Fluids is the low-risk HydroGym backend [I]. LIBERO/robosuite/MuJoCo headless EGL rendering on aarch64: not confirmed [U]. If rendering fails, LIBERO evaluation can't run on Isambard and must move elsewhere.
- **Containers:** only aarch64 images; multi-node needs brics/nccl + aws-ofi-nccl and brics/apptainer-multi-node [V] https://docs.isambard.ac.uk/user-documentation/guides/nccl/.

## 6. Reviewer objections

**NeurIPS / ICLR / ICML [I]**
1. "It is already known that loss doesn't predict control" (cites 2607.10362, 2609.32322, 2608.10145, 2609.36599). The negative half of the result is not new.
2. "Your finding is about CEM with an L2 latent cost, not about world models" (cites 2608.12959).
3. "Frozen encoder, so this is predictor scaling only." "Why V-JEPA 2.1 and not DINOv3 or a jointly trained LeJEPA?"
4. "1–3 seeds, no intervals; differences are inside seed noise" (cites rliable).
5. "LIBERO numbers are 50 points below VLA policies; why does this matter?"
6. "Your compute range covers ~4 decades at most; extrapolating one half-decade is a weak test."
7. "The generative comparison isn't matched-compute."
8. "World-in-World already reported data scaling and inference-compute scaling in closed loop."
9. "Two substrates, each under-powered."

**Physics / fusion venues (e.g. Nuclear Fusion, PPCF, JCP) [I]**
1. "ISM turbulence is not tokamak plasma; the 'plasma control' framing is wrong."
2. "The Well has no actuators; the HydroGym flows are canonical cylinder/cavity cases."
3. "No breakeven accounting against classical solvers" (cites 2605.15399).
4. "RL4F success is agreement with another learned model, not with a plasma."
5. "Where is the physical insight? A scaling exponent of a surrogate is not physics."

**UK funding [V from UK_FIT.md notes]:** the Dec 2026 UKRI call excludes projects that "focus primarily on advancing AI methods". This study is a methods study, so it only fits as AI-methods research, whatever the substrate.

## 7. Timeline (one person, ~6 months) [I]
- Stage 0 infrastructure: containers on aarch64, feature cache, LIBERO rendering on Isambard, planner, protocol freeze, seed-variance pilot. 4–6 weeks if nothing breaks. Rendering or solver builds on ARM can add weeks.
- Stage 1 small grid plus fits: 3–4 weeks.
- Stage 2 main grid: 6–8 weeks. The top decade needs multi-node runs in 24 h chunks with queue waits between them.
- Stage 3 pre-registered run plus evaluation: 2–3 weeks.
- Writing: 3–4 weeks.
- Total for **one** substrate: 18–25 weeks. This leaves no room for the physics substrate, which needs its own data generation (HydroGym CFD) and pipeline. Two substrates in 6 months is not realistic. One substrate is tight but possible.
- The field moves monthly (WM_UPDATE_OCT2026.md). A 6-month project is likely to be partly overtaken before submission.

## 8. The 10 most likely ways the project dies (ranked)

| # | Failure | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Planning success is flat in predictor scale because the planner's cost function is the ceiling (2608.12959) | High | Stage 0: two predictor sizes × {latent L2, oracle simulator cost}. If the oracle-cost arm is also flat, there is no headroom; kill or switch the task |
| 2 | Seed and protocol variance is larger than the scale effect (14-point seed spread; 84%→8% from goal construction) | High | Measure seed SD in Stage 0; ≥3 seeds per fitted point; paired episodes; rliable intervals; freeze protocol before Stage 1 |
| 3 | Judged incremental: World-in-World, JEPA-WMs, the Atari study, VIScore, DRPE and 2607.10362 each cover part of it | High | Make the headline "reachable-state error is scale-predictable and predicts success at the pre-registered top run"; preprint early |
| 4 | Physics substrate fails: The Well has no actions, MHD_256 has 100 trajectories and is ISM, HydroGym is another system, CFD is expensive | High | Drop it, or generate action-conditioned data in HydroGym (JAX-Fluids) and call it fluid control, not plasma |
| 5 | Result is "loss scales, decisions don't", which reviewers already expect | Med-high | Pre-commit to the positive test (metric (c) predicts success); report the negative honestly as a secondary finding |
| 6 | Storage caps the data axis (35 TB of features vs 5 TiB scratch with a 60-day purge; project quota 20 or 200 TiB) | Med-high | Pool tokens, stride clips, keep features in $PROJECTDIR, set max D from the quota before designing the grid |
| 7 | Timeline and hours run out (one person, unknown end date, unused hours lost) | Med-high | One substrate only; dated stage gates; spend to schedule; get the end date now |
| 8 | No decision headroom in LIBERO: policies at ~97%, CEM planners near 0% or redefined tasks not comparable | Med | Stage-0 headroom gate with an oracle-dynamics planner; choose tasks in the mid range; report regret, not just success |
| 9 | aarch64 infrastructure blocks evaluation (LIBERO/MuJoCo EGL, Firedrake/NEK, Stim) plus 24 h chunking | Med | 1-GPU smoke tests of every simulator in week 1; run simulators on Grace CPUs; checkpoint/resume from day one |
| 10 | u6xn's Access Terms scope doesn't cover this study | Med (unknown) | Get written confirmation from the allocator before Stage 0. If refused: none on u6xn; apply for an AIRR Gateway allocation instead |

Not in the top 10 but real: the latent-vs-generative comparison can't be matched-compute (call it a reference point); InternW0-Δ data and V-JEPA 2.1 weight licences are unconfirmed (use OXE/DROID and check the weights' licence); UK domain-first calls won't fund a methods study (use general AI routes).

