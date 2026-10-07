# Explorer report: making the scaling question well-posed (2026-10-07)

Labels: **[V]** = I read it in the source this session (arXiv abstract page, dataset card, licence file, Nature page, ACL Anthology). **[I]** = my inference. **[U]** = I could not confirm it. All GPU-hour figures are my estimates [I]; none of them has been measured.

Method: arXiv abstract pages and the arXiv API, raw GitHub files, nature.com, aclanthology.org, proceedings.mlr.press. The arXiv API, Semantic Scholar, OpenAlex and DuckDuckGo started rate-limiting partway through, so some lookups fell back to arXiv HTML search. The MarkTechPost page for JEPA-Anything did not load, so I used the arXiv paper instead. Nothing was cloned, downloaded or run.

## 0. What the new evidence says, and why the original question is ill-posed

Checked claims [V]:
- **Control Theory of Predictability**, https://arxiv.org/abs/2607.10362
  - The planner queries the states its candidate actions reach, and those are off the data manifold.
  - Planner suboptimality is at most 2x the plan-cost discrepancy at the committed plan. Data-averaged error "neither bounds nor tracks it".
  - In latent-MPC experiments, validation error is "essentially uncorrelated" with success, while fidelity on planner-reachable states tracks it.
- **DRPE**, https://arxiv.org/abs/2609.32322
  - Factored gridworld, 55 models. Total error vs success: Spearman -0.25. DRPE: -0.84 (-0.98 within the controlled family).
  - A 1% difference in total error gave 97% vs 37% success. Model rankings reverse across tasks.
- **Objective Is the Bottleneck**, https://arxiv.org/abs/2608.12959
  - LeWM on TwoRoom. Latent squared distance tracks true distance at only r=0.426.
  - Changing only the objective (no retraining) raised offset-100 success from 26% to 98%.
  - Across 4 checkpoints, long-horizon success ranks inversely with prediction accuracy.
- **VIScore**, https://arxiv.org/abs/2608.11174: a combined encoder+predictor+planner score gets Spearman >0.75 with success on seen and unseen models.
- **Rubik's cube video**, https://arxiv.org/abs/2609.36599
  - Validation MSE follows an approximate power law but does not reliably indicate state accuracy.
  - At ~0.1 PF-days, 70M gets 44.6% vs 0.3% for 1B; 1B reaches 83.7% at 3.14 PF-days.
- **Scaling Laws Are Unreliable**, https://aclanthology.org/2025.findings-emnlp.877/: downstream scaling is predictable in 39% of cases.
- **Schaeffer et al.**, https://arxiv.org/abs/2406.04391 (listed in PMLR v267, ICML 2025): the chain of transformations from NLL to the downstream metric degrades predictability; you need to predict the probability mass on specific wrong answers too.
- **Relative Scaling Laws**, https://arxiv.org/abs/2510.24626: 255 IsoFLOP models, 1e18–1e20 FLOPs. Gaps between test populations converge, shift or split with scale.
- **Atari WM scaling**, https://arxiv.org/abs/2605.08578: per-environment scaling regimes differ; joint training makes scaling monotonic; fidelity transfers to control (median 0.770).
- **SufficientPlan**, https://arxiv.org/abs/2610.08350: the sufficient planning budget varies per model–task pair.
- **Breakeven complexity**, https://arxiv.org/abs/2605.15399: neural PDE solvers pay off more as problems get harder.
- **HydroGym**, https://www.nature.com/articles/s41586-026-10917-6 (Nature, Sept 2026): 60+ environments, Re up to 4x10^5; zero-shot transfer to a 3D wing section cut local skin friction by 38%.
- **OpenWAM**, https://arxiv.org/abs/2610.07922: LIBERO-Long 68.4 -> 97.8%; 84.0% on four held-out LIBERO-90 tasks.
- **JEPA-Anything**, https://arxiv.org/abs/2609.20800: orthogonal predictive factorisation across 7 domains.

**Why the question as worded is ill-posed [I].**
1. "Decision quality" is not a property of the model alone. Swapping the objective moved success by 72 points (2608.12959). A law of success against training compute that leaves the planner and objective free is confounded.
2. Loss is measured on the data distribution, but the planner uses the model elsewhere (2607.10362). Data loss and decision quality can scale differently for a structural reason.
3. The protocol alone can swing results. The same released LeWM weights scored 14% under the appendix protocol and 84% under the repo config (https://arxiv.org/abs/2608.10145 [V]). Pre-register the protocol.

**Design rule [I]: a 2x2 oracle factorial.** Dynamics {learned model, true simulator} x cost {latent/learned cost, true task cost}, with planner and budget held fixed.
- S(true, true) is the ceiling.
- S(true, true) − S(learned, true) is the loss you can blame on the predictor.
- S(true, true) − S(true, latent) is the loss you can blame on the objective.

This needs a simulator you can reset to an arbitrary state. That rules out video-only setups, and weakens RL4F, whose evaluation dynamics are learned from DIII-D data.

## 1. Three well-posed reformulations

### F1. Scaling of planner-reachable fidelity vs data fidelity
- **Question.** As training compute C grows, does the error on the states the planner queries, E_reach(C), fall at the same rate as held-out data error, E_data(C)? Which one predicts regret at scale?
- **Definitions.**
  - E_reach: k-step error on the planner's candidate sequences (final-iteration CEM candidates or elite set), scored against the true simulator from the same start state.
  - E_rel (DRPE variant): the same error, restricted to the state dimensions the cost depends on.
  - Regret: true-cost return of the oracle cell minus that of the executed plan.
- **Prediction (fix before running).**
  - (a) α_reach < α_data, with the bootstrap 95% CI of the ratio excluding 1.
  - (b) A law fitted on runs ≤ C_max/30 predicts top-run regret better with E_reach (or E_rel) as the proxy than with E_data.
  - Also test iso-E_data pairs (within 5%) that differ in architecture or data mix; raw correlation over the grid is confounded by compute.
- **Primary metric.** Absolute error in predicted top-run regret, E_reach-based vs E_data-based, with bootstrap CI.
- **Falsifier.** The ratio CI contains 1 and E_data predicts as well. Then the original question is well-posed on this substrate. That is a publishable negative result.

### F2. Relative shift scaling (gap laws)
- **Question.** For one interpolation shift and one extrapolation shift fixed in advance, how do R_L(C) = L_shift/L_id and R_J(C) = regret_shift/regret_id scale with compute?
- **Prediction.**
  - Interpolation: slope of log R_L against log C is below 0.
  - Extrapolation: slope ≥ 0.
  - The R_J slope differs from the R_L slope in sign, or by more than its CI.
- **Primary metric.** The slopes with bootstrap CIs; pre-register the expected signs.
- **Falsifier.** Both slopes negative and equal within CI.
- **Must fix in advance.** No test-time adaptation in the main arm. JEPA-TTT (https://arxiv.org/abs/2610.00722 [V]) cut latent error 83% and raised planning 153% under shift through test-time training alone.

### F3. Train–plan exchange rate, objective held fixed
- **Model [I].** S(C,B) = S_max − a·C^−α − b·B^−β. Exchange rate ρ = −∂log C/∂log B at fixed S.
- **Prediction.**
  - (a) ρ is non-zero only below a sufficient budget B*(C), consistent with 2610.08350.
  - (b) B* changes under shift; pre-register the sign. More inference compute helping (World-in-World, https://arxiv.org/abs/2510.18135 [V]) argues for "grows". Planner exploitation of model errors (Hi-LeWM, https://arxiv.org/abs/2607.12547 [V]) argues for "shrinks".
- **Primary metric.** ρ at S = 0.5·S(true,true), plus B*(C) in-distribution and under shift.
- **Falsifier.** ρ ≈ 0 everywhere, or ρ unchanged under shift.
- **Needs the true-cost arm.** With latent cost, more search can lower success (2608.12959).

## 2. Minimal viable experiments (<500 GPU-h each)

### 2a. Video / LIBERO
- **Data and models.**
  - LIBERO-90 demos (LIBERO: 4 suites, 130 tasks, teleop demos, https://arxiv.org/abs/2306.03310 [V]) with a frozen V-JEPA 2.1 encoder.
  - 4 predictor sizes (5M–300M) x 3 demo fractions.
  - Fixed CEM at 3 budgets; latent-cost and true-cost arms.
- **Shift.** LIBERO-Plus camera viewpoint, graded. VLAs fell from 95% to below 30% under modest perturbations (https://arxiv.org/abs/2510.13626 [V]).
- **Decision metric.** Success on paired initial states.
- **Oracle.** Cloned MuJoCo state for every candidate, frames encoded by the frozen encoder. This also gives E_reach.
- **Cost.** ~250–400 GPU-h [I].
- **Kill.**
  - The predictor accounts for <10 points of the latent-cost gap.
  - S(true,true) < 50%.
  - The spread across models is less than 2x its CI.

### 2b. The Well
- **No actuators**, so the decision must be an inverse problem.
- **MHD_64** (71.6 GB; MHD_256 is 4.58 TB): 10 ICs x 10 (Ms ∈ {0.5, 0.7, 1.5, 2.0, 7.0} x MA ∈ {0.7, 2.0}), isothermal ISM turbulence [V card]. Every IC exists at every parameter pair, so the dataset is an exact oracle [I].
- **Decision.** Identify (Ms, MA) from the first k frames using a parameter-conditioned surrogate. Oracle: a grid lookup on the true trajectories.
- **Shift.** Hold out Ms=1.5 (interpolation) and Ms=7.0 (extrapolation).
- **Cost.** ~150–300 GPU-h [I].
- **Caveats.** The data axis is tiny (~80 training trajectories), and the fit to UK fusion priorities is weak.
- **Kill.** The smallest model already identifies ≥95%, or every model is at chance.
- **HydroGym (MIT [V])** is the control half, with Re as the shift axis; aarch64 support is [U].

### 2c. Fusion-adjacent
- **TORAX** (Apache-2.0 [V]; differentiable JAX core transport, verified against RAPTOR, https://arxiv.org/abs/2406.06718)
  - Through Gym-TORAX (ITER ramp-up env, https://arxiv.org/abs/2510.11283). Exact action set: [U].
  - Generate rollouts on CPU. Train 4 sizes x 3 data sizes. MPC on the env reward.
  - Oracle: TORAX in the loop, plus gradient optimisation through TORAX.
  - E_rel is defined exactly. Shift: transport-model setting, or out-of-range actuators.
  - < 50 GPU-h [I].
  - Kill: the smallest surrogate already has ~zero regret, or TORAX is too slow.
- **FreeGSNKE** (LGPL-3.0 [V]; FPDT closed-loop MAST-U, https://arxiv.org/abs/2603.28513, https://arxiv.org/abs/2604.00781)
  - Coil-current schedule to reach a target shape; oracle: FreeGSNKE in the loop.
  - In-distribution reference: FNO error ∝ N^−0.68 (https://arxiv.org/abs/2608.05555).
- **RL4F** (https://arxiv.org/abs/2606.07550 [V]): its env is learned from DIII-D data. Use it as an external check only. Code and licence: [U].
- **TokaMark** (https://arxiv.org/abs/2602.10132): no decisions, so F2 loss-gap only. https://arxiv.org/abs/2607.11915 shows NRMSE and alarm TPR ranking imputation methods in opposite orders.

## 3. Further 2025–26 papers [V]
- **Operator-on-F**, https://arxiv.org/abs/2607.04464. Closest prior to F1: TD-MPC2 sweep over 5 sizes. Reward error vs return Spearman −0.30; operator error −0.90. Small n, one task.
- **AD-WM**, https://arxiv.org/abs/2609.30264: factual error does not follow closed-loop success order; elite regret does. Hard-start success 3.7 -> 52.0%.
- **ACA**, https://arxiv.org/abs/2610.04539: alternative actions get lower prediction error than the factual one despite worse outcomes.
- **ATLAS**, https://arxiv.org/abs/2609.36333: regularising the latent marginal does not fix planning geometry; biggest gains on high-novelty episodes.
- **Accuracy vs mechanism consistency**, https://arxiv.org/abs/2610.01842: the lowest-error config is at or below chance on action-response direction in 4 of 8 datasets.
- **JEPA-TTT**, https://arxiv.org/abs/2610.00722: test-time training dominates under shift.
- **LeWM reproduction**, https://arxiv.org/abs/2608.10145: 14% vs 84% from protocol alone.
- **Hi-LeWM**, https://arxiv.org/abs/2607.12547: planner exploitation of model errors.
- **World-in-World**, https://arxiv.org/abs/2510.18135: controllability matters more than visual quality; more inference compute helps.
- **Revisiting downstream scaling**, https://arxiv.org/abs/2512.08894. Counter-evidence: at a fixed token/parameter ratio, a direct power law predicts downstream accuracy well. Suggests fitting regret on compute directly.
- **DIRECT**, https://arxiv.org/abs/2606.12402: test-time compute is not a uniform lever.
- **DeepJEPA**, https://arxiv.org/abs/2610.00368: uniformly deeper transitions waste compute and can hurt planning.
- **TCV surrogate**, https://arxiv.org/abs/2606.09487: in fusion, closed-loop "control-equivalent" checks are already the norm.
- **Walrus sim-to-lab**, https://arxiv.org/abs/2606.01470: usefulness is judged on a physical statistic (growth rate α), not loss.

**Read across [I].** The gap between loss and decisions is now established. In the abstracts I read, nobody fits how it scales with compute and shift against a pre-registered run.

## 4. Pilots (≤1 week, 1 GPU or CPU)
- **P1. TORAX decoupling test.**
  - 5 surrogates spanning ~30x, plus the TORAX-in-the-loop oracle. Measure E_data, E_reach, E_rel and regret.
  - If regret doesn't vary, or E_data already ranks regret perfectly: move fusion to FreeGSNKE, or use TORAX for F3 only.
- **P2. LeWM size sweep.**
  - 5 widths on TwoRoom/PushT, three costs (latent, true, learned reachability head). LeWM is ~15M params on one GPU (https://arxiv.org/abs/2603.19312); code for the reproduction: [U].
  - If true-cost success is saturated at the smallest width, there is no headroom. If latent-cost success is non-monotone in width, fix the objective before any grid.
- **P3. LIBERO headroom check, no training.**
  - True-dynamics CEM, latent vs true cost, at 3 LIBERO-Plus camera-shift levels.
  - If S(true, latent) < 30%, video is objective-bound: demote it or redesign the cost first.

## 5. Pareto ranking (scores 1–5: novelty / feasibility / UK fit / trajectory fit) [I]
- A. F1 on TORAX + FreeGSNKE: 4/5/5/3 (**Pareto**)
- B. F1 on LIBERO: 3/3/2/5 (**Pareto**)
- C. F2 on The Well: 4/4/2/2 (dominated)
- D. F3 on TORAX: 4/4/5/3 (dominated by A)
- E. F3 on LIBERO: 3/2/2/5 (dominated)
- F. One recipe, F1+F3, fusion main + LIBERO second: 4/3/4/5 (**Pareto**, best balance)
- G. HydroGym + The Well: 3/2/3/3 (dominated)

**Recommendation.**
1. Run P1 and P3 first.
2. If E_reach separates from E_data in P1, make option F the study.
3. Drop The Well as a decision substrate.
4. Pre-register the planner, budgets, cost arms, shift axis, top-run prediction and protocol.

**Risks.**
- TORAX may saturate at small model sizes, in which case most of the GPU budget goes to LIBERO and to more seeds.
- The u6xn award scope still needs checking before any non-EO study.

