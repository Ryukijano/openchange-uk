# Deep think before any plan — v2 (2026-10-07)

Synthesis of my own review plus three independent sessions: explorer (EXPLORER_REPORT_2.md), pessimist (PESSIMIST_REPORT_2.md), verifier (VERIFICATION_REPORT_2.md). [V] = read in the primary source; [I] = inference; [U] = unverified. No data downloaded, nothing trained, no Isambard access.

## 1. Where all three reviews and I agree

1. **The question as worded is mostly answered, and not by scale.** Data-averaged prediction error structurally neither bounds nor tracks planner suboptimality (2607.10362 [V]); with total error fixed, success swings 97% vs 37% depending on where the error sits (2609.32322 [V]); on LeWorldModel the planner cost, not the predictor, was the ceiling, and swapping only the objective took 26% to 98% (2608.12959 [V]); the same released LeWM weights score 14% or 84% depending on protocol (2608.10145 [V]); video MSE follows a power law but does not predict state accuracy (2609.36599 [V]). A headline of "loss scales, decisions don't" would be read as confirmatory.
2. **The open version is narrower and sharper:** does the error on the states the planner actually reaches (E_reach, or its decision-relevant restriction E_rel) follow a predictable scaling law, and does it predict regret at a pre-registered, held-out larger run better than data loss does? Nobody has shown E_reach/DRPE to be scale-predictable (explorer, pessimist, my search). Closest prior: Operator-on-F 2607.04464 (TD-MPC2, 5 sizes, one task: reward error vs return ρ −0.30, operator error −0.90), World-in-World (data and inference-compute scaling for generative WMs), JEPA-WMs, the Atari WM scaling paper, Subramanian 2023 (SciML scaling with parameters pushed OOD), Jones 2021 (train vs test compute). Each has a piece; none has E_reach + shift + pre-registered extrapolation.
3. **Measuring E_reach needs a simulator you can reset to arbitrary states.** That rules out real-robot video as the evaluation substrate and rules out RL4F (its "environment" is a 25-member NN ensemble fitted to DIII-D, so success there is agreement between two learned models [V]).
4. **The planner objective must be an experimental factor.** Design rule (explorer): a 2x2 oracle factorial, dynamics {learned, true simulator} x cost {latent/learned, true task cost}, planner and budget fixed. S(true,true) is the ceiling; S(true,true) − S(learned,true) is what the predictor costs; S(true,true) − S(true,latent) is what the objective costs. Planner budget is swept in FLOPs, not samples. Protocol and episode list frozen and published before Stage 1.
5. **Seeds:** one seed at the top run is indefensible. Measure seed SD at two sizes in Stage 0, size the grid from it, >= 3 seeds per fitted point, report regret and final distance as continuous metrics alongside binary success.
6. **One substrate for the first paper.** One substrate is 18–25 weeks for one person (pessimist). The "same recipe on two substrates" idea is dropped for paper one.
7. **Two hard stops before any hour is spent:** the u6xn Access Terms scope clause (if the award letter says Earth observation, this study needs the allocator's written OK [V]), and the project end date (hours expire; BriCS cannot extend [V]).

## 2. Substrate verdicts

| substrate | exact resettable truth | actions in training data | honest data axis | shift axis | UK fit | trajectory fit | main risk |
|---|---|---|---|---|---|---|---|
| TORAX (Apache-2.0, JAX 1D tokamak transport; Gym-TORAX ITER ramp-up env) | yes | yes (generate) | yes, unlimited | transport model, actuator ranges, geometry | fusion, named priority | medium | surrogates of a reduced 1D code may saturate at small sizes; "it's a surrogate of TORAX, not a plasma" |
| FreeGSNKE (LGPL-3, UKAEA; free-boundary equilibrium evolution, MAST-U) | yes | yes (coil voltages) | yes | shot regime, geometry | fusion, UK-native | medium | slow magnetic control, narrower dynamics |
| HydroGym (MIT; cylinder/cavity/pinball… Re to 4e5) | yes | yes | yes | Reynolds number | clean energy / engineering | low–medium | Docker images amd64 only; Firedrake on aarch64 [U]; per-step cost unpublished; CFD eval "hours on HPC" |
| The Well (CC-BY-4.0 data) | no reset, no actions | no | no (10 trajectories per cell in MHD; better: rayleigh_benard 1,750, shear_flow 1,120, euler 10,000) | parameter grids | astrophysics/fluids; MHD is ISM turbulence, not plasma control | low | decision metric limited to inverse design; fitting loss here and decisions elsewhere breaks the logic |
| LIBERO + frozen V-JEPA 2.1 | yes (MuJoCo clone) | yes (demos) | demo fractions only; OXE/DROID pretraining does not match the sim | LIBERO-Plus perturbations | general AI / robotics only | high | policies at 95–99% while CEM planners sit far lower; MuJoCo EGL on aarch64 [U]; storage for features |
| RL4F / TokaMark (CC-BY-4.0 official) | no (learned env) / no actions | — | — | — | fusion | — | not ground truth; TokaMark is a loss-gap check only |

Verifier corrections already applied: TokaMark is CC-BY-4.0 not SA; InternW0-Δ "open data" is false as worded (third-party licences); VIScore v2 was withdrawn by arXiv, do not cite; 288 Grace cores per node, 96 GB per GH200; the full 1e17–3e21 grid is ~23k GPU-h per seed not 10–12k, closed-loop evaluation ~21k GPU-h at the notes' own per-episode figure, feature cache 50–60 TB for 5k hours of video at full tokens, so BUDGET_DESIGN must be redone once the substrate is fixed.

## 3. The formulation I would now defend

**F1. Scaling of planner-reachable fidelity vs data fidelity, under shift.**
- Definitions: E_data = k-step held-out rollout error. E_reach = the same error on the planner's final-iteration candidate sequences, scored against the true simulator from the same start state. E_rel = E_reach restricted to the state dimensions the cost depends on. Regret = true return of the oracle cell minus the executed plan.
- Pre-registered predictions: (a) the exponent of E_reach differs from that of E_data (bootstrap CI of the ratio excludes 1); (b) a law fitted on runs <= C_max/30 predicts top-run regret with smaller error using E_reach/E_rel than E_data; (c) under an extrapolation shift the shifted/in-distribution ratio of E_reach does not fall with compute (slope >= 0), while the interpolation-shift ratio does.
- Falsifier: ratio CI contains 1 and E_data predicts as well. Then the original question was well-posed on this substrate; publishable as a negative result.
- Secondary: F2 gap laws (relative scaling of shifted vs in-distribution loss and regret) fall out of the same runs. F3 (train vs plan compute exchange rate) only in the true-cost arm, after the objective is validated.

**Substrate: fusion simulators, TORAX first, FreeGSNKE as the same-domain second environment.** Reasons: exact resettable truth; action-conditioned data generated on the Grace CPUs (288 cores per node, co-scheduled with the GPUs); unlimited data so the data axis is real; shift axes that are physically meaningful (transport model, actuator range, geometry); named UK priority and a UKAEA ecosystem (FreeGSNKE is theirs; SUNRISE; Fusion Computing Lab) for follow-on compute; and the framing can be honest: "learned surrogates of tokamak transport and equilibrium codes under regime shift, scored by the control decisions they support." Not "plasma control" on astrophysical MHD.

Video/LIBERO drops to a no-training pilot (P3 below). If it shows the video setting is objective-bound (S(true, latent) < 30%), it stays out of paper one. The trajectory link is preserved through the method (latent predictors, CEM planners, JEPA-style training) rather than the substrate.

## 4. Pilots that decide this (CPU or 1 GPU, <= 1 week each, runnable here without Isambard)

- **P1 TORAX decoupling test (first).** Generate rollouts in Gym-TORAX on CPU; train ~5 surrogates spanning ~30x in size; MPC with TORAX-in-the-loop as the oracle. Measure E_data, E_reach, E_rel, regret per size. Kill if regret does not vary across sizes, if E_data already ranks regret perfectly, or if the smallest surrogate has ~zero regret (saturation). Also measures TORAX step time, which sets the data-generation budget.
- **P2 LeWM width sweep (method check).** 5 widths on TwoRoom/PushT with three costs (latent L2, true, learned reachability). Tells us whether the 2x2 factorial separates predictor from objective at all, and whether latent-cost success is monotone in width. ~15M-param models, one GPU.
- **P3 LIBERO headroom, no training.** True-dynamics CEM with latent vs true cost at three LIBERO-Plus camera-shift levels. If S(true, latent) < 30%, video is objective-bound.
- **P0 (paper, not compute):** read the u6xn award letter for scope and end date; confirm the project storage quota column (20 or 200 TiB); check TORAX and JAX CUDA wheels on aarch64 inside the NGC container during the JupyterHub smoke.

## 5. Kill criteria before Stage 1 (any GPU grid)
- Headroom: S(true,true) − best simple controller must exceed a pre-set margin (register 20% of range); otherwise the decision axis is flat.
- Separation: in P1, Spearman(regret, E_reach) − Spearman(regret, E_data) > 0.3, or the study has no reason to exist on this substrate.
- Objective: S(true,true) − S(true,latent) must be smaller than S(true,true) − S(learned,true) at the mid size, or fix the objective first (2608.12959).
- Signal: between-size differences larger than 2x the seed CI at two sizes.
- Cost: data generation + in-loop evaluation for the full design <= 25k GPU-h-equivalent, else shrink.

## 6. Risk register
| risk | likelihood | mitigation |
|---|---|---|
| TORAX surrogates saturate at small sizes (1D transport too easy) | medium–high | P1 decides in a week; fall back to FreeGSNKE (2D equilibrium) or HydroGym (JAX-Fluids backend) |
| Planner cost is the ceiling | high | 2x2 factorial; three cost arms; report laws per arm |
| Seed/protocol variance > scale effect | high | seed SD in Stage 0; >= 3 seeds per point; frozen protocol; paired episodes |
| Judged incremental | high | headline is E_reach scale-predictability + pre-registered top-run prediction; preprint early |
| "Surrogate of a reduced code, not physics" | medium | state it plainly; add TokaMark/MAST loss-gap transfer as an external check; breakeven accounting vs TORAX itself |
| u6xn scope clause | unknown | read award letter; written OK or AIRR Gateway instead |
| aarch64 builds (JAX CUDA wheels exist [V]; Firedrake/MuJoCo EGL [U]) | medium | JupyterHub smoke in week 1; simulators on Grace CPUs |
| Hours expire / 24 h walltime | certain | end date now; checkpoint-resume; spend to schedule |
| Storage | medium | TORAX states are small; this risk mostly disappears with the fusion substrate |

## 7. Gate verdicts (gap-to-topic)
- Original "loss vs decisions across scale" (any substrate): open = partly closed; contribution = unclear. **PIVOT**.
- F1 on TORAX/FreeGSNKE: open = yes (no prior scaling study of planner-reachable error found by three independent searches); contribution = yes if exponents separate, publishable negative otherwise; feasible = conditional on P1 (saturation, step time) and the u6xn scope. **GO conditional on P1 and P0.**
- F1 on LIBERO/video: open = yes; contribution = weaker (predictor-only scaling on a frozen encoder, 50 points behind policies); feasible = medium (rendering on aarch64, storage, mismatch with OXE/DROID). **PIVOT to pilot P3 only.**
- The Well as decision substrate: **NO-GO** (no actions, no reset). Optional cheap F2 chapter on rayleigh_benard/shear_flow if time remains.
- HydroGym: **hold** until a Firedrake/JAX-Fluids aarch64 build-and-time pilot exists.

## 8. What I need from you
1. The u6xn award wording (scope) and end date.
2. Whether a fusion-simulator substrate is acceptable to you given it is further from your video trajectory than LIBERO. The method (latent predictors, CEM, JEPA-style training) carries the trajectory; the substrate carries the UK case and the exact ground truth.
3. Permission to run P1 (TORAX on CPU, here) and P2 (LeWM widths, CPU/1 GPU here). Both are local, no downloads beyond pip packages and public code, no Isambard.
