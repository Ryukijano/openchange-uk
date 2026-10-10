# How to tackle the AIRR work: execution plan

Date: 9 October 2026. Companion to MASTERPLAN.md (the science and the gates). This document is about doing the work: order, owners, acceptance criteria, compute per step, and the repository shape. Nothing here authorises a cluster job; every Isambard step is run by you after review.

## 0. What today's checks found

Installed on this VM (x86, CPU only, in a throwaway venv) to check the software, not to run the pilot.

| Item | Finding | Source |
|---|---|---|
| Gym-TORAX | v1.1.1 on PyPI, MIT, Python >= 3.11, pins torax==1.4.*; also depends on pyqt5, matplotlib, xarray, gymnasium >= 1.2. Installed in under a minute with uv. | [PyPI](https://pypi.org/project/gymtorax/), [repo](https://github.com/antoine-mouchamps/gymtorax) |
| TORAX | v1.4.3, needs jax >= 0.10 and imas-python; TORAX docs say Python 3.12+, but the wheel installed and ran on 3.11. | [TORAX install](https://torax.readthedocs.io/en/latest/installation.html) |
| Built-in task | IterHybrid-v0: ITER hybrid ramp-up, 150 s of plasma time at a 1 s control step. Actions: Ip (1), ECRH power/location/width (3), NBI power/location/width (3). Observation: 93 profiles + 101 scalars = 2,536 numbers. | run on this VM |
| Speed on this VM's CPU | reset 3.6 s; first step 9.3 s (JIT compile); steady steps 0.12 to 0.14 s. One episode is roughly 150 steps, so about 20 to 25 s after warm-up. | run on this VM |
| Counterfactual replay | gym reset() takes no state. TORAX has a file restart (restart: do_restart, filename, time) and Gym-TORAX's config loader supports it, selecting the nearest saved time and checking action scalars. So snapshot = save state file, restore = restart from it. Exactness and the nearest-time rounding must be tested. | gymtorax/torax_wrapper/config_loader.py, torax/_src/orchestration/initial_state.py |
| JAX on Isambard-AI | Official matrix: JAX supported by pip (jax[cuda12], aarch64 wheels exist for CUDA 12 and 13) and by the NGC container nvcr.io/nvidia/jax:25.04-py3 with Apptainer --nv. No conda route. | [Isambard ML packages](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/), [JAX install](https://docs.jax.dev/en/latest/installation.html) |
| RoboJEPA code | Paper gives github.com/facebookresearch/robo_jepa; the GitHub API answered 403 to me and the project page says "coming soon". Still unverified. The plan does not need it. | [arXiv HTML](https://arxiv.org/html/2610.10515) |
| FreeGSNKE | LGPL-3.0, pip-installable, UKAEA. Later same-domain extension, not now. | [repo](https://github.com/FusionComputingLab/freegsnke) |

Two consequences for the design:

1. Direct simulator planning is affordable for a 1D code. At 0.13 s per step, a CEM planner with 64 candidates and a 10-step horizon costs about 80 s per decision, so about 3.5 h per episode single-process. Spread over Grace cores (72 per GPU share) that is of the order of 10 minutes per episode. Good: the ground-truth planner is cheap enough to run on hundreds of episodes. Bad: a learned surrogate is not obviously needed economically for TORAX, which weakens H3 (verification) and must be stated plainly in the paper. The study's main claim (training versus planning compute under shift) does not depend on the surrogate being necessary in deployment.
2. The data axis is unlimited and cheap: 500 episodes is about 3 CPU-hours here. Storage per episode is small (2,536 floats per step). The repository's 500-sample pilot rule still applies until you approve more.

## 1. The two projects and how they fit

| | OpenChange-UK (project A) | Decision-versus-compute study (project B) |
|---|---|---|
| State | Repo scaffold, provenance, Slurm checker, split proposal marked PROPOSED, no approved sources, no data, no runs | Masterplan only; no code |
| Framework | PyTorch (NGC pytorch image, frozen EO encoders) | JAX for TORAX; predictors in JAX or PyTorch |
| Likely u6xn fit | Probably what the award names (unverified) | Needs the award wording checked; may need the allocator's agreement or a Gateway application |
| Compute | Small: adapters on frozen encoders. My guess is under 1,000 NHR including seeds; replace with measurement | Main consumer: up to 20,000 NHR ceiling in stages |
| Paper | Benchmark/transfer paper, remote-sensing or ML venue | Empirical ML paper, ICML/NeurIPS/ICLR type |

Recommendation: keep them in separate repositories with the same guardrails. Copy scripts/check_sbatch.py, the Python hooks, the provenance logger and the Makefile pattern from openchange-uk into the new repo instead of mixing two frameworks and two data policies in one tree. Both fit under the 32-GPU project limit at once, so they do not block each other on the cluster; they compete for your review time.

Order: while the award question for B is open, project A can move (approve sources, accept the split, 1-GPU smoke), because it is in scope today and burns the calendar otherwise. Project B's first four weeks are CPU-local and need no permission beyond "yes, build the harness". If you want only B, say so and A waits.

## 2. Order of work for project B

Weeks are elapsed time for one person plus the agent, not effort. Compute column: local means this VM's CPU; 1 GPU means a human-run Isambard job of at most 30 minutes.

### Week 1: harness and replay (local)

| # | Task | Owner | Done when |
|---|---|---|---|
| B1 | New repo with the openchange-uk guardrails, lockfile (uv, Python 3.11, gymtorax 1.1.1, torax 1.4.3, jax 0.10.x), make check | agent | make check passes; CI runs lint and tests |
| B2 | Determinism test: same seed, same action sequence, twice; compare every profile and scalar | agent | max abs and rel differences recorded; tolerance written down; test in CI |
| B3 | Replay test: run to t, save state file, restart from it with action sequence u1 and separately u2; compare the u1 branch with an uninterrupted run | agent | difference within the declared tolerance, or a documented failure and a workaround (in-memory copy of the sim state) |
| B4 | Timing: cold start, JIT, warm step, per-episode cost, memory, on 1, 8 and 72 processes | agent locally; you later on one Grace share | numbers in reports/timing.md with machine, versions and commit |
| B5 | Provenance logger: config path and hash, git SHA, seed, environment digest, simulator versions, start/end, metrics, artefact paths | agent | every script writes one JSON record; schema tested |

### Week 2: headroom and the shift axis (local; one human smoke)

| # | Task | Owner | Done when |
|---|---|---|---|
| B6 | Baselines on IterHybrid-v0: the built-in reference trajectory, a tuned PI on one or two channels, random within limits, open-loop ramps | agent | 20 paired episodes each, mean and spread of the task cost |
| B7 | Reference planner: CEM or MPPI calling the true simulator through the replay mechanism, 3 budgets | agent | cost curve versus budget; wall-clock per decision |
| B8 | Headroom verdict: gap between the best simple controller and the simulator planner, relative to seed and initial-condition spread | agent drafts, you decide | if the gap is inside the noise, TORAX fails M2 for this task; try one other task definition before switching substrate |
| B9 | Shift axis: choose two physical knobs from the TORAX config (candidates: transport coefficient multipliers, pedestal height, Z_eff or impurity content, density, geometry file) and check that the planner's chosen actions and cost change by more than the noise when the knob moves | agent proposes, a plasma person or the TORAX docs confirm the ranges are physical | table of knob, range, cost change, action change |
| H1 | Isambard smoke, 1 GPU, 30 min: pull nvcr.io/nvidia/jax:25.04-py3 to $PROJECTDIR/sif-images, install gymtorax in a venv inside the container or with pip jax[cuda12], run B2 and B4 | you, via JupyterHub or Clifton | log committed with versions, timings on Grace cores and on the GPU, NHR charged |

Cost definitions must be fixed in B6 before any model is trained: the task cost is a physical tracking or scenario cost with constraints (q_min, q95, density limit, actuator limits), not the Gym-TORAX reward shaping unless we adopt it explicitly and say so.

### Week 3: do small models matter (local or 1 GPU)

| # | Task | Owner | Done when |
|---|---|---|---|
| B10 | Pilot data: up to 500 episodes from approved configs, mixed controllers, whole-episode manifests hashed | agent | manifests committed; no test-informed selection |
| B11 | Three predictors: linear or low-rank, small MLP, small transformer, each with 3 seeds, trained on the same data | agent | rollout error on held-out episodes; training cost logged |
| B12 | Decision test: plan with each predictor, execute in the simulator, compare realised cost and action-ranking agreement with the reference planner | agent | table of model size versus realised cost and ranking agreement, with seeds |
| B13 | Plan-error diagnostic prototype: error on planner-visited states, measured against the simulator, for each model | agent | correlation with realised cost on the pilot (exploratory, labelled as such) |

### Week 4: price it and decide (no compute)

| # | Task | Owner | Done when |
|---|---|---|---|
| B14 | Cost model from measured numbers: data generation, training per size, planning per budget, evaluation per episode, in NHR | agent | budget-ceilings.csv replaced by measured figures |
| B15 | M2 report: headroom, shift sensitivity, model-size effect, costs, risks | agent | you accept, narrow, or switch substrate |
| B16 | Literature re-audit for the narrowed claim (last two years, direct papers only) | agent | list of nearest papers and the stated delta |

### Weeks 5 to 8: exploration grid on Isambard (M3, cap 1,500 NHR)

Fractional grid: 3 to 4 model sizes, 3 planner budgets, 2 shift levels, 3 seeds, paired episodes. Job arrays of 30-minute chunks with checkpoint-resume unless you raise the walltime rule. Output: effect sizes, noise, a power calculation, and a priced main plan.

### Weeks 9 to 10: protocol freeze (M4)

Pre-registration document in the repo: primary endpoint, practical margin, sample size, immutable manifests with hashes, fit candidates including no-trend, the held-out model or compute point, and the allocation prediction written before the main run.

### Weeks 11 to 18: main runs and closed-loop evaluation (M5)

Matched-compute training, evaluation with paired episodes and independent seeds, cost ledger per job, failure log. Multi-node only if a model does not fit one GH200 or walltime demands it, and only after a 1/4/8-GPU scaling measurement with brics/nccl.

### Weeks 19 to 24: robustness, paper, release (M6)

Score the held-out prediction once. Uncertainty over seeds, initial conditions and regimes. Local reproduction path tested on a laptop-class CPU for the small configuration. Code, manifests and logs released with the paper.

## 3. Who does what

Agent: code, tests, CI, documentation, analysis scripts, Slurm templates (1 node, explicit GPUs, 30 minutes), literature audits, draft pre-registration, draft paper sections, memory of decisions.

You: award facts (unit, balance, end date, scope), every cluster action, approval of data sources and splits, choice at each gate, the physics sanity check of the task and shift ranges (ask a tokamak person if one is reachable; otherwise rely on TORAX's documented example scenarios and say so), authorship and venue.

Review load: expect to review and run one or two cluster jobs per week in weeks 2 to 8 and more in M5. If that is too much, the calendar stretches; it does not change the plan.

## 4. Repository shape for project B

Proposed name: torax-decision-scaling (or any name you prefer).

- configs/: sim/ (TORAX scenario files and shift knobs), model/, planner/, eval/; every run names one config per axis
- src/: sim/ (wrapper with save/restore and timing), control/ (PI, open-loop, CEM/MPPI), models/, metrics/ (task cost, plan error, regret), provenance/
- tests/: determinism, replay, cost function, manifest hashing, provenance schema
- manifests/: hashed episode lists per split and shift level, immutable once accepted
- slurm/: 00_pull_sif, 00_smoke, 01_generate, 02_train_1gpu, 03_eval; all pass check_sbatch
- reports/: timing, headroom, M2 report, pre-registration, cost ledger
- docs/: PLAN (this plan), WORKFLOW, decisions log

Make targets: check, test-replay, headroom, pilot-data, train-small, eval-small, price. Each writes a provenance record.

## 5. Isambard set-up for project B (draft for you to run; verify partition and module names in the docs)

Container route, which keeps CUDA and JAX versions matched:

    # on a compute node via JupyterHub terminal or an interactive job of at most 30 minutes
    mkdir -p $PROJECTDIR/sif-images
    apptainer pull $PROJECTDIR/sif-images/jax_25.04-py3.sif docker://nvcr.io/nvidia/jax:25.04-py3
    apptainer exec --nv $PROJECTDIR/sif-images/jax_25.04-py3.sif python3 -c "import jax; print(jax.default_backend(), jax.devices())"

Then a venv that reuses the container's JAX:

    apptainer exec --nv $PROJECTDIR/sif-images/jax_25.04-py3.sif bash -c '
      python3 -m venv --system-site-packages $PROJECTDIR/venvs/torax
      source $PROJECTDIR/venvs/torax/bin/activate
      python3 -m pip install "gymtorax==1.1.1"
      MPLBACKEND=Agg python3 -c "import gymtorax, torax, jax; print(gymtorax.__version__, torax.__version__, jax.__version__, jax.devices())"'

Checks before trusting it: the container's JAX must satisfy torax's jax >= 0.10 pin (if not, use the pip route jax[cuda12] in a plain venv, which the Isambard docs also support); pyqt5 may lack an aarch64 wheel, in which case install with MPLBACKEND=Agg and report whether gymtorax imports without it. Set JAX_COMPILATION_CACHE_DIR under $PROJECTDIR so JIT is paid once per node type. TMPDIR on JupyterHub should point at $SCRATCHDIR/tmp (known issue).

Accounting note: a CPU-only TORAX job on an Isambard-AI node is billed by the largest fraction requested, so asking for 72 cores and no GPU still costs about a quarter node-hour per hour. Data generation should therefore share nodes with training or evaluation rather than run alone.

## 6. Gates restated with today's numbers

- M1 passes when B2 and B3 hold within tolerance on this VM and on one Grace share, and H1's log is committed.
- M2 passes when the simulator planner beats the best simple controller by more than the paired-episode spread, at least one shift knob changes the optimal actions beyond noise, and the small-model table shows a model-size effect on realised cost. If any of the three fails, the report says so and names one fallback (a different TORAX task definition first, then RoboCasa or HydroGym).
- No model larger than the Week-3 transformer is trained before M2 is accepted.
- The 1,500 NHR exploration cap and the later caps stay as in MASTERPLAN.md until replaced by measured prices in B14.

## 7. Execution risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| File restart is not bit-exact or rounds to saved times | medium | B3 measures it; fallback is an in-memory copy of ToraxApp state; last resort is re-simulating from t=0 with the prefix, which is affordable at 0.13 s per step |
| Task saturates with a PI controller | medium to high (pessimist's warning) | B8 decides; try a harder task (disturbance, tighter constraints, shifted regime) before switching substrate |
| pyqt5 or imas-python fail to install on aarch64 | medium | test in H1; headless install; report upstream |
| 30-minute walltime makes M5 painful | high if kept | job arrays plus resume from week 1; you decide whether to raise it at M3 |
| Scope: u6xn not approved for this study | unknown | ask now; project A proceeds meanwhile |
| Reviewer says "1D simulator, so what" | medium | state it; the claim is about compute allocation under shift with exact ground truth, and FreeGSNKE is the same-domain extension |
| Solo review bottleneck on cluster jobs | high | batch jobs weekly; keep everything resumable; agent prepares exact commands |

## 8. Decisions needed from you

1. Build the project B harness now (weeks 1 to 4 are local CPU, no Isambard, no data beyond simulator output)? 
2. Separate repository for project B, named as you like, with the openchange-uk guardrails copied?
3. Project A in parallel (approve sources, accept the split, 1-GPU smoke), or paused?
4. The award facts: unit and balance, end date, approved description.
