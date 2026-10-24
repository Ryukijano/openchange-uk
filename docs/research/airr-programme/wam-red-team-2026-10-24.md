# Red-team: WAM track, Ryukijano/openchange-uk @ ed23fd2

Review of the WAM track, 24 Oct 2026. "Verified" means I read it in the repo or in the arXiv abstract (export.arxiv.org API). "Inference" means my own reasoning. Nothing was run on Isambard.

## Short answers

1. **Gap.** The "causal latent vs AR-generative, history as an axis" question is real but narrow. As written, "at matched compute" can't be won: Long-WAM's pretraining alone (~30.7k H100-h) is 1.5x our whole budget. Reframe it against Long-WAM's released checkpoints (take 3).
2. **Is it a WAM?** Not yet. `wam.py` is an action-conditioned latent predictor with an action head. It has no closed-loop harness for RoboCasa, LIBERO, ManiSkill or MetaWorld (take 1).
3. **Grace-CPU verification.** Mostly a gimmick as a research claim. Grace cores do help with evaluation throughput (take 5).
4. **Benchmark and data.** Use RoboCasa GR-1 as the single benchmark, with RoboCasa365, OXE and DROID (CC-BY-4.0) as data. I verified no licensed UK-application robot dataset, and none is in `configs/sources.yaml`.
5. **Sharpest version.** See the end of this report.

## Takes, by severity

### FATAL

**T1. "Track A: causal clean-JEPA world-action model (WAM)"** (`src/openchange/jepa/wam.py` L1–10) and the kill rule "positive closed-loop history gain" (`COMPUTE_AND_DATA_PLAN` §7.2).
- **Why weak:** nothing in the repo closes the loop. Grepping `*.py`/`*.yaml` for robocasa, libero, maniskill, mujoco, gym and env.step finds only a docstring (L1319–1320). The planner `_rollout_cost` (L644–653) takes a generic `dynamics(z, a)` callable on flat vectors and scores L2 distance to a goal latent. It is not connected to `CausalPatchJEPAWorldModel`, to an environment, or to success. All WAM tests use synthetic tensors on the CPU (`tests/test_jepa_wam.py`, e.g. L139, L325–366). The kill criterion therefore can't be evaluated today.
- **Sharper:** "Before training anything, the GR-1 harness reproduces Long-WAM's released checkpoints at H=0 and H=19.2 s, within the paper's 63.3%/78.7% plus or minus our seed SD." Until that passes, call the module an action-conditioned latent predictor.

**T2. "Fixed prediction origin ... so history length H is not confounded with episode stage"** (`wam.py` L1322–1328, L1374–1377).
- **Why weak:** the fix pins the anchor to `total_t - fut_span`, which is always the last frames of the episode. Consequences:
  - `__len__` = episodes × strata (L1360–1361), so each episode gives exactly 5 training windows.
  - Every target is the episode ending. `target_progress` is therefore about 1 for every sample (L1397), so the progress head can learn a constant.
  - Inference: in demos the robot is often slowing or idle at the end, which pushes training towards the action-blind regime that `DEFENSIVE_FORMULATION...` L25 warns about.
- **Second confound:** `context_frames` is fixed at 4 (L1336). H only changes the spacing between those 4 frames (L1382–1386). At 38.4 s the frames are about 12.8 s apart, so "more history" is also "sparser history". Long-WAM's drop at 38.4 s could be reproduced or hidden by this alone.
- **Sharper:** for each episode, sample many anchors with `t_anchor ≥ max(H)` and reuse the same anchor across all strata. Report per-stratum truncation. Add a fixed-stride arm, where frame count grows with H, next to the fixed-count arm.

### MAJOR

**T3. "against the AR-generative backbone at matched compute"** (`wam-bridge.md` L19, L29, L32).
- **Compute math:**
  - The budget is 20,000 GH200-h (`COMPUTE_AND_DATA_PLAN` L32–36). `wam-bridge.md` L43 still prices against the stale 80k figure.
  - Long-WAM's pretrain is ~30.7k H100-h (`wam-bridge.md` L13, not re-verified past the abstract). That is 1.5x the entire allocation. Moonshot 1 (2,250 NHR = 9,000 GPU-h) is 29% of it.
  - Inference: assuming H100-class dense bf16 (~989 TFLOP/s) and 40% MFU, 9,000 GPU-h is about 1.3e22 FLOPs. Across the planned ~108 runs (`wam-bridge.md` L43) that is about 1e20 FLOPs each. Enough for 20M–300M latent predictors; nowhere near enough for an AR video generator pretrained on 10k hours.
  - A from-scratch generative arm at our scale would be small and undertrained. If it showed no history gain, that would say nothing about Long-WAM.
- **Sharper:** "How much of Long-WAM's 0→19.2 s gain (+15.4 pts) does a causal latent WAM recover, and at what fraction of its pretraining and inference compute?" The generative arm becomes fixed reference points (released checkpoints), not a matched sweep.

**T4. "Nobody has combined them"** (`wam-bridge.md` L19, L26).
- **Why weak:** there are two different LeWAM papers on arXiv. The doc cites 2610.12407 (bidirectional, diffusion-steering MPC). The code cites 2610.12818 (L142, L678), which the doc never discusses. Its abstract describes an end-to-end JEPA with a joint latent-prediction plus action flow-matching objective and an isotropic Gaussian constraint on the latent. That is essentially `FlowMatchingWAMHead` (L1058) plus SIGReg.
- RoboJEPA (2610.10515) already shows that latent-WM planning performance "improves predictably with compute".
- The space left open is causal masking plus history as an axis, not "clean-JEPA WAM".
- **Sharper:** cite both LeWAMs and RoboJEPA as prior art. Claim only the history × compute relationship for causal latent WAMs in closed loop.

**T5. "GPU latent planning with Grace-CPU simulator verification"** (`README.md` L26; `COMPUTE_AND_DATA_PLAN` §4.4).
- **Why weak (inference):**
  - In simulation the exact simulator is available, so "verify top-k in the sim before acting" is simulator MPC with a learned proposal. The obvious baseline is to skip the world model.
  - On a real robot there is no simulator to verify against.
  - Simulator state is kilobytes, so NVLink-C2C bandwidth doesn't matter here.
  - GPU-parallel simulators such as ManiSkill3 (https://arxiv.org/abs/2410.00425) and MuJoCo MJX/Playground already run thousands of environments on the GPU.
- **Sharper:** drop it as a paper. Keep the real GH200 advantage, which is infrastructure: run CPU MuJoCo/RoboCasa rollouts on Grace cores next to policy inference on the GPU, so that evaluating the ~96-run grid in closed loop is cheap.

**T6. "decoder-free" as a strength, with "a usable HF demo" as the goal** (`wam-bridge.md` L29; `COMPUTE_AND_DATA_PLAN` L8–10).
- **Why weak:** a latent-only model has nothing to show. A Space user can't see a 1024-d latent rollout.
- **Sharper:** train a small post-hoc decoder (stop-grad, labelled "visualisation only"), or retrieve nearest real frames from the latent bank. Make latency the headline of the demo.

**T7. Objective defaults that are untested at scale** (`wam.py` L1280, L503, L1286 + L1297).
- **What the code does:** `pixel_wam_loss` defaults to `target_mode="live"`, meaning gradients flow into the target encoder. SIGReg defaults to `patch_flat` on predicted latents only, not on the encoder outputs that LeWM regularises (per the 2603.19312 abstract). `patch_flat` also treats spatially correlated patches as independent samples.
- **Inference:** that leaves an unguarded collapse or rescaling path for the encoder.
- **Cost:** the world model runs forward twice per step (once inside `wam_loss`, again at L1297), which doubles predictor compute.
- **Sharper:** follow LeWM's placement first (SIGReg on encoder embeddings, one forward pass). Treat history stratification as a single ablation.

### MINOR

**T8. "2x2 Oracle Factorial"** (`AI_BUILDER_MASTER_PLAN` L28). `S(true,true)` is hard-coded to 0, and `S(true,latent)` is 0 unless supplied (`wam.py` L627–633). Two of four cells are placeholders.

**T9. "history-stratified SIGReg" as a method contribution** (`wam-bridge.md` L36). The doc itself marks the premise UNVERIFIED. Strata are keyed on the realised H (L1398), so truncated long-history samples fall into lower strata, mixing episode length into the stratum.

**T10. "4x GH200 ... MVE runner"** (`slurm/06` L9–12, L43). It runs CPU unit tests on 4 GPUs. The plan admits this (`COMPUTE_AND_DATA_PLAN` L154–155). Cheap (0.42 NHR), but mislabelled.

**T11. UK application.** `configs/sources.yaml` has no robotics source (grep finds no robocasa/lerobot/libero), and the PI scope question is still open (`COMPUTE_AND_DATA_PLAN` §8.4). I verified no open, action-labelled dataset for nuclear, offshore, agri or warehouse work. Don't claim one. A truthful UK angle is "sovereign-compute, real-time-capable open WAM", not a specific sector.

## Keep

1. **Honest history bookkeeping:** `realized_hist_s`, `requested_history_seconds`, the `history_truncated` flag, transition-aligned actions, and tests that pin them down (`wam.py` L1398–1410; `tests/test_jepa_wam.py` L368–402). Fix where the anchor goes (T2); keep the bookkeeping.
2. **Kill rules that score control, not latent MSE** (`COMPUTE_AND_DATA_PLAN` §4.1 "must beat ... on control, not latent MSE", §7.2), and the licence quarantine of AgiBot World and EgoDex (`wam-bridge.md` L41).
3. **The history strata match Long-WAM's exactly** (`HISTORY_STRATA_SECONDS`, L31). That is what makes direct comparison with released checkpoints possible.

## Sharpest version (paper + demo)

**Paper: "History without generation."**
- **Setup:** RoboCasa GR-1 only. The abstract confirms Long-WAM reports it, with a 63.3→78.7% gain; per our earlier notes it isn't saturated, unlike LIBERO-Long at 94.5→99.5%. Use a frozen encoder (V-JEPA 2.1, or Long-WAM's own visual encoder if its licence allows) with pre-encoded latents. Train a causal latent predictor plus flow-matching action expert at 3 sizes (20M/80M/300M) × 5 history lengths × 2 seeds × 2 frame regimes.
- **Comparison:** against Long-WAM's released checkpoints at the same H values.
- **Outputs:** the history-gain curve, compute-to-match, and control latency.
- **Cost (estimate):** about 1,000–2,000 NHR including evaluation.
- **Either result is publishable:** "latents recover X% of the gain at Y% of the compute", or "they don't, and here is what latents lose".

**Demo:** a GR-1 scene viewer on HF. Pick a task and a history length, watch the policy act from logged rollouts, see retrieved or decoded imagined frames, and compare a ms-per-action meter against the generative baseline.

**Decisive first experiment:** fix T2. Then run the 20M causal model on GR-1 at H=0 and H=19.2 s for 3 seeds. If the gain is below 3 points, or below 3x the seed SD, stop the track.
