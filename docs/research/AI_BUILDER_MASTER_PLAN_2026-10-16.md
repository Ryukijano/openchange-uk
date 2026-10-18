# Open-Source AI Builder (`u6xn`) — 3-Track Master Blueprint & Scaling Law Plan

**Prepared:** 16 October 2026  
**Award:** DSIT × NVIDIA Open-Source AI Builder Fund (`AI Builder: NV-Nemotron` / BriCS `u6xn`, Isambard-AI Phase 2)  
**Allocation:** **20,000 GH200 GPU-hours = 5,000 Node-Hours (NHR)** through 15 March 2027 (4× NVIDIA GH200 96 GB HBM3e + 288 Grace ARM Neoverse-V2 CPU cores + 460 GB LPDDR5X per node)

---

## 1. Executive Summary & Architectural Unification

This blueprint unifies the research and engineering across all three tracks into runnable, unit-tested codebases while keeping the publication archive in `/home/ubuntu/repos/openchange-uk` completely untouched:

| Track | Workspace | Core SOTA Question | Implemented & Tested Modules | Compute Budget |
|---|---|---|---|---:|
| **Track A: Causal Clean-JEPA World-Action Model (WAM)** | `/home/ubuntu/repos/openchange-jepa` (`src/openchange/jepa/wam.py`) | Does long-horizon manipulation history ($0 \to 38.4\text{ s}$) require 30k-GPU-hr pixel generation (*Long-WAM*) or only Theorem-1 causal structure in a decoder-free Clean-JEPA with Diffusion-Steered Planning? And does planner-reached error ($E_{\text{reach}}$) separate from rollout MSE ($E_{\text{data}}$)? | `causal_neighborhood_mask`, `bidirectional_neighborhood_mask`, `HistoryStratifiedSIGReg`, `CausalPatchJEPAWorldModel`, `LeWMPixelPatchEncoder`, `EndToEndPixelWAM`, `LeRobotWindowDataset`, `compute_reachable_error_and_factorial`, `diffusion_steered_plan` (`tests/test_jepa_wam.py`) | **9,000 GPU-h** (2,250 NHR) |
| **Track B: `Agentic-SfM 2.0` Multi-Specialist 2D/3D Router (`Qwen3.5` & `Nemotron-3`)** | `/home/ubuntu/repos/agentic-sfm` (`src/agentic_sfm/`, `tools_server/`) | How does diagnostic-conditioned routing across `LoFTR`, `MASt3R`, `VGGT`, and `MapAnything` (`facebook/map-anything-apache`) + view-graph surgery (`bridge_components`, `verify_doppelganger`) scale with tool budget $B \in \{1..32\}$ and RL compute $C_{\text{RL}}$? | `_record_agent_match_union`, `_match_mapanything`, `select_diagnostic_specialist_matcher`, $\epsilon$-forced specialist exploration (`matcher_explore_eps`), `keep_best_match` floor, `baseline_delta_reward`, `scripts/eval_budget_scaling.py`, `configs/phase2_isambard_4xgh200.yaml`, `jobs/isambard_4xgh200_hybrid.slurm` (`tests/test_track_b_monotonic_and_scaling.py`) | **5,000 GPU-h** (1,250 NHR) |
| **Track C: GPU-Native Open-Science Spatiotemporal World Model (`Sci-JEPA-WM`) & Operational Guarantees** | `/home/ubuntu/repos/openchange-jepa` (`src/openchange/jepa/science_wm.py`, `guarantees.py`, `runtime.py`) | How does a forcing-conditioned spatiotemporal Clean-JEPA world model (`PhysicalForcingEncoder` baseline + anomaly + cumulative stress, `MultiSensorContinuumPatchStem`, `FactorizedSpatiotemporalJEPAWorldModel`) forecast multi-sensor EO (`SSL4EO-S12 v1.1`, `TerraMesh`) and continuum PDE fields (`The Well`) on 4×GH200 while certifying Q1–Q4 guarantees and GPU counterfactual observation value? | `PhysicalForcingEncoder`, `MultiSensorContinuumPatchStem`, `FactorizedSpatiotemporalJEPAWorldModel`, `gpu_counterfactual_observation_value` (`tests/test_jepa_science_wm.py`), `configure_precision_and_determinism` (GH200 TF32/SyncBN fix), `rankme_score`, `lidar_score`, `selection_spearman`, `fit_orthogonal_procrustes`, `StratifiedConformalAlertCalibrator` (`tests/test_jepa_guarantees.py`, `slurm/06_wam_and_science_gpu.sbatch`) | **3,600 GPU-h** (900 NHR) |
| **Shared Containers, Smokes & Reserve** | Both workspaces | Container preflight (`/projects/u6xn/sif-images/pytorch_26.09-py3.sif` & `pytorch_25.05-py3.sif` aarch64), 4-GPU SyncBN verification, and mid-January reserve release for the winning track | `slurm/00_smoke.sbatch`, `slurm/05_jepa_cuda_checks.sbatch`, `slurm/06_wam_and_science_gpu.sbatch` | **2,400 GPU-h** (600 NHR) |

---

## 2. Track A: Causal Clean-JEPA WAM (`src/openchange/jepa/wam.py`)

### 2.1 Why This Solves Your ICML 2026 `Causal-JEPA` Reproduction Findings
In `Ryukijano/causal-jepa-icml2026-agent-repro`, Theorem 1 held mathematically, yet object-slot C-JEPA dropped from $88.67\%$ to **$49.33\%$** on Push-T CEM planning and ran **$2.1\times$ slower** than patch DINO-WM.
- **`causal_neighborhood_mask(grid_size, timesteps, spatial_radius)`:** Enforces Theorem 1 directly on dense patch tokens $(u, t)$ ($\|u - u'\|_\infty \le r,\; t' \le t$ plus causal CLS tokens), eliminating fragile `VideoSAUR` object-slot extraction while strictly blocking future gradient leakage.
- **`HistoryStratifiedSIGReg`:** Applies FP32 Epps–Pulley characteristic-function Gaussian regularization across the 5 *Long-WAM* history strata $H \in \{0.0, 2.4, 9.6, 19.2, 38.4\}\text{ s}$ so context length is not encoded as a nuisance shift.
- **`CausalPatchJEPAWorldModel` + `diffusion_steered_plan`:** FiLM-modulated causal world model paired with policy-noise-space trajectory optimization (*LeWAM* / *MBDPO* style) to prevent CEM from exploiting off-manifold predictor errors, evaluated via `compute_reachable_error_and_factorial` ($E_{\text{data}}$, $E_{\text{reach}}$, $E_{\text{rel}}$, and the $2\times 2$ Oracle Factorial).

### 2.2 4-Tier License-Clean Data Ladder
1. **Tier 0 (<8.5 GB, Week-1 Pilot):** `Push-T` (<2 GB) + `ember-lab-berkeley/robocasa365-target-atomic` (6.23 GB, 9,126 episodes, 31 hrs @ 20 fps, `LeRobot v3` flat format, MIT / CC-BY-4.0).
2. **Tier 1 (499 GB, Primary Closed-Loop Scaling Grid):** `huiwon/robocasa365` (2,200+ hours across 365 kitchen tasks and 2,500 scenes, CC-BY-4.0).
3. **Tier 2 (36.8 GB $\to$ 1.5 TB sharded, Real-Robot Video & Action Scaling):** `BridgeData V2` (36.8 GB, 60k episodes, CC-BY-4.0) + `DROID` (350 hrs, CC-BY-4.0) + streamed shards of `DAGroup-PKU/RoVid-X` (10k+ hrs CC-BY-4.0, decoded on 288 Grace CPU cores).
4. **Tier 3 (Zero-Extra-Pretraining Surgical Transfer Appendix):** Downstream temporal probe transfer on `CholecTrack20` / `JIGSAWS` vs. `DINO-Endo`.

---

## 3. Track B: `Agentic-SfM` Scene-Scale RL (`/home/ubuntu/repos/agentic-sfm`)

### 3.1 Root-Cause Fixes for the 62% vs. 88% Pair-Level Gap
1. **Monotonic Correspondence Union (`tools_server/server.py`):**
   - `tool_crop` now tracks `parent_id` and cumulative `origin_xy` across chained crops.
   - `_record_agent_match_union` shifts crop-matched keypoints by `origin_xy` back into root full-frame coordinates and union-merges them (deduplicated at 0.1 px) into `_match_store`, ensuring a crop match **can only add correspondences and never overwrite or shrink the full-frame baseline**.
2. **Strict Full-Frame Floor & $\Delta R_{\text{geom}}$ Reward (`geometry.py`, `pose_rewards.py`, `verl_agent_loop.py`):**
   - `keep_best_match` requires a cropped candidate to **strictly** exceed an uncropped full-frame match in quality.
   - `compute_pair_reward` adds `baseline_delta_reward` ($\max(0, \text{AUC}_{\text{current}} - \text{AUC}_{\text{full-frame}})$), and `verl_agent_loop.py` whitelists `first_call_validity_weight`, `gate_geometry_on_validity`, and `baseline_delta_weight` so post-SFT GRPO runs can turn off the `+0.5` syntax bonus.
3. **Budget $\times$ Compute Scaling Law & Quad-GH200 Pipeline (`scripts/eval_budget_scaling.py`, `configs/phase2_isambard_4xgh200.yaml`, `jobs/isambard_4xgh200_hybrid.slurm`):**
   - Evaluates and fits $E(B, C_{\text{RL}}) = E_\infty + a B^{-\alpha} + c C_{\text{RL}}^{-\beta}$ across tool budgets $B \in \{1, 2, 4, 8, 16, 32\}$ with GPU0 (`vLLM` dynamic LoRA), GPU1 (`LoFTR` + `MASt3R` + `VGGT-1B` + 288-core `pycolmap`), GPU2 (trainable LoRA), and GPU3 (frozen KL anchor).

---

## 4. Track C: `OpenChange-UK` Guarantees & Conformal Alerts (`src/openchange/jepa/guarantees.py`)

1. **GH200 CUDA SyncBN / TF32 Parity Fix (`src/openchange/jepa/runtime.py`):**
   - `configure_precision_and_determinism` explicitly sets `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, and `torch.set_float32_matmul_precision("highest")` when `precision == "fp32"`, eliminating the 10-bit TF32 mantissa drift ($\approx 3.23 \times 10^{-4}$) between $M=64$ single-rank and $M=32$ two-rank projector GEMMs on Hopper GH200.
2. **Operational Embedding Guarantees & Conformal Alerting (`src/openchange/jepa/guarantees.py`):**
   - **Q1:** `rankme_score`, `lidar_score`, and `selection_spearman` (with sign-trap protection and top-1 selection regret).
   - **Q2:** `fit_orthogonal_procrustes`, `apply_orthogonal_procrustes`, and `evaluate_version_upgrade` (testing zero-re-embedding model upgrades on held-out geographic blocks).
   - **Q3/Q4:** `cross_unit_distance_ratio` (measuring S2A/S2C and S1B/S1C hardware-swap stability).
   - **Part 2:** `StratifiedConformalAlertCalibrator` (season $\times$ land-cover split-conformal quantiles, empirical FAR, detection lead time, S1-only cloudy lead-time gain, and Richardson economic cost-loss value) + `score_observation_reveal_policies`.
