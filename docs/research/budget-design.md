# Design argument: how to spend 80k GH200-hours (2026-10-07)

All numbers are estimates to be replaced by Stage-0 measurements. Assumed effective throughput 3.5e14 FLOP/s per GH200 (35% of ~1e15 BF16 dense).

## 1. The hours do not go where people think
- Frozen-encoder feature extraction is cheap: ~1B-param encoder over 1 hour of 4 fps video is ~7e15 FLOPs, ~20–40 s. 5k hours of video ≈ 50 GPU-h. Plus I/O overhead, call it <500 GPU-h.
- Storage, not compute, caps the data axis. 16-frame clips at 4096 tokens x 1024 dims x bf16 ≈ 8 MB per clip; 1 hour of video ≈ 900 clips ≈ 7 GB. 5k hours ≈ 35 TB of features. Need token pooling (2–4x) or stride to fit Isambard scratch. Decide the max D from the storage quota, then the grid.
- The training grid is moderate. A 3e21-FLOP top run is ~2,400 GPU-h. A grid at 4 points/decade over 1e17–3e21 with ~4 model sizes per FLOP level is ~70 runs, summing to ~10–12k GPU-h for one seed; the top decade is ~70% of that.
- Closed-loop planning evaluation is the hidden cost. CEM in latent space: 100 samples x 3 iterations x 10-step horizon x 100 env steps = 3e5 predictor forwards per episode. For a 1B predictor that is ~1.5e17 FLOPs, ~7 min per episode. 500 episodes (10 tasks x 50 seeds) x 70 models x 5 planner budgets is far more than the training grid. Evaluation must be designed first, not last.

## 2. Evaluation design to keep it affordable
- Planner-budget sweep (option C) on ~12 models (3 sizes x 4 compute levels), not all 70.
- Paired seeds across models and budgets (same initial states) so differences need fewer episodes; 20–30 episodes per task is often enough when paired.
- Batch rollouts: run the N samples of CEM as one batched forward; run LIBERO on Grace CPUs (72 cores/node) in parallel with GPU planning.
- Loss, shifted loss and calibration are cheap (one forward per clip); compute them for every model and seed. Only success is expensive.
- Distribution shift must cost evaluation only, no retraining: held-out LIBERO-90 tasks (OpenWAM's split), held-out embodiment/camera subsets of OXE vs DROID, and for physics, held-out PDE parameters.

## 3. Staged spend with kill criteria (GPU-hours)
| Stage | Spend | Kill / go rule |
|---|---|---|
| 0 Pilot | ≤1,000 | Measured MFU, feature-cache throughput, two tiny predictors, headroom: true-state planner vs scripted policy on LIBERO. Kill the video substrate if the headroom gap is small. |
| 1 Small grid | ~8,000 | 1e17–1e19, 3 seeds. Fit laws for loss, shifted loss and success. Write down the Stage-2 prediction before running it. |
| 2 Main grid | ~15,000 | 1e19–1e21, 2 seeds mid, 1 seed top. Compare with Stage-1 predictions. |
| 3 Pre-registered test | ~5,000 | One 3e21 run plus eval. The headline is how far it lands from the prediction. |
| Eval (planning) | ~12,000 | Across stages, following the rules above. |
| Physics substrate (The Well) | ~12,000 | Same recipe; 2D datasets are 1-GPU jobs; one 3D dataset justifies multi-node. Decision metric via HydroGym (CPU-heavy) or inverse design. |
| Generative comparator | ~4,000 | Cosmos 3 / OpenWAM fine-tune at 2 sizes, same eval. |
| Reserve | ~15,000 | Reruns, bugs, a second shift axis. |
| Total | ~72,000 | of 80,000 |

## 4. Multi-node is a walltime issue, not a scale issue
A 1B predictor on cached features fits one GH200. But workq caps jobs at 24 h, so a 2,400 GPU-h run on 4 GPUs is 25 days; on 32 GPUs it is ~3 days in 4 resumable chunks. Multi-node (brics/nccl + aws-ofi) is needed to finish top runs before hours expire, and every run needs checkpoint/resume. Everything below ~1e20 FLOPs is a 1-GPU job-array member.

## 5. What not to spend on
- Training a generative WAM from scratch: 5B models on 10–20k hours are an industrial race (OpenWAM, InternW0-Δ, Cosmos 3).
- Pixel-space evaluation benchmarks (WorldMark etc.): crowded and not decision-relevant.
- A single large run instead of a grid: no law, no prediction, no paper.

## 6. Decisions to argue
1. Shift axis: held-out tasks, held-out embodiment/camera, or both (the second costs more data engineering, no GPU).
2. Seeds vs grid density at fixed budget: 3 seeds at low compute and 1 at the top (my proposal) vs 2 seeds everywhere.
3. Physics substrate: 12k GPU-h (two-substrate paper) or fold it into reserve (one-substrate paper, more seeds).
4. Planner budget sweep: 5 budgets x 12 models (my proposal) vs 3 x 20.
5. Project end date and quota for scratch storage: both bound the design and I still don't have them.
