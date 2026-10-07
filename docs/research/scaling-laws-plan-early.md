# Scaling-laws study on u6xn: draft plan (2026-10-07)

Surgical world models are dropped because there isn't enough data. The goal now is to understand scaling laws. Anything marked "check" or "estimate" hasn't been verified.

## What already exists (searched today)

| Work | What it found | What it leaves open |
|---|---|---|
| Pearce et al., "Scaling Laws for Pre-training Agents and World Models" (ICML 2025, arXiv 2411.04434) | World models and imitation learning follow LLM-style power laws. The coefficients depend heavily on the tokenizer, task and architecture. | Uses game and offline data. Loss is measured in-distribution only. Doesn't test whether lower loss means better decisions. |
| "Scaling Laws of Motion Forecasting and Planning" (arXiv 2506.08228) | Driving motion forecasting improves as a power law in compute, on about 500k hours of driving data. | Proprietary data. Not action-conditioned control. |
| TESSERA v2 (arXiv 2607.03949) | 395 EO pretraining runs. **Pretraining loss barely predicts downstream performance (|r| < 0.2).** | EO only. Leaves open whether the same disconnect happens for world models. |
| PhilEO scaling (arXiv 2506.14765) | EO models at 44M–300M parameters. CNNs are competitive in low-shot settings; ViTs win at 23 TB. | Small size range. |
| AlphaQubit 2 (arXiv 2512.07737) | Neural decoders reach near-optimal error rates for surface and colour codes and run in real time. | The abstract doesn't report a scaling-law study (check the full paper). |
| "Accuracy on the line" (Miller et al. 2021) | Out-of-distribution accuracy often tracks in-distribution accuracy linearly. | Image classifiers, not world models or decoders. |

**The gap:** nobody seems to have measured, for a world model trained on simulator data with exact ground truth, how three things scale together:
1. in-distribution loss;
2. loss under shift (unseen layouts, noise or sizes);
3. **decision quality** when the model is used to act, judged by the simulator.

TESSERA v2's finding (loss doesn't predict downstream) is the sort of result that could repeat here, and it would matter.

## The question

> As parameters, data and compute grow, do (1) held-out loss, (2) loss under shift, and (3) the quality of decisions made with the model all follow power laws? Are the exponents the same, and does loss predict decisions?

## Why use a simulator

A clean scaling study needs data you can grow cheaply by 100× or more, exact labels, and a shift you control. Simulators give all three. They also avoid licensing and patient-data problems.

| Testbed | Data | Shift axis | Decision metric | Notes |
|---|---|---|---|---|
| **SUMO junctions** (JunctionLab) | State-action trajectories, unlimited | Unseen junction layouts, demand surges, sensor dropout | Delay and spillback when planning with the model, scored by SUMO | Builds on your traffic work. SUMO runs on CPU, so data generation speed must be measured first. |
| **Stim QEC** (syndrome-net) | Syndrome shots, very fast to generate, exact labels | Unseen noise models and code distances | Logical error rate against MWPM | Cheap data. Fits the quantum priority. |

**Recommendation:** use SUMO as the main testbed, since a world model is the subject. Then repeat the key fits on Stim to see whether the conclusions hold in a second domain. A result in two domains is much stronger than one.

## Rough compute estimate

Assumptions (estimates, to be measured in the first week):
- training FLOPs ≈ 6 × parameters × tokens;
- an H100 at about 35% utilisation gives roughly 1.3e18 FLOPs per GPU-hour;
- so 80k GPU-hours ≈ 1e23 FLOPs in total.

Examples:
- 1B parameters × 20B tokens ≈ 1.2e20 FLOPs ≈ 100 GPU-hours ≈ 25 NHR.
- A Chinchilla-style IsoFLOP grid: 5 compute budgets from 1e19 to 1e21, 7 model sizes per budget, 1 seed. That's about 1e22 FLOPs ≈ 8k GPU-hours ≈ **2,000 NHR (10%)**.
- Adding seeds on the small runs, the shift and decision evaluations, and a 3e21 run gives about 25–35% of the budget.

So the budget can cover a full grid in both testbeds and still leave room. The likely limit is how fast the simulators can generate data, not GPUs.

## How it would run on Isambard

- **Small and medium runs (most of the grid):** one GPU each, many at once as Slurm job arrays. Multi-node is wasted on them.
- **Large runs (1B+ parameters):** data-parallel or FSDP across 2–16 nodes, using NCCL over Slingshot as in the Isambard NCCL guide.
- **Scaling ladder first:** 1 GPU, then 4, 8 and 16 GPUs. Measure throughput and scaling efficiency before trusting any large run.
- **Data generation:** each job is charged for at least 1 GPU (0.25 NHR) per 72 CPU cores. Generating on the local PC, or packing generation into GPU jobs, may be cheaper. Measure it.
- **Time limits:** the repo's 30-minute cap must be raised for real training. You set the new limits.

## Keeping it honest (pre-registered)

- Fix the functional form before seeing results, e.g. L(N, D) = E + A/N^α + B/D^β.
- Fix the shift splits and the decision metric in advance too.
- **Hold out the largest run:** fit on the smaller runs, write the prediction down, then run it. Report the miss whatever it is.
- Report the spread across seeds, and fits that don't work.

## First four weeks

1. Measure SUMO and Stim data-generation throughput, and GPU utilisation on GH200 at 1 GPU and 1 node.
2. Build the model family: one transformer recipe across 1M–1B parameters. Fix the tokenizer, because Pearce et al. show it changes the coefficients.
3. Run a small grid (up to 1e19 FLOPs) to check the fits are stable.
4. Freeze the pre-registration, then run the full grid.

## Decisions for you

1. Main testbed: SUMO, Stim, or both?
2. New job time limits for scaling tests and training.
3. The u6xn project end date.
