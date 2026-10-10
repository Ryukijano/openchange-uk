# Red-team: scaling-laws study on u6xn (80k GPU-hours), SUMO junctions and Stim QEC

Role: pessimist. I looked for every way this could fail, mislead, or be hard to publish. Sources are web pages and paper abstracts or HTML. Anything I could not check is marked **UNVERIFIED**. I did not touch Isambard, download datasets, create repos, or run expensive compute.

**Local measurements (not from a URL).** I ran two small benchmarks (under 1 minute each) on this VM's x86 CPU (Intel Xeon Platinum 8559C, single core). Grace is aarch64, so Grace numbers will differ (**UNVERIFIED**). I pip-installed `eclipse-sumo`, `stim` 1.16.0 and `pymatching` 2.4.0.
- SUMO, 3×3 grid from `netgenerate`, `randomTrips -p 1.0`, 3,600 s simulated, about 90 vehicles in the network at once. Plain `sumo` with no control: about 1 s wall time at step 1 s, and about 6 s at step 0.1 s. Under TraCI, setting signal phases every 10 steps and reading each vehicle's position and speed every step: **about 205 steps/s and about 18k vehicle observations/s per core.**
- Stim rotated surface-code memory, p = 0.001, rounds = d, 200k shots each:
  - d=5: 120 detectors, 1.33M shots/s sampled, 2.9M shots/s decoded by PyMatching, MWPM logical error rate 1.3e-4.
  - d=7: 336 detectors, 230k shots/s sampled, 857k shots/s decoded, MWPM logical error rate 1.5e-5 (only **3 failures** in 200k shots).
  - d=11: 1,320 detectors, 65k shots/s sampled, 185k shots/s decoded, **0 failures** in 200k shots.
  - Mean detector firing rate is about 1.5–1.7%.

---

## 0. Corrections to the draft plan

1. **Isambard's 32-GPU-per-project cap probably does not apply to Phase 2.** On the job-scheduling page, the 32-GPU QoS sits in the first tab (Phase 1). The Phase 2 tab lists 256 running and 512 pending-plus-running jobs per user, and a maximum array size of 1,000. https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/ (This is my reading of a tabbed page. Confirm before relying on it.)
2. **A 1-GPU allocation comes with 72 Grace cores and 216 GB of RAM, charged at 0.25 NHR/h.** So "CPU work costs a GPU" really means each 72-core CPU-hour costs 1 GPU-hour from the allocation. https://docs.isambard.ac.uk/user-documentation/guides/accounting/
3. **For SUMO, data generation is probably not the bottleneck. Closed-loop decision evaluation is.** At about 18k vehicle observations/s/core (x86, TraCI), and assuming about 4 tokens per vehicle observation (**my assumption**), the 58B tokens needed for a compute-optimal 1e21 run take about 224 core-hours, or about 3 Grace-socket-hours if cores scale linearly (**UNVERIFIED**). The draft says "the likely limit is how fast the simulators can generate data". On these numbers that's wrong for the token count. It could be right for *distinct information*, as section 6 explains.
4. **The 35% MFU assumption** (1.3e18 FLOPs per GPU-hour) is optimistic for the 1M–100M-parameter models that make up most of a grid. Small models are limited by kernel launches and data loading, not FLOPs (**UNVERIFIED**; measure it in week 1). If small-model MFU is 5–10%, the FLOPs-to-GPU-hours conversion for the bottom of the grid is off by 3–7×.

---

## 1. Fitting and extrapolation

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 1.1 | **Too narrow a range.** The plan's 1e19–1e21 grid spans 2 orders of magnitude in compute. Kaplan's trends span more than 7. Analysing at small scale is one of the two main reasons Kaplan's exponent was biased (N_opt ∝ C^0.73 vs Chinchilla's C^0.50). | High | https://arxiv.org/abs/2001.08361 ; https://arxiv.org/abs/2406.12907 | Cover at least 3 decades of compute (e.g. 1e17–1e21), and at least 4 points per decade on each axis. **Kill:** if the held-out top run misses its predicted loss by more than the bootstrap 90% interval, report that the law doesn't extrapolate. Don't refit and re-announce it. |
| 1.2 | **Too few points, and confidence intervals that are too narrow.** Besiroglu et al. found Chinchilla's Approach 3 estimates didn't match Approaches 1–2 and didn't fit the data reconstructed from the plots. Its CIs were so narrow they "would require over 600,000 experiments" when likely fewer than 500 were run. | High | https://arxiv.org/abs/2404.10102 | Fit with Huber loss on log-loss, using several initialisations. Bootstrap over *runs*, not over evaluation tokens. Report all three Chinchilla approaches. If they disagree on the exponent by more than the CI, say so. |
| 1.3 | **Which fitting choices matter.** A methodology paper on scaling-law estimation recommends using intermediate checkpoints, fitting on source models close in size to the target, and running several seeds for small models when seed variance matters. | Med | https://arxiv.org/abs/2410.11840 | Use those recommendations. Pre-register which checkpoints count (e.g. the last 20% of training, after cooldown). |
| 1.4 | **Counting parameters.** Kaplan counted non-embedding parameters. Chinchilla counted total parameters. This choice alone shifts the fitted exponent a lot at small scale. For tokenised simulator states, embeddings and the output head can be a large share of a 1M–10M model. | High | https://arxiv.org/abs/2406.12907 ; https://arxiv.org/abs/2406.19146 | Report fits with both counts. Also count FLOPs including the last layer, which is a factor Porian et al. identify. Pre-register which count the headline result uses (total). |
| 1.5 | **Warmup and optimiser tuning.** Porian et al. reproduced Kaplan's law and traced the gap to three things: last-layer compute, warmup length, and scale-dependent optimiser tuning. After fixing them they matched Chinchilla. They also found AdamW β2 tuning is essential at small batch sizes. | High | https://arxiv.org/abs/2406.19146 | Scale warmup with run length. Tune β2. Use a fixed rule for LR and batch size per size (section 2). |
| 1.6 | **LR schedule.** Chinchilla argued that cosine schedules need to match the run length. Porian et al. found careful LR decay "not essential". Hägele et al. show constant LR plus a cooldown scales as reliably as cosine and lets one run serve several data budgets. | Med | https://arxiv.org/abs/2203.15556 ; https://arxiv.org/abs/2406.19146 ; https://arxiv.org/abs/2405.18392 | Use constant LR with cooldown branches at 3–4 data budgets per model size. This cuts the cost of the D axis. Run one cosine check at mid-scale. **Kill:** if the cooldown and cosine endpoints differ by more than the seed spread, use full cosine runs and shrink the grid. |
| 1.7 | **Functional form.** Chinchilla's form is L = E + A/N^α + B/D^β. Simulator data may need a broken power law (section 4). Choosing the form after seeing the data is a garden-of-forking-paths problem. | Med | https://arxiv.org/abs/2210.14891 | Pre-register the Chinchilla form as primary and a smoothly broken power law as secondary. Report both, with leave-top-out error. |

## 2. Per-size hyperparameter tuning and its cost

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 2.1 | **Fixed hyperparameters across sizes produce fake exponents.** If the LR is right for 10M parameters and too high for 1B, the large models look worse and the fitted exponent comes out too shallow. That is the "scale-dependent optimiser tuning" factor in Porian et al. | High | https://arxiv.org/abs/2406.19146 | Fit LR(N) and batch(N) laws from sweeps at 3–4 small sizes, as Porian et al. do, and extrapolate them. Check that the largest size is near-optimal with a 3-point LR bracket. |
| 2.2 | **muP is not a free pass.** Everett et al. trained tens of thousands of models (3 optimisers × 4 parameterisations, 14 sizes up to 26.8B). They found that all parameterisations, not just muP, can transfer hyperparameters if the per-layer LR is set right, and that the best prescription was often excluded by earlier assumptions. Tensor Programs V shows muP transfer works for width, not automatically for depth or data. | Med | https://arxiv.org/abs/2407.05872 ; https://arxiv.org/abs/2203.03466 | Use muP (or SP with per-layer LR), scale width only, and keep depth fixed or scale it with a stated rule. Verify transfer: the LR optimum at the 3 smallest widths should stay within one grid step. **Kill:** if it moves more than 2 steps, tune per size and budget for the cost. |
| 2.3 | **Tuning cost.** A 5-LR × 2-batch sweep at 4 small sizes is 40 runs. At small scale they're cheap in FLOPs but expensive in wall-clock and researcher time. Brackets at large sizes cost about 3× the largest run. | Med | My arithmetic, not a URL | Budget 15–20% of GPU-hours for tuning and say so in the paper. Do the tuning before the pre-registration freeze. |
| 2.4 | **Instability at high LR.** Wortsman et al. show attention-logit growth and output-logit divergence appear in *small* models at high LR, and that LR sensitivity changes with scale. | Med | https://arxiv.org/abs/2309.14322 | Use QK-layernorm and z-loss from the start. Log max attention logit. Drop and report diverged runs; don't quietly re-run them. |

## 3. Seeds and variance

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 3.1 | **Seed variance can be as large as adjacent-size gaps.** Madaan et al. measure seed variance and monotonicity of benchmarks and find they are often large enough to decide comparisons. DataDecide used 3 seeds. | High | https://arxiv.org/abs/2406.10229 ; https://arxiv.org/abs/2504.11393 | Use at least 3 seeds for every run below 1e19 FLOPs and 2 at the top. Report seed SD next to the gap between adjacent sizes. **Kill:** if seed SD is more than half the gap between adjacent model sizes, the N exponent isn't identified. Widen the size spacing. |
| 3.2 | **Simulator seed vs init seed.** Two sources of randomness: model initialisation and data order, and the SUMO/Stim RNG seed that generates the training set. They need to vary separately. | Med | SUMO randomness: https://sumo.dlr.de/docs/FAQ.html (general docs; per-seed variance UNVERIFIED) | Use a fixed evaluation set for all runs, plus one experiment with 3 independent training-data seeds at a single size. |
| 3.3 | **Logical-error-rate estimates are very noisy.** With k failures, relative SE ≈ 1/√k. At d=7, p=0.001 I saw 3 failures in 200k shots. About 1e7 shots are needed for 10% relative error at LER 1e-5, and 1e8 at 1e-6. | High | Local measurement; binomial arithmetic. LER magnitudes are consistent with https://arxiv.org/abs/2303.15933 (not cross-checked) | Evaluate near threshold (p ≈ 0.003–0.007) where LER is about 1e-2–1e-3, or report log-likelihood per shot as well. Pre-register shot counts per checkpoint. |
| 3.4 | **Closed-loop traffic metrics are heavy-tailed.** Delay and spillback depend on rare gridlock events. | Med | UNVERIFIED for this setup | Use at least 50 demand seeds per evaluation and report a median and a tail quantile. Compute the bootstrap CI before claiming any exponent on decision quality. |

## 4. Broken or inflected scaling

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 4.1 | **The curve may bend inside the range you can afford.** Caballero et al. show saturation, double descent and delayed inflections are common across domains, and fit them with a smoothly broken power law (BNSL). Simulator data with a low irreducible floor is a likely place for this. | High | https://arxiv.org/abs/2210.14891 | Check for a break with leave-one-decade-out validation. If a BNSL fits significantly better, say "broken" and report where the break is. **Kill (for the headline claim):** if the break sits within half a decade of the top of the range, you can't extrapolate. Publish it as a characterisation, not a law. |
| 4.2 | **Data complexity changes the exponents.** Pandey shows scaling laws depend on data complexity (PCFG experiments) and that gzip-compressibility predicts the shift. More compressible data favours more parameters relative to data. | High | https://arxiv.org/abs/2405.16684 | Measure gzip ratio and n-gram entropy of the tokenised SUMO and Stim streams before training. Report them next to the exponents. This is also a cheap way to see that the data are "too easy" before spending GPU-hours. |

## 5. Downstream metrics that don't follow loss

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 5.1 | **Loss may not predict decisions.** In the meta-analysis by Lourie et al., predictable downstream scaling holds only 39% of the time, and "seemingly benign changes" flip it. TESSERA v2 (395 EO runs) finds |r| < 0.2 between pretraining loss and downstream performance. | High (also an opportunity) | https://arxiv.org/abs/2507.00885 ; https://arxiv.org/abs/2607.03949 | This is the study's actual question, so a null result is publishable *only* if it is clean: tight seeds, enough decision headroom (5.3), and a pre-registered metric. |
| 5.2 | **Metric artefacts.** Schaeffer et al. show nonlinear or discontinuous metrics produce apparent emergence, while continuous metrics scale smoothly. Schaeffer et al. 2024 trace the loss of predictability to the step from log-likelihood to multiple-choice accuracy. A discrete metric like "LER below MWPM" or "fraction of episodes without gridlock" will look emergent. | High | https://arxiv.org/abs/2304.15004 ; https://arxiv.org/abs/2406.04391 | Pre-register a continuous decision metric (mean delay; LER as log-probability) plus the thresholded one. Report both on the same axes. |
| 5.3 | **No headroom in the decision metric.** If a perfect model (SUMO itself as the planner's model) only beats an actuated or max-pressure controller by a few percent, decision quality can't show scaling. For QEC, MWPM is already strong at these noise levels, and the gap to near-optimal decoders is limited (AlphaQubit-class decoders beat MWPM, but the size of the gap depends on the noise model; exact factor UNVERIFIED). | High | Objective mismatch: https://arxiv.org/abs/2002.04523 ; closed-loop evaluation of world models: https://arxiv.org/abs/2510.18135 ; https://arxiv.org/abs/2512.07737 | **Week-1 kill test:** run the planner with the true simulator as its model vs the best hand-written controller. If oracle minus baseline is less than 3× the seed SD of the metric, drop decision quality as an axis or change the task. For Stim, measure the MWPM vs maximum-likelihood or correlated-matching gap at the chosen p and d first. |
| 5.4 | **The planner is a confounder.** Lower model loss can give worse control: objective mismatch in MBRL. Results depend on planner horizon and samples. | Med | https://arxiv.org/abs/2002.04523 | Freeze one planner (e.g. CEM with fixed horizon and samples) before training. Report one planner-sensitivity check at 2 model sizes. |

## 6. Simulator data too easy or low-entropy

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 6.1 | **SUMO is close to deterministic given the seed.** Car-following plus fixed signals plus Poisson insertion means most next-state uncertainty comes from a few randomness sources (driver imperfection σ, insertion times). The irreducible loss E is then a property of SUMO's RNG settings, not of traffic. Calling that a "scaling law of traffic world models" would mislead. | High | https://sumo.dlr.de/docs/FAQ.html (simulation is microscopic and seeded; the claim that randomness is concentrated in a few parameters is my inference, UNVERIFIED) | Run with σ = 0 and σ = default and report E for both. Report E as a simulator property. **Kill:** if a 10M model reaches within 5% of E at 1e18 FLOPs, the task is too easy for a 3-decade study. Add multi-junction networks, heterogeneous vehicle types or partial observability. |
| 6.2 | **Stim syndromes are sparse.** About 1.5–1.7% of detectors fire (local measurement). The label is 1 bit per shot, and the Bayes-optimal error is tiny (LER around 1e-5 at d=7, p=0.001). Cross-entropy will plateau near zero, and almost all the remaining signal sits in rare hard shots. | High | Local measurement; https://arxiv.org/abs/2103.02202 | Work near threshold, or use per-shot log-likelihood with error bars that account for the rare-event tail. Scale code distance as a third axis. |
| 6.3 | **"Unlimited" data isn't unlimited information.** Muennighoff et al. find up to about 4 epochs of repeated data costs almost nothing, but beyond that, extra compute is worth less and eventually nothing. A simulator with few distinct layouts or demand patterns behaves like repeated data even when every token is new. | Med | https://arxiv.org/abs/2305.16264 | Track "unique scenarios" (layout × demand × seed) as the data axis, not raw tokens. Run one experiment holding tokens fixed and varying scenario diversity. |

## 7. Data-generation throughput

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 7.1 | **SUMO is single-threaded per simulation.** Its FAQ says there is no multi-node parallelisation for the microsim. Routing can use threads, but a general speedup isn't guaranteed. TraCI goes over a socket; libsumo avoids that overhead. More vehicles and smaller step lengths slow the simulation. | Med (raw tokens), **High (closed-loop eval)** | https://sumo.dlr.de/docs/FAQ.html ; https://sumo.dlr.de/docs/Libsumo.html | Generate with embarrassingly parallel libsumo processes (72 per GPU allocation). Charged at 0.25 NHR/h, ~224 core-hours costs about 3 GPU-hours, which is negligible. The real cost is closed-loop evaluation: each control step needs model inference while SUMO waits, at about 205 steps/s/core before any model call. Count it as checkpoints × shifts × seeds × episode length, and measure it in week 1. |
| 7.2 | **Stim is too cheap to be the bottleneck, and that cuts both ways.** 2B d=7 shots is about 2.4 core-hours at my measured x86 rate. AlphaQubit 2 trains on 2–7B examples, and the 2023 recurrent decoder used about 2B and still hadn't converged at larger distances. So the GPU cost is in training, not data. That's fine for a scaling study, but "cheap labels" alone isn't a contribution. | Med | https://arxiv.org/abs/2512.07737 ; https://arxiv.org/abs/2310.05900 | Make the question one Stim makes uniquely clean: the exact Bayes-optimal gap vs N, D and d, with matched noise shift. Don't present "the data are cheap" as novel. |
| 7.3 | **Stim on Grace may be slower than on x86.** The Stim paper credits 256-bit SIMD. Its build has x86 SSE2/AVX paths and a generic polyfill, and I saw no ARM-specific SIMD path. | Low–Med | https://arxiv.org/abs/2103.02202 ; https://github.com/quantumlib/Stim/blob/main/setup.py | Benchmark on 1 GPU allocation in week 1. Even a 10× slowdown is affordable (UNVERIFIED). |

## 8. Tokenizer and representation

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 8.1 | **Coefficients depend on the tokenizer.** Pearce et al. find world-model and imitation-learning power laws, but the coefficients are "heavily influenced by the tokenizer, task & architecture". Your exponents describe your encoding as much as traffic or QEC. | High | https://arxiv.org/abs/2411.04434 | Freeze one tokenizer. Run a reduced grid (3 sizes × 3 data budgets) with a second tokenizer, e.g. per-vehicle discrete bins vs lane-occupancy grid. Report how much the exponent moves. **Kill:** if it moves more than the cross-domain difference you want to claim, you can't claim a cross-domain difference. |
| 8.2 | **Earlier IL work found loss and return correlated, so it may not transfer.** Tuyls et al. found IL loss and mean return scale smoothly with compute and are strongly correlated (Atari, NetHack). That is the opposite of TESSERA. Which one you reproduce may depend on the task. | Med | https://arxiv.org/abs/2307.09423 ; https://arxiv.org/abs/2607.03949 | Cite both. Frame the study as "under which conditions" rather than "do they or don't they". |

## 9. Novelty risk (searched arXiv 2025–2026; absence is UNVERIFIED)

| Testbed | Closest prior work found | Sev | Mitigation |
|---|---|---|---|
| **Traffic world models** | Scaling laws of motion forecasting and planning, about 500k driving hours: https://arxiv.org/abs/2506.08228 · end-to-end driving data scaling: https://arxiv.org/abs/2412.02689 · DriveVLA-W0 (world models amplify data scaling): https://arxiv.org/abs/2510.12796 · OpenCity traffic foundation model (scaling): https://arxiv.org/abs/2408.10269 · Enactor, a simulator-to-surrogate traffic world model: https://arxiv.org/abs/2603.18266 · long-term traffic simulation with structured autoregressive modelling: https://arxiv.org/abs/2606.31209 | Med | I found no paper that fits N/D/C laws on **SUMO** world models with closed-loop decision quality (absence UNVERIFIED; full texts of 2603.18266 and 2606.31209 not read). Read those two before committing. |
| **World models in general** | Pearce et al. https://arxiv.org/abs/2411.04434 · Tuyls et al. https://arxiv.org/abs/2307.09423 · World-in-World closed-loop evaluation https://arxiv.org/abs/2510.18135 | High | The plan's "gap" (loss vs shift vs decisions together) is narrow. Reviewers will ask why it isn't just Pearce et al. plus World-in-World. Your answer has to be exact-ground-truth shift and decision headroom. |
| **Neural QEC decoders** | AlphaQubit 2 https://arxiv.org/abs/2512.07737 · scalable neural decoders https://arxiv.org/abs/2510.22724 · efficient foundation decoders (transfer across distances) https://arxiv.org/abs/2606.27119 · "Rethink the role of neural decoders" (5 architecture families, accuracy vs latency, d≤9) https://arxiv.org/abs/2605.12046 · QAdapt (noise-adaptive pre-decoding, 110 OOD noise settings) https://arxiv.org/abs/2607.28422 · colour-code pre-decoders https://arxiv.org/abs/2607.10058 · qubit-centric Transformer https://arxiv.org/abs/2510.11593 · decoder confidence as a logical-gap proxy https://arxiv.org/abs/2606.08758 | **High** | This field is crowded and moving fast in 2026. Several of these papers report data or size effects (whether any fits a formal N/D/C law is UNVERIFIED, since full texts weren't read). Stim should be a replication domain, not the headline. If you want it as headline, read 2605.12046 and 2606.27119 in full first. **Kill:** if either fits loss or LER vs N and D, drop QEC novelty claims. |
| **Real freeway "scaling law"** | https://arxiv.org/abs/2507.09530 is about traffic physics, not ML scaling. Not a conflict. | Low | Avoid the name clash in the title. |

## 10. Isambard practicalities

| # | Caveat | Sev | Evidence | Mitigation / kill criterion |
|---|---|---|---|---|
| 10.1 | **24 h walltime on workq, 8 h on the interactive reservation (max 4 nodes, 1 job per user).** Longer work must checkpoint and resume. | Med | https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/ | Checkpoint every 1–2 h. Make resume a tested code path, and kill and resume one run on purpose in week 2. |
| 10.2 | **Queue waits.** No public wait-time data. The docs say limits exist to stop monopolising the queue. | Med (UNVERIFIED) | https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/ | Use job arrays of 1-GPU runs (array max 1,000). Use `--time-min` so backfill can start jobs. Log submit-to-start time from week 1 and replan if the median wait is over 12 h. |
| 10.3 | **CPU work is charged per GPU.** The minimum unit is 1 GPU = 0.25 NHR/h, which includes 72 cores. Using 2 GPUs but all 288 cores is charged as a full node. `--exclusive` charges a full node by accident. | Med | https://docs.isambard.ac.uk/user-documentation/guides/accounting/ | Put 72 SUMO workers in each 1-GPU job. Never use `--exclusive` for generation. Or generate locally (DGX Spark is also aarch64). |
| 10.4 | **aarch64 packages.** Stim 1.16.0 on PyPI: no Linux aarch64 wheel in the checked metadata, but conda-forge has a linux-aarch64 build. PyMatching 2.4.0: no Linux aarch64 wheel. jax-cuda12 plugin/pjrt: aarch64 wheels exist, but compatibility with NGC PyTorch 25.05 is UNVERIFIED. SUMO aarch64 build: UNVERIFIED (build from source or apt inside the container). | Med | https://pypi.org/project/stim/ ; https://anaconda.org/conda-forge/stim ; https://pypi.org/project/pymatching/ ; https://pypi.org/project/jax-cuda12-plugin/ ; https://docs.isambard.ac.uk/specs/ ("x86_64 … will not work") | Build one Apptainer image on top of NGC PyTorch 25.05 with Stim, PyMatching and SUMO compiled from source. Use PyTorch only and drop JAX. **Kill:** if the image isn't working by end of week 1, run QEC data generation off-cluster. |
| 10.5 | **Multi-node over Slingshot.** NCCL needs the aws-ofi-nccl plugin with `NCCL_NET="AWS Libfabric"`. The guide shows about 162.69 GB/s bus bandwidth with it and about 2.32 GB/s without, so a missing plugin silently costs about 70×. Known issue: GLIBC mismatches in multi-node NGC containers; 25.05 is listed as known-working. | Med | https://docs.isambard.ac.uk/user-documentation/guides/nccl/ ; https://docs.isambard.ac.uk/service-status/known_issues/ | Only the top 1–2 grid points (more than 1B parameters) need multi-node. Run nccl-tests at 2 nodes before any multi-node training, and require bus bandwidth above 100 GB/s. **Kill:** if 4-node scaling efficiency is below 70%, run the top point on 1 node for longer instead. |
| 10.6 | **Project end and expiring NHR.** Storage and access end with the project, and project storage isn't archival. Usage should be spread evenly; some allocations require a minimum per month or quarter, and hours outside it are lost. BriCS can't extend dates; the allocating body decides. u6xn's end date: UNVERIFIED. | **High** | https://docs.isambard.ac.uk/user-documentation/information/project-policies/ ; https://docs.isambard.ac.uk/user-documentation/guides/accounting/ ; https://docs.isambard.ac.uk/policies/resource_management/ ; https://docs.isambard.ac.uk/access/ | Find the end date and any usage-profile rule before planning. Copy checkpoints, fits and eval logs off the cluster every week. |
| 10.7 | **Storage.** Phase 2 project storage is 200 TiB and scratch is 5 TiB. Scratch files not accessed for 60 days are deleted. | Low | https://docs.isambard.ac.uk/user-documentation/information/system-storage/ | Tokenised data (around 100 GB) fits. Generate data on the fly where possible. Keep only final and cooldown checkpoints. |

## 11. Is 80k GPU-hours small?

| Study | What is verified | Compute |
|---|---|---|
| Chinchilla | More than 400 models, 70M–16B+ parameters, 5–500B tokens https://arxiv.org/abs/2203.15556 | Final 70B model: 6·N·D with D = 1.4T tokens (taken from the paper body, not re-checked) ≈ 5.9e23 FLOPs ≈ **450k H100-hours** at the plan's 35% MFU. That's more than 5× all of u6xn for one run. Total for the grid: UNVERIFIED. |
| Kaplan | More than 7 orders of magnitude https://arxiv.org/abs/2001.08361 | Total UNVERIFIED |
| Porian et al. | ~22.3K A100-hours https://arxiv.org/abs/2406.19146 | Smaller than u6xn |
| Hägele et al. | ~2.5–3k GPU-hours (as captured during research; exact figure not re-checked) https://arxiv.org/abs/2405.18392 | Smaller |
| Everett et al. | Tens of thousands of models up to 26.8B https://arxiv.org/abs/2407.05872 | Total UNVERIFIED (almost certainly far more than u6xn) |
| DataDecide | 25 corpora × sizes up to 1B × 3 seeds, up to 100B tokens https://arxiv.org/abs/2504.11393 | Total UNVERIFIED |
| AlphaQubit 2 | 2–7B training examples https://arxiv.org/abs/2512.07737 | UNVERIFIED |

**Verdict:** 80k H100-hours is *more* than the published methodology papers (Porian, Hägele) used. It is enough for a careful 1e17–3e21 study in **one** domain with seeds. It is not enough to claim laws that extrapolate to frontier scale, and not enough for two full domains with tuning, seeds, a second tokenizer and closed-loop evaluation. Count GPU-hours, not FLOPs: at 1e17–1e19, overhead dominates.

## 12. Risk to a single researcher's time

| # | Caveat | Sev | Evidence | Mitigation |
|---|---|---|---|---|
| 12.1 | Two simulators, a tokenizer, a model family, a muP/LR-transfer check, an HPC container, checkpoint and resume, multi-node, a closed-loop planner, a fitting and bootstrap pipeline, and a pre-registration. That's a team's worth of infrastructure, and it competes with a full-time technician job (UNVERIFIED workload). | **High** | No URL; judgement. Scale of comparable studies: https://arxiv.org/abs/2406.19146 ; https://arxiv.org/abs/2504.11393 | One domain. One model codebase (e.g. a nanoGPT-style trainer). Off-the-shelf fitting code. A dated milestone list with a go/no-go every 2 weeks. |
| 12.2 | Work spread evenly across the project (10.6) conflicts with "build infrastructure first, then burn the budget". | Med | https://docs.isambard.ac.uk/policies/resource_management/ | Start small-grid runs in week 2 so usage is steady from the start. |

## 13. UKRI/AIRR relevance

| Testbed | Evidence | Assessment |
|---|---|---|
| **Stim QEC** | AIRR's AI-for-Science priorities explicitly name quantum technologies https://docs.isambard.ac.uk/access/ ; routes and sizes: https://apply.isambard.ac.uk/ukri | Strongest written fit. Matches the user's Qiskit/QEC profile. Weakest on novelty (section 9). |
| **SUMO traffic** | Transport isn't named in the AIRR AI-for-Science text I read (UNVERIFIED for every call). It could fit "AI-driven scientific discovery" or AI safety/evaluation framing. | Fit is plausible but would need to be argued. The "does loss predict decisions" angle reads as AI evaluation methodology, which is arguably more fundable than traffic itself (my interpretation). |
| u6xn route | 80k GPU-hours exceeds Gateway (10k) and Rapid Access (20k), so it's probably AI Open Access or similar https://apply.isambard.ac.uk/ukri | Which route, and what it committed to: UNVERIFIED. If u6xn was awarded for a different stated project, check that a change of topic is allowed. |

---

## The 5 things that would most likely sink this

1. **No decision headroom.** Oracle-model planning barely beats a hand-written controller (SUMO) or MWPM (Stim), so the decision axis is flat and the main question has no answer (5.3).
2. **The range is too narrow, or the scaling is broken.** Two decades of compute, plus a simulator that saturates early (6.1, 4.1), give a fit that can't extrapolate. Reviewers can cite Besiroglu and Porian against it (1.1, 1.2, 1.5).
3. **The exponents are artefacts of choices.** Untuned LR and warmup per size, the parameter count, or the tokenizer move the exponent more than the effect being claimed (2.1, 1.4, 8.1).
4. **Scooped or crowded.** 2026 neural-decoder papers already cover data and size effects, and Pearce et al. plus closed-loop world-model evaluation cover most of the traffic angle (section 9).
5. **Time and calendar.** One researcher building two pipelines on aarch64 HPC with 24 h jobs, against a project end date that is unknown and can't be extended (10.6, 12.1).

## The minimum design that survives them

1. **One domain: SUMO.** Stim is optional, as a 3-size × 3-data replication only if time is left. (Reason: QEC novelty risk; SUMO has the decision axis.)
2. **Week 1 go/no-go, no training:**
   - Find the u6xn end date and any usage rules.
   - Build an aarch64 container with SUMO from source.
   - Measure libsumo throughput on 1 GPU allocation (72 cores).
   - Gzip and entropy of the tokenised stream.
   - **Oracle-planner vs max-pressure/actuated headroom across 50 seeds.** Kill if headroom is less than 3× seed SD.
   - Make SUMO harder before any GPU spend: σ > 0, several junctions, partial observability.
3. **Hyperparameter rules from a small sweep:**
   - muP or SP with per-layer LR; width-only scaling at fixed depth.
   - LR/batch/β2 sweeps at 4 sizes from 1M to 30M; check LR transfer.
   - QK-norm and z-loss.
   - Budget about 15% of GPU-hours.
4. **Grid:**
   - 1e17–3e21 FLOPs (3.5 decades), at least 4 points per decade.
   - Constant LR with cooldown branches for the D axis.
   - 3 seeds below 1e19, 2 above.
   - Count total parameters and FLOPs including the head.
   - Hold out the top run and a separate shift split (unseen layouts and demand).
   - Estimated about 25–40k GPU-hours (estimate, UNVERIFIED until MFU is measured).
5. **Fitting, frozen before the full grid:**
   - Chinchilla form as primary, BNSL as secondary.
   - Huber loss on log-loss, bootstrap over runs.
   - Report all three Chinchilla approaches and both parameter counts.
   - Write down the prediction for the held-out run before running it.
6. **Decision axis:**
   - One frozen planner.
   - A continuous metric (mean delay) and a thresholded one (gridlock rate).
   - 50 demand seeds per evaluation.
   - Report the correlation of loss with decisions with a CI, whatever it turns out to be.
7. **One robustness check:** a second tokenizer at 3×3 points. Report how much the exponent moves.
8. **Infrastructure:** single-GPU job arrays for everything below 1B. Multi-node only for the top 1–2 points, after nccl-tests show more than 100 GB/s and more than 70% efficiency at 4 nodes. Checkpoint every 1–2 h. Copy results off the cluster weekly.
9. **The claim is narrow:** "On a controlled simulator with exact ground truth, here is how loss, shift-loss and decision quality scale, and whether they agree". Not "scaling laws for traffic".
