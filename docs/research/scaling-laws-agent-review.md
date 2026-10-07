# Scaling-laws study: review by three agents (2026-10-07)

Three separate sessions worked on this: an explorer looking for ideas, a pessimist looking for caveats, and a verifier fact-checking. Their full reports are attached, with sources. This file is my merge of them. Where they disagree, I say so.

## Bottom line

- **Compute isn't the limit.** 80k H100-hours is more than Porian et al. used to resolve the Kaplan/Chinchilla gap (about 22k A100-hours). A careful study in one domain is affordable.
- **The claimed gap was too broad.** Whether loss predicts decisions under scaling has been studied before:
  - Tuyls 2023, return vs imitation-learning loss in Atari/NetHack;
  - Hilton 2023, RL scaling;
  - Waymo 2506.08228, closed-loop planning that scales with compute;
  - World-in-World 2025.
  
  There is also counter-evidence: Lambert 2020 on objective mismatch, and a 2026 LunarLander result (2607.01736) where model loss kept improving after control had collapsed.
- **What still looks open:** none of the agents found a study that does both of these:
  - (a) a **controlled, simulator-scored shift** (unseen layouts, noise or sizes) together with loss and decision quality;
  - (b) a **latent (JEPA) vs generative world-model scaling comparison measured on planning success**.
  
  Neural QEC decoders have no fitted compute-optimal law yet, but the area is crowded in 2025–26, so it works better as a second domain than as the headline.
- **The study fails if the decision axis is flat.** If planning with a perfect model barely beats a simple controller, there's nothing to scale. Test this before spending GPU time.

## Candidate studies

| | A. Latent vs generative world models on planning | B. SUMO junction: in-distribution loss, shift and control | C. QEC decoders vs code distance |
|---|---|---|---|
| Question | Does lower loss mean better plans, and does that differ between JEPA and pixel/generative models as both scale? Test-time planning compute is a second axis. | Do in-distribution loss, shifted loss and control quality scale together? | Compute-optimal decoder size vs code distance and noise |
| Builds on | LeWM, jepa-wms, C-JEPA PushT | JunctionLab, Cosmos-Sentinel | syndrome-net |
| Main risk | Evaluation protocol: one LeWM checkpoint moved from 14% to 84% success just by changing the protocol (2608.10145). The planner and goals must be frozen. | Low entropy (SUMO's RNG sets the irreducible loss); no headroom over max-pressure control | Crowded field; logical error rate needs about 1e7 shots per checkpoint at 1e-5; no aarch64 wheels |
| UK fit | AI-driven discovery (weak) | Safer streets mission (not an AI for Science priority) | Quantum (named priority) |
| Explorer's estimate | about 5.5k GPU-h | 1–7% of budget | needs multi-node at d ≥ 15 |
| Pessimist's view | not reviewed separately | The one domain that survives, if headroom exists | Replication only |

The explorer ranked A first, then C, then The Well (PDE surrogates) or B. The pessimist's minimal design is B alone. **My view:** A or B as the core, chosen by the week-1 tests below, with C as a small replication if there's time.

## Corrections to my earlier plan

- **Waymo 2506.08228** does cover planning and closed-loop metrics, and reports a loss–metric correlation. Its data is 447k hours, not "about 500k". My table was wrong.
- **Pearce et al.** use game data and RT-1 robot data. They also cite strong loss–return correlations (stronger than −0.94).
- **Grid range:** 1e19–1e21 is only 2 decades. Widen it to about 1e17–3e21, with at least 4 points per decade.
- **Model sizes:** a 1M–1B family probably won't reach the compute-optimal size at 1e21. The Chinchilla rule of thumb gives about 2.9B, though this is an inference and the world-model ratio may differ. Check it on the small grid.
- **Utilisation:** 35% MFU is optimistic for models under 100M. The GH200 BF16 peak is unverified; the H100 SXM dense figure is 989 TFLOPS.
- **Hyperparameters:** without per-size LR, warmup and β2 tuning (or muP), the exponents are artefacts. Budget about 15% for sweeps.
- **Realistic cost** with seeds, a wider range and sweeps: **25–40k GPU-hours (30–50%)** for one domain. My earlier figure of 10% only covered the bare grid.
- **Tokenizer:** the coefficients depend on it (Pearce). Run a second tokenizer at 3×3 points and report how much the exponent moves.
- **Downstream predictability:** only 39% of downstream scaling cases are predictable (2507.00885), and TESSERA v2 found |r| < 0.2 between loss and downstream. Pre-register continuous decision metrics.

## Isambard facts (verified on docs.isambard.ac.uk)

- **Walltime:** maximum 24 h on `workq`; the interactive queue is 8 h and 4 nodes. Large runs need checkpoint and resume.
- **Job limits:** no per-job node cap is documented for Phase 2. The 32-GPU QoS is listed under Phase 1. Per user: 256 running jobs and arrays of up to 1,000.
- **Multi-node:** use `brics/nccl`, which includes aws-ofi-nccl. Measured all-reduce is about 163 GB/s with it and about 2.3 GB/s without. Containers use `brics/apptainer-multi-node` plus `/host/adapt.sh`.
- **Power and CPU:** each GH200 is power-capped at 660 W, shared between CPU and GPU. CPU-only work is still charged 0.25 NHR/h per 72 cores.
- **Hours can be lost:** NHR expires at project end, and some allocations lose hours outside a minimum monthly or quarterly usage. **Check u6xn's end date and usage rules first.**
- **Software:** SUMO/libsumo have aarch64 wheels on PyPI. (conda-forge "sumo" is a different package.) Stim and PyMatching have **no** aarch64 wheels, so they must be built from source. PyTorch and JAX are fine.
- **Data generation:** for SUMO this is probably not the bottleneck. An x86 test gave about 205 TraCI steps/s per core, so about 224 core-hours for 58B tokens (Grace speed unmeasured). Closed-loop evaluation is the slow part.

## Week-1 go/no-go (no GPU training)

1. Get u6xn's end date and usage rules.
2. **Headroom test:**
   - B: an oracle planner using the true simulator vs max-pressure/actuated control, over 50 seeds.
   - A: planning with true dynamics vs a policy baseline.
   - Kill the decision axis if headroom is less than 3× the seed SD.
3. **Entropy test:** compress the tokenised stream and estimate the irreducible loss. Make SUMO harder before spending GPU time (noise, several junctions, partial observation).
4. **aarch64 container:** SUMO from wheels, Stim from sdist.
5. **One human-run 1-GPU job:** measure MFU at 3 model sizes on GH200.

Steps 2–4 can run on CPU here or on your local PC, without Isambard.

## If it goes ahead: design rules (pessimist's minimum design)

- Use one frozen planner and continuous metrics (mean delay or success rate), with 50 seeds per evaluation.
- Use the Chinchilla form as the primary fit and BNSL as secondary. Fit with Huber loss on log-loss, bootstrap the CIs, and report both parameter counts.
- Hold out the largest run and a separate shift split. Write down the prediction before running it.
- Use 1-GPU job arrays below 1B parameters. Go multi-node only for the top 1–2 points, after nccl-tests show more than 70% efficiency at 4 nodes.
- Keep the claim narrow: "on a controlled simulator with exact ground truth, here is how loss, shifted loss and decision quality scale, and whether they agree".
