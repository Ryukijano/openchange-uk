# Problem choice for u6xn: UK priorities, your track record, and 20,000 node-hours

Written 2026-10-07. Sources are linked. "Check" marks a claim I haven't verified. I've separated what I found from what I conclude.

## 1. The budget in GPU terms

From the Isambard accounting guide (https://docs.isambard.ac.uk/user-documentation/guides/accounting/):

- 1 node = 4 GH200 superchips. Each has an H100 with 96 GB, plus a 72-core Grace CPU. 1 NHR = 4 GPU-hours.
- 20,000 NHR = 80,000 GPU-hours. For example:
  - 8 nodes (32 GPUs) for 24 h = 192 NHR, about 1% of the budget;
  - 16 nodes (64 GPUs) for 72 h = 1,152 NHR, about 6%.
- NHR not used by the project end date is lost. A job won't start if the remaining NHR can't cover its requested walltime.

So the budget is big enough for several multi-node post-training runs of 10–20B-parameter video models. It is not enough to pretrain a foundation model, and it shouldn't go on cheap probing work.

## 2. What the UK is funding (AIRR and the AI for Science strategy)

- AIRR routes on Isambard-AI (https://apply.isambard.ac.uk/ukri/):
  - **Gateway**: up to 10k GPU-h for 3 months. Rolling.
  - **Rapid Access**: up to 20k GPU-h. Rolling, SMEs only.
  - **AI Open Access**: 50k–1.4M GPU-h over 6 or 12 months. Fixed deadlines; the last round closed 17 July 2026.
- AI Open Access and the earlier "AI for Science" call (200k–1M GPU-h, closed Dec 2025) target the AI for Science priority areas:
  - materials science;
  - nuclear fusion;
  - **medical research**;
  - engineering biology;
  - **quantum technologies**;
  - **AI-driven research and scientific discovery** (new models and virtual systems for automated discovery).
- They give extra weight to the government missions: growing the economy, **an NHS fit for the future**, **safer streets**, opportunity for all, and clean energy. The Gateway route also lists the Industrial Strategy sectors.
- Eligibility: the AI for Science call said research technical professionals count as academic staff and can lead. AI Open Access requires lecturer-equivalent leads. Check the terms of the next round.
- The AI for Science Strategy (Nov 2025, https://www.gov.uk/government/publications/ai-for-science-strategy/ai-for-science-strategy) directs up to £137m. It stresses data, compute at scale, interdisciplinary teams and autonomous labs.

Environmental and coastal EO is **not** on the priority list. It fits only loosely under clean energy. That matters if u6xn's results feed into a future AIRR bid.

## 3. Your track record (from GitHub, Hugging Face, LinkedIn, ORCID)

- **2023–24, MSc in Advanced CS & AI at Leeds:** future-frame prediction for robotic surgery. Earlier work: a pothole-detection paper (YOLOv7 + ESRGAN).
- **Nov 2025 onward, Research Technician in the Leeds AIMS group** (Sharib Ali): medical computer vision, working with Leeds Teaching Hospitals clinicians.
- **Published:** ISBI 2026, "Self-Supervised Vision Transformer for Surgical Phase Recognition in Endoscopic Submucosal Dissection" (now on IEEE Xplore).
- **Surgical video lineage:**
  - DINO/V-JEPA2 phase models (Cholec_Vjepa-2, dino-endo, ai-endo);
  - GOT-JEPA surgical tool tracking on CholecTrack20;
  - **ESD-WORLD**: a LoRA of NVIDIA Cosmos-H-Surgical Predict 15B on ESD video, trained with FSDP on 3× L40S on AIRE, reaching iteration 600.
- **World models and control:** LeWorldModel, jepa-wms, C-JEPA PushT planning, a MARL world model in Isaac Sim, Gemma-GR00T VLA, an MPC vs VLA vs diffusion-MPC study, and agentic SfM with GRPO.
- **Traffic:** Cosmos-Sentinel, which combines V-JEPA2 collision gating, Cosmos Reason 2 and Cosmos Predict 2.5 for what-if rollouts. Your JunctionLab paste comes from this.
- **Quantum:** Qiskit Advocate, Bradford Quantum Hackathon winner, NQCC hackathon. syndrome-net: Stim QEC with an NVIDIA Ising pre-decoder, RL decoders and calibration environments.
- **Other:** the Conjugate ADC harness (Iterate Hack), and cuda-blackwell-labs for the DGX Spark (GB10).

**My reading:** your work runs from detection, to self-supervised video representations, to **predictive world models**, to planning and control. Surgical video is the strongest part of it: a peer-reviewed paper, clinical collaborators, and a 15B world-model prototype already training. EO is new to you, with no prior record.

## 4. Re-ranked problems for 80k GPU-hours with multi-node training

Each one keeps the identity running through your work: **a predictive model that knows when it's wrong, judged against ground truth rather than on looks.**

### A. Surgical world models scored on clinical consistency, not FVD (best fit)
- Scale ESD-WORLD to Isambard: Cosmos-H-Surgical Predict with FSDP across 2–8 nodes, plus an action- or instrument-conditioned V-JEPA 2.1 predictor as a cheaper comparison.
- Evaluate predicted futures with your own phase recogniser and tool tracker: is the predicted phase, tool presence or tool motion consistent with what actually happened?
- Report calibration, and when the model should defer.
- Fits **medical research** and the **NHS** mission. Compute: 30–50% of the budget.
- **Blocker:** patient video governance. Check whether Isambard's terms allow this data and what your NHS data-sharing agreement permits. Public fallbacks: Cholec80, CholecT50, CholecTrack20 (licences to check).

### B. Neural QEC decoders trained on unlimited simulated data (strong AI-for-Science fit)
- Train transformer decoders on Stim syndrome data, which is free, unlimited and has exact labels. Data-parallel training scales across nodes easily.
- Honest evaluation:
  - logical error rate against MWPM and BP-OSD;
  - generalisation to unseen code distances and noise models;
  - a calibrated "defer to MWPM" signal;
  - decode latency.
- Builds directly on syndrome-net. Fits **quantum technologies**. No licence or patient-data problems.
- I recall DeepMind's AlphaQubit as the reference here; check what is now state of the art. Compute: 20–30%.

### C. Traffic world models checked against simulator truth (JunctionLab + Cosmos-Sentinel)
- Train SUMO junctions with action-conditioned forecasting, building on the RESCO and Traffic-Alpha baselines.
- Post-train Cosmos Predict 2.5 on SUMO renders and score its futures on SUMO's real queues and counts.
- Fits **safer streets**. Compute: 10–20%.

### D. OpenChange-UK EO transfer (the current repo)
- The leakage-aware benchmark and the TESSERA embeddings-vs-fine-tuning comparison need little compute: under 5%.
- Worth finishing because the harness exists, but it shouldn't drive the allocation.

### Starting split (a proposal for you to change)
- 10% smoke tests, scaling ladder and debugging
- A 40%, B 25%, C 15%, D 5%
- 5% held back

## 5. Multi-node plan (once the smoke passes)

- **Scaling ladder:** 1 GPU, then 1 node (4 GPU), then 2 nodes, then 4–8 nodes. At each step, measure throughput and scaling efficiency before going bigger.
- **NCCL over Slingshot:** use the setup in Isambard's NCCL and containers guides (now installed as the isambard-nccl and isambard-containers skills). Write checkpoints to `$PROJECTDIR` and keep active data on `$SCRATCHDIR`.
- **Job time limit:** the repo currently caps jobs at 30 minutes. For multi-hour multi-node runs you need to set a new cap per job class, for example scaling tests at 1 h or less and training at 24 h or less with resumable checkpoints.
- **Local PC:** if it's a DGX Spark (GB10), it's aarch64 like GH200, so the same arm64 NGC containers run on both. That makes it a good place to develop and run inference before spending NHR. (I'm inferring this from your cuda-blackwell-labs repo.)

## 6. Decisions needed from you

1. Which tracks: A, B, C or D, and in what proportions?
2. The u6xn project end date, so NHR isn't lost.
3. For A: can ESD video legally go onto Isambard?
4. The new time limits for scaling and training jobs.
