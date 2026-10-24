# Moonshots on top of agentic SfM (24 Oct 2026)

Inputs: your 880 X bookmarks and 794 likes (scraped from the logged-in VM browser, 24 Oct), the HF/GitHub/portfolio inventory, and the agentic-sfm README. "Verified" = I read the source. "Inference" = my judgement.

## What your bookmarks say you actually care about

Counts over 1,674 posts (keyword buckets, overlapping): robotics/VLA/RL 349, world models/JEPA 129, 3D/SfM/matching 97, architecture 85, GPU/kernels 33, EO 13. Most-saved accounts after HuggingPapers: @gabriberton (38), @rsasaki0109 (28, 3D/SLAM), @CSProfKGD, @zhenjun_zhao (3D paper digests).

The 3D posts you save are not "better matcher" posts. They are: Déjà View (looped transformer for multi-view 3D), Zebra3D (linear attention for scalable 3D), Scal3R (kilometre-scale online 3D on a frozen backbone), Trust3R (uncertainty for feed-forward 3D), GATOR (agentic 3D object reconstruction), LoMa on AstroLoc, and "agents + an exact verifier" (Phasecraft, 2610.31352). Plus a run of looped-transformer and attention-residual (MSAR) posts. EO barely appears. Inference: your taste is agentic 3D + architecture + compute, and the plan should follow that.

## The stack

### M0. Gate (about 1 GPU-day, no training)
HardMatch val (100 pairs) through {LoMa-G, RoMa-Ω, MASt3R, VGGT, LoFTR} × {full, 2-3 crops}. Report best single tool, per-pair oracle, per-category oracle, bootstrap CI. Kill pair routing if oracle − LoMa-G < 5 mAA. This decides whether M1 is a routing paper or a scene-level paper.

### M1. Agentic SfM at scale (main paper + demo)
Verified: no paper does per-pair VLM matcher/crop routing for SfM (search 24 Oct; nearest are AstraLOD3 2609.28061, GATOR 2610.11215 which is object-level edit-render-review, SPAgent/SpaceTools which are QA). Your README:81 shows +7.6 points per-pair oracle headroom on ScanNet.
Contribution: a learned router/agent that recovers a stated share of the oracle gap over the best single matcher on HardMatch and WxBS, at a stated cost, plus scene-level decisions (which pairs to match, when to bridge or prune) on hard scenes.
Baselines it must beat: LoMa-G alone; a feature-based router (GBM over inlier stats and DINO similarity) with no VLM.
Demo: two hard photos in, the tool chosen and the pose out. Same shape as AstroLoc.
Compute: inference-heavy. RL in the tens to low hundreds of GH200-hours (inference, your earlier runs were 3×L40S for 50 epochs).

### M2. A verifier the agent can trust (fixes your two recorded failures)
Verified: pair agent lost to LoFTR (62% vs 88%); scene GRPO collapsed to LoFTR with 0 MASt3R/VGGT calls (README:31,45). Both are reward and stop-signal problems. Trust3R (ICML 2026, 2605.19539) shows an evidential head on feed-forward 3D gives 25% lower AURC than MASt3R's confidence on ScanNet++. The Phasecraft post you saved is the same pattern: agent proposes, exact verifier checks.
Contribution: a two-view/scene geometry verifier (does this pose and pointmap agree with the images?) trained with your JEPA/latent-prediction skills, used (a) as the agent's stop/accept signal and (b) as a dense GRPO reward, with anti-collapse constraints (minimum tool diversity, cost-aware reward).
Decisive experiment: risk–coverage of the verifier vs MASt3R confidence and Trust3R on your frozen ScanNet pairs and HardMatch; then one GRPO run with verifier reward vs the collapsed baseline, same seeds. Kill if the verifier does not beat Trust3R's AURC or GRPO still collapses.
This is where "world model" skills survive: as a learned critic inside a task with a metric.

### M3. Looped 3D transformer at scale on GH200 (the compute moonshot)
Verified: Déjà View (NeurIPS 2026 oral, 2605.30215, code released) loops one VGGT-style block K times, 117M params, beats π³ and Depth Anything 3-G with 8-10× fewer params, and K is an inference-time compute knob. Same group's Scal3R does kilometre-scale online 3D. Your bookmarks also hold LoopFormer (elastic-depth looping), MSAR (multi-stream attention residuals), and Zebra3D (linear attention 3D).
Open question (inference): nobody has shown the scaling law of looped 3D blocks against data and K at matched FLOPs, or what residual/attention scheme inside the loop makes depth-elastic 3D stable. That is a genuine use of 20k GH200-hours: a training sweep, not a demo. It also plugs straight into M1/M2 (a faster, elastic specialist and verifier backbone) and into your GH200/Blackwell kernel work (a looped block is one kernel to make fast).
Decisive experiment: reproduce Déjà View at small scale from the released code; then a matched-compute sweep over {loop count K, data size, residual scheme ∈ {plain, attention-residual, MSAR-like}} on two benchmarks. Kill if no variant beats the released checkpoint at equal FLOPs after the first 1k GH200-hours. Cost the full sweep only after reading their training section; I have not.

### Not recommended
- Robot WAM history studies: crowded (2609.24048, 2610.21095, 2610.18774, 2610.15305, 2610.22323). Your own bookmark says it: "every new system swaps the video backbone, training data, …".
- EO anomaly feed as a moonshot: CloudFerro Anomaly Watch and Google's AlphaEarth change products exist; 13 of 1,674 saved posts are EO.
- Satellite LoMa: Nordström (LoMa author) replied on 11 Oct that a satellite version "should be coming". Do not race them.

## Order
M0 now (1 GPU-day) → M1 and M2 together (same harness, same data) → M3 as the Isambard allocation's main training line once M0's answer fixes the specialist pool. One paper each; one demo (M1).
