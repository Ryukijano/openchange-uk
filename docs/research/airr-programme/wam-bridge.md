# WAM bridge: where world-action models fit the AIRR plan

Status: design note, 10 Oct 2026. Companion to CLEAN_JEPA_PLAN_V2.md. Nothing here has been run. Items marked UNVERIFIED were not checked against a primary source this session.

## 0. The problem with plan v2

Plan v2 (clean JEPA on Earth observation) answers "massive open data + clean publication", but it does not touch world-action models (WAMs), which are a stated goal. This note asks whether the same methodology (clean JEPA, pre-registered scaling grid, matched baselines, three operational guarantees) can be pointed at WAMs instead, and what the honest cost is.

## 1. What changed this week (all 7–8 Oct 2026)

| Paper | What it does | What it leaves open |
|---|---|---|
| [Long-WAM](https://arxiv.org/abs/2610.10528) (NVIDIA/MIT/HKU) | AR video pretrain (LongLive2.0-Robot, ~10,000 window-equivalent h: RoVid-X 82.7%, EgoDex 9.0%, AgiBot World 4.8%, EgoVerse 2.6%, VITRA 0.8%) then causal world-action adaptation. RoboCasa GR-1 success 63.3→78.7% as history goes 0→19.2 s, drops at 38.4 s; LIBERO-Long 94.5→99.5%, saturates at 2.4 s. Central claim: **a bidirectionally pretrained init shows no net gain from history; AR pretraining makes history usable.** Pretrain ≈30,720 H100 GPU-h on 64 H100; WAMs on 16 GB200. Code Apache-2.0, checkpoints on HF (five GR-1 history lengths). | Five history points, two benchmarks, one model size, one backbone family. No law in history × model × data, no matched-compute comparison, and no test of whether the history benefit needs *pixel generation* or only *causal structure*. |
| [RoboJEPA](https://arxiv.org/abs/2610.10515) (Meta) | Action-conditioned latent world model on frozen V-JEPA 2.1, 22M–8B, 23 datasets; imagination error follows a second-order power law in compute; planning success correlates with it. Checkpoints and code promised. | Frozen encoder (laws are about the predictor only). History handled only as a max trained length. Its own Fig. 11: on the short-horizon obstacle-reach task "reward correlates with compute, but no clean frontier emerges and smaller models often beat larger ones". No action head, no policy. |
| [LeWAM](https://arxiv.org/abs/2610.12407) (JEPA WAM, diffusion-steering MPC) | ViT-tiny + SIGReg, one bidirectional transformer for forward/backward/inverse/policy modes; 3 context frames; PushT/robomimic. Finding: planning in the policy's noise space stops MPC exploiting dynamics errors. | Tiny scale, 3-frame history, sim only. Note the name clash with [XuejiFang/LeWAM](https://github.com/XuejiFang/LeWAM) (Baidu, "Latent Evolving WAM", I-JEPA-Huge). |
| [LaWAM](https://arxiv.org/abs/2606.15768) | Latent WAM on a frozen VFM, latent subgoals instead of video; LIBERO 98.6%. | No scaling, no history study. |
| [RoboTTT](https://arxiv.org/abs/2607.15275) | VLA with test-time-training fast weights; context 1K→8K timesteps gives +62%; "context length as a new scaling axis". | Two context points, one architecture, policy not world model. |

Reading: the field has just established (i) latent world models scale predictably (RoboJEPA), (ii) history helps WAMs *if* the pretraining is causal (Long-WAM), (iii) clean JEPA latents can carry a WAM at small scale (LeWAM). **Nobody has combined them**: a clean, decoder-free, *causal* JEPA pretrained on the same open robot video, adapted to actions the same way, with history length as a pre-registered scaling axis against the AR-generative backbone at matched compute.

## 2. Candidate topic (one sentence)

**Does the benefit of visual history in world-action models come from causal pretraining structure or from pixel generation, and does it follow a law in history length, model size and data that survives shift?**

### Gate 1, open?
Five closest: Long-WAM (history benefit, one backbone, no law); RoboJEPA (laws, no history axis, no action head, frozen encoder); RoboTTT (context axis, policy only, two points); LeWAM (clean-JEPA WAM, tiny, 3 frames); LeVJEPA (clean video JEPA, bidirectional masking, no actions; UNVERIFIED whether a causal variant exists). Earlier work on latent-vs-generative world models does not vary history. **open: yes**, but the window is weeks, not months: Long-WAM's authors can run the latent arm themselves.

### Gate 2, contribution?
"Unlike Long-WAM, which attributes the history gain to AR pretraining with one generative backbone, we pretrain a causal clean-JEPA (no decoder, no EMA, SIGReg) on the same open video at matched FLOPs and fit the first joint law of closed-loop success in history length × compute × data, which tells practitioners whether a 30k-GPU-hour video generator is needed to use history or a far cheaper latent predictor suffices, and how much history to buy per unit of decision-time compute."

Three results, each publishable with either sign:
1. **Structure vs generation.** Causal clean-JEPA vs AR-generative vs bidirectional JEPA, same data, matched pretrain FLOPs, same action adaptation. Outcome A: latent causal gets the history gain at a fraction of the cost (applied win: edge WAMs without video decoding). Outcome B: it does not; generation carries information history needs (a real finding about what latents drop).
2. **History law.** Success and imagination error vs history length at 4 sizes × 3 data fractions; is there a predictable optimum (Long-WAM's drop at 38.4 s) and does it move with scale? Pre-register the prediction for the held-out largest run.
3. **The EO-plan guarantees, ported.** (a) Label-free selection: does clean-JEPA pretraining loss rank WAM checkpoints by closed-loop success (RoboJEPA says yes for imagination error, no for reward on one task; the "Control Theory of Predictability" and "Not All Errors Matter" papers say no in general). (b) Version compatibility: can an action expert trained on world-model version k be reused on version k+1 after a Procrustes fit, so retraining the policy per backbone (what Long-WAM ships per benchmark) is unnecessary? (c) Substitution: held-out camera/embodiment, the analogue of SAR-for-optical.

Method contribution, parallel to availability-stratified SIGReg: **history-stratified SIGReg**, Gaussianise the latent conditional on how much history was available so the embedding does not encode context length as nuisance, which is exactly what a mixed-history training batch invites. Same ablation grid as plan v2. UNVERIFIED that pooled SIGReg actually encodes history length; this is pilot P2.

**contribution: yes if result 1 or 2 lands; unclear if only 3.**

### Gate 3, feasible?
**Data.** Open and large: RoVid-X 4M clips / 10K+ h, CC-BY-4.0, HF ([DAGroup-PKU/RoVid-X](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X), size category >1 TB; bytes UNVERIFIED), and it is 82.7% of Long-WAM's own mix. Open X-Embodiment CC-BY-4.0. RoboCasa365: code MIT, assets and 600 h human + 1,600 h synthetic demos CC-BY-4.0, GR-1 humanoid tasks with real headroom (63–79%), which fixes the LIBERO saturation problem from deep-think. Exclude or quarantine: AgiBot World (CC-BY-NC-SA), EgoDex (CC-BY-NC-ND; derivative-weights question), EgoVerse/VITRA (licence UNVERIFIED). Storage: RoVid-X raw will not fit 5 TiB scratch; pre-encode to latents once in $PROJECTDIR, as in plan v2.

**Compute.** Long-WAM's generative pretrain alone is ~30.7k H100-h, i.e. ~38% of the 80k GH200-GPU-h allocation if GH200≈H100 (UNVERIFIED rate). So the matched-compute comparison must run at 1/10–1/3 of that scale (ViT-S/B/L latent predictors, the generative arm fine-tuned from LongLive2.0-Robot or trained small) with the law extrapolated, not replicated at 5B. Closed-loop evaluation on RoboCasa GR-1 is the hidden cost again (MuJoCo on Grace cores, policy on GPU). Rough grid: 3 backbones × 4 sizes × 4 history lengths × 2 seeds ≈ 96 adaptation runs plus ~12 pretrains; unpriced until a 1-GPU throughput pilot. The 24 h walltime and 32-GPU cap apply as in MASTERPLAN.

**Skill/time.** Closest to the user's trajectory (LeWM/ESD-WORLD, V-JEPA). Heavy engineering: RoboCasa eval harness, two backbones, streaming history. Realistic for one person: 5–6 months to a submission, not the 5 weeks to CVPR.

**Minimal viable experiment (P1, no Isambard).** Pre-encode RoboCasa365 GR-1 demos for 5 tasks; train a 20M causal clean-JEPA predictor and a 20M bidirectional one on the latents at 3 history lengths (0, 2.4 s, 9.6 s); attach the same flow-matching action head; measure closed-loop success. If the causal arm shows no monotone history gain at this scale while Long-WAM's released checkpoints do, the headline is at risk (or is Outcome B). Needs one GPU (DGX Spark or a 1-GPU Isambard smoke); CPU-only VM can build the harness and data pipeline but not run it.

### Verdict
**PIVOT-ready GO.** Gate 1 yes (narrow, closing), gate 2 yes conditional on results 1–2, gate 3 yes at reduced scale with the pilot gate. verdict_reason: the question is open and sharper than plan v2's EO headline for this user's goals, but it is a race against the Long-WAM/RoboJEPA groups, and the compute makes it a laws-at-small-scale paper, not a frontier model.

## 3. How it sits beside plan v2

| | Plan v2 (clean JEPA on EO) | WAM bridge (clean JEPA WAM) |
|---|---|---|
| Serves | OpenChange-UK, UK AI-for-Science fit, u6xn wording (if EO) | stated WAM goal, AIMS/robotics trajectory |
| Data | 20 TB Copernicus-Pretrain, CC-BY | 10K h RoVid-X + RoboCasa365, CC-BY |
| Closed-loop truth | none (probes only) | simulator with reset (RoboCasa) |
| Race risk | low (narrow gap, slow field) | high (two groups active this month) |
| Compute | FLOPs cheap, bytes expensive | both moderate; evaluation dominates |
| Shared method | SIGReg stratified on availability | SIGReg stratified on history length |
| Shared guarantees | selection, version compatibility, substitution | same three |

Two honest options:
- **A. WAM leads, EO is the OpenChange-UK track.** Project B becomes this note; plan v2's Q1–Q3 run on EO only as far as OpenChange-UK needs. Fits the goal; needs the allocator's OK if u6xn names EO.
- **B. EO leads, WAM is paper two.** Safer on compute and permissions, slower on the goal, and the WAM gap may close meanwhile.

Not recommended: both at full scale in one paper (deep-think's pessimist costed one substrate at 18–25 weeks).

## 4. Immediate local steps if A
1. Verify licences for EgoVerse and VITRA; decide the corpus (RoVid-X + OXE + RoboCasa365 only is the clean choice).
2. Check LeVJEPA/LeJEPA reference code for a causal mask path; if absent, specify the causal clean-JEPA objective (SIGReg per time step vs pooled) in writing before coding.
3. Build the RoboCasa365 GR-1 eval harness with dry-run and the 500-sample pilot rule; measure sim steps/s on CPU.
4. Download nothing until the source gate passes; log bytes/sample for RoVid-X from the HF metadata only.
5. Re-cost BUDGET_DESIGN for this grid after a 1-GPU throughput number exists.
