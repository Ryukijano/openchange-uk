# Science testbeds we could train scaling studies on (2026-10-07)

Companion to VIDEO_WORLD_MODELS.md. Same rules: links given, "unverified" where I didn't read the primary number, nothing run.

## 1. What the UK funds
AIRR's AI for Science calls (https://www.ukri.org/opportunity/airr-compute-opportunity-ai-for-science/) list: material science, nuclear fusion, medical research, engineering biology, quantum technologies, AI-driven discovery. The AI for Science Strategy (https://www.gov.uk/government/publications/ai-for-science-strategy) sets missions in the same areas. Isambard's own 2026 user stories (https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2026/stories-from-a-sovereign-ai-supercomputer-.html) span protein structure, time-series foundation models, dementia, farming, AI safety. Weather/climate is funded but under NERC/Met Office rather than the AI for Science list.

## 2. Where scaling laws already exist (you'd be measured against these)

| Domain | Done | Open |
|---|---|---|
| Weather | "Scaling Laws of Global Weather Models" (ICML 2026, https://arxiv.org/abs/2602.22962): data beats params at fixed compute, wide beats deep. Continual-training IsoFLOPs (https://arxiv.org/abs/2603.25687). Per-channel/horizon analysis shows pooled scaling hides late-lead degradation (https://arxiv.org/abs/2604.05068) | Crowded by Hoefler's group and ECMWF; not a gap you can own with 80k GPU-h |
| PDE surrogates | Poseidon (https://arxiv.org/abs/2405.19101), MPP, DPOT, Walrus (Polymathic, 19 scenarios, MIT, Nov 2025, https://github.com/PolymathicAI/walrus) show pretraining helps. A 60k-measurement benchmark (https://arxiv.org/abs/2605.29283) finds physics FMs are "conditional, not universal generalists": generality depends on regime, scale, IC, model size | No compute-optimal law that jointly fits in-distribution loss, OOD loss (Reynolds/parameter shift) and a decision metric. OOD is bad: FNO hits 47% rel. L2 under a 10x Re shift, barely better than retrieval baselines (https://arxiv.org/abs/2605.30112) |
| Materials / MLIPs | Power laws for Equiformer/transformer property prediction (https://arxiv.org/abs/2509.21811); UMA family (NeurIPS 2025, https://arxiv.org/abs/2506.23971); compositional-generalisation benchmark (https://arxiv.org/abs/2605.08988) | Loss-vs-decision (does lower force MAE mean better discovery hit-rate on Matbench Discovery?) partly covered by Matbench's own metrics |
| Power grids | ACOPF surrogates: MSE and physics-aware losses both follow power laws but at different rates; constraint violation scales differently (https://arxiv.org/abs/2609.16282) | Closest existing paper to "loss vs decision quality" in science |
| Protein LMs | Compute-optimal pLMs (NeurIPS 2024), encoder-decoder laws (ICLR 2026); but PLMs "scale poorly", mid-size often beats largest (https://arxiv.org/abs/2603.07710) | Crowded; not your area |
| QEC decoders | Data scale matters more than architecture up to d=9 (https://arxiv.org/abs/2605.12046); CNN decoders reach 1e-10 logical error on Gross code (https://arxiv.org/abs/2604.08358); "foundation decoders" (https://arxiv.org/abs/2606.27119) | Crowded, as the pessimist said. Stim/PyMatching need aarch64 source builds |
| Fusion | FAIR-MAST open diagnostic data (UKAEA, S3 Zarr, https://www.ukaea.org/service/fair-mast/); TokaMark benchmark on HF (Feb 2026, https://huggingface.co/datasets/UKAEA-IBM-STFC/tokamark-dataset) | No scaling study I found. Data is sparse, noisy experimental diagnostics, not unlimited simulation |
| Fluid control | HydroGym (Nature, Aug 2026, https://www.nature.com/articles/s41586-026-10917-6): 60+ open flow-control environments, Re up to 4e5, MIT licence | Nobody has trained world models on it or measured how model scale maps to control performance |
| Downstream laws in general | "Scaling Laws Are Unreliable for Downstream Tasks" (https://arxiv.org/abs/2507.00885) | This is the fight your project is in |

## 3. Open data that is licence-clean and big enough
- **The Well** (Polymathic): 16 datasets, 15 TB, uniform grids, HDF5, Hugging Face hosting; code BSD-3; data licence listed per dataset on HF (check each, I didn't confirm all). Best single source for a multi-physics study. https://polymathic-ai.org/the_well/
- **ERA5** via Copernicus licence rev. 12: free, worldwide, any lawful purpose including adaptation; Anemoi training-ready ERA5 is open. https://ecds.ecmwf.int/licences/licence-to-use-copernicus-products
- **OMat24 / OMol25 / OC20** (Meta FAIR): large DFT datasets; licences vary (CC-BY 4.0 for OMat24 per earlier knowledge; unverified here).
- **Matbench Discovery** (MIT code, WBM test set): discovery hit-rate as a decision metric.
- **HydroGym**: environments, not a dataset; you generate trajectories. Solver cost per step is the limit.
- **FAIR-MAST / TokaMark**: open, UK, fusion-aligned, but experimental and small.
- **CAMELS** (cosmology): public, thousands of simulations, but far from your trajectory.

## 4. Candidate science studies

### S1. Physics world models on The Well, scored on OOD and on a decision (recommended science core)
- Question: for autoregressive surrogates trained on The Well, how do in-distribution loss, loss under parameter shift (Re, Mach, Rayleigh, IC complexity) and a downstream decision metric scale with params, data (number of physical scenarios and trajectories) and compute? Do they agree?
- Decision metric options with exact ground truth: (a) control in HydroGym using the surrogate as the planner's model (world-model MPC; same machinery as the video track); (b) design/inverse problems (pick the parameter that hits a target field) scored by the true solver.
- Prior: Walrus/Poseidon give pretraining recipes; 2605.29283 gives the OOD protocol; nobody fitted joint laws or added a planning metric. Related in spirit to ACOPF 2609.16282.
- Compute: 2D datasets fit 1-GPU jobs across the grid; 3D (turbulence, MHD) justifies multi-node for the top runs. Rough budget 20-35k GPU-h including seeds and tuning. Data is already generated, which removes the simulator bottleneck we worried about with SUMO.
- Risks: HydroGym solvers may be slow for closed-loop evaluation at high Re; need a CPU-only headroom test (true-solver MPC vs the paper's PPO/no-control baselines) exactly as planned before. The Well loaders at 15 TB need Isambard scratch planning.
- UK fit: fusion/engineering only indirectly (MHD and fluids), unless you pick the MHD datasets and say so.

### S2. Fluid control world models (HydroGym only)
- Narrower version of S1: generate trajectories in HydroGym, train world models of increasing size, plan inside them, measure control performance vs Re shift and vs model scale. Clean "latent vs generative" and "train vs plan compute" axes reused from the video track. Smaller data, more simulator cost.

### S3. Materials: force-field accuracy vs discovery decisions
- Train MLIPs across sizes on OMat24, measure energy/force loss, OOD loss on unseen chemistries (compositional benchmark 2605.08988), and discovery hit-rate on Matbench Discovery. Strong UK fit (materials is first on the AIRR list). But Meta FAIR, Microsoft and Orbital already run this at scales we cannot match, and the equivariant architectures are a long way from your world-model stack.

### S4. QEC decoders
- Keep as optional replication (unchanged from earlier reports). Crowded; aarch64 source builds.

### S5. Fusion plasma dynamics on FAIR-MAST / TokaMark
- Highest UK alignment, lowest data volume, no simulator ground truth for a decision metric. Good for a small transfer/evaluation paper, poor for a scaling law.

## 5. How this fits with the video track
S1/S2 and the video track (VIDEO_WORLD_MODELS.md, option A+C) are the same experiment on two substrates: a predictive model, a frozen planner, in-distribution loss, shifted loss, decision quality, and the train-vs-plan compute frontier. One codebase, two data domains. That is the strongest framing I can see: "does the loss-to-decision relation hold across scale, and does it depend on the domain?" If the budget allows only one, the physics substrate has the cleaner shift axis and exact ground truth; the video substrate has the better link to your trajectory and to robotics.

## 6. Week-1 checks (local, no cluster)
1. The Well: read each dataset's licence on HF; stream one 2D dataset (e.g. turbulent_radiative_layer_2D, 6.9 GB tier) and time the loader.
2. HydroGym: install, run the cylinder and cavity environments on CPU; time one step; true-solver MPC vs no-control vs PPO baseline on 20 seeds for headroom.
3. Train a 5M and 50M-parameter surrogate for 1 GPU-hour each on one Well dataset to size the grid (MFU, loss curve, memory).
4. Decide whether the decision metric is control (HydroGym) or inverse design (true solver).
