# Compute, storage and dataset plan — 21 October 2026

Status: **PLANNING DOCUMENT.** Nothing here approves a data source, a split, a
job longer than 30 minutes, or a result claim. Every dataset still needs an
approved entry in `configs/sources.yaml` before any fetch. Sizes and costs in the
"proposed" sections are estimates unless marked *measured*.

Goal this plan serves: a ready-to-use public Hugging Face demo for UK earth
observation and world/action-model work. Research and training are chosen to get
there, not for architectural novelty on its own.

## 1. What we actually have (measured on Isambard, 21 Oct 2026)

### Storage

| Location | Limit | Used | Whose |
|---|---:|---:|---|
| `/projects/u6xn` (`$PROJECTDIR`, Lustre project ID `1483808123`) | **200 TB**, 51.2M files | 22.73 GB | **Shared by the `u6xn` project**, not reserved to one user. We can write there as project members. |
| `/scratch/u6xn/ryukijano.u6xn` (`$SCRATCHDIR`, project ID `813702`) | **5 TB**, 1.024M files | 60 KB | Personal scratch |
| `/scratch/u6xn` parent | 400 TB shared pool | 2.97 TB | Shared scratch; our personal path is still capped at 5 TB |
| `$HOME` `/home/u6xn/ryukijano.u6xn` | 101 GB | 1.74 GB | Personal; not for datasets |

Current project contents: `data/pilot_v1` (~1.8 GB, 480 SSL4EO-S12 pairs) and
`sif-images` (~21 GB). **Nothing on Isambard is backed up.** Anything we must keep
needs a copy elsewhere (for example Hugging Face Hub for manifests, adapters and
release checkpoints).

Measured with `lfs quota -h -p <project-id> /lus/lfs1aip2` and `quota`.

### Compute

- Planning balance: **5,000 NHR** (AI Builder award, through 15 March 2027 —
  [PLAN.md](../PLAN.md)). 1 NHR = one node-hour = 4 GH200 GPU-hours, so this is
  **20,000 GH200 GPU-hours**. Some October drafts (`budget-design.md`,
  `airr-programme/masterplan.md`) wrongly treated 20,000 as NHR.
- The live remaining balance is **not visible through `sacctmgr`**: our
  association has no `GrpTRESMins` set. Check the portal
  (https://portal.isambard.ac.uk) for the authoritative number.
- QOS limits seen from `sacctmgr show qos` (verify in docs before relying on them):
  `workq_qos` max wall 1 day, `interactive_qos` 8 h, `restricted48` 2 days.
  Our association QOS shows as `normal`. No per-project concurrent-GPU cap was
  visible; verify in docs.
- The repo rule is still **1 node, ≤4 GPUs, ≤30 minutes per job** unless the
  human explicitly approves more.

### What a node gives us (from job `7527080`)

50 clean-JEPA steps on 480 real pairs, 4×GH200, bf16, global batch 32: 95 s
wall including startup, ~30k tokens/s, **~471 MB peak memory per GPU out of
~96 GB**, loader wait ~0.009 s. The pilot is tiny relative to the hardware;
larger batches and ViT-S/B/L backbones fit on one node.

### Gate-1 result against its kill rule (job `7560286`)

Kill rule: `defensive_split_cfm` must beat `clean_jepa` on shared-subspace
Procrustes residual before we spend on a large SSL4EO grid.

- Measured: `defensive_split_cfm` 0.0372 vs `clean_jepa` 0.0514 — **passes that
  half of the rule** on a 50-step, 160-location screen
  ([report](../../reports/jepa-gate1/2026-10-20/SUMMARY.md)).
- Not measured: whether the private-subspace SIGReg conflict is removed.
- This is a bounded objective screen, not a downstream benchmark.

## 2. Storage layout

| Tier | Path | Use | Rules |
|---|---|---|---|
| Persistent corpus | `/projects/u6xn/datasets/`, `/projects/u6xn/latents/`, `/projects/u6xn/releases/`, `/projects/u6xn/sif-images/` | Immutable shards + SHA-256 manifests, pre-encoded latents, release checkpoints, containers | Read-mostly in training. Only sources that pass `make validate-sources`. Mirror manifests and release checkpoints off-cluster. |
| Job scratch | `$SCRATCHDIR/jobs/${SLURM_JOB_ID}/` | Tar extraction, conversion, rolling checkpoints, COLMAP workspaces | Keep latest 2 rolling checkpoints; promote milestones to `releases/` with `.sha256`; clean job scratch at exit. |
| Node-local | `$LOCALDIR` (verify size in docs) | Hot prefetch, SQLite/COLMAP DBs, simulator state | Wiped when the job ends. |

Because `/projects/u6xn` is shared with other project members, we keep our
footprint to roughly **≤75 TB** and leave the rest free.

## 3. Datasets to focus on

Only the SSL4EO-S12 numbers below are measured (HF API listing at revision
`97a91733`). Everything else is an estimate to check during source approval.

### EO track (main path to the UK demo)

| Priority | Dataset | Raw size | Why |
|---|---|---:|---|
| E1 | SSL4EO-S12 v1.1 S1GRD + S2L2A, train+val | **1.09 TB (measured)** | Paired SAR/optical at scale for the existing trainer; direct continuation of the pilot. |
| E2 | SSL4EO-S12 v1.1, all 7 modalities (adds S2L1C, S2RGB, LULC, DEM, NDVI) | **2.26 TB (measured)** | Adds terrain, land cover and NDVI targets for probes. |
| E3 | Cloud / time-series sets (SEN12MS-CR-TS, GreenEarthNet) + ERA5 forcing | ~2.5 TB (estimate) | Forecast-through-cloud and weather-conditioned vegetation tasks. |
| E4 | UK held-out evaluation data (Sentinel-1/2 over UK regions, UK flood/coastal labels) | TBD | Needed for any UK claim; NASA/global data stays auxiliary. |

NISAR products are the strongest new auxiliary SAR source but are provisional;
treat as later auxiliary data, not a replacement for Sentinel-1.

### World-action-model track

| Priority | Dataset | Raw size | Why |
|---|---|---:|---|
| A0 | `robocasa365-target-atomic` (+ Push-T) | ~6–9 GB (estimate) | Small first pilot for the repaired WAM sampler; one memory-dependent task. |
| A1 | Full RoboCasa365, BridgeData V2 | ~0.9 TB (estimate) | Scale-up if A0 shows positive history gain. |
| A2 | DROID, Open X-Embodiment / RoVid-X subsets | ~2–11 TB (estimate) | Only after A0/A1 gates; licences must be checked per subset. |

### Reconstruction track (Agentic-SfM, separate demo)

| Priority | Dataset | Raw size | Why |
|---|---|---:|---|
| B1 | MegaDepth + ScanNet (frozen 30-scene eval) + ScanNet++ | ~2.8 TB (estimate) | Hard-pair corpus for router training. ScanNet-family licences are non-commercial/academic — check before use. |
| B2 | DL3DV-10K subset, Doppelgangers++, LaMAR | ~1.5–3 TB (estimate) | Symmetry and outdoor failure cases. |
| B3 | NASA astronaut photography + Copernicus GLO-30 DEM | ~0.8 TB (estimate) | High-tilt geo-registration failures. |

## 4. Moonshot ideas (proposals, not commitments)

The useful advantage here is not training a giant pixel video generator. It is
200 TB of storage plus Grace CPUs next to each GPU: store pre-encoded latents
once, then run many cheap model variants and closed-loop evaluations.

1. **Latent WAM data lake + compute/history scaling study.** Encode robot video
   once with frozen backbones into `/projects/u6xn/latents/`, then train many
   small causal latent WAMs across model size, data hours, history length and
   planning compute. Question: when does history actually improve closed-loop
   control per unit compute? Rough cost: ~2,200 NHR, ~38 TB. Must beat
   DINO-WM / Long-WAM-style baselines on control, not latent MSE.
2. **All-weather EO model on the full SSL4EO-S12 archive.** Split-subspace
   clean-JEPA + flow bridge (S1→S2) vs TESSERA-style and VICReg controls, then
   UK forecasting/change tasks under cloud. Rough cost: ~900 NHR, ~12 TB. The
   current observation-value function must be redesigned before any
   "which pass to fetch" claim.
3. **Agentic-SfM router at scale.** Router over LoFTR / MASt3R / VGGT /
   MapAnything trained on ~100k–250k verified hard pairs, measured against the
   best fixed specialist at equal budget. Rough cost: ~1,250 NHR, ~18 TB. The
   GRPO off-policy token-rewrite issue must be fixed first.
4. **GPU imagination + CPU verification ("speculative planning").** GPU rolls
   out many candidate action sequences in latent space; Grace CPU cores verify
   the top-k in an exact simulator before acting. Question: does this beat pure
   neural planning and pure simulator MPC in decision quality per second?
   Rough cost: ~400 NHR, ~5 TB.

These four together exceed what we should commit up front. Pick one primary
(recommended: 2 for the UK demo, with 1 as the WAM research study) and gate the
rest.

## 5. Proposed `/projects/u6xn` footprint

| Directory | Planned | Contents |
|---|---:|---|
| `sif-images/` | ~0.5 TB | NGC PyTorch 26.09 / 25.05 images + `.sha256` |
| `datasets/eo/` | ~12 TB | `pilot_v1`, full SSL4EO-S12, cloud/time-series + ERA5 cube |
| `datasets/wam/` | ~14 TB | RoboCasa, BridgeData V2, DROID, OXE/RoVid-X subsets |
| `latents/` | ~28 TB | Pre-encoded bf16 latents |
| `datasets/sfm/` | ~15 TB | SfM scenes + cached correspondences/pointmaps |
| `releases/` | ~3.5 TB | Release checkpoints, calibration tables, provenance |
| **Total** | **~73 TB** | ~37% of the shared 200 TB |

## 6. Compute plan (5,000 NHR)

- **Phase 0 — 30-minute gates (~1.25 NHR).** CUDA checks (`05`), Gate-1 screen
  (`07`, done: job `7560286`), WAM/science GPU check (`06`). Note `06` currently
  reserves GPUs but runs CPU unit tests; it is not a training run.
- **Phase 1 — staging and encoding (~100 NHR).** Stage approved shards into
  `/projects/u6xn/datasets/` via transfer jobs (not login-node copies), using
  scratch as the unpack buffer; pre-encode latents.
- **Phase 2 — main grids.** Needs explicit approval to raise the 30-minute cap.
  Run resumable 1-node slices chained with `--dependency=afterok`; many
  independent 1-node jobs rather than multi-node jobs.

| Track | Under 5,000 NHR |
|---|---:|
| EO all-weather model (Moonshot 2) | ~900 NHR |
| WAM study (Moonshot 1) | ~2,250 NHR |
| Agentic-SfM router (Moonshot 3) | ~1,250 NHR |
| Phase 0/1 + reserve | ~600 NHR |

The split above is an upper bound per track. Release reserve only to tracks
that pass their gates.

## 7. Kill criteria

1. **EO:** no large SSL4EO grid until the defensive arm also shows the SIGReg
   conflict is removed and beats clean-JEPA on a downstream UK probe.
2. **WAM:** on A0, the repaired sampler must show non-collapsed action
   sensitivity and a positive closed-loop history gain at identical prediction
   origins before encoding large video corpora.
3. **SfM:** after SFT warm-start, no single specialist may take almost all
   selections; the router must beat the best fixed specialist at equal budget.
4. **Scratch hygiene:** at most 2 rolling checkpoints per run in scratch.

## 8. Next steps

1. Choose the first corpus to approve (recommended: E1, SSL4EO-S12
   S1GRD + S2L2A train+val, 1.09 TB) and add exact pinned entries to
   `configs/sources.yaml`.
2. Write a Slurm transfer job that stages it into `/projects/u6xn/datasets/eo/`
   and builds checksummed manifests and a geographic split.
3. Confirm the live NHR balance on the portal.
4. Ask the PI whether the `u6xn` award scope covers robotics corpora as well as EO.
