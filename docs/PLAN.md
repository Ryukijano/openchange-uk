# OpenChange-UK plan

Status: provisional. The smoke job has not been run. Region names and the time cutoff are still unset.

## Decision

Build an open UK benchmark and transfer-learning toolkit for environmental change, starting with coastal shoreline and land-cover change. Do not grow NV-Disruptron into a larger product, and do not pretrain a foundation model.

The flagship is evaluation-first:

> OpenChange-UK: a strict-transfer benchmark for open Earth-observation models on environmental change detection.

V-JEPA 2.1 is one later backbone inside that benchmark. A positive or negative transfer result is both publishable. The benchmark is the artefact that should outlast either result.

## Claim

An open pretrained encoder, adapted with a small number of parameters, should beat a frozen encoder and a small from-scratch baseline when the test set is a held-out stretch of UK coast and a later time window.

Random image splits are not a result.

## What stays out

- Fine-tuning Nemotron into another chatbot.
- Spending the allocation on pretraining.
- A generic climate dashboard.
- Clinical, surgical, or robotic deployment.
- Treating Earth-2 as the change-detection backbone. Weather or hazard fields are an ablation only.
- Redistributing imagery before the licence is checked.

## What is reused from NV-Disruptron

NV-Disruptron was a London mobility hackathon stack, not a foundation-model paper. The reusable pattern is observation, a temporal persistence filter, a geospatial index, and an explanation tied to a public source.

| Old piece | Role here |
|---|---|
| LocateAnything-style grounding | Weak labels and visual checks. Not the benchmark label. |
| Temporal tracking | Separate persistent change from cloud, tide, shadow, and season. |
| RAPIDS / cuSpatial | Patch index, area statistics, regional splits. |
| Retrieval | Evidence cards over public documents. |
| Nemotron | Readable evidence cards after a model exists. |
| Cosmos Reason | Hypothesis text. Never the ground truth. |
| cuOpt | Out of the scientific core. |

Provenance stays in the paper. The project name is OpenChange-UK.

## v1 task

Given two public observations of the same coastal location, predict a change map, an optional change class (erosion, accretion, other land-cover transition), and an uncertainty score. Metadata covers resolution, cloud mask, tide or season if available, and the split id.

Flood extent is the second task, after the coastal harness works. Urban heat is a runner-up, and any map from it is a research estimate, not a warning.

## Backbones

Pilot, after a licence and an aarch64 check, two open encoders. Starting candidates: Prithvi and Clay. Confirm sizes and licences before the pilot; do not treat that pair as final.

Later, on the same splits:

- TESSERA (CVPR 2026; Sentinel-1/2 annual embeddings)
- SatMAE
- V-JEPA 2.1 dense features on image sequences

Earth-2 context is scored as the difference between vision-only and vision-plus-weather. A zero gain is a result.

## Evaluation

- Pixel IoU and F1
- Boundary F1
- Per-class mean IoU when classes exist
- Expected calibration error
- Correlation between uncertainty and error
- Variance across held-out regions
- Geographic transfer and temporal transfer, reported separately
- Leakage test: no train region in the test region list, and no test date before the cutoff

## Baseline ladder

1. Difference image plus a small U-Net.
2. Siamese U-Net.
3. Frozen encoder plus a linear decoder.
4. LoRA or adapters on that encoder.
5. A small temporal fusion module.
6. The same model with Earth-2 context.
7. An ablation that removes each piece that the report claims matters.

Loss for the first real fit, when training is allowed: cross-entropy, Dice, a boundary term, and a calibration term. Weights are chosen on the pilot, not in this document.

## Data

`configs/sources.yaml` is empty on purpose. The download job exits without fetching until a source is added with a licence note. Manifests and checksums are the released data product. Unclear licences are not redistributed.

## Compute

Award: AI Builder, BriCS project `u6xn`, Isambard-AI Phase 2, through 15 March 2027. Balance used for planning: 5000 NHR. One NHR is one node-hour, which is four GH200 GPU-hours. One GPU for one wall-hour is 0.25 NHR.

These are envelopes, not jobs to submit. They sum to the allocation:

| Phase | GPU-hours | NHR |
|---|---:|---:|
| Environment and smoke | 50 | 12.5 |
| Manifests and patch build | 400 | 100 |
| Frozen and small baselines | 2,000 | 500 |
| Adapters and temporal variants | 8,000 | 2,000 |
| Holdout, context, and uncertainty ablations | 5,000 | 1,250 |
| Repeat seeds and release checkpoints | 2,500 | 625 |
| Failures and revision | 2,050 | 512.5 |
| Total | 20,000 | 5,000 |

Order of runs: 10-minute 1-GPU smoke, then a dry-run, then a 1-epoch 1-GPU fit, then scale. No multi-node job first. Agent-written scripts keep `--time` at or under 30 minutes until a longer limit is requested in writing.

Smoke cost if the template is used as written: 10 minutes times 1 GPU is about 0.042 NHR. The 4-GPU dry-run template is 30 minutes times 4 GPUs = 0.5 NHR. Do not submit the 4-GPU file in week 1.

## Sequence

1. Pull `nvcr.io/nvidia/pytorch:25.05-py3` as an aarch64 SIF and run `slurm/00_smoke.sbatch`.
2. Commit the log, the SIF identity, and the git SHA.
3. Freeze the one-page spec: label definition, regions, cutoff, metrics.
4. Hand-check 100 to 500 pairs. Fit nothing larger than the dry-run.
5. Two backbones only.
6. Open the split design for comment.
7. Add V-JEPA 2.1 after that harness runs.
8. Publish weights, manifests, cards, and the compute account only after a baseline number exists.

## How the agent is allowed to help

Cursor stays on this machine. It may edit this repo. It may not `ssh`, `sbatch`, `srun`, or `scancel`. Login-node agents, tmux, and screen are out. Heavy work is a Slurm job the human submits.

## Release set

- Data builder and manifests
- Fixed splits, baselines, metrics, leakage tests
- Adapter weights, config, and inference
- A map with the pair, the mask, the uncertainty, and the source
- A report with cards, methods, limitations, and the NHR account

Acknowledgement for papers: Isambard-AI National AI Research Resource (AIRR), DSIT via UKRI/STFC.