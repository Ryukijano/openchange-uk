# OpenChange-UK

Open benchmark for geographic and temporal transfer in UK environmental change detection. The first task is coastal shoreline and land-cover change. Flood mapping, urban heat, and V-JEPA 2.1 are later experiments inside the same harness, not separate projects and not a pretraining run.

This repository does not submit jobs. Review commands against [docs.isambard.ac.uk](https://docs.isambard.ac.uk) and run them yourself.

## Why this shape

A 2026 audit of geospatial foundation models found 152 papers evaluated on 401 benchmarks, with little shared protocol ([arXiv:2605.12678](https://arxiv.org/abs/2605.12678)). OpenChange-UK is a small, strict-holdout benchmark, not another maximal model wrapper.

V-JEPA 2.1 has public code and checkpoints ([arXiv:2603.14482](https://arxiv.org/abs/2603.14482), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)). It enters only after two open Earth-observation encoders run on a hand-checked pilot. Tessera is a Cambridge Sentinel-1/2 model announced in a CVPR 2026 paper; the model itself launched in 2025 ([ESA](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Tessera_AI_model_offers_accessible_way_to_view_Earth)). TESSERA v2 is a separate preprint ([arXiv:2607.03949](https://arxiv.org/abs/2607.03949)).

## Smoke before any training

Isambard-AI Phase 2 is Arm64 GH200. BriCS does not install PyTorch for you. The documented GPU path is the NGC image `nvcr.io/nvidia/pytorch:25.05-py3` with Apptainer `--nv` ([ML packages](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/)).

On the login node, after `clifton auth` and `ssh u6xn.aip2.isambard` (full Linux steps in [docs/WORKFLOW.md](docs/WORKFLOW.md)):

```bash
mkdir -p "$PROJECTDIR/sif-images"
apptainer pull "$PROJECTDIR/sif-images/pytorch_25.05-py3.sif" docker://nvcr.io/nvidia/pytorch:25.05-py3
sbatch slurm/00_smoke.sbatch
squeue --me
```

`$PROJECTDIR` is the documented place for shared container images ([storage](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)). Set `OPENCHANGE_SIF` to use another path. The job writes `logs/airr-smoke-<jobid>.out` and `outputs/smoke-slurm<jobid>/{manifest.json,smoke.json}`.

Ten minutes on one GPU is about 0.042 NHR. One GPU is one GH200. A full node is four GPUs and one NHR per wall-hour. Do not train until that log shows `aarch64`, a GH200, and a bfloat16 matmul, and the container file plus the git commit are recorded.

`slurm/02_train_1gpu.sbatch` and `slurm/03_train_4gpu.sbatch` are dry-run scaffolds capped at 30 minutes. They refuse to fit a model. Raising `--time` needs an explicit decision. `03` requests four GPUs, so 30 minutes is 0.5 NHR.

## Local checks

```bash
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
make check
```

`make check` runs lint, unit tests, the Slurm template checker, the source-approval check, a dry-run, and the evaluation stub. It needs no GPU, no network, and no Isambard login. `make smoke` only prints the `sbatch` line.

## Data sources

Nothing is fetched until an entry in `configs/sources.yaml` passes `scripts/validate_sources.py`. Each entry needs a URL, licence and licence URL, attribution, expected size in bytes, use, intended splits, redistribution terms, and who approved it and when. A future fetcher must call `openchange.sources.require_approved(source_id, split)` first.

## Layout

See `reports/SPEC.md` for the task, splits, metrics, and the stop rules.

The working plan is [docs/PLAN.md](docs/PLAN.md). The cluster loop is [docs/WORKFLOW.md](docs/WORKFLOW.md). Papers and BibTeX are in [papers/](papers/README.md).
