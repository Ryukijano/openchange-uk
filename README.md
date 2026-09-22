# OpenChange-UK

Open benchmark for geographic and temporal transfer in UK environmental change detection. The first task is coastal shoreline and land-cover change. Flood mapping, urban heat, and V-JEPA 2.1 are later experiments inside the same harness, not separate projects and not a pretraining run.

This repository does not submit jobs. Review commands against [docs.isambard.ac.uk](https://docs.isambard.ac.uk) and run them yourself.

## Why this shape

A 2026 audit of geospatial foundation models found 152 papers evaluated on 401 benchmarks, with little shared protocol ([arXiv:2605.12678](https://arxiv.org/abs/2605.12678)). OpenChange-UK is a small, strict-holdout benchmark, not another maximal model wrapper.

V-JEPA 2.1 has public code and checkpoints ([arXiv:2603.14482](https://arxiv.org/abs/2603.14482), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)). It enters only after two open Earth-observation encoders run on a hand-checked pilot. Tessera is a Cambridge Sentinel-1/2 model announced in a CVPR 2026 paper; the model itself launched in 2025 ([ESA](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Tessera_AI_model_offers_accessible_way_to_view_Earth)). TESSERA v2 is a separate preprint ([arXiv:2607.03949](https://arxiv.org/abs/2607.03949)).

## Smoke before any training

Isambard-AI Phase 2 is Arm64 GH200. BriCS does not install PyTorch for you. The documented GPU path is the NGC image `nvcr.io/nvidia/pytorch:25.05-py3` with Apptainer `--nv` ([ML packages](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/)).

On the login node, after `clifton auth`:

```bash
mkdir -p "$HOME/sif-images"
apptainer pull "$HOME/sif-images/pytorch_25.05-py3.sif" docker://nvcr.io/nvidia/pytorch:25.05-py3
mkdir -p logs
sbatch slurm/00_smoke.sbatch
squeue --me
```

Ten minutes on one GPU is about 0.042 NHR. One GPU is one GH200. A full node is four GPUs and one NHR per wall-hour. Do not train until that log shows `aarch64`, a GH200, and a bfloat16 matmul, and the container file plus the git commit are recorded.

`slurm/02_train_1gpu.sbatch` and `slurm/03_train_4gpu.sbatch` are dry-run scaffolds capped at 30 minutes. They refuse to fit a model. Raising `--time` needs an explicit decision. `03` requests four GPUs, so 30 minutes is 0.5 NHR.

## Local checks

```bash
python -m pip install -e ".[dev]" PyYAML
make test
make dry-run
make evaluate
```

`make smoke` only prints the `sbatch` line.

## Layout

See `reports/SPEC.md` for the task, splits, metrics, and the stop rules.

The working plan is [docs/PLAN.md](docs/PLAN.md). Papers and BibTeX are in [papers/](papers/README.md).
