#!/bin/bash
#SBATCH --job-name=airr-pull-sif
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs "${PROJECTDIR}/sif-images"

SIF="${PROJECTDIR}/sif-images/pytorch_25.05-py3.sif"

echo "=== PROVENANCE ==="
date -Is || date
hostname
uname -m
echo "git_sha=$(git rev-parse HEAD)"
echo "config=containers/apptainer.def"
echo "seed=0"
echo "sif=${SIF}"
env | sort

if [[ -f "${SIF}" ]]; then
  echo "Container already present"
  ls -lh "${SIF}"
  exit 0
fi

apptainer pull "${SIF}" docker://nvcr.io/nvidia/pytorch:25.05-py3
ls -lh "${SIF}"