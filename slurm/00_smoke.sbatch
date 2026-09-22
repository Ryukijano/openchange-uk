#!/bin/bash
#SBATCH --job-name=airr-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs outputs

SIF="${OPENCHANGE_SIF:-${HOME}/sif-images/pytorch_25.05-py3.sif}"

echo "=== PROVENANCE ==="
date -Is || date
hostname
uname -m
git rev-parse HEAD 2>/dev/null || echo "NO_GIT_SHA"
echo "config=configs/smoke.yaml"
echo "seed=0"
echo "sif=${SIF}"
echo "=== ENV ==="
env | sort

if [[ ! -f "${SIF}" ]]; then
  echo "Missing container: ${SIF}"
  echo "On the login node, before sbatch:"
  echo "  mkdir -p \"\$HOME/sif-images\""
  echo "  apptainer pull \"\$HOME/sif-images/pytorch_25.05-py3.sif\" docker://nvcr.io/nvidia/pytorch:25.05-py3"
  exit 2
fi

echo "=== GPU ==="
nvidia-smi

apptainer exec --nv --bind "${PWD}:/work" "${SIF}" \
  python3 /work/scripts/smoke.py \
  --config /work/configs/smoke.yaml \
  --report /work/outputs/smoke.json