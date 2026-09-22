#!/bin/bash
#SBATCH --job-name=airr-dry-4gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Dry-run scaffold for a full node. Do not submit during week 1.
# 30 min x 4 GPUs = 0.5 NHR. Do not raise --time without an explicit decision.
# Do not start multi-node from this file.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "=== PROVENANCE ==="
hostname
uname -m
git rev-parse HEAD 2>/dev/null || echo "NO_GIT_SHA"
echo "config=configs/baseline.yaml"
echo "seed=0"
env | sort

SIF="${OPENCHANGE_SIF:-${HOME}/sif-images/pytorch_25.05-py3.sif}"
if [[ ! -f "${SIF}" ]]; then
  echo "Missing container: ${SIF}"
  exit 2
fi

apptainer exec --nv --bind "${PWD}:/work" "${SIF}" \
  python3 /work/scripts/train.py --config /work/configs/baseline.yaml --dry-run