#!/bin/bash
#SBATCH --job-name=airr-sources
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Manifest check only. This job must not become a training download.
# One GPU is requested because Phase 2 compute is allocated as GH200s.
# Verify a non-GPU path in the docs before removing --gpus.
# 10 min x 1 GPU is about 0.042 NHR. Do not raise --time past 00:30:00
# without an explicit decision.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "=== PROVENANCE ==="
hostname
uname -m
git rev-parse HEAD 2>/dev/null || echo "NO_GIT_SHA"
echo "config=configs/sources.yaml"

SIF="${OPENCHANGE_SIF:-${HOME}/sif-images/pytorch_25.05-py3.sif}"
if [[ ! -f "${SIF}" ]]; then
  echo "Missing container: ${SIF}"
  exit 2
fi

apptainer exec --nv --bind "${PWD}:/work" "${SIF}" \
  python3 /work/scripts/validate_sources.py