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

if [[ -n "${OPENCHANGE_SIF:-}" ]]; then
  SIF="${OPENCHANGE_SIF}"
elif [[ -n "${PROJECTDIR:-}" && -f "${PROJECTDIR}/sif-images/pytorch_25.05-py3.sif" ]]; then
  SIF="${PROJECTDIR}/sif-images/pytorch_25.05-py3.sif"
else
  SIF="${HOME}/sif-images/pytorch_25.05-py3.sif"
fi

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
  echo "Submit slurm/00_pull_sif.sbatch first."
  echo "Expected image: ${PROJECTDIR}/sif-images/pytorch_25.05-py3.sif"
  exit 2
fi

echo "=== GPU ==="
nvidia-smi

apptainer exec --nv --bind "${PWD}:/work" "${SIF}" \
  python3 /work/scripts/smoke.py \
  --config /work/configs/smoke.yaml \
  --report /work/outputs/smoke.json