#!/bin/bash
#SBATCH --job-name=airr-sources
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Source manifest validation only. It never fetches. The same check runs locally
# and in CI for free (make validate-sources), so this job is optional.
# One GPU is requested because Phase 2 compute is allocated as GH200s; verify a
# CPU-only path in the docs before removing --gpus.
# 10 min x 1 GPU is about 0.042 NHR. Do not raise --time past 00:30:00 without
# an explicit decision.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs outputs

CONFIG=configs/sources.yaml
SIF_NAME=pytorch_25.05-py3.sif
if [[ -n "${OPENCHANGE_SIF:-}" ]]; then
  SIF="${OPENCHANGE_SIF}"
elif [[ -n "${PROJECTDIR:-}" ]]; then
  SIF="${PROJECTDIR}/sif-images/${SIF_NAME}"
else
  SIF="${HOME}/sif-images/${SIF_NAME}"
fi

OPENCHANGE_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
if [[ "${OPENCHANGE_GIT_SHA}" == unavailable ]]; then
  OPENCHANGE_GIT_DIRTY=unknown
elif git diff --quiet HEAD --; then
  OPENCHANGE_GIT_DIRTY=0
else
  OPENCHANGE_GIT_DIRTY=1
fi
OPENCHANGE_SIF_SHA256=not-recorded
if [[ -f "${SIF}.sha256" ]]; then
  OPENCHANGE_SIF_SHA256="$(cut -d' ' -f1 "${SIF}.sha256")"
fi
export OPENCHANGE_SIF="${SIF}" OPENCHANGE_GIT_SHA OPENCHANGE_GIT_DIRTY OPENCHANGE_SIF_SHA256
export OPENCHANGE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3

echo "=== PROVENANCE ==="
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo "git_sha=${OPENCHANGE_GIT_SHA}"
echo "git_dirty=${OPENCHANGE_GIT_DIRTY}"
echo "config=${CONFIG}"
echo "seed=$(awk '/^seed:/ {print $2}' "${CONFIG}" 2>/dev/null || echo none)"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "sif=${SIF}"
echo "sif_sha256=${OPENCHANGE_SIF_SHA256}"
echo "image=${OPENCHANGE_IMAGE}"
echo "=== SLURM/GPU ENV (filtered) ==="
env | grep -E '^(SLURM_JOB_|SLURM_GPUS|SLURM_NNODES|SLURM_NODELIST|CUDA_VISIBLE_DEVICES)' \
  | grep -Ev '(TOKEN|SECRET|PASSWORD|KEY)' | sort || true

if [[ ! -f "${SIF}" ]]; then
  echo "Missing container: ${SIF}"
  echo "Pull it first (see README.md) or set OPENCHANGE_SIF."
  exit 2
fi
echo "apptainer=$(apptainer --version)"

apptainer exec --nv --bind "${PWD}:/work" --pwd /work --env PYTHONPATH=/work/src "${SIF}" \
  python3 scripts/validate_sources.py --config "${CONFIG}"
