#!/bin/bash
#SBATCH --job-name=airr-pull-sif
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# OPENCHANGE_CONTAINER_PULL
# Optional. The docs pull NGC images with apptainer on the login node; use this
# job only if a login-node pull is too slow or gets killed. It pulls the container,
# not data. 30 min x 1 GPU = 0.125 NHR at most.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

IMAGE=docker://nvcr.io/nvidia/pytorch:25.05-py3
SIF="${OPENCHANGE_SIF:-${PROJECTDIR:?PROJECTDIR is not set}/sif-images/pytorch_25.05-py3.sif}"
mkdir -p "$(dirname "${SIF}")"

echo "=== PROVENANCE ==="
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "config=containers/apptainer.def"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "image=${IMAGE}"
echo "sif=${SIF}"
echo "apptainer=$(apptainer --version)"

if [[ -f "${SIF}" ]]; then
  echo "Container already present"
else
  apptainer pull "${SIF}" "${IMAGE}"
fi
ls -lh "${SIF}"
if [[ ! -f "${SIF}.sha256" ]]; then
  sha256sum "${SIF}" > "${SIF}.sha256"
fi
cat "${SIF}.sha256"
