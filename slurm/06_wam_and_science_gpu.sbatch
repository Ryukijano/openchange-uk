#!/bin/bash
#SBATCH --job-name=jepa-wam-science-4gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 4x GH200 GPU benchmark & MVE runner for:
#   1. Track A: EndToEndPixelWAM (Causal vs. Bidirectional MVE on LeRobotWindowDataset)
#   2. Track C: FactorizedSpatiotemporalJEPAWorldModel (Sci-JEPA-WM multi-sensor EO + continuum PDE
#      latent dynamics with 3-pathway physical forcing and GPU counterfactual VoI ranking).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${OPENCHANGE_EXPECTED_GIT_SHA:?Set the reviewed exact git commit hash}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${OPENCHANGE_EXPECTED_GIT_SHA}" ]] || [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing: expected commit mismatch or uncommitted/untracked changes."
  exit 2
fi
DEFAULT_SIF="/projects/u6xn/sif-images/pytorch_26.09-py3.sif"
if [[ ! -f "${DEFAULT_SIF}" ]]; then
  DEFAULT_SIF="${PROJECTDIR:-${HOME}}/sif-images/pytorch_25.05-py3.sif"
fi
SIF="${OPENCHANGE_SIF:-${DEFAULT_SIF}}"
if [[ ! -f "${SIF}" ]]; then
  echo "Container missing: stage the documented aarch64 NGC image before submission."
  exit 2
fi
mkdir -p logs outputs
export OPENCHANGE_GIT_SHA="${ACTUAL_SHA}" OPENCHANGE_GIT_DIRTY=0
export OPENCHANGE_SIF="${SIF}"
OPENCHANGE_SIF_SHA256="$(sha256sum "${SIF}" | cut -d' ' -f1)"
export OPENCHANGE_SIF_SHA256
echo "git_sha=${ACTUAL_SHA} seed=0"
echo "arch=$(uname -m) sif=${OPENCHANGE_SIF} sif_sha256=${OPENCHANGE_SIF_SHA256}"
echo "slurm_job_id=${SLURM_JOB_ID:-none} host=$(hostname) world_size=4"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

RUN=(apptainer exec --nv --bind "${PWD}:/work" --pwd /work --env PYTHONPATH=/work/src "${SIF}")

echo "=== 1. Track A (WAM) & Track C (Sci-JEPA-WM + Guarantees) GPU unit suites ==="
"${RUN[@]}" python3 -m unittest tests/test_jepa_wam.py tests/test_jepa_science_wm.py tests/test_jepa_guarantees.py -v
echo "done"
