#!/bin/bash
#SBATCH --job-name=jepa-cuda-checks
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# GH200 checks the CPU suite skips: the JEPA unittest suite with real GPUs
# (CUDA SyncBatchNorm 1 vs 2 ranks, bf16 training and exact restart), then the
# 4-GPU distributed trainer on synthetic tensors (dry-run, 50-step fit).
# No data. 20 min x 4 GPUs is at most 0.33 NHR.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${OPENCHANGE_EXPECTED_GIT_SHA:?Set the reviewed exact git commit hash}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${OPENCHANGE_EXPECTED_GIT_SHA}" ]] || [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing: expected commit mismatch or uncommitted/untracked changes."
  exit 2
fi
SIF="${OPENCHANGE_SIF:-${PROJECTDIR:-${HOME}}/sif-images/pytorch_25.05-py3.sif}"
if [[ ! -f "${SIF}" ]]; then
  echo "Container missing: stage the documented aarch64 NGC image before submission."
  exit 2
fi
CONFIG=configs/jepa/gh200_synthetic.yaml
mkdir -p logs outputs
export OPENCHANGE_GIT_SHA="${ACTUAL_SHA}" OPENCHANGE_GIT_DIRTY=0
export OPENCHANGE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3 OPENCHANGE_SIF="${SIF}"
OPENCHANGE_SIF_SHA256="$(sha256sum "${SIF}" | cut -d' ' -f1)"
export OPENCHANGE_SIF_SHA256
echo "git_sha=${ACTUAL_SHA} config=${CONFIG} seed=0"
echo "arch=$(uname -m) image=${OPENCHANGE_IMAGE} sif_sha256=${OPENCHANGE_SIF_SHA256}"
echo "slurm_job_id=${SLURM_JOB_ID:-none} host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

RUN=(apptainer exec --nv --bind "${PWD}:/work" --pwd /work --env PYTHONPATH=/work/src "${SIF}")

echo "=== 1. JEPA unittest suite on GPUs ==="
"${RUN[@]}" python3 -c "import torch, yaml, numpy; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"
"${RUN[@]}" python3 -m unittest discover -s tests -p "test_jepa_*.py" -v

TRAIN=(python3 -m torch.distributed.run --standalone --nproc_per_node=4
  scripts/train_jepa_distributed.py --config "${CONFIG}" --device cuda
  --workers 2 --prefetch 2 --log-every 10 --sigreg-chunk 128 --output-root outputs)

echo "=== 2. 4-GPU dry-run ==="
"${RUN[@]}" "${TRAIN[@]}" --dry-run

echo "=== 3. 4-GPU synthetic fit (50 steps, plumbing only) ==="
"${RUN[@]}" "${TRAIN[@]}" --synthetic-fit
echo "done"
