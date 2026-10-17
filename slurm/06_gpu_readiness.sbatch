#!/bin/bash
#SBATCH --job-name=gpu-readiness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Human submission only. Default: one GPU. See docs/GPU_READINESS.md for 2/4-GPU mode.
# No data, weights, package installs or container pulls inside this job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${OPENCHANGE_EXPECTED_GIT_SHA:?Set the reviewed full commit hash}"
: "${OPENCHANGE_EXPECTED_SIF_SHA256:?Set the reviewed container SHA-256}"
: "${OPENCHANGE_SIF:?Set the absolute path to the staged Arm64 NGC SIF}"
CONFIG=configs/gpu_readiness.yaml
MODE="${OPENCHANGE_GPU_MODE:-single}"
GPUS="${SLURM_GPUS:?Missing allocated GPU count}"

if [[ "${SLURM_JOB_NUM_NODES:-}" != 1 || "$(uname -m)" != aarch64 ]]; then
  echo "Refusing: requires one aarch64 compute node."
  exit 2
fi
case "${MODE}:${GPUS}" in
  single:1|distributed:2|distributed:4) ;;
  *) echo "Refusing: single needs 1 GPU; distributed needs 2 or 4 GPUs."; exit 2 ;;
esac
OPENCHANGE_GIT_SHA="$(git rev-parse HEAD)"
if [[ "${OPENCHANGE_GIT_SHA}" != "${OPENCHANGE_EXPECTED_GIT_SHA}" || -n "$(git status --porcelain)" ]]; then
  echo "Refusing: commit mismatch or uncommitted/untracked changes."
  exit 2
fi
if [[ ! -f "${OPENCHANGE_SIF}" || "${OPENCHANGE_SIF}" != /* || ! -f "${CONFIG}" ]]; then
  echo "Refusing: stage the container and config before submission."
  exit 2
fi
OPENCHANGE_SIF_SHA256="$(sha256sum "${OPENCHANGE_SIF}" | cut -d' ' -f1)"
if [[ "${OPENCHANGE_SIF_SHA256}" != "${OPENCHANGE_EXPECTED_SIF_SHA256}" ]]; then
  echo "Refusing: container SHA-256 differs from the reviewed image."
  exit 2
fi
OPENCHANGE_GIT_DIRTY=0
OPENCHANGE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3
export OPENCHANGE_GIT_SHA OPENCHANGE_GIT_DIRTY OPENCHANGE_IMAGE OPENCHANGE_SIF_SHA256
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export NCCL_DEBUG=INFO TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Avoid inherited local-venv paths; keep Apptainer's documented GPU library handling.
unset PYTHONHOME PYTHONPATH RANK LOCAL_RANK WORLD_SIZE
mkdir -p outputs
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) arch=$(uname -m)"
echo "git_sha=${OPENCHANGE_GIT_SHA} git_dirty=0 config=${CONFIG} seed=0"
echo "config_sha256=$(sha256sum "${CONFIG}" | cut -d' ' -f1)"
echo "slurm_job_id=${SLURM_JOB_ID} mode=${MODE} gpus=${GPUS} image=${OPENCHANGE_IMAGE}"
echo "sif=${OPENCHANGE_SIF} sif_sha256=${OPENCHANGE_SIF_SHA256}"
echo "apptainer=$(apptainer --version)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
RUN=(apptainer exec --nv --bind "${PWD}:/work" --pwd /work
  --env PYTHONPATH=/work/src "${OPENCHANGE_SIF}")
if [[ "${MODE}" == single ]]; then
  "${RUN[@]}" python3 scripts/check_gpu_env.py --config "${CONFIG}" --mode single
else
  "${RUN[@]}" python3 -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${GPUS}" --max_restarts=0 \
    scripts/check_gpu_env.py --config "${CONFIG}" --mode distributed
fi
