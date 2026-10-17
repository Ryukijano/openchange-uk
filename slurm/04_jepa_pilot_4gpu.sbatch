#!/bin/bash
#SBATCH --job-name=jepa-pilot-4gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Human submission only, after the documented one-GPU container smoke.
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
MODE="${OPENCHANGE_JEPA_MODE:-dry-run}"
CONFIG=configs/jepa/gh200_synthetic.yaml
EXTRA=()
BINDS=(--bind "${PWD}:/work")
case "${MODE}" in
  dry-run|synthetic-fit) ;;
  local-fit)
    : "${OPENCHANGE_JEPA_MANIFEST:?Set the reviewed local pilot manifest}"
    : "${OPENCHANGE_DATA_ROOT:?Set its pre-staged local array root}"
    CONFIG=configs/jepa/gh200_local_pilot.yaml
    MANIFEST="$(realpath "${OPENCHANGE_JEPA_MANIFEST}")"
    DATA_ROOT="$(realpath "${OPENCHANGE_DATA_ROOT}")"
    EXTRA=(--manifest "${MANIFEST}" --sources-config configs/sources.yaml)
    BINDS+=(--bind "${DATA_ROOT}:${DATA_ROOT}:ro")
    ;;
  *) echo "Unsupported mode: ${MODE}"; exit 2 ;;
esac
mkdir -p logs outputs
export OPENCHANGE_GIT_SHA="${ACTUAL_SHA}" OPENCHANGE_GIT_DIRTY=0
export OPENCHANGE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3 OPENCHANGE_SIF="${SIF}"
OPENCHANGE_SIF_SHA256="$(sha256sum "${SIF}" | cut -d' ' -f1)"
export OPENCHANGE_SIF_SHA256
echo "git_sha=${ACTUAL_SHA} config=${CONFIG} mode=${MODE} seed=0"
echo "arch=$(uname -m) image=${OPENCHANGE_IMAGE} sif_sha256=${OPENCHANGE_SIF_SHA256}"
echo "slurm_job_id=${SLURM_JOB_ID:-none} world_size=4"
apptainer exec --nv "${BINDS[@]}" --pwd /work --env PYTHONPATH=/work/src "${SIF}" \
  python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_jepa_distributed.py --config "${CONFIG}" --"${MODE}" --device cuda \
  --workers "${OPENCHANGE_JEPA_WORKERS:-2}" --prefetch 2 --log-every 10 \
  --sigreg-chunk "${OPENCHANGE_JEPA_CHUNK:-128}" \
  --profile-steps "${OPENCHANGE_JEPA_PROFILE_STEPS:-0}" --output-root outputs "${EXTRA[@]}"
