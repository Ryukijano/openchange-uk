#!/bin/bash
#SBATCH --job-name=jepa-gate1-4gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Bounded Gate-1 objective screen on the approved pilot manifest.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${OPENCHANGE_EXPECTED_GIT_SHA:?Set the reviewed exact git commit hash}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${OPENCHANGE_EXPECTED_GIT_SHA}" ]] || [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing: expected commit mismatch or uncommitted/untracked changes."
  exit 2
fi
: "${OPENCHANGE_JEPA_MANIFEST:?Set the reviewed local pilot manifest}"
: "${OPENCHANGE_DATA_ROOT:?Set its pre-staged local array root}"
MANIFEST="$(realpath "${OPENCHANGE_JEPA_MANIFEST}")"
DATA_ROOT="$(realpath "${OPENCHANGE_DATA_ROOT}")"
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
echo "git_sha=${ACTUAL_SHA} mode=gate1_local_fit seed=0"
echo "arch=$(uname -m) sif=${SIF} sif_sha256=${OPENCHANGE_SIF_SHA256}"
echo "slurm_job_id=${SLURM_JOB_ID:-none} world_size=4"

ARMS=(
  "configs/jepa/gh200_local_pilot.yaml|jepa-gh200-local-pilot"
  "configs/jepa/gate1_defensive.yaml|jepa-gate1-defensive"
  "configs/jepa/gate1_tessera.yaml|jepa-gate1-tessera"
  "configs/jepa/gate1_vicreg.yaml|jepa-gate1-vicreg"
)
for arm in "${ARMS[@]}"; do
  CONFIG="${arm%%|*}"
  JOB="${arm##*|}"
  RUN_DIR="outputs/${JOB}-slurm${SLURM_JOB_ID}"
  echo "=== arm config=${CONFIG} job=${JOB} ==="
  apptainer exec --nv --bind "${PWD}:/work" --bind "${DATA_ROOT}:${DATA_ROOT}:ro" \
    --pwd /work --env PYTHONPATH=/work/src "${SIF}" \
    python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/train_jepa_distributed.py --config "${CONFIG}" --local-fit --device cuda \
    --manifest "${MANIFEST}" --sources-config configs/sources.yaml \
    --workers "${OPENCHANGE_JEPA_WORKERS:-2}" --prefetch 2 --log-every 10 \
    --sigreg-chunk "${OPENCHANGE_JEPA_CHUNK:-128}" \
    --profile-steps "${OPENCHANGE_JEPA_PROFILE_STEPS:-0}" --output-root outputs
  CHECKPOINT="${RUN_DIR}/checkpoints/step_000050.pt"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing expected Gate-1 checkpoint: ${CHECKPOINT}"
    exit 4
  fi
  apptainer exec --nv --bind "${PWD}:/work" --bind "${DATA_ROOT}:${DATA_ROOT}:ro" \
    --pwd /work --env PYTHONPATH=/work/src "${SIF}" \
    python3 scripts/evaluate_jepa_gate1.py --config "${CONFIG}" \
    --manifest "${MANIFEST}" --sources-config configs/sources.yaml \
    --checkpoint "${CHECKPOINT}" --device cuda --eval-steps 5 \
    --output "${RUN_DIR}/gate1_diagnostics.json"
done
echo "Gate-1 four-arm screen completed"
