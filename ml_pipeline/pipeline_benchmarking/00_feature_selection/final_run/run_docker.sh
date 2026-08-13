#!/usr/bin/env bash
# Docker launcher: mounts ml_pipeline root and runs in this stage directory.
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_ROOT="${STAGE_DIR}"
while [[ "${ML_ROOT}" != "/" && ! -d "${ML_ROOT}/shared" ]]; do
  ML_ROOT="$(dirname "${ML_ROOT}")"
done
if [[ ! -d "${ML_ROOT}/shared" ]]; then
  echo "Could not locate ml_pipeline root (shared/ missing)." >&2
  exit 1
fi
IMAGE="${DOCKER_IMAGE:-inference-gpu:latest}"
NAME="${CONTAINER_NAME:-ml_pipeline_stage}"
REL="$(python3 -c "import os.path; print(os.path.relpath('${STAGE_DIR}', '${ML_ROOT}'))")"

docker run --rm --gpus all \
  --name "${NAME}" \
  -v "${ML_ROOT}:/workspace/ml_pipeline" \
  -w "/workspace/ml_pipeline/${REL}" \
  -e PYTHONPATH=/workspace/ml_pipeline \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  bash -lc 'pip install -q -r /workspace/ml_pipeline/requirements.txt 2>/dev/null || true; exec "$@"' _ "$@"
