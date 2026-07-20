#!/usr/bin/env bash
# Pilot bulk-trim launcher (relative to ml_pipeline root).
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_ROOT="${STAGE_DIR}"
while [[ "${ML_ROOT}" != "/" && ! -d "${ML_ROOT}/shared" ]]; do
  ML_ROOT="$(dirname "${ML_ROOT}")"
done
IMAGE="${DOCKER_IMAGE:-inference-gpu:latest}"
docker run --rm --gpus all \
  -v "${ML_ROOT}:/workspace/ml_pipeline" \
  -w /workspace/ml_pipeline/pipeline_benchmarking/feature_selection/bulk_features_trim \
  -e PYTHONPATH=/workspace/ml_pipeline \
  "${IMAGE}" \
  bash -lc 'python eval_bulk_trim_pilot.py'
