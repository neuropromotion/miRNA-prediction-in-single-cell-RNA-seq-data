#!/usr/bin/env bash
set -euo pipefail

IMAGE="${INFERENCE_GPU_IMAGE:-inference-gpu:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE03="$(cd "${SCRIPT_DIR}/.." && pwd)"
ML_PIPELINE="$(cd "${STAGE03}/../.." && pwd)"
LOG="${STAGE03}/results/fttransformer_speed_50/run.log"

mkdir -p "$(dirname "$LOG")"

docker run --rm \
  --gpus all \
  -v "${ML_PIPELINE}:/workspace/ml_pipeline" \
  -w /workspace/ml_pipeline/pipeline_benchmarking/model_selection \
  -e PYTHONPATH="/workspace/ml_pipeline:/workspace/ml_pipeline/pipeline_benchmarking/model_selection" \
  -e STAGE03_BATCH="${STAGE03_BATCH:-512}" \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -c "pip install -q --no-deps rtdl rtdl-revisiting-models && python3 speed_test/fttransformer_speed_50.py" \
  2>&1 | tee -a "$LOG"
