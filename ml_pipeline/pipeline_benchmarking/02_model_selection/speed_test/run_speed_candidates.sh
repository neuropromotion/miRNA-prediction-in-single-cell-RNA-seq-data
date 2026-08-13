#!/usr/bin/env bash
# Speed test: LassoNet + GANDALF on 5 miRNA (same targets as original benchmark)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE03="$(cd "${SCRIPT_DIR}/.." && pwd)"
ML_PIPELINE="$(cd "${STAGE03}/../.." && pwd)"
NAME="${1:-final_speed_candidates}"
IMAGE="${DOCKER_IMAGE:-inference-gpu:latest}"
DEPS='pip install -q lassonet pytorch-tabular pytorch-lightning joblib'

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --cpus=8 \
  --memory=24g \
  -v "${ML_PIPELINE}:/workspace/ml_pipeline" \
  -w "/workspace/ml_pipeline/pipeline_benchmarking/model_selection" \
  -e PYTHONPATH="/workspace/ml_pipeline:/workspace/ml_pipeline/pipeline_benchmarking/model_selection" \
  -e STAGE03_DEVICE=cuda \
  -e STAGE03_BATCH=512 \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -c "${DEPS} && python3 speed_test/speed_benchmark_candidates.py"

echo "Started $NAME"
echo "  docker logs -f $NAME"
echo "  results: ${STAGE03}/results/speed_candidates/"
