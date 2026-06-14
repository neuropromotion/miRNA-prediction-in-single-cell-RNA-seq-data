#!/usr/bin/env bash
# Speed benchmark: 7 screen models on 5 miRNA (train + infer timing)
set -euo pipefail
cd "$(dirname "$0")"

NAME="${1:-final_stage03_speed}"
IMAGE="${DOCKER_IMAGE:-inference-gpu:latest}"
DEPS='pip install -q --default-timeout=600 optuna tabm pytorch-tabnet rtdl rtdl-revisiting-models joblib'

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --cpus=16 \
  --memory=32g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage03_models/speed_test \
  -e PYTHONPATH=/workspace/FINAL_VERSION/stage03_models \
  -e STAGE03_DEVICE=cuda \
  -e STAGE03_BATCH=512 \
  "$IMAGE" \
  bash -c "${DEPS} && python3 speed_benchmark.py"

echo "Started $NAME — tail with:"
echo "  docker logs -f $NAME"
