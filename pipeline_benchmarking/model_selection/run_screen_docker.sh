#!/usr/bin/env bash
# Model screen: 50 miRNA × 7 models (XGB, DL, TabM, TabNet)
set -euo pipefail
cd "$(dirname "$0")"
STAGE03="$(cd .. && pwd)"

NAME="${1:-final_stage03_screen}"
IMAGE="${DOCKER_IMAGE:-inference-gpu:latest}"
DEPS='pip install -q --default-timeout=600 optuna tabm pytorch-tabnet rtdl rtdl-revisiting-models joblib'

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --cpus=16 \
  --memory=32g \
  -v /home/amismailov:/workspace \
  -w "/workspace/FINAL_VERSION/stage03_models/model_screen" \
  -e PYTHONPATH="/workspace/FINAL_VERSION/stage03_models" \
  -e STAGE03_DEVICE=cuda \
  -e STAGE03_BATCH=512 \
  -e STAGE03_MODELS="${STAGE03_MODELS:-all}" \
  "$IMAGE" \
  bash -c "${DEPS} && python3 run_model_screen.py"

echo "Started $NAME"
echo "  docker logs -f $NAME"
echo "  tail -f ${STAGE03}/journal.log"
