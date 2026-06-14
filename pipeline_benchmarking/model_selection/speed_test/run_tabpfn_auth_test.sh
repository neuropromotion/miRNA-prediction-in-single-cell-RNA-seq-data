#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

docker run --rm --gpus all \
  -e TABPFN_TOKEN="${TABPFN_TOKEN:-}" \
  -e HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage03_models/speed_test \
  -e PYTHONPATH=/workspace/FINAL_VERSION/stage03_models \
  final-tabpfn3:latest \
  python3 test_tabpfn_auth.py
