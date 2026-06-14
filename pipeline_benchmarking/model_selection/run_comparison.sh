#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

docker run --rm \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage03_models/model_screen/comparison \
  -e PYTHONPATH=/workspace/FINAL_VERSION/stage03_models \
  inference-gpu:latest \
  bash -c "pip install -q matplotlib seaborn && python3 plot_screen_comparison.py"
