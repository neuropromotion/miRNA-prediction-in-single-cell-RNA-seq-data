#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${COMPARISON_CONTAINER:-final_stage01_comparison}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --cpus=16 \
  --memory=64g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage01_features \
  inference-gpu:latest \
  bash -c "pip install -q matplotlib && python3 train_eval_comparison.py && python3 plot_r2_comparison.py" \
  2>&1 | tee run_comparison.log
echo "Started container: $NAME"
echo "Logs: docker logs -f $NAME"
