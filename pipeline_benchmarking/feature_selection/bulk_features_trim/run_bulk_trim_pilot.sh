#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${TRIM_CONTAINER:-final_bulk_trim_pilot}"
docker rm -f "$NAME" 2>/dev/null || true
docker run --rm --name "$NAME" \
  --cpus=16 \
  --memory=32g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage01_features \
  inference-gpu:latest \
  python3 eval_bulk_trim_pilot.py
echo "Done. See results/bulk_trim_pilot/"
