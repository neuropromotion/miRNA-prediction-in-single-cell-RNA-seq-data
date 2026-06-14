#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${BULK_TRIM_CONTAINER:-final_stage01_bulk_trim}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --cpus=16 \
  --memory=32g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage01_bulk_trim \
  inference-gpu:latest \
  python3 run_full.py
echo "Started: $NAME"
echo "Logs:   docker logs -f $NAME"
echo "Journal: tail -f /home/amismailov/FINAL_VERSION/stage01_bulk_trim/journal.log"
