#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${FULL_CONTAINER:-final_stage01_full}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --cpus=16 \
  --memory=64g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage01_full \
  inference-gpu:latest \
  python3 run_full.py
echo "Started: $NAME"
echo "Logs:   docker logs -f $NAME"
echo "Journal: tail -f /home/amismailov/FINAL_VERSION/stage01_full/journal.log"
