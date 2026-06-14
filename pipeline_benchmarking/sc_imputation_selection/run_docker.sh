#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${IMPUTE_CONTAINER:-final_stage02_impute}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --cpus=16 \
  --memory=32g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage02_impute \
  inference-gpu:latest \
  bash -c "pip install -q fancyimpute magic-impute && python3 run_screen.py"
echo "Started: $NAME"
echo "Logs:   docker logs -f $NAME"
echo "Journal: tail -f /home/amismailov/FINAL_VERSION/stage02_impute/journal.log"
