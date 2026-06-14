#!/usr/bin/env bash
# Build TabPFN-3 image and run 5-miRNA speed benchmark.
set -euo pipefail
cd "$(dirname "$0")"
STAGE03="$(cd .. && pwd)"

NAME="${1:-final_stage03_tabpfn3}"
IMAGE="${TABPFN3_IMAGE:-final-tabpfn3:latest}"

TOKEN="${TABPFN_TOKEN:-${CASCADE_TABPFN_TOKEN:-}}"
HF="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "$HF" ]]; then
  echo "Set HF_TOKEN (HuggingFace read token with Prior-Labs/tabpfn_3 access)." >&2
  exit 1
fi

echo "Building $IMAGE ..."
docker build -f Dockerfile.tabpfn3 -t "$IMAGE" .

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --cpus=16 \
  --memory=32g \
  -v /home/amismailov:/workspace \
  -w /workspace/FINAL_VERSION/stage03_models/speed_test \
  -e PYTHONPATH=/workspace/FINAL_VERSION/stage03_models \
  -e TABPFN_TOKEN="$TOKEN" \
  -e HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
  -e STAGE03_DEVICE=cuda \
  -e TABPFN_VERSION=v3 \
  -e TABPFN_MAX_TRAIN=0 \
  "$IMAGE"

echo "Started $NAME — tail: docker logs -f $NAME"
