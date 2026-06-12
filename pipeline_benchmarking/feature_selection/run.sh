#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 run_feature_screen.py
python3 train_eval_baseline.py
