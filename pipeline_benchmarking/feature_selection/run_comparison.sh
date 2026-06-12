#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 train_eval_comparison.py
python3 plot_r2_comparison.py
