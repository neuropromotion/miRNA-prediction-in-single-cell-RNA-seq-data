#!/usr/bin/env bash
# Evaluate final stack on sc_TEST (K1–K10) → test_metrics/K1_K10_test_metrics.csv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_PIPELINE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${ML_PIPELINE}${PYTHONPATH:+:${PYTHONPATH}}"
export FINAL_DEVICE="${FINAL_DEVICE:-cpu}"
export CATBOOST_TASK="${CATBOOST_TASK:-CPU}"
cd "${SCRIPT_DIR}"
exec python3 evaluate_sc_test.py "$@"
