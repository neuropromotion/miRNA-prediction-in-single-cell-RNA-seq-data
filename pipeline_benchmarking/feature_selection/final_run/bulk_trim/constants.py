"""Constants for FINAL_VERSION stage01_bulk_trim (327 miRNA)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "stage00_splits"
STAGE = Path(__file__).resolve().parent
RESULTS = STAGE / "results"
FEATURE_SOURCE = ROOT / "stage01_full" / "results"

SEED = 42

K_OPTIONS = (50, 100, 150, 200)
MIN_BULK_ONLY = 50
MIN_BASELINE_BULK_R2 = 0.4
MAX_REL_DROP = 0.10
MAX_ABS_DROP = 0.02
R2_THRESHOLD = 0.4

XGB_SHALLOW = {
    "n_estimators": 80,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": SEED,
}

XGB_DEFAULT = {
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": SEED,
}
