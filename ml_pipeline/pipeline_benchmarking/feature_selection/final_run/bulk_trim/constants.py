"""Constants for bulk-only feature trimming."""

from __future__ import annotations

from pathlib import Path
import sys

ML_PIPELINE = Path(__file__).resolve().parents[4]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import SPLITS  # noqa: E402

STAGE = Path(__file__).resolve().parent
ROOT = ML_PIPELINE
RESULTS = STAGE / "results"
FEATURE_SOURCE = STAGE.parent / "results"

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
