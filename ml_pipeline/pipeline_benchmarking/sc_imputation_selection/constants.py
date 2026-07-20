"""Constants for scRNA imputation benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

ML_PIPELINE = Path(__file__).resolve().parents[2]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import (  # noqa: E402
    FEATURES,
    FROZEN,
    INFERENCE_DIR,
    NE_MODULE,
    PILOT_TARGETS,
    SPLITS,
)

STAGE = Path(__file__).resolve().parent
ROOT = ML_PIPELINE
RESULTS = STAGE / "results"

SEED = 42

IMPUTE_METHODS = (
    "raw",
    "knn_k5",
    "knn_k10",
    "ne",
    "softimpute",
    "magic",
)

SOFTIMPUTE_MAX_ITERS = 100
MAGIC_KNN = 5
MAGIC_T = 3

XGB_DEFAULT = {
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": SEED,
}

NE_CONFIG = {
    "n_pca": 50,
    "k": 20,
    "alpha": 0.9,
    "order": 2,
    "self_weight": 1.5,
}
