"""Constants for FINAL_VERSION stage02 imputation pilot."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "stage00_splits"
STAGE = Path(__file__).resolve().parent
RESULTS = STAGE / "results"
FEATURES = ROOT / "stage01_bulk_trim" / "results" / "selected_features.json"
PILOT_TARGETS = ROOT / "stage01_features" / "selected_targets.txt"
INFERENCE_DIR = ROOT.parent / "inference"
NE_MODULE = ROOT.parent / "train_blend_ne"

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
