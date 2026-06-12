"""Shared paths and constants for FINAL_VERSION stage01."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "stage00_splits"
STAGE = Path(__file__).resolve().parent
RESULTS = STAGE / "results"

SEED = 42
N_PILOT_TARGETS = 50

SPEARMAN_THR_HIGH = 0.2
SPEARMAN_THR_LOW = 0.1
MIN_SPEARMAN_FEATURES = 100
MAX_SPEARMAN_FEATURES = 3000
SPEARMAN_CHUNK = 512

MAX_FINAL_FEATURES = 800
MAX_MODALITY_FEATURES = 800

# Adaptive score cut (XGB importance / MI): keep score >= rel_frac * max(score)
ADAPTIVE_REL_FRAC = 0.01
ADAPTIVE_MIN_FEATURES = 30

LINEAR_CV = 3
LINEAR_ALPHAS = 30
LINEAR_MAX_ITER = 8000
LINEAR_MAX_SAMPLES = 8000
LINEAR_MAX_POOL = 1500
ENET_L1_RATIOS = (0.2, 0.5, 0.7, 0.9, 0.95, 1.0)

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

METHODS = (
    "method_a_lasso",
    "method_d_elasticnet",
    "method_b_xgb_imp",
    "method_c_mi",
    "method_e_spearman",
)
