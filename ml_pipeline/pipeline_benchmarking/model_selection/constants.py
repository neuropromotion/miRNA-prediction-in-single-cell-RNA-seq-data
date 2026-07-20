"""Constants for the unified 11-model stage03 screen."""

from __future__ import annotations

from pathlib import Path
import sys

ML_PIPELINE = Path(__file__).resolve().parents[2]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import FEATURES, INNER_VAL_FRAC, KNN_K, PILOT_TARGETS, SEED, STAGE03  # noqa: E402

STAGE = Path(__file__).resolve().parent
RESULTS = STAGE03 / "results"

OPTUNA_TRIALS = 15
EARLY_STOPPING_ROUNDS = 30

XGB_DEFAULT = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}

SCREEN_MODELS = (
    "xgb_default",
    "xgb_optuna",
    "catboost_optuna",
    "dcnv2",
    "realmlp",
    "resnet",
    "tabm",
    "tabnet",
    "fttransformer",
    "gandalf",
    "lassonet",
)

MODEL_LABELS = {
    "xgb_default": "XGB default",
    "xgb_optuna": "XGB Optuna",
    "catboost_optuna": "CatBoost Optuna",
    "dcnv2": "DCNv2",
    "realmlp": "RealMLP",
    "resnet": "ResNet",
    "tabm": "TabM",
    "tabnet": "TabNet",
    "fttransformer": "FT-Transformer",
    "gandalf": "GANDALF",
    "lassonet": "LassoNet",
}

# Training groups used by shell launchers (can run in parallel across groups).
MODEL_GROUPS = {
    "batch1": (
        "xgb_default",
        "xgb_optuna",
        "dcnv2",
        "realmlp",
        "resnet",
        "tabm",
        "tabnet",
    ),
    "ft": ("fttransformer",),
    "candidates": (
        "catboost_optuna",
        "gandalf",
        "lassonet",
    ),
}

TEST_METRIC_COLS = (
    "inner_val_r2",
    "test_bulk_r2",
    "test_k1_r2",
    "test_pb_K2_r2",
    "test_pb_K3_r2",
    "test_pb_K4_r2",
    "test_pb_K5_r2",
    "test_pb_K10_r2",
)
