"""Constants for stage03 model screen (50-miRNA pilot, 7 models)."""

from __future__ import annotations

from pathlib import Path

from shared.paths import FEATURES, INNER_VAL_FRAC, KNN_K, PILOT_TARGETS, SEED, STAGE03

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
    "dcnv2",
    "realmlp",
    "resnet",
    "tabm",
    "tabnet",
)

MODEL_LABELS = {
    "xgb_default": "XGB default",
    "xgb_optuna": "XGB Optuna",
    "dcnv2": "DCNv2",
    "realmlp": "RealMLP",
    "resnet": "ResNet",
    "tabm": "TabM",
    "tabnet": "TabNet",
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
