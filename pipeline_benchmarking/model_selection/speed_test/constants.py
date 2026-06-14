"""Constants for FINAL_VERSION stage03 model screen."""

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
PILOT_DIR = ROOT.parent / "pilot_borderline"
TABM_DIR = ROOT.parent / "train_blend_tabm"

SEED = 42
INNER_VAL_FRAC = 0.15
SPEED_N_TARGETS = 5
OPTUNA_TRIALS_SPEED = 5
OPTUNA_TRIALS = 15
EARLY_STOPPING_ROUNDS = 30

XGB_DEFAULT = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}

KNN_K = 5

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
