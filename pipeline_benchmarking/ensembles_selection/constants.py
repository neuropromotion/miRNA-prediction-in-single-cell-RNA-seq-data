"""Constants for stage04 ensembles v2."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(__file__).resolve().parent
STAGE03 = ROOT / "stage03_models"
STAGE03_RESULTS = STAGE03 / "results"
PILOT_DIR = ROOT.parent / "pilot_borderline"
RESULTS = STAGE / "results"

BASE_MODELS = (
    "xgb_optuna",
    "catboost_optuna",
    "tabm",
    "resnet",
)

MODEL_SHORT = {
    "xgb_optuna": "xgb",
    "catboost_optuna": "catboost",
    "tabm": "tabm",
    "resnet": "resnet",
}

BASE_MODEL_LABELS = {
    "xgb_optuna": "XGB Optuna",
    "catboost_optuna": "CatBoost Optuna",
    "tabm": "TabM",
    "resnet": "ResNet",
}


def model_set_name(models: tuple[str, ...]) -> str:
    if len(models) == len(BASE_MODELS):
        return "quad4"
    return "_".join(MODEL_SHORT[m] for m in models)


def build_ensemble_sets() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for r in (2, 3, 4):
        for combo in combinations(BASE_MODELS, r):
            out[model_set_name(combo)] = combo
    return out


ENSEMBLE_SETS: dict[str, tuple[str, ...]] = build_ensemble_sets()

ENSEMBLE_METHODS = (
    "blend",
    "soup_uniform",
    "soup_greedy",
    "soup_pruned",
    "stack",
)

METHOD_LABELS = {
    "blend": "Blend (grid)",
    "soup_uniform": "Soup uniform",
    "soup_greedy": "Soup greedy",
    "soup_pruned": "Soup pruned",
    "stack": "Stack (Ridge)",
}

# Weight tuning: K1 + pseudo-bulk PB only (no real bulk).
TUNE_SPLITS = (
    "test_k1",
    "test_pb_K2",
    "test_pb_K3",
    "test_pb_K4",
    "test_pb_K5",
    "test_pb_K10",
)

EVAL_SPLITS = (
    "inner_val",
    "test_bulk",
    *TUNE_SPLITS,
)

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

R2_THRESHOLD = 0.4
BLEND_GRID_STEP = 0.1
SOUP_MIX_STEP = 0.05
SOUP_PRUNE_PASSES = 2
RIDGE_ALPHAS = tuple(float(x) for x in np.logspace(-2, 4, 40))
SAFETY_GATE = True
