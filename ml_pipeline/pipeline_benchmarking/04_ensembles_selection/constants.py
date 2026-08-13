"""Constants for ensembles_selection_v4.

Protocol:
  - tune weights on inner_val K1 + PB (no bulk, no outer_val)
  - report / rank on outer_val K1 + PB + bulk
  - all pairs + triples from the 4-model pool; no full quadruple

Base artifacts:
  - xgb_optuna, dcnv2, tabm  → model_selection (AdamW screen where applicable)
  - tabpack                  → model_tuning (Muon / paper protocol)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ML_PIPELINE = Path(__file__).resolve().parents[2]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import MODEL_SELECTION, MODEL_TUNING, SHARED  # noqa: E402

ROOT = ML_PIPELINE
STAGE = Path(__file__).resolve().parent
STAGE03 = MODEL_SELECTION
STAGE03_RESULTS = STAGE03 / "results"
MODEL_TUNING_RESULTS = MODEL_TUNING / "results"
PILOT_DIR = SHARED
RESULTS = STAGE / "results"
TABLES = STAGE / "tables"

BASE_MODELS = (
    "xgb_optuna",
    "tabpack",
    "dcnv2",
    "tabm",
)

# Mixed roots: solos live in different stage result trees.
MODEL_ARTIFACT_ROOTS: dict[str, Path] = {
    "xgb_optuna": STAGE03_RESULTS / "xgb_optuna",
    "tabpack": MODEL_TUNING_RESULTS / "tabpack",
    "dcnv2": STAGE03_RESULTS / "dcnv2",
    "tabm": STAGE03_RESULTS / "tabm",
}

MODEL_SHORT = {
    "xgb_optuna": "xgb",
    "tabpack": "tabpack",
    "dcnv2": "dcnv2",
    "tabm": "tabm",
}

BASE_MODEL_LABELS = {
    "xgb_optuna": "XGB Optuna",
    "tabpack": "TabPack Muon",
    "dcnv2": "DCNv2 AdamW",
    "tabm": "TabM AdamW",
}


def model_set_name(models: tuple[str, ...]) -> str:
    return "_".join(MODEL_SHORT[m] for m in models)


def build_ensemble_sets() -> dict[str, tuple[str, ...]]:
    """All size-2/3 subsets of the pool (exclude the full 4-model set)."""
    out: dict[str, tuple[str, ...]] = {}
    for r in (2, 3):
        for combo in combinations(BASE_MODELS, r):
            out[model_set_name(combo)] = combo
    return out


ENSEMBLE_SETS: dict[str, tuple[str, ...]] = build_ensemble_sets()

ENSEMBLE_METHODS = (
    "blend",
    "avg_uniform",
    "stack",
)

METHOD_LABELS = {
    "blend": "Blend (simplex grid)",
    "avg_uniform": "Uniform average",
    "stack": "Stack (Ridge)",
}

# Tune: SC part of inner_val only (K1 + PB cohorts).
TUNE_SPLITS = (
    "inner_val_k1",
    "inner_val_pb_K2",
    "inner_val_pb_K3",
    "inner_val_pb_K4",
    "inner_val_pb_K5",
    "inner_val_pb_K10",
)

# Holdout metrics for reporting / ranking.
EVAL_SPLITS = (
    "outer_val_k1",
    "outer_val_bulk",
    "outer_val_pb_K2",
    "outer_val_pb_K3",
    "outer_val_pb_K4",
    "outer_val_pb_K5",
    "outer_val_pb_K10",
)

REPORT_METRIC_COLS = (
    "outer_val_k1_r2",
    "outer_val_bulk_r2",
    "outer_val_pb_K2_r2",
    "outer_val_pb_K3_r2",
    "outer_val_pb_K4_r2",
    "outer_val_pb_K5_r2",
    "outer_val_pb_K10_r2",
)

# SC cohorts used for cross-K aggregate scores (exclude bulk).
K_COHORT_METRIC_COLS = (
    "outer_val_k1_r2",
    "outer_val_pb_K2_r2",
    "outer_val_pb_K3_r2",
    "outer_val_pb_K4_r2",
    "outer_val_pb_K5_r2",
    "outer_val_pb_K10_r2",
)

PRIMARY_RANK_COL = "median_outer_val_k1_r2"

R2_THRESHOLD = 0.4
BLEND_GRID_STEP = 0.1
RIDGE_ALPHAS = tuple(float(x) for x in np.logspace(-2, 4, 40))
SAFETY_GATE = True
