"""Constants for model_tuning A/B (Muon / EMA / TabPack paper; no Optuna)."""

from __future__ import annotations

from pathlib import Path
import sys

ML_PIPELINE = Path(__file__).resolve().parents[2]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import (  # noqa: E402
    FEATURES,
    INNER_VAL_FRAC,
    KNN_K,
    MODEL_SELECTION,
    MODEL_TUNING,
    PILOT_TARGETS,
    SEED,
)

STAGE = Path(__file__).resolve().parent
RESULTS = MODEL_TUNING / "results"
BASELINE_SCREEN = MODEL_SELECTION / "results"

# Kept for config dumps; Optuna is disabled in this A/B stage.
OPTUNA_TRIALS = 0
EARLY_STOPPING_ROUNDS = 30

TABPACK_N_MODELS = 32
TABPACK_PROTOCOL = "paper"

BATCH_SIZE = 512
TABM_BATCH_SIZE = 4096

TUNING_MODELS = (
    "xgb_optuna",  # baseline from model_selection
    "tabm_muon",
    "tabm_adamw_ema",
    "dcnv2_muon",
    "tabpack",
)

TRAIN_MODELS = (
    "tabm_muon",
    "tabm_adamw_ema",
    "dcnv2_muon",
    "tabpack",
)

MODEL_LABELS = {
    "xgb_optuna": "XGB Optuna",
    "tabm_muon": "TabM Muon",
    "tabm_adamw_ema": "TabM AdamW+EMA",
    "dcnv2_muon": "DCNv2 Muon",
    "tabpack": "TabPack Muon",
}

OUTER_VAL_METRIC_COLS = (
    "inner_val_r2",
    "outer_val_bulk_r2",
    "outer_val_k1_r2",
    "outer_val_pb_K2_r2",
    "outer_val_pb_K3_r2",
    "outer_val_pb_K4_r2",
    "outer_val_pb_K5_r2",
    "outer_val_pb_K10_r2",
)

K_COHORT_METRIC_COLS = (
    "outer_val_k1_r2",
    "outer_val_pb_K2_r2",
    "outer_val_pb_K3_r2",
    "outer_val_pb_K4_r2",
    "outer_val_pb_K5_r2",
    "outer_val_pb_K10_r2",
)
