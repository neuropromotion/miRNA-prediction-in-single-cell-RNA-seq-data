"""Top-level constants for final_train pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINAL_VERSION = ROOT.parent
WORKSPACE = FINAL_VERSION.parent

BULK_SOURCE = FINAL_VERSION / "bulk_TRAIN"
SC_SOURCE = FINAL_VERSION / "sc_TRAIN"
SC_TEST = FINAL_VERSION / "sc_TEST"

SPLITS = ROOT / "splits"
FEATURES_SRC = FEATURES

RESULTS = ROOT / "results"
ENSEMBLE_RESULTS = RESULTS / "ensemble"

MODELS = ("catboost_optuna", "tabm", "resnet")
STACK_MODELS = ("catboost_optuna", "tabm", "resnet")

SEED = 42
VAL_FRAC = 0.2
TRANSFORM = "log2(x+1)"

OPTUNA_TRIALS = 15
EARLY_STOPPING_ROUNDS = 30

PILOT_BORDERLINE = WORKSPACE / "pilot_borderline"
TABM_DIR = WORKSPACE / "train_blend_tabm"
INFERENCE_DIR = WORKSPACE / "inference"

PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")
