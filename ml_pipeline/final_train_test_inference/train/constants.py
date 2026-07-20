"""Constants for final model training."""

from __future__ import annotations

from pathlib import Path
import sys

TRAIN_DIR = Path(__file__).resolve().parent
FINAL = TRAIN_DIR.parent
ML_PIPELINE = FINAL.parent
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import INFERENCE_DIR, SHARED, SPLITS  # noqa: E402

ROOT = TRAIN_DIR
FINAL_VERSION = FINAL
WORKSPACE = ML_PIPELINE

# Raw sources are optional; prepared splits under data/splits are required.
BULK_SOURCE = ML_PIPELINE / "data" / "raw" / "bulk_TRAIN"
SC_SOURCE = ML_PIPELINE / "data" / "raw" / "sc_TRAIN"
SC_TEST = ML_PIPELINE / "data" / "raw" / "sc_TEST"

FEATURES = FINAL / "selected_features.json"
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

PILOT_BORDERLINE = SHARED
TABM_DIR = SHARED
# train/impute.py expects model_loader here
INFERENCE_DIR = INFERENCE_DIR

PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")
