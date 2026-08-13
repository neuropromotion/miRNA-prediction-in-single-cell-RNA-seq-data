from __future__ import annotations

from pathlib import Path

# GITHUB/ == ml_pipeline root
ML_PIPELINE = Path(__file__).resolve().parents[1]
SHARED = Path(__file__).resolve().parent

DATA = ML_PIPELINE / "data"
SPLITS = DATA / "splits"
FROZEN = DATA / "frozen"

PIPELINE_BENCH = ML_PIPELINE / "pipeline_benchmarking"
FEATURE_SELECTION = PIPELINE_BENCH / "00_feature_selection"
SC_IMPUTATION = PIPELINE_BENCH / "01_imputation_selection"
MODEL_SELECTION = PIPELINE_BENCH / "02_model_selection"
MODEL_TUNING = PIPELINE_BENCH / "03_optimizer_selection"
ENSEMBLES = PIPELINE_BENCH / "04_ensembles_selection"

FINAL = ML_PIPELINE / "final_train_test_inference"
FINAL_TRAIN = FINAL / "train"
FINAL_INFERENCE = FINAL / "inference"
FINAL_TEST_METRICS = FINAL / "test_metrics"

# Backward-compatible aliases used by stage03-style code
STAGE03 = MODEL_SELECTION
ROOT = ML_PIPELINE

FEATURES = FROZEN / "selected_features.json"
if not FEATURES.is_file():
    FEATURES = FEATURE_SELECTION / "final_run" / "results" / "selected_features.json"
if not FEATURES.is_file():
    FEATURES = FINAL / "selected_features.json"

PILOT_TARGETS = FROZEN / "selected_targets.txt"
if not PILOT_TARGETS.is_file():
    PILOT_TARGETS = FEATURE_SELECTION / "selected_targets.txt"

# Vendored helpers (no external /home/... deps)
PILOT_DIR = SHARED
TABM_DIR = SHARED
INFERENCE_DIR = ML_PIPELINE / "deps" / "imputation"
NE_MODULE = ML_PIPELINE / "deps" / "imputation"

SEED = 42
INNER_VAL_FRAC = 0.15
KNN_K = 5
