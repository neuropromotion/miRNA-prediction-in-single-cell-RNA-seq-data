"""Central paths for the ml_pipeline workspace (GITHUB / future ml_pipeline/)."""

from __future__ import annotations

from pathlib import Path

# GITHUB/ == ml_pipeline root
ML_PIPELINE = Path(__file__).resolve().parents[1]
SHARED = Path(__file__).resolve().parent

DATA = ML_PIPELINE / "data"
SPLITS = DATA / "splits"
FROZEN = DATA / "frozen"

PIPELINE_BENCH = ML_PIPELINE / "pipeline_benchmarking"
FEATURE_SELECTION = PIPELINE_BENCH / "feature_selection"
SC_IMPUTATION = PIPELINE_BENCH / "sc_imputation_selection"
MODEL_SELECTION = PIPELINE_BENCH / "model_selection"
MODEL_TUNING = PIPELINE_BENCH / "model_tuning"
ENSEMBLES = PIPELINE_BENCH / "ensembles_selection"
ENSEMBLES_V2 = PIPELINE_BENCH / "ensembles_selection_v2"
ENSEMBLES_V3 = PIPELINE_BENCH / "ensembles_selection_v3"
ENSEMBLES_V4 = PIPELINE_BENCH / "ensembles_selection_v4"

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
