"""Final TEST metrics on eval half (bootstrap R² + MSE for SC@assigned_K and bulk)."""

from __future__ import annotations

from pathlib import Path

TEST_METRICS_DIR = Path(__file__).resolve().parent
FTTI = TEST_METRICS_DIR.parent
ML_PIPELINE = FTTI.parent
TRAIN_DIR = FTTI / "train"
OPTIMAL_K_DIR = FTTI / "Optimal_K"

# Protocol (match Optimal_K)
SEED = 42
N_BOOTSTRAP = 1000

ENSEMBLE_ID = "tabpack_dcnv2_tabm_stack"
STACK_MODELS = ("tabpack", "dcnv2", "tabm")
COHORTS = ("K1", "K2", "K3", "K4", "K5", "K10")

# Inputs from Optimal_K (proto = assignments from tune half; not production config)
SPLIT_PATH = OPTIMAL_K_DIR / "results" / "test_split.json"
PROTO_PREDICTION_CONFIG = OPTIMAL_K_DIR / "results" / "proto_prediction_config.json"
# Legacy fallback if Optimal_K was run before the proto rename
_LEGACY_PREDICTION_CONFIG = OPTIMAL_K_DIR / "results" / "prediction_config.json"
PRED_CACHE = OPTIMAL_K_DIR / "results" / "pred_cache"

# Models (only needed if pred_cache miss)
MODELS_ROOT = TRAIN_DIR / "results"
WEIGHTS_DIR = MODELS_ROOT / "ensemble" / ENSEMBLE_ID / "weights"
SELECTED_FEATURES = FTTI / "selected_features.json"
SC_TEST = ML_PIPELINE / "data" / "sc_TEST"
BULK_TEST = ML_PIPELINE / "data" / "bulk_TEST"
K1_REF = ML_PIPELINE / "data" / "splits" / "sc_k1" / "X_train.parquet"

# Outputs
RESULTS = TEST_METRICS_DIR / "results"
FIGURES = TEST_METRICS_DIR / "figures"
TABLES = TEST_METRICS_DIR / "tables"
JOURNAL = RESULTS / "journal.log"

# Final production config (eval-half test metrics; copy to inference/ when ready)
PREDICTION_CONFIG_PATH = RESULTS / "prediction_config.json"

PER_TARGET_PATH = TABLES / "per_target_bootstrap_summary.csv"
SC_SUMMARY_PATH = TABLES / "sc_summary.csv"
BULK_SUMMARY_PATH = TABLES / "bulk_summary.csv"
OVERALL_PATH = TABLES / "overall_summary.csv"
