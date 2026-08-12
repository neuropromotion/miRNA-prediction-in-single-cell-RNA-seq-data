"""Optimal_K on held-out TEST (tune half): bootstrap R² → eligibility + cohort.

Protocol (fixed):
  B=1000, split 50/50 (shared positional idx for all SC cohorts + separate bulk),
  eligible iff median(R²_bulk)≥0.5 AND max_K median(R²_K)≥0.5,
  optimal_K = smallest K among those with m_K ≥ max(m) − δ (δ=0.05).

Split is persisted to results/test_split.json for reuse by test_metrics (eval half).
"""

from __future__ import annotations

from pathlib import Path

OPTIMAL_K_DIR = Path(__file__).resolve().parent
FTTI = OPTIMAL_K_DIR.parent
ML_PIPELINE = FTTI.parent
TRAIN_DIR = FTTI / "train"

# --- Protocol knobs ---
SEED = 42
SPLIT_FRAC_TUNE = 0.5
N_BOOTSTRAP = 1000
MEDIAN_THRESHOLD = 0.4
DELTA = 0.05

COHORTS = ("K1", "K2", "K3", "K4", "K5", "K10")  # preference order (small → large)
PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")
STACK_MODELS = ("tabpack", "dcnv2", "tabm")
ENSEMBLE_ID = "tabpack_dcnv2_tabm_stack"

# Data
SC_TEST = ML_PIPELINE / "data" / "sc_TEST"
BULK_TEST = ML_PIPELINE / "data" / "bulk_TEST"
K1_REF = ML_PIPELINE / "data" / "splits" / "sc_k1" / "X_train.parquet"
SELECTED_FEATURES = FTTI / "selected_features.json"
ZERO_EXPRESSED = TRAIN_DIR / "zero_expressed_mirs.txt"

# Models / stack weights (v2 TabPack with inference_bundle)
MODELS_ROOT = TRAIN_DIR / "results"
WEIGHTS_DIR = MODELS_ROOT / "ensemble" / ENSEMBLE_ID / "weights"

# Outputs
RESULTS = OPTIMAL_K_DIR / "results"
PRED_CACHE = RESULTS / "pred_cache"
FIGURES = OPTIMAL_K_DIR / "figures"
TABLES = OPTIMAL_K_DIR / "tables"
SPLIT_PATH = RESULTS / "test_split.json"
DECISIONS_PATH = TABLES / "decisions.csv"
BOOT_SUMMARY_PATH = TABLES / "bootstrap_medians.csv"
COHORT_COUNTS_PATH = TABLES / "cohort_counts.csv"
PREDICTION_CONFIG_PATH = RESULTS / "proto_prediction_config.json"
JOURNAL = RESULTS / "journal.log"

ASSIGNMENT_RULE = (
    f"TEST tune half (seed={SEED}, frac={SPLIT_FRAC_TUNE}): "
    f"B={N_BOOTSTRAP} bootstrap R² medians; "
    f"eligible if m_bulk≥{MEDIAN_THRESHOLD} and max_K m_K≥{MEDIAN_THRESHOLD}; "
    f"optimal_K = smallest K with m_K ≥ max(m)−{DELTA}."
)
