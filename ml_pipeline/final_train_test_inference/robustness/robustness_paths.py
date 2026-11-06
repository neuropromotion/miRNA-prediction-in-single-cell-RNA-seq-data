"""Paths and constants for robustness v4 (LOO weights + importance budget)."""

from __future__ import annotations

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FTTI = THIS_DIR.parent
ML_PIPELINE = FTTI.parent

SC_TEST = ML_PIPELINE / "data" / "sc_TEST"
INFERENCE_DIR = FTTI / "inference"
CONFIG_PATH = INFERENCE_DIR / "prediction_config.json"
SC_FEATURES_PATH = FTTI / "sc_features.json"
K1_REF_PATH = ML_PIPELINE / "data" / "splits" / "sc_k1" / "X_train.parquet"

OPTIMAL_K_DIR = FTTI / "Optimal_K"
SPLIT_PATH = OPTIMAL_K_DIR / "results" / "test_split.json"

BOOTSTRAP_SUMMARY_PATH = (
    FTTI / "test_metrics" / "tables" / "per_target_bootstrap_summary.csv"
)
GENE_MAPPING_PATH = INFERENCE_DIR / "ensembl_gene_mapping.csv"

PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")
K_SPLITS = ("K1", *PB_COHORTS)

OUT_DIR = THIS_DIR / "results"
TABLES_DIR = OUT_DIR / "tables"
WEIGHTS_DIR = OUT_DIR / "feature_weights"
FIGURES_DIR = OUT_DIR / "figures"
ROBUSTNESS_CONFIG_PATH = OUT_DIR / "robustness_config.json"
JOURNAL_PATH = OUT_DIR / "journal.log"

# Calibration on eval half (same retention rule as v3)
RETENTION_RATIO = 0.50
RETENTION_STAT = "median"
SEARCH_N_MASKS = 200
CONFIRM_N_MASKS = 500
BUDGET_EPS = 0.01
B_MAX = 1.0
SEED = 0

USE_FULL_TEST = False

_DEFAULT_DATA = ML_PIPELINE / "data"
DEFAULT_INFERENCE_H5AD = {
    "healthy_integrated_L": _DEFAULT_DATA / "healthy_integrated_L.h5ad",
    "RCC_integrated_full": _DEFAULT_DATA / "RCC_integrated_full.h5ad",
}

# Relative to ``results/`` (directory containing robustness_config.json)
WEIGHTS_DIR_REL = "feature_weights"
