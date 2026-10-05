"""Paths and defaults for robustness v3 (bisect dropout on eval half)."""

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

PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")

OUT_DIR = THIS_DIR / "results"
TABLES_DIR = OUT_DIR / "tables"
ROBUSTNESS_CONFIG_PATH = OUT_DIR / "robustness_config.json"
JOURNAL_PATH = OUT_DIR / "journal.log"

# Bisect on absent fraction p in [0, P_MAX]; stop when interval width < BISECT_EPS.
P_MAX = 0.95
BISECT_EPS = 0.01

# Monte Carlo masks per bisect probe vs final confirmation at max p.
SEARCH_N_MASKS = 200
CONFIRM_N_MASKS = 1000

# stat(R² masks) >= RETENTION_RATIO * r2_reference
RETENTION_RATIO = 0.50
# "median" (default) or "q25" — set via CLI --retention-stat
RETENTION_STAT = "median"

USE_FULL_TEST = False
SEED = 0
