"""Constants for final model training.

Winner from ensembles_selection_v4 (median outer_val K1):
  tabpack_dcnv2_tabm_stack
  = TabPack Muon (paper) + DCNv2 AdamW + TabM AdamW, Ridge stack
"""

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

# Raw TRAIN sources (optional; prepared splits under data/splits are required).
BULK_SOURCE = ML_PIPELINE / "data" / "bulk_TRAIN"
SC_SOURCE = ML_PIPELINE / "data" / "sc_TRAIN"
# Held-out TEST lives next to evaluate_* scripts (not under data/).
SC_TEST = FINAL / "test_metrics" / "test_evaluating" / "sc_TEST"
BULK_TEST = FINAL / "test_metrics" / "test_evaluating" / "bulk_TEST"

FEATURES = FINAL / "selected_features.json"
FEATURES_SRC = FEATURES

# miRNAs excluded from final train / stack (zero / near-zero expression).
ZERO_EXPRESSED_MIRS = TRAIN_DIR / "zero_expressed_mirs.txt"

RESULTS = ROOT / "results"
ENSEMBLE_RESULTS = RESULTS / "ensemble"

# Base models for the v4 winner stack.
MODELS = ("tabpack", "dcnv2", "tabm")
STACK_MODELS = ("tabpack", "dcnv2", "tabm")
ENSEMBLE_ID = "tabpack_dcnv2_tabm_stack"

# Recipes (match model_selection / model_tuning winners).
TABPACK_PROTOCOL = "paper"  # MuonAdamWPack
TABPACK_N_MODELS = 32
# v2: persists inference_bundle.pt (ensemble weights) for live predict / production.
TABPACK_EXPERIMENT_NAMESPACE = "mirna_final_v2"

SEED = 42
VAL_FRAC = 0.2
TRANSFORM = "log2(x+1)"

EARLY_STOPPING_ROUNDS = 30

PILOT_BORDERLINE = SHARED
TABM_DIR = SHARED
INFERENCE_DIR = INFERENCE_DIR

PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")
