"""Shared paths for stage03 (speed_test + model_screen)."""

from __future__ import annotations

from pathlib import Path

STAGE03 = Path(__file__).resolve().parents[1]
ROOT = STAGE03.parent
SPLITS = ROOT / "stage00_splits"
FEATURES = ROOT / "stage01_bulk_trim" / "results" / "selected_features.json"
PILOT_TARGETS = ROOT / "stage01_features" / "selected_targets.txt"
INFERENCE_DIR = ROOT.parent / "inference"
NE_MODULE = ROOT.parent / "train_blend_ne"
PILOT_DIR = ROOT.parent / "pilot_borderline"
TABM_DIR = ROOT.parent / "train_blend_tabm"

SEED = 42
INNER_VAL_FRAC = 0.15
KNN_K = 5
