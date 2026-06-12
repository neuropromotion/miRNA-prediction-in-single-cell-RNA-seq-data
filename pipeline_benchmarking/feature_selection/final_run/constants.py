"""Constants for FINAL_VERSION stage01_full (327 miRNA ElasticNet)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "stage00_splits"
STAGE = Path(__file__).resolve().parent
RESULTS = STAGE / "results"

SEED = 42

SPEARMAN_THR_HIGH = 0.2
SPEARMAN_THR_LOW = 0.1
MIN_SPEARMAN_FEATURES = 100
MAX_SPEARMAN_FEATURES = 3000
SPEARMAN_CHUNK = 512

MAX_MODALITY_FEATURES = 800

LINEAR_CV = 3
LINEAR_ALPHAS = 30
LINEAR_MAX_ITER = 8000
LINEAR_MAX_SAMPLES = 8000
LINEAR_MAX_POOL = 1500
ENET_L1_RATIOS = (0.2, 0.5, 0.7, 0.9, 0.95, 1.0)
