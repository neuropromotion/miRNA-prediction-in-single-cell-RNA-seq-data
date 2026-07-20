"""Constants for full ElasticNet feature selection (all miRNAs)."""

from __future__ import annotations

from pathlib import Path
import sys

ML_PIPELINE = Path(__file__).resolve().parents[3]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import SPLITS  # noqa: E402

STAGE = Path(__file__).resolve().parent
ROOT = ML_PIPELINE
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
