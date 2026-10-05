"""Load Optimal_K config/split and pred cache (recompute if missing)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from config import (
    ML_PIPELINE,
    MODELS_ROOT,
    PRED_CACHE,
    SELECTED_FEATURES,
    SPLIT_PATH,
    TRAIN_DIR,
)

os.environ.setdefault("FINAL_MODELS_ROOT", str(MODELS_ROOT))

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))


def load_prediction_config() -> dict:
    """Load Optimal_K proto (assignments). Final config is written by evaluate.finalize."""
    from build_prediction_config import load_proto_path

    path = load_proto_path()
    return json.loads(path.read_text(encoding="utf-8"))


def load_split() -> dict:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"missing {SPLIT_PATH}")
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def eligible_assignments(cfg: dict) -> list[tuple[str, str]]:
    """Return sorted list of (target, optimal_k)."""
    out: list[tuple[str, str]] = []
    for t in cfg["eligible_mirs"]:
        k = None
        for cohort in cfg["cohorts"]:
            if t in cfg.get(cohort, {}):
                k = cohort
                break
        if k is None:
            raise KeyError(f"{t} in eligible_mirs but missing from cohort maps")
        out.append((t, k))
    return out


def cache_path(target: str) -> Path:
    return PRED_CACHE / f"{target.replace('/', '_')}.npz"


def load_cached_pair(target: str, cohort: str) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(target)
    if not path.is_file():
        raise FileNotFoundError(f"pred cache missing for {target}: {path}")
    z = np.load(path)
    key_y, key_p = f"{cohort}_y", f"{cohort}_pred"
    if key_y not in z.files or key_p not in z.files:
        raise KeyError(f"{target}: cache lacks {cohort}")
    return np.asarray(z[key_y], dtype=np.float64), np.asarray(z[key_p], dtype=np.float64)


def ensure_cached_pair(target: str, cohort: str) -> tuple[np.ndarray, np.ndarray]:
    """Load from Optimal_K pred_cache; if missing, predict that target fully and cache."""
    path = cache_path(target)
    if path.is_file():
        return load_cached_pair(target, cohort)

    # Lazy recompute via Optimal_K helpers (rare).
    ok_dir = str(TRAIN_DIR.parent / "Optimal_K")
    if ok_dir not in sys.path:
        sys.path.insert(0, ok_dir)
    from data_loading import load_bulk_split, load_features, load_sc_splits  # type: ignore
    from predict_stack import predict_target_all_cohorts  # type: ignore

    features = json.loads(SELECTED_FEATURES.read_text(encoding="utf-8"))
    genes = features[target]
    preds = predict_target_all_cohorts(target, genes, load_sc_splits(), load_bulk_split(), force=True)
    return preds[cohort]
