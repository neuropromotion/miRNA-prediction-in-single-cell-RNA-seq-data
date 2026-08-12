"""Stack predictions on TEST cohorts (tabpack+dcnv2+tabm) with disk cache."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from config import (
    COHORTS,
    ML_PIPELINE,
    MODELS_ROOT,
    PRED_CACHE,
    STACK_MODELS,
    TRAIN_DIR,
    WEIGHTS_DIR,
)
from data_loading import SplitData

# Point model_trainers at train/results (TabPack v2 bundles).
os.environ["FINAL_MODELS_ROOT"] = str(MODELS_ROOT)

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from data import select_features  # noqa: E402
from model_trainers import load_artifact, predict_one  # noqa: E402
from stack import apply_fit, fit_from_dict  # noqa: E402


def _load_fit(target: str):
    path = WEIGHTS_DIR / f"{target}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing stack weights: {path}")
    return fit_from_dict(json.loads(path.read_text(encoding="utf-8")))


def cache_path(target: str) -> Path:
    safe = target.replace("/", "_")
    return PRED_CACHE / f"{safe}.npz"


def predict_target_all_cohorts(
    target: str,
    genes: list[str],
    sc_splits: dict[str, SplitData],
    bulk: SplitData,
    *,
    force: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {cohort: (y_true, y_pred)} for COHORTS + bulk. Cache full-length arrays."""
    path = cache_path(target)
    if path.is_file() and not force:
        z = np.load(path)
        out = {}
        for name in (*COHORTS, "bulk"):
            out[name] = (
                np.asarray(z[f"{name}_y"], dtype=np.float64),
                np.asarray(z[f"{name}_pred"], dtype=np.float64),
            )
        return out

    fit = _load_fit(target)
    arts = {m: load_artifact(m, target) for m in STACK_MODELS}
    all_splits = {**sc_splits, "bulk": bulk}
    arrays: dict[str, np.ndarray] = {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, sp in all_splits.items():
        if target not in sp.y.columns:
            raise KeyError(f"{target} missing from {name} Y")
        x = select_features(sp.x, genes).to_numpy(dtype=np.float32)
        y = sp.y[target].to_numpy(dtype=np.float64)
        cols = [predict_one(m, arts[m], x) for m in STACK_MODELS]
        mat = np.column_stack(cols)
        pred = apply_fit(fit, mat, STACK_MODELS)
        out[name] = (y, pred)
        arrays[f"{name}_y"] = y.astype(np.float64)
        arrays[f"{name}_pred"] = pred.astype(np.float64)

    PRED_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return out
