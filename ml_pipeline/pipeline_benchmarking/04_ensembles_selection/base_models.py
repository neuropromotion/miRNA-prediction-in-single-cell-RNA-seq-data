"""Load solo artifacts (mixed model_selection / model_tuning) and predict splits."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    from .constants import BASE_MODELS, ML_PIPELINE, MODEL_ARTIFACT_ROOTS, PILOT_DIR
    from .metrics import clip_nonneg
except ImportError:
    from constants import BASE_MODELS, ML_PIPELINE, MODEL_ARTIFACT_ROOTS, PILOT_DIR
    from metrics import clip_nonneg

if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from dl_trainers import (  # noqa: E402
    _predict_torch_regressor,
    _resolve_device,
    load_torch_artifacts,
    predict_tabm,
)
from shared.data import ModalityBundle, select_features  # noqa: E402
from shared.io_splits import PB_COHORTS, pb_cohort_mask  # noqa: E402
from shared.tabpack_trainer import load_tabpack_screen  # noqa: E402

DEVICE = os.environ.get("STAGE04_DEVICE", os.environ.get("STAGE03_DEVICE", "cuda"))


def model_dir(model_name: str, target: str) -> Path:
    root = MODEL_ARTIFACT_ROOTS[model_name]
    return root / "models" / target


def model_exists(model_name: str, target: str) -> bool:
    d = model_dir(model_name, target)
    if model_name.startswith("xgb"):
        return (d / "model.json").exists()
    if model_name == "tabpack":
        return (d / "preds.npz").exists() and (d / "meta.json").exists()
    if model_name == "tabm":
        return (d / "tabm.pt").exists()
    # torch regressors (dcnv2, …)
    return (d / "model.pt").exists() and (d / "meta.json").exists()


def load_artifact(model_name: str, target: str):
    d = model_dir(model_name, target)
    if model_name.startswith("xgb"):
        import xgboost as xgb

        m = xgb.XGBRegressor()
        m.load_model(str(d / "model.json"))
        return m
    if model_name == "tabpack":
        return load_tabpack_screen(d)
    return d


def _predict_torch(model_dir_path: Path, x: np.ndarray) -> np.ndarray:
    artifacts = load_torch_artifacts(model_dir_path)
    artifacts["device"] = str(_resolve_device(DEVICE))
    return _predict_torch_regressor(artifacts, x)


def predict_one(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    if model_name.startswith("xgb"):
        return clip_nonneg(artifact.predict(x))
    if model_name == "tabm":
        return clip_nonneg(predict_tabm(artifact, x, DEVICE))
    if model_name == "tabpack":
        raise RuntimeError("tabpack uses cached preds.npz; call predict_all_splits")
    return clip_nonneg(_predict_torch(artifact, x))


def _x_y(x_df, y_df, target: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = select_features(x_df, genes).to_numpy(dtype=np.float32)
    y = y_df[target].to_numpy(dtype=np.float64)
    return x, y


def _inner_val_masks(bundle: ModalityBundle) -> dict[str, np.ndarray]:
    """Boolean masks over full mixed inner_val (same order as bundle.x_val_inner / val_pred)."""
    mod = np.asarray(bundle.val_modality)
    n = len(mod)
    out: dict[str, np.ndarray] = {}

    out["inner_val_k1"] = mod == "k1"

    m_pb = mod == "pb"
    # Build PB cohort masks on the full inner_val index, not only the pb subframe,
    # so they align with tabpack val_pred row order.
    idx = bundle.x_val_inner.index
    for cohort in PB_COHORTS:
        mask = np.zeros(n, dtype=bool)
        pb_pos = np.flatnonzero(m_pb)
        if len(pb_pos):
            sub_mask = pb_cohort_mask(idx[m_pb], cohort)
            mask[pb_pos[np.asarray(sub_mask)]] = True
        out[f"inner_val_pb_{cohort}"] = mask
    return out


def _inner_val_frames(bundle: ModalityBundle) -> dict[str, tuple]:
    """Slice inner_val into K1 + PB cohorts (exclude bulk)."""
    masks = _inner_val_masks(bundle)
    x = bundle.x_val_inner
    y = bundle.y_val_inner
    return {key: (x.loc[mask], y.loc[mask]) for key, mask in masks.items()}


def _predict_tabpack_splits(bundle: ModalityBundle, artifact: dict) -> dict[str, np.ndarray]:
    """Map TabPack cached preds.npz onto ensemble split names."""
    val_pred = np.asarray(artifact["val_pred"], dtype=np.float64)
    if len(val_pred) != len(bundle.x_val_inner):
        raise ValueError(
            f"tabpack val_pred length {len(val_pred)} != inner_val {len(bundle.x_val_inner)}"
        )

    preds: dict[str, np.ndarray] = {}
    for key, mask in _inner_val_masks(bundle).items():
        preds[key] = clip_nonneg(val_pred[mask])

    outer = artifact["outer_preds"]
    # Cached keys: bulk, k1, pb_K2, …
    preds["outer_val_bulk"] = clip_nonneg(np.asarray(outer["bulk"], dtype=np.float64))
    preds["outer_val_k1"] = clip_nonneg(np.asarray(outer["k1"], dtype=np.float64))
    for cohort in PB_COHORTS:
        preds[f"outer_val_pb_{cohort}"] = clip_nonneg(
            np.asarray(outer[f"pb_{cohort}"], dtype=np.float64)
        )
    return preds


def predict_all_splits(
    bundle: ModalityBundle,
    target: str,
    genes: list[str],
    model_name: str,
) -> dict[str, np.ndarray]:
    if model_name not in BASE_MODELS:
        raise ValueError(f"Unknown base model {model_name!r}")
    if not model_exists(model_name, target):
        raise FileNotFoundError(f"Missing artifact: {model_dir(model_name, target)}")

    artifact = load_artifact(model_name, target)

    if model_name == "tabpack":
        return _predict_tabpack_splits(bundle, artifact)

    preds: dict[str, np.ndarray] = {}

    for key, (x_df, y_df) in _inner_val_frames(bundle).items():
        if len(x_df) == 0:
            preds[key] = np.zeros(0, dtype=np.float64)
            continue
        x, _ = _x_y(x_df, y_df, target, genes)
        preds[key] = predict_one(model_name, artifact, x)

    x, _ = _x_y(bundle.x_outer_val_bulk, bundle.y_outer_val_bulk, target, genes)
    preds["outer_val_bulk"] = predict_one(model_name, artifact, x)

    x, _ = _x_y(bundle.x_outer_val_k1, bundle.y_outer_val_k1, target, genes)
    preds["outer_val_k1"] = predict_one(model_name, artifact, x)

    for cohort in PB_COHORTS:
        key = f"outer_val_pb_{cohort}"
        x, _ = _x_y(bundle.x_outer_val_pb[cohort], bundle.y_outer_val_pb[cohort], target, genes)
        preds[key] = predict_one(model_name, artifact, x)

    return preds


def true_all_splits(bundle: ModalityBundle, target: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, (_x_df, y_df) in _inner_val_frames(bundle).items():
        out[key] = y_df[target].to_numpy(dtype=np.float64) if len(y_df) else np.zeros(0, dtype=np.float64)

    out["outer_val_bulk"] = bundle.y_outer_val_bulk[target].to_numpy(dtype=np.float64)
    out["outer_val_k1"] = bundle.y_outer_val_k1[target].to_numpy(dtype=np.float64)
    for cohort in PB_COHORTS:
        out[f"outer_val_pb_{cohort}"] = bundle.y_outer_val_pb[cohort][target].to_numpy(dtype=np.float64)
    return out
