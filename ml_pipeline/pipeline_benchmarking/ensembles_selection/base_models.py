"""Load stage03 base models and produce split-wise predictions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    from .constants import BASE_MODELS, ML_PIPELINE, PILOT_DIR, STAGE03_RESULTS
    from .metrics import clip_nonneg
except ImportError:
    from constants import BASE_MODELS, ML_PIPELINE, PILOT_DIR, STAGE03_RESULTS
    from metrics import clip_nonneg

if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from dl_trainers import predict_tabm, predict_torch_model  # noqa: E402
from shared.data import ModalityBundle, select_features  # noqa: E402
from shared.io_splits import PB_COHORTS  # noqa: E402

DEVICE = os.environ.get("STAGE04_DEVICE", os.environ.get("STAGE03_DEVICE", "cuda"))


def model_dir(model_name: str, target: str) -> Path:
    return STAGE03_RESULTS / model_name / "models" / target


def model_exists(model_name: str, target: str) -> bool:
    d = model_dir(model_name, target)
    if model_name.startswith("xgb"):
        return (d / "model.json").exists()
    if model_name == "catboost_optuna":
        return (d / "model.cbm").exists()
    if model_name == "tabm":
        return (d / "tabm.pt").exists()
    return (d / "model.pt").exists() and (d / "meta.json").exists()


def load_artifact(model_name: str, target: str):
    d = model_dir(model_name, target)
    if model_name.startswith("xgb"):
        import xgboost as xgb

        m = xgb.XGBRegressor()
        m.load_model(str(d / "model.json"))
        return m
    if model_name == "catboost_optuna":
        from catboost import CatBoostRegressor

        m = CatBoostRegressor()
        m.load_model(str(d / "model.cbm"))
        return m
    return d


def predict_one(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    if model_name.startswith("xgb"):
        return clip_nonneg(artifact.predict(x))
    if model_name == "catboost_optuna":
        return clip_nonneg(artifact.predict(x))
    if model_name == "tabm":
        return clip_nonneg(predict_tabm(artifact, x, DEVICE))
    return clip_nonneg(predict_torch_model(artifact, x))


def _x_y(bundle: ModalityBundle, target: str, genes: list[str], x_df, y_df) -> tuple[np.ndarray, np.ndarray]:
    x = select_features(x_df, genes).to_numpy(dtype=np.float32)
    y = y_df[target].to_numpy(dtype=np.float64)
    return x, y


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
    preds: dict[str, np.ndarray] = {}

    x, _ = _x_y(bundle, target, genes, bundle.x_val_inner, bundle.y_val_inner)
    preds["inner_val"] = predict_one(model_name, artifact, x)

    x, _ = _x_y(bundle, target, genes, bundle.x_outer_val_bulk, bundle.y_outer_val_bulk)
    preds["outer_val_bulk"] = predict_one(model_name, artifact, x)

    x, _ = _x_y(bundle, target, genes, bundle.x_outer_val_k1, bundle.y_outer_val_k1)
    preds["outer_val_k1"] = predict_one(model_name, artifact, x)

    for cohort in PB_COHORTS:
        key = f"outer_val_pb_{cohort}"
        x, _ = _x_y(bundle, target, genes, bundle.x_outer_val_pb[cohort], bundle.y_outer_val_pb[cohort])
        preds[key] = predict_one(model_name, artifact, x)

    return preds


def true_all_splits(bundle: ModalityBundle, target: str) -> dict[str, np.ndarray]:
    out = {
        "inner_val": bundle.y_val_inner[target].to_numpy(dtype=np.float64),
        "outer_val_bulk": bundle.y_outer_val_bulk[target].to_numpy(dtype=np.float64),
        "outer_val_k1": bundle.y_outer_val_k1[target].to_numpy(dtype=np.float64),
    }
    for cohort in PB_COHORTS:
        out[f"outer_val_pb_{cohort}"] = bundle.y_outer_val_pb[cohort][target].to_numpy(dtype=np.float64)
    return out
