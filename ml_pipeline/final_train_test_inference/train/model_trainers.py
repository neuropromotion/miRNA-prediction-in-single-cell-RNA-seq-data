"""Train / predict wrappers for final_train base models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from .constants import RESULTS
    from .data import TrainBundle, select_features
except ImportError:
    from final_train_test_inference.train.constants import RESULTS
    from final_train_test_inference.train.data import TrainBundle, select_features

DEVICE = os.environ.get("FINAL_DEVICE", os.environ.get("STAGE04_DEVICE", "cuda"))
BATCH_SIZE = int(os.environ.get("FINAL_BATCH", "512"))


def model_dir(model_name: str, target: str) -> Path:
    return RESULTS / model_name / "models" / target


def model_exists(model_name: str, target: str) -> bool:
    d = model_dir(model_name, target)
    if model_name == "catboost_optuna":
        return (d / "model.cbm").exists()
    if model_name == "tabm":
        return (d / "tabm.pt").exists() and (d / "meta.json").exists()
    return (d / "model.pt").exists() and (d / "meta.json").exists()


def _arrays(bundle: TrainBundle, target: str, genes: list[str]) -> dict:
    return {
        "x_train": select_features(bundle.x_train, genes).to_numpy(dtype=np.float32),
        "y_train": bundle.y_train[target].to_numpy(dtype=np.float64),
        "sw_train": bundle.sw_train,
        "x_val": select_features(bundle.x_val, genes).to_numpy(dtype=np.float32),
        "y_val": bundle.y_val[target].to_numpy(dtype=np.float64),
        "sw_val": bundle.sw_val,
    }


def train_one(model_name: str, bundle: TrainBundle, target: str, genes: list[str]) -> object:
    arr = _arrays(bundle, target, genes)
    out_dir = model_dir(model_name, target)
    if model_name == "catboost_optuna":
        try:
            from .catboost_trainer import train_catboost_optuna
        except ImportError:
            from final_train_test_inference.train.catboost_trainer import train_catboost_optuna

        return train_catboost_optuna(arr, out_dir)
    try:
        from .torch_trainers import train_resnet_model, train_tabm_model
    except ImportError:
        from final_train_test_inference.train.torch_trainers import train_resnet_model, train_tabm_model

    if model_name == "tabm":
        return train_tabm_model(arr, out_dir, DEVICE, BATCH_SIZE)
    if model_name == "resnet":
        return train_resnet_model(arr, out_dir, DEVICE, BATCH_SIZE)
    raise ValueError(f"Unknown model {model_name!r}")


def predict_one(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    if model_name == "catboost_optuna":
        try:
            from .catboost_trainer import predict_catboost
        except ImportError:
            from final_train_test_inference.train.catboost_trainer import predict_catboost

        return predict_catboost(artifact, x)
    try:
        from .torch_trainers import predict_resnet_model, predict_tabm_model
    except ImportError:
        from final_train_test_inference.train.torch_trainers import predict_resnet_model, predict_tabm_model

    if model_name == "tabm":
        return predict_tabm_model(artifact, x, DEVICE)
    return predict_resnet_model(artifact, x)


def load_artifact(model_name: str, target: str):
    d = model_dir(model_name, target)
    if model_name == "catboost_optuna":
        try:
            from .catboost_trainer import load_catboost
        except ImportError:
            from final_train_test_inference.train.catboost_trainer import load_catboost

        return load_catboost(d)
    return d
