"""Dispatch train/predict for final_train base models: tabpack / dcnv2 / tabm."""

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


def _results_root() -> Path:
    """Override with FINAL_MODELS_ROOT (e.g. ../models after publish_models.sh)."""
    env = os.environ.get("FINAL_MODELS_ROOT", "").strip()
    return Path(env) if env else RESULTS


def model_dir(model_name: str, target: str) -> Path:
    return _results_root() / model_name / "models" / target


def model_exists(model_name: str, target: str) -> bool:
    d = model_dir(model_name, target)
    if model_name == "tabpack":
        return (
            (d / "preds.npz").exists()
            and (d / "meta.json").exists()
            and (d / "inference_bundle.pt").exists()
        )
    if model_name == "tabm":
        return (d / "tabm.pt").exists() and (d / "meta.json").exists()
    if model_name == "dcnv2":
        return (d / "model.pt").exists() and (d / "meta.json").exists()
    return False


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
    out_dir = model_dir(model_name, target)

    if model_name == "tabpack":
        try:
            from .tabpack_trainer import train_tabpack_model
        except ImportError:
            from final_train_test_inference.train.tabpack_trainer import train_tabpack_model

        return train_tabpack_model(bundle, target, genes, out_dir)

    arr = _arrays(bundle, target, genes)
    try:
        from .torch_trainers import train_dcnv2_model, train_tabm_model
    except ImportError:
        from final_train_test_inference.train.torch_trainers import (
            train_dcnv2_model,
            train_tabm_model,
        )

    if model_name == "tabm":
        return train_tabm_model(arr, out_dir, DEVICE, BATCH_SIZE)
    if model_name == "dcnv2":
        return train_dcnv2_model(arr, out_dir, DEVICE, BATCH_SIZE)
    raise ValueError(f"Unknown model {model_name!r}; expected tabpack / dcnv2 / tabm")


def load_artifact(model_name: str, target: str):
    d = model_dir(model_name, target)
    if model_name == "tabpack":
        try:
            from .tabpack_trainer import load_tabpack
        except ImportError:
            from final_train_test_inference.train.tabpack_trainer import load_tabpack

        return load_tabpack(d)
    return d


def predict_one(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    """Live predict for tabpack / dcnv2 / tabm."""
    if model_name == "tabpack":
        try:
            from shared.tabpack_trainer import predict_tabpack
        except ImportError as exc:
            raise ImportError("shared.tabpack_trainer.predict_tabpack required") from exc

        if isinstance(artifact, dict):
            md = Path(
                artifact.get("model_dir")
                or model_dir("tabpack", str(artifact["target"]))
            )
        else:
            md = Path(artifact)
        return predict_tabpack(md, x, device=DEVICE)
    try:
        from .torch_trainers import predict_dcnv2_model, predict_tabm_model
    except ImportError:
        from final_train_test_inference.train.torch_trainers import (
            predict_dcnv2_model,
            predict_tabm_model,
        )

    if model_name == "tabm":
        return predict_tabm_model(artifact, x, DEVICE)
    if model_name == "dcnv2":
        return predict_dcnv2_model(artifact, x)
    raise ValueError(f"Unknown model {model_name!r}")


# Re-exports for run_train / run_stack
def tabpack_preds_by_split(artifact: dict, bundle: TrainBundle | None = None):
    try:
        from .tabpack_trainer import preds_by_split
    except ImportError:
        from final_train_test_inference.train.tabpack_trainer import preds_by_split

    return preds_by_split(artifact, bundle)


def eval_tabpack_row(artifact: dict, bundle: TrainBundle, target: str) -> dict:
    try:
        from .tabpack_trainer import eval_row
    except ImportError:
        from final_train_test_inference.train.tabpack_trainer import eval_row

    return eval_row(artifact, bundle, target)
