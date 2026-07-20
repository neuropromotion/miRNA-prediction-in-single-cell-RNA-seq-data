"""DL trainers with modality-weighted loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from metrics import clip_nonneg
from dl_trainers import (
    predict_tabm,
    predict_torch_model,
    train_resnet,
    train_tabm,
)


def train_tabm_model(arr: dict, model_dir: Path, device: str, batch_size: int) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_tabm(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        arr["sw_train"],
        arr["sw_val"],
        model_dir,
        device,
        batch_size,
    )
    return model_dir


def train_resnet_model(arr: dict, model_dir: Path, device: str, batch_size: int) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_resnet(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        arr["sw_train"],
        arr["sw_val"],
        model_dir,
        device,
        batch_size,
    )
    return model_dir


def predict_tabm_model(model_dir: Path, x: np.ndarray, device: str) -> np.ndarray:
    return clip_nonneg(predict_tabm(model_dir, x, device))


def predict_resnet_model(model_dir: Path, x: np.ndarray) -> np.ndarray:
    return clip_nonneg(predict_torch_model(model_dir, x))
