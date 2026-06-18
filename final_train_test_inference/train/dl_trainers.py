"""ResNet + TabM trainers with modality-weighted loss (final_train)."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from dataclasses import dataclass

import joblib
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch import Tensor

from final_train.constants import SEED, TABM_DIR
from final_train.shared.metrics import weighted_rmse


@dataclass
class LabelStats:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std + self.mean).astype(np.float64)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> LabelStats:
        return cls(mean=float(d["mean"]), std=float(d["std"]))


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_standard_scaler(x_train: np.ndarray) -> sklearn.preprocessing.StandardScaler:
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    return sklearn.preprocessing.StandardScaler().fit(x_train + noise)


def _make_tabm_preprocessing(x_train: np.ndarray) -> sklearn.preprocessing.QuantileTransformer:
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    n_quantiles = max(min(len(x_train) // 30, 1000), 10)
    return sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=10**9,
    ).fit(x_train + noise)


def _weighted_mse(pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    w = weight / weight.sum().clamp_min(1e-12)
    return (w * (pred - target) ** 2).sum()


def _tabm_weighted_loss(y_pred: Tensor, y_true: Tensor, k: int, weight: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    y_true = y_true.repeat_interleave(k)
    w = weight.repeat_interleave(k)
    return _weighted_mse(y_pred, y_true, w)


def _train_torch_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sw_train: np.ndarray,
    sw_val: np.ndarray,
    preprocessing: sklearn.preprocessing.StandardScaler,
    device: str,
    arch: str,
    batch_size: int = 512,
    patience: int = 20,
    max_epochs: int = 200,
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
) -> tuple[dict[str, Any], dict[str, float]]:
    dev = _resolve_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
    sw_train = np.asarray(sw_train, dtype=np.float64).reshape(-1)
    sw_val = np.asarray(sw_val, dtype=np.float64).reshape(-1)

    x_train_t = preprocessing.transform(x_train).astype(np.float32)
    x_val_t = preprocessing.transform(x_val).astype(np.float32)
    label_stats = LabelStats(mean=float(y_train.mean()), std=float(max(y_train.std(), 1e-6)))
    y_train_z = label_stats.transform(y_train)

    def forward(xb: Tensor) -> Tensor:
        return model(xb).squeeze(-1).float()

    model = model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train_cpu = torch.as_tensor(x_train_t)
    y_train_cpu = torch.as_tensor(y_train_z)
    sw_train_cpu = torch.as_tensor(sw_train, dtype=torch.float32)
    train_size = len(x_train_t)

    best_state = deepcopy(model.state_dict())
    best_rmse = math.inf
    best_epoch = -1
    remaining_patience = patience

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(train_size)
        for batch_start in range(0, train_size, batch_size):
            idx = perm[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            xb = x_train_cpu[idx].to(dev, non_blocking=True)
            yb = y_train_cpu[idx].to(dev, non_blocking=True)
            wb = sw_train_cpu[idx].to(dev, non_blocking=True)
            pred = forward(xb)
            loss = _weighted_mse(pred, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        pred_val_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(x_val_t), batch_size):
                xb = torch.as_tensor(x_val_t[start : start + batch_size], device=dev)
                pred_val_parts.append(forward(xb).cpu().numpy())
        pred_val = label_stats.inverse(np.concatenate(pred_val_parts))
        rmse = weighted_rmse(y_val, pred_val, sw_val)

        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience < 0:
                break
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(best_state)
    artifacts = {
        "model_state": best_state,
        "preprocessing": preprocessing,
        "label_stats": {"mean": label_stats.mean, "std": label_stats.std},
        "best_epoch": best_epoch,
        "val_rmse": best_rmse,
        "val_metric": "weighted_rmse",
        "device": str(dev),
        "arch": arch,
    }
    info = {"best_epoch": best_epoch, "val_rmse": best_rmse, "epochs_ran": epoch + 1}
    return artifacts, info


def save_torch_artifacts(model_dir: Path, artifacts: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts["model_state"], model_dir / "model.pt")
    joblib.dump(artifacts["preprocessing"], model_dir / "preprocessing.joblib")
    meta = {
        "label_stats": artifacts["label_stats"],
        "device": artifacts["device"],
        "arch": artifacts["arch"],
        "model_hparams": artifacts["model_hparams"],
        "best_epoch": artifacts["best_epoch"],
        "val_rmse": artifacts["val_rmse"],
        "val_metric": artifacts.get("val_metric", "rmse"),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_torch_artifacts(model_dir: Path) -> dict[str, Any]:
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "model_state": torch.load(model_dir / "model.pt", map_location="cpu"),
        "preprocessing": joblib.load(model_dir / "preprocessing.joblib"),
        "label_stats": meta["label_stats"],
        "device": meta["device"],
        "arch": meta["arch"],
        "model_hparams": meta["model_hparams"],
        "best_epoch": meta["best_epoch"],
        "val_rmse": meta["val_rmse"],
    }


def _predict_torch_regressor(artifacts: dict[str, Any], x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    dev = torch.device(artifacts["device"])
    preprocessing: sklearn.preprocessing.StandardScaler = artifacts["preprocessing"]
    label_stats = LabelStats(**artifacts["label_stats"])
    x_t = preprocessing.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)

    arch = artifacts["arch"]
    hparams = artifacts["model_hparams"]
    if arch == "ResNet":
        from rtdl_revisiting_models import ResNet

        model = ResNet(d_in=x.shape[1], d_out=1, **hparams)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(artifacts["model_state"])
    model.to(dev)
    model.eval()

    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x_t), batch_size):
            batch = torch.as_tensor(x_t[start : start + batch_size], device=dev)
            out.append(model(batch).squeeze(-1).float().cpu().numpy())
    return np.clip(label_stats.inverse(np.concatenate(out)), 0.0, None)


def predict_torch_model(model_dir: Path, x: np.ndarray) -> np.ndarray:
    return _predict_torch_regressor(load_torch_artifacts(model_dir), x)


def train_resnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sw_train: np.ndarray,
    sw_val: np.ndarray,
    model_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    from rtdl_revisiting_models import ResNet

    n_features = x_train.shape[1]
    hparams = {
        "n_blocks": 4,
        "d_block": 256,
        "d_hidden_multiplier": 2.0,
        "dropout1": 0.1,
        "dropout2": 0.1,
    }
    model = ResNet(d_in=n_features, d_out=1, **hparams)
    preprocessing = _fit_standard_scaler(x_train)
    artifacts, info = _train_torch_regressor(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        sw_train,
        sw_val,
        preprocessing,
        device=device,
        arch="ResNet",
        batch_size=batch_size,
    )
    artifacts["model_hparams"] = hparams
    save_torch_artifacts(model_dir, artifacts)
    return info


def _tabm_bundle_cls():
    sys.path.insert(0, str(TABM_DIR))
    from tabm_wrapper import TabMBundle

    return TabMBundle


def train_tabm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sw_train: np.ndarray,
    sw_val: np.ndarray,
    model_dir: Path,
    device: str,
    batch_size: int,
    patience: int = 20,
    max_epochs: int = 200,
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
) -> dict[str, float]:
    import tabm

    TabMBundle = _tabm_bundle_cls()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = _resolve_device(device)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
    sw_train = np.asarray(sw_train, dtype=np.float64).reshape(-1)
    sw_val = np.asarray(sw_val, dtype=np.float64).reshape(-1)

    preprocessing = _make_tabm_preprocessing(x_train)
    x_train_t = preprocessing.transform(x_train).astype(np.float32)
    label_stats = LabelStats(mean=float(y_train.mean()), std=float(max(y_train.std(), 1e-6)))
    y_train_z = label_stats.transform(y_train)

    n_num_features = x_train.shape[1]
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=1,
        arch_type="tabm-mini",
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    k = int(model.backbone.k)
    train_size = len(x_train_t)

    x_train_cpu = torch.as_tensor(x_train_t)
    y_train_cpu = torch.as_tensor(y_train_z)
    sw_train_cpu = torch.as_tensor(sw_train, dtype=torch.float32)

    best_state = deepcopy(model.state_dict())
    best_rmse = math.inf
    best_epoch = -1
    remaining_patience = patience

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(train_size)
        for batch_start in range(0, train_size, batch_size):
            idx = perm[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            xb = x_train_cpu[idx].to(dev, non_blocking=True)
            yb = y_train_cpu[idx].to(dev, non_blocking=True)
            wb = sw_train_cpu[idx].to(dev, non_blocking=True)
            pred = model(xb).squeeze(-1).float()
            loss = _tabm_weighted_loss(pred, yb, k, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bundle = TabMBundle(
            model=model,
            preprocessing=preprocessing,
            label_stats=label_stats,
            device=dev,
            k=k,
            n_num_features=n_num_features,
        )
        pred_val = bundle.predict(x_val)
        rmse = weighted_rmse(y_val, pred_val, sw_val)
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience < 0:
                break
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(best_state)
    final_bundle = TabMBundle(
        model=model,
        preprocessing=preprocessing,
        label_stats=label_stats,
        device=dev,
        k=k,
        n_num_features=n_num_features,
        arch_type="tabm-mini",
        use_embeddings=False,
    )
    final_bundle.save(model_dir)
    return {
        "best_epoch": best_epoch,
        "val_rmse": best_rmse,
        "val_metric": "weighted_rmse",
        "device": str(dev),
        "k": k,
        "epochs_ran": epoch + 1,
    }


def predict_tabm(model_dir: Path, x: np.ndarray, device: str) -> np.ndarray:
    TabMBundle = _tabm_bundle_cls()
    bundle = TabMBundle.load(model_dir, device=device)
    return bundle.predict(x)
