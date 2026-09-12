"""TabM trainer with modality-weighted loss (final_train)."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn.preprocessing
import torch
from torch import Tensor

try:
    from .constants import SEED, TABM_DIR
    from .metrics import weighted_rmse
except ImportError:
    from final_train_test_inference.train.constants import SEED, TABM_DIR
    from final_train_test_inference.train.metrics import weighted_rmse


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


def _fit_noisy_quantile(x_train: np.ndarray) -> sklearn.preprocessing.QuantileTransformer:
    """Numerical policy for TabM (noisy quantile → N(0,1))."""
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    n_quantiles = max(min(len(x_train) // 30, 1000), 10)
    return sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=10**9,
    ).fit(x_train + noise)


_make_tabm_preprocessing = _fit_noisy_quantile


def _weighted_mse(pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    w = weight / weight.sum().clamp_min(1e-12)
    return (w * (pred - target) ** 2).sum()


def _tabm_weighted_loss(y_pred: Tensor, y_true: Tensor, k: int, weight: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    y_true = y_true.repeat_interleave(k)
    w = weight.repeat_interleave(k)
    return _weighted_mse(y_pred, y_true, w)


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
    epoch = -1

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
