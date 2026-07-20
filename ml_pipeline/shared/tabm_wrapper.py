"""TabM training/inference wrapper for per-target miRNA regression."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
from torch import Tensor

SEED = 42


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


@dataclass
class TabMBundle:
    model: nn.Module
    preprocessing: sklearn.preprocessing.QuantileTransformer
    label_stats: LabelStats
    device: torch.device
    k: int
    n_num_features: int
    arch_type: str = "tabm-mini"
    use_embeddings: bool = False

    def predict(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x_t = self.preprocessing.transform(x)
        out: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(x_t), batch_size):
                batch = torch.as_tensor(x_t[start : start + batch_size], device=self.device)
                pred = self.model(batch).squeeze(-1).float().mean(dim=1)
                out.append(pred.cpu().numpy())
        pred_z = np.concatenate(out)
        return np.clip(self.label_stats.inverse(pred_z), 0.0, None)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "tabm.pt")
        joblib.dump(self.preprocessing, model_dir / "preprocessing.joblib")
        meta = {
            "k": self.k,
            "n_num_features": self.n_num_features,
            "label_stats": self.label_stats.to_dict(),
            "device": str(self.device),
            "arch_type": getattr(self, "arch_type", "tabm-mini"),
            "use_embeddings": getattr(self, "use_embeddings", False),
        }
        with (model_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path, device: str | None = None) -> TabMBundle:
        with (model_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        dev = torch.device(device or meta.get("device", "cpu"))
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")

        n_num_features = int(meta["n_num_features"])
        arch_type = meta.get("arch_type", "tabm-mini")
        use_embeddings = bool(meta.get("use_embeddings", False))

        state = torch.load(model_dir / "tabm.pt", map_location=dev)
        if "use_embeddings" not in meta:
            use_embeddings = any(k.startswith("num_module") for k in state)
            if use_embeddings:
                arch_type = "tabm"

        num_embeddings = None
        if use_embeddings:
            import rtdl_num_embeddings

            num_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_num_features)

        model = tabm.TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=[],
            d_out=1,
            num_embeddings=num_embeddings,
            arch_type=arch_type,
        )
        model.load_state_dict(state)
        model.to(dev)
        model.eval()

        preprocessing = joblib.load(model_dir / "preprocessing.joblib")
        return cls(
            model=model,
            preprocessing=preprocessing,
            label_stats=LabelStats.from_dict(meta["label_stats"]),
            device=dev,
            k=int(meta["k"]),
            n_num_features=n_num_features,
            arch_type=arch_type,
            use_embeddings=use_embeddings,
        )


def _make_preprocessing(x_train: np.ndarray) -> sklearn.preprocessing.QuantileTransformer:
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    n_quantiles = max(min(len(x_train) // 30, 1000), 10)
    return sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=10**9,
    ).fit(x_train + noise)


def _tabm_loss(y_pred: Tensor, y_true: Tensor, k: int) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    y_true = y_true.repeat_interleave(k)
    return nn.functional.mse_loss(y_pred, y_true)


def _val_rmse(bundle: TabMBundle, x_val: np.ndarray, y_val: np.ndarray) -> float:
    pred = bundle.predict(x_val)
    return float(np.sqrt(np.mean((y_val - pred) ** 2)))


def train_tabm_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cuda",
    batch_size: int = 4096,
    patience: int = 20,
    max_epochs: int = 200,
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
    gradient_clipping_norm: float = 1.0,
) -> tuple[TabMBundle, dict[str, Any]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)

    preprocessing = _make_preprocessing(x_train)
    x_train_t = preprocessing.transform(x_train).astype(np.float32)
    x_val_t = preprocessing.transform(x_val).astype(np.float32)

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
    train_size = len(x_train_t)
    k = int(model.backbone.k)

    x_train_cpu = torch.as_tensor(x_train_t)
    y_train_cpu = torch.as_tensor(y_train_z)

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
            pred = model(xb).squeeze(-1).float()
            loss = _tabm_loss(pred, yb, k)
            loss.backward()
            if gradient_clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
            optimizer.step()
            del xb, yb, pred, loss

        bundle = TabMBundle(
            model=model,
            preprocessing=preprocessing,
            label_stats=label_stats,
            device=dev,
            k=k,
            n_num_features=n_num_features,
        )
        val_rmse = _val_rmse(bundle, x_val, y_val)
        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
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
    info = {
        "best_epoch": best_epoch,
        "val_rmse": best_rmse,
        "device": str(dev),
        "k": k,
        "epochs_ran": epoch + 1,
    }
    return final_bundle, info


def load_tabm(model_dir: Path, device: str | None = None) -> TabMBundle | None:
    if not (model_dir / "tabm.pt").exists():
        return None
    return TabMBundle.load(model_dir, device=device)


def predict_tabm(bundle: TabMBundle, x: np.ndarray) -> np.ndarray:
    return bundle.predict(x)
