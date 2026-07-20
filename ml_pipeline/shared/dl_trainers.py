"""Per-architecture trainers for phase-2 solo DL screen."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch import Tensor

from pilot_constants import SEED
from dl_utils import LabelStats, val_rmse

TABM_DIR = Path(__file__).resolve().parent


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_standard_scaler(x_train: np.ndarray) -> sklearn.preprocessing.StandardScaler:
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    return sklearn.preprocessing.StandardScaler().fit(x_train + noise)


def _ft_infer_batch_size(n_features: int, default: int = 2048) -> int:
    # attention memory ~ O(batch * n_features^2); keep peak under ~4GB activations
    return max(8, min(default, 12000 // max(n_features, 1)))


def _train_torch_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
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
    use_ft_fast = arch == "FTTransformer" and dev.type == "cuda"
    if use_ft_fast:
        torch.backends.cudnn.benchmark = True

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)

    x_train_t = preprocessing.transform(x_train).astype(np.float32)
    x_val_t = preprocessing.transform(x_val).astype(np.float32)
    label_stats = LabelStats(mean=float(y_train.mean()), std=float(max(y_train.std(), 1e-6)))
    y_train_z = label_stats.transform(y_train)

    def forward(xb: Tensor) -> Tensor:
        if arch == "FTTransformer":
            return model(xb, None).squeeze(-1).float()
        return model(xb).squeeze(-1).float()

    model = model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    if use_ft_fast:
        x_train_t_tensor = torch.as_tensor(x_train_t, device=dev)
        y_train_t_tensor = torch.as_tensor(y_train_z, device=dev)
        scaler = torch.cuda.amp.GradScaler()
    else:
        x_train_cpu = torch.as_tensor(x_train_t)
        y_train_cpu = torch.as_tensor(y_train_z)
    train_size = len(x_train_t)

    best_state = deepcopy(model.state_dict())
    best_rmse = math.inf
    best_epoch = -1
    remaining_patience = patience

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(train_size, device=dev if use_ft_fast else None)
        for batch_start in range(0, train_size, batch_size):
            idx = perm[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if use_ft_fast:
                xb = x_train_t_tensor[idx]
                yb = y_train_t_tensor[idx]
                with torch.cuda.amp.autocast():
                    pred = forward(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                xb = x_train_cpu[idx].to(dev, non_blocking=True)
                yb = y_train_cpu[idx].to(dev, non_blocking=True)
                pred = forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            del xb, yb, pred, loss

        model.eval()
        pred_val_parts: list[np.ndarray] = []
        with torch.inference_mode():
            val_bs = min(batch_size, 256) if arch == "FTTransformer" else batch_size
            for start in range(0, len(x_val_t), val_bs):
                xb = torch.as_tensor(x_val_t[start : start + val_bs], device=dev)
                if use_ft_fast:
                    with torch.cuda.amp.autocast():
                        pred_val_parts.append(forward(xb).float().cpu().numpy())
                else:
                    pred_val_parts.append(forward(xb).cpu().numpy())
        pred_val = label_stats.inverse(np.concatenate(pred_val_parts))
        rmse = val_rmse(y_val, pred_val)

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
        "device": str(dev),
        "arch": arch if arch != "FTTransformer" else "FTTransformer",
    }
    info = {"best_epoch": best_epoch, "val_rmse": best_rmse, "epochs_ran": epoch + 1}
    return artifacts, info


def _predict_torch_regressor(
    artifacts: dict[str, Any],
    x: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    dev = torch.device(artifacts["device"])
    preprocessing: sklearn.preprocessing.StandardScaler = artifacts["preprocessing"]
    label_stats = LabelStats(**artifacts["label_stats"])
    x = np.asarray(x, dtype=np.float32)
    x_t = preprocessing.transform(x).astype(np.float32)

    arch = artifacts["arch"]
    n_features = x.shape[1]
    hparams = artifacts["model_hparams"]
    if arch == "FTTransformer":
        from rtdl import FTTransformer

        model = FTTransformer.make_baseline(
            n_num_features=n_features,
            cat_cardinalities=None,
            d_out=1,
            **hparams,
        )
        batch_size = _ft_infer_batch_size(n_features, batch_size)
    elif arch == "ResNet":
        from rtdl_revisiting_models import ResNet

        model = ResNet(d_in=n_features, d_out=1, **hparams)
    elif arch == "MLP":
        from rtdl_revisiting_models import MLP as RealMLP

        model = RealMLP(d_in=n_features, d_out=1, **hparams)
    elif arch == "DCNv2":
        from dcnv2_model import DCNv2

        model = DCNv2(d_in=n_features, d_out=1, **hparams)
    else:
        raise ValueError(f"Unknown arch for prediction: {arch}")

    model.load_state_dict(artifacts["model_state"])
    model.to(dev)
    model.eval()

    out: list[np.ndarray] = []
    use_amp = arch == "FTTransformer" and dev.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(x_t), batch_size):
            batch = torch.as_tensor(x_t[start : start + batch_size], device=dev)
            if arch == "FTTransformer":
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(batch, None).squeeze(-1).float()
                else:
                    pred = model(batch, None).squeeze(-1).float()
            else:
                pred = model(batch).squeeze(-1).float()
            out.append(pred.cpu().numpy())
    pred_z = np.concatenate(out)
    return np.clip(label_stats.inverse(pred_z), 0.0, None)


def save_torch_artifacts(model_dir: Path, artifacts: dict[str, Any]) -> None:
    model_dir = Path(model_dir)
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
    }
    with (model_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_torch_artifacts(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
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


def predict_torch_model(model_dir: Path, x: np.ndarray) -> np.ndarray:
    artifacts = load_torch_artifacts(model_dir)
    return _predict_torch_regressor(artifacts, x)


def train_fttransformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    from rtdl import FTTransformer

    n_features = x_train.shape[1]
    hparams = {
        "d_token": 32,
        "n_blocks": 2,
        "attention_dropout": 0.2,
        "ffn_d_hidden": 64,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.0,
    }
    model = FTTransformer.make_baseline(
        n_num_features=n_features,
        cat_cardinalities=None,
        d_out=1,
        **hparams,
    )
    preprocessing = _fit_standard_scaler(x_train)
    artifacts, info = _train_torch_regressor(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        preprocessing,
        device=device,
        arch="FTTransformer",
        batch_size=min(batch_size, 64),
        lr=1e-3,
    )
    artifacts["model_hparams"] = hparams
    save_torch_artifacts(model_dir, artifacts)
    return info


# Stubs for imports used by model_trainers when training other models
def train_resnet(*args, **kwargs):
    from rtdl_revisiting_models import ResNet

    x_train, y_train, x_val, y_val, model_dir, device, batch_size = args[:7]
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
        model, x_train, y_train, x_val, y_val, preprocessing,
        device=device, arch="ResNet", batch_size=batch_size,
    )
    artifacts["model_hparams"] = hparams
    save_torch_artifacts(model_dir, artifacts)
    return info


def train_realmlp(*args, **kwargs):
    from rtdl_revisiting_models import MLP as RealMLP

    x_train, y_train, x_val, y_val, model_dir, device, batch_size = args[:7]
    n_features = x_train.shape[1]
    hparams = {"n_blocks": 5, "d_block": 256, "dropout": 0.1}
    model = RealMLP(d_in=n_features, d_out=1, **hparams)
    preprocessing = _fit_standard_scaler(x_train)
    artifacts, info = _train_torch_regressor(
        model, x_train, y_train, x_val, y_val, preprocessing,
        device=device, arch="MLP", batch_size=batch_size,
    )
    artifacts["model_hparams"] = hparams
    save_torch_artifacts(model_dir, artifacts)
    return info


def train_dcnv2(*args, **kwargs):
    from dcnv2_model import DCNv2

    x_train, y_train, x_val, y_val, model_dir, device, batch_size = args[:7]
    n_features = x_train.shape[1]
    low_rank = min(32, max(8, n_features // 4))
    hparams = {
        "n_cross_layers": 3,
        "cross_low_rank": low_rank,
        "d_deep": 256,
        "n_deep_layers": 3,
        "dropout": 0.1,
    }
    model = DCNv2(d_in=n_features, d_out=1, **hparams)
    preprocessing = _fit_standard_scaler(x_train)
    artifacts, info = _train_torch_regressor(
        model, x_train, y_train, x_val, y_val, preprocessing,
        device=device, arch="DCNv2", batch_size=batch_size,
    )
    artifacts["model_hparams"] = hparams
    save_torch_artifacts(model_dir, artifacts)
    return info


def train_tabm(*args, **kwargs):
    sys.path.insert(0, str(TABM_DIR))
    from tabm_wrapper import train_tabm_regressor

    x_train, y_train, x_val, y_val, model_dir, device, batch_size = args[:7]
    bundle, info = train_tabm_regressor(
        x_train, y_train, x_val, y_val, device=device, batch_size=batch_size,
    )
    bundle.save(model_dir)
    return info


def predict_tabm(model_dir: Path, x: np.ndarray, device: str) -> np.ndarray:
    sys.path.insert(0, str(TABM_DIR))
    from tabm_wrapper import TabMBundle

    bundle = TabMBundle.load(model_dir, device=device)
    return bundle.predict(x)


def train_tabnet(*args, **kwargs):
    from pytorch_tabnet.tab_model import TabNetRegressor

    x_train, y_train, x_val, y_val, model_dir, device, batch_size = args[:7]
    max_epochs = args[7] if len(args) > 7 else 100
    patience = args[8] if len(args) > 8 else 20
    n_features = x_train.shape[1]
    width = min(64, max(16, n_features // 8))
    model = TabNetRegressor(
        n_d=width, n_a=width, n_steps=5, gamma=1.5, lambda_sparse=1e-3,
        optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", seed=SEED, verbose=0,
        device_name=device if device == "cuda" and torch.cuda.is_available() else "cpu",
    )
    model.fit(
        x_train.astype(np.float32),
        y_train.reshape(-1, 1).astype(np.float32),
        eval_set=[(x_val.astype(np.float32), y_val.reshape(-1, 1).astype(np.float32))],
        eval_metric=["rmse"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=min(256, batch_size),
        num_workers=0,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir))
    return model


def predict_tabnet(model_dir: Path, x: np.ndarray) -> np.ndarray:
    from pytorch_tabnet.tab_model import TabNetRegressor

    zip_path = model_dir.with_suffix(".zip") if model_dir.suffix != ".zip" else model_dir
    model = TabNetRegressor()
    model.load_model(str(zip_path))
    return model.predict(x.astype(np.float32)).reshape(-1)
