"""Train/predict wrappers for stage03 model screen."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from model_screen.constants import (
    EARLY_STOPPING_ROUNDS,
    OPTUNA_TRIALS,
    SEED,
    XGB_DEFAULT,
)
from model_screen.metrics import clip_nonneg, r2
from shared.data import ModalityBundle, select_features
from shared.paths import PILOT_DIR

sys.path.insert(0, str(PILOT_DIR))

from dl_trainers import (  # noqa: E402
    predict_tabm,
    predict_tabnet,
    predict_torch_model,
    train_dcnv2,
    train_realmlp,
    train_resnet,
    train_tabm,
    train_tabnet,
)

DEVICE = os.environ.get("STAGE03_DEVICE", "cuda")
BATCH_SIZE = int(os.environ.get("STAGE03_BATCH", "512"))
TABNET_EPOCHS = int(os.environ.get("STAGE03_TABNET_EPOCHS", "100"))
TABNET_PATIENCE = int(os.environ.get("STAGE03_TABNET_PATIENCE", "20"))


def _arrays(
    bundle: ModalityBundle,
    target: str,
    genes: list[str],
) -> dict[str, np.ndarray | np.ndarray]:
    sw = bundle.sample_weight
    return {
        "x_train": select_features(bundle.x_train, genes).to_numpy(dtype=np.float32),
        "y_train": bundle.y_train[target].to_numpy(dtype=np.float64),
        "sw": sw,
        "x_val": select_features(bundle.x_val_inner, genes).to_numpy(dtype=np.float32),
        "y_val": bundle.y_val_inner[target].to_numpy(dtype=np.float64),
    }


def _predict_xgb(model: xgb.XGBRegressor, x: np.ndarray) -> np.ndarray:
    return clip_nonneg(model.predict(x))


def train_xgb_default(arr: dict, model_dir: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        **XGB_DEFAULT,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        arr["x_train"],
        arr["y_train"],
        sample_weight=arr["sw"],
        eval_set=[(arr["x_val"], arr["y_val"])],
        verbose=False,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.json"))
    return model


def _suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 3.0, log=True),
    }


def train_xgb_optuna(arr: dict, model_dir: Path) -> xgb.XGBRegressor:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb(trial)
        model = xgb.XGBRegressor(
            **XGB_DEFAULT,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )
        model.fit(
            arr["x_train"],
            arr["y_train"],
            sample_weight=arr["sw"],
            eval_set=[(arr["x_val"], arr["y_val"])],
            verbose=False,
        )
        pred = model.predict(arr["x_val"])
        return r2(arr["y_val"], pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])

    model = xgb.XGBRegressor(
        **XGB_DEFAULT,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best,
    )
    model.fit(
        arr["x_train"],
        arr["y_train"],
        sample_weight=arr["sw"],
        eval_set=[(arr["x_val"], arr["y_val"])],
        verbose=False,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.json"))
    (model_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return model


def train_torch(name: str, arr: dict, model_dir: Path) -> Path:
    fn = {
        "dcnv2": train_dcnv2,
        "realmlp": train_realmlp,
        "resnet": train_resnet,
    }[name]
    model_dir.mkdir(parents=True, exist_ok=True)
    fn(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        model_dir,
        DEVICE,
        BATCH_SIZE,
    )
    return model_dir


def train_tabm_model(arr: dict, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_tabm(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        model_dir,
        DEVICE,
        BATCH_SIZE,
    )
    return model_dir


def train_tabnet_model(arr: dict, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_tabnet(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        model_dir,
        DEVICE,
        BATCH_SIZE,
        TABNET_EPOCHS,
        TABNET_PATIENCE,
    )
    return model_dir


def predict_model(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    if model_name.startswith("xgb"):
        return _predict_xgb(artifact, x)
    if model_name == "tabm":
        return clip_nonneg(predict_tabm(artifact, x, DEVICE))
    if model_name == "tabnet":
        return clip_nonneg(predict_tabnet(artifact, x))
    return clip_nonneg(predict_torch_model(artifact, x))


def load_artifact(model_name: str, model_dir: Path):
    if model_name.startswith("xgb"):
        model = xgb.XGBRegressor()
        model.load_model(str(model_dir / "model.json"))
        return model
    return model_dir


def train_one(
    model_name: str,
    bundle: ModalityBundle,
    target: str,
    genes: list[str],
    model_dir: Path,
):
    arr = _arrays(bundle, target, genes)
    if model_name == "xgb_default":
        return train_xgb_default(arr, model_dir)
    if model_name == "xgb_optuna":
        return train_xgb_optuna(arr, model_dir)
    if model_name in ("dcnv2", "realmlp", "resnet"):
        train_torch(model_name, arr, model_dir)
        return model_dir
    if model_name == "tabm":
        return train_tabm_model(arr, model_dir)
    if model_name == "tabnet":
        return train_tabnet_model(arr, model_dir)
    raise ValueError(f"Unknown model {model_name!r}")
