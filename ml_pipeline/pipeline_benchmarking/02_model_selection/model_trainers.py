"""Train/predict wrappers for stage03 model screen."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from model_screen_final_11.constants import (
    EARLY_STOPPING_ROUNDS,
    OPTUNA_TRIALS,
    SEED,
    XGB_DEFAULT,
)
from model_screen_final_11.candidate_trainers import (
    load_catboost,
    load_gandalf,
    load_lassonet,
    predict_catboost,
    predict_gandalf,
    predict_lassonet,
    train_catboost_optuna,
    train_gandalf,
    train_lassonet,
)
from model_screen_final_11.metrics import clip_nonneg, r2
from shared.data import ModalityBundle, select_features
from shared.paths import PILOT_DIR

sys.path.insert(0, str(PILOT_DIR))

from dl_trainers import (  # noqa: E402
    predict_tabm,
    predict_tabnet,
    predict_torch_model,
    train_dcnv2,
    train_fttransformer,
    train_realmlp,
    train_resnet,
    train_tabm,
    train_tabnet,
)
from shared.tabr_trainer import (  # noqa: E402
    load_tabr,
    predict_tabr,
    save_tabr,
    train_tabr,
)
from shared.tabpack_trainer import load_tabpack_screen, train_tabpack_screen  # noqa: E402
from shared.io_splits import PB_COHORTS  # noqa: E402

DEVICE = os.environ.get("STAGE03_DEVICE", "cuda")
BATCH_SIZE = int(os.environ.get("STAGE03_BATCH", "512"))
TABNET_EPOCHS = int(os.environ.get("STAGE03_TABNET_EPOCHS", "100"))
TABNET_PATIENCE = int(os.environ.get("STAGE03_TABNET_PATIENCE", "20"))


def _xgb_base_params() -> dict:
    """XGB defaults; set XGB_DEVICE=cuda (or STAGE03_DEVICE=cuda with XGB_DEVICE unset→cpu for trees).

    Explicit: XGB_DEVICE=cuda|cpu. Default remains CPU hist unless XGB_DEVICE=cuda.
    """
    params = dict(XGB_DEFAULT)
    xgb_device = os.environ.get("XGB_DEVICE", "cpu").strip().lower()
    if xgb_device in {"cuda", "gpu"}:
        # XGBoost>=2.0: device=cuda + tree_method=hist
        params["device"] = "cuda"
        params["tree_method"] = "hist"
        params["n_jobs"] = 1
    return params


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
        **_xgb_base_params(),
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
    base = _xgb_base_params()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb(trial)
        model = xgb.XGBRegressor(
            **base,
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
        **base,
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
        "fttransformer": train_fttransformer,
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


def _outer_parts(bundle: ModalityBundle, target: str, genes: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    parts: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "bulk",
            select_features(bundle.x_outer_val_bulk, genes).to_numpy(dtype=np.float32),
            bundle.y_outer_val_bulk[target].to_numpy(dtype=np.float64),
        ),
        (
            "k1",
            select_features(bundle.x_outer_val_k1, genes).to_numpy(dtype=np.float32),
            bundle.y_outer_val_k1[target].to_numpy(dtype=np.float64),
        ),
    ]
    for cohort in PB_COHORTS:
        parts.append(
            (
                f"pb_{cohort}",
                select_features(bundle.x_outer_val_pb[cohort], genes).to_numpy(dtype=np.float32),
                bundle.y_outer_val_pb[cohort][target].to_numpy(dtype=np.float64),
            )
        )
    return parts


def train_tabr_model(arr: dict, model_dir: Path):
    model, _info = train_tabr(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        model_dir=model_dir,
        device=DEVICE,
        variant="tabr",
    )
    try:
        save_tabr(model, model_dir)
    except Exception:
        # Metrics can still be computed from in-memory model; dump is best-effort.
        pass
    return model


def train_tabpack_model(bundle: ModalityBundle, target: str, genes: list[str], model_dir: Path) -> dict:
    arr = _arrays(bundle, target, genes)
    return train_tabpack_screen(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        _outer_parts(bundle, target, genes),
        model_dir=model_dir,
        target=target,
    )


def predict_model(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    if model_name.startswith("xgb"):
        return _predict_xgb(artifact, x)
    if model_name == "catboost_optuna":
        return predict_catboost(artifact, x)
    if model_name == "lassonet":
        return predict_lassonet(artifact, x)
    if model_name == "gandalf":
        return predict_gandalf(artifact, x)
    if model_name == "tabm":
        return clip_nonneg(predict_tabm(artifact, x, DEVICE))
    if model_name == "tabnet":
        return clip_nonneg(predict_tabnet(artifact, x))
    if model_name == "tabr":
        return clip_nonneg(predict_tabr(artifact, x))
    if model_name == "tabpack":
        raise TypeError("tabpack uses cached outer preds; call eval_tabpack_row()")
    return clip_nonneg(predict_torch_model(artifact, x))


def eval_tabpack_row(artifact: dict, y_val: np.ndarray) -> dict:
    """Build metrics row pieces from TabPack cached predictions."""
    row = {"inner_val_r2": r2(y_val, artifact["val_pred"]), "status": "ok"}
    for name, pred in artifact["outer_preds"].items():
        # y is not stored in artifact; caller fills outer_val_* via true labels
        row[f"_pred_{name}"] = pred
    return row


def load_artifact(model_name: str, model_dir: Path):
    if model_name.startswith("xgb"):
        model = xgb.XGBRegressor()
        model.load_model(str(model_dir / "model.json"))
        return model
    if model_name == "catboost_optuna":
        return load_catboost(model_dir)
    if model_name == "lassonet":
        return load_lassonet(model_dir)
    if model_name == "gandalf":
        return load_gandalf(model_dir)
    if model_name == "tabr":
        return load_tabr(model_dir)
    if model_name == "tabpack":
        return load_tabpack_screen(model_dir)
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
    if model_name == "catboost_optuna":
        return train_catboost_optuna(arr, model_dir)
    if model_name == "lassonet":
        return train_lassonet(arr, model_dir)
    if model_name == "gandalf":
        return train_gandalf(arr, model_dir, DEVICE, BATCH_SIZE)
    if model_name in ("dcnv2", "realmlp", "resnet", "fttransformer"):
        train_torch(model_name, arr, model_dir)
        return model_dir
    if model_name == "tabm":
        return train_tabm_model(arr, model_dir)
    if model_name == "tabnet":
        return train_tabnet_model(arr, model_dir)
    if model_name == "tabr":
        return train_tabr_model(arr, model_dir)
    if model_name == "tabpack":
        return train_tabpack_model(bundle, target, genes, model_dir)
    raise ValueError(f"Unknown model {model_name!r}")
