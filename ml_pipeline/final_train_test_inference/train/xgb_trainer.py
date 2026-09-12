"""XGB Optuna trainer for final_train (recipe from 02_model_selection)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

try:
    from .constants import EARLY_STOPPING_ROUNDS, OPTUNA_TRIALS, SEED
    from .metrics import clip_nonneg, r2
except ImportError:
    from final_train_test_inference.train.constants import (
        EARLY_STOPPING_ROUNDS,
        OPTUNA_TRIALS,
        SEED,
    )
    from final_train_test_inference.train.metrics import clip_nonneg, r2

XGB_DEFAULT = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}


def _xgb_base_params() -> dict:
    params = dict(XGB_DEFAULT)
    xgb_device = os.environ.get("XGB_DEVICE", "cpu").strip().lower()
    if xgb_device in {"cuda", "gpu"}:
        params["device"] = "cuda"
        params["tree_method"] = "hist"
        params["n_jobs"] = 1
    return params


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
    """arr: x_train, y_train, sw_train, x_val, y_val (final_train style)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = _xgb_base_params()
    sw = arr["sw_train"]

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
            sample_weight=sw,
            eval_set=[(arr["x_val"], arr["y_val"])],
            verbose=False,
        )
        return r2(arr["y_val"], model.predict(arr["x_val"]))

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
        sample_weight=sw,
        eval_set=[(arr["x_val"], arr["y_val"])],
        verbose=False,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.json"))
    (model_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": "xgb_optuna",
                "optuna_trials": OPTUNA_TRIALS,
                "xgb_device": os.environ.get("XGB_DEVICE", "cpu"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def load_xgb(model_dir: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(Path(model_dir) / "model.json"))
    return model


def predict_xgb(model: xgb.XGBRegressor, x: np.ndarray) -> np.ndarray:
    return clip_nonneg(model.predict(x))


def xgb_exists(model_dir: Path) -> bool:
    d = Path(model_dir)
    return (d / "model.json").is_file() and (d / "meta.json").is_file()
