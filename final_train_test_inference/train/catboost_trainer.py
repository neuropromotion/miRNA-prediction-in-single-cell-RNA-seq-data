"""CatBoost + Optuna trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

from final_train.constants import EARLY_STOPPING_ROUNDS, OPTUNA_TRIALS, SEED
from final_train.shared.metrics import clip_nonneg, weighted_r2

CATBOOST_TASK = os.environ.get("CATBOOST_TASK", "CPU")


def _suggest_catboost(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 300, 1200),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def train_catboost_optuna(arr: dict, model_dir: Path) -> CatBoostRegressor:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_pool = Pool(arr["x_train"], arr["y_train"], weight=arr["sw_train"])
    val_pool = Pool(arr["x_val"], arr["y_val"], weight=arr["sw_val"])

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_catboost(trial)
        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=SEED,
            task_type=CATBOOST_TASK,
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        pred = model.predict(arr["x_val"])
        return weighted_r2(arr["y_val"], pred, arr["sw_val"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best["iterations"] = int(best["iterations"])
    best["depth"] = int(best["depth"])

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=SEED,
        task_type=CATBOOST_TASK,
        verbose=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best,
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.cbm"))
    (model_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return model


def predict_catboost(model: CatBoostRegressor, x: np.ndarray) -> np.ndarray:
    return clip_nonneg(model.predict(x))


def load_catboost(model_dir: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(model_dir / "model.cbm"))
    return model
