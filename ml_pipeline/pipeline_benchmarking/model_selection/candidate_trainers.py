"""LassoNet, GANDALF, CatBoost+Optuna trainers for stage03 screen."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler

from model_screen_final_11.constants import EARLY_STOPPING_ROUNDS, OPTUNA_TRIALS, SEED
from model_screen_final_11.metrics import clip_nonneg, r2

GANDALF_EPOCHS = int(__import__("os").environ.get("GANDALF_MAX_EPOCHS", "50"))
GANDALF_PATIENCE = int(__import__("os").environ.get("GANDALF_PATIENCE", "10"))
GANDALF_STAGES = int(__import__("os").environ.get("GANDALF_STAGES", "6"))
LASSONET_PATIENCE = int(__import__("os").environ.get("LASSONET_PATIENCE", "15"))
CATBOOST_TASK = __import__("os").environ.get("CATBOOST_TASK", "CPU")
# CatBoost GPU OOMs / segfaults on wide feature matrices (~800+); fall back to CPU.
CATBOOST_GPU_MAX_FEATURES = int(__import__("os").environ.get("CATBOOST_GPU_MAX_FEATURES", "400"))


def _resolve_catboost_task(n_features: int) -> str:
    wanted = (CATBOOST_TASK or "CPU").strip().upper()
    if wanted != "GPU":
        return "CPU"
    if n_features > CATBOOST_GPU_MAX_FEATURES:
        return "CPU"
    return "GPU"


def _fit_scaler(x_train: np.ndarray) -> StandardScaler:
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    return StandardScaler().fit(x_train + noise)


def train_lassonet(arr: dict, model_dir: Path) -> dict:
    from lassonet import LassoNetRegressor

    scaler = _fit_scaler(arr["x_train"])
    x_tr = scaler.transform(arr["x_train"]).astype(np.float32)
    x_va = scaler.transform(arr["x_val"]).astype(np.float32)
    model = LassoNetRegressor(
        hidden_dims=(128, 64),
        verbose=0,
        patience=LASSONET_PATIENCE,
        backtrack=True,
    )
    path = model.path(
        x_tr,
        arr["y_train"],
        X_val=x_va,
        y_val=arr["y_val"],
        return_state_dicts=True,
    )
    best = min(path, key=lambda s: s.val_loss)
    model.load(best.state_dict)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    meta = {
        "arch": "LassoNet",
        "n_selected": int(len(best.selected)),
        "val_loss": float(best.val_loss),
        "lambda": float(best.lambda_),
        "hidden_dims": [128, 64],
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    torch.save(best.state_dict, model_dir / "state_dict.pt")
    return load_lassonet(model_dir)


def predict_lassonet(artifact: dict, x: np.ndarray) -> np.ndarray:
    x_s = artifact["scaler"].transform(x).astype(np.float32)
    pred = artifact["model"].predict(x_s)
    return clip_nonneg(np.asarray(pred, dtype=np.float64).reshape(-1))


def load_lassonet(model_dir: Path) -> dict:
    from lassonet import LassoNetRegressor

    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    hidden = tuple(meta.get("hidden_dims", [128, 64]))
    model = LassoNetRegressor(hidden_dims=hidden, verbose=0)
    state = torch.load(model_dir / "state_dict.pt", map_location="cpu")
    model.load(state)
    return {
        "model": model,
        "scaler": joblib.load(model_dir / "scaler.joblib"),
    }


def train_gandalf(arr: dict, model_dir: Path, device: str, batch_size: int) -> dict:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig

    scaler = _fit_scaler(arr["x_train"])
    x_tr = scaler.transform(arr["x_train"]).astype(np.float32)
    x_va = scaler.transform(arr["x_val"]).astype(np.float32)
    cols = [f"f{i}" for i in range(x_tr.shape[1])]
    train_df = pd.DataFrame(x_tr, columns=cols)
    train_df["target"] = arr["y_train"]
    val_df = pd.DataFrame(x_va, columns=cols)
    val_df["target"] = arr["y_val"]

    use_gpu = device == "cuda" and torch.cuda.is_available()
    data_config = DataConfig(target=["target"], continuous_cols=cols, categorical_cols=[])
    trainer_config = TrainerConfig(
        batch_size=min(batch_size, 512),
        max_epochs=GANDALF_EPOCHS,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        early_stopping="valid_loss",
        early_stopping_patience=GANDALF_PATIENCE,
        progress_bar="none",
    )
    model_config = GANDALFConfig(task="regression", gflu_stages=GANDALF_STAGES, learning_rate=1e-3)
    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=OptimizerConfig(),
        trainer_config=trainer_config,
        experiment_config=None,
    )
    model.fit(train=train_df, validation=val_df)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "gandalf"
    model.save_model(str(save_path))
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(cols, model_dir / "columns.joblib")
    meta = {"arch": "GANDALF", "gflu_stages": GANDALF_STAGES, "save_path": str(save_path)}
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return load_gandalf(model_dir)


def predict_gandalf(artifact: dict, x: np.ndarray) -> np.ndarray:
    x_s = artifact["scaler"].transform(x).astype(np.float32)
    cols = artifact["columns"]
    df = pd.DataFrame(x_s, columns=cols)
    pred = artifact["model"].predict(df)
    col = "target_prediction" if "target_prediction" in pred.columns else pred.columns[-1]
    return clip_nonneg(pred[col].to_numpy(dtype=np.float64))


def load_gandalf(model_dir: Path) -> dict:
    from pytorch_tabular import TabularModel

    return {
        "model": TabularModel.load_model(str(model_dir / "gandalf")),
        "scaler": joblib.load(model_dir / "scaler.joblib"),
        "columns": joblib.load(model_dir / "columns.joblib"),
    }


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
    n_features = int(np.asarray(arr["x_train"]).shape[1])
    task = _resolve_catboost_task(n_features)
    devices = "0" if task == "GPU" else None
    train_pool = Pool(arr["x_train"], arr["y_train"], weight=arr["sw"])
    val_pool = Pool(arr["x_val"], arr["y_val"])

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_catboost(trial)
        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=SEED,
            task_type=task,
            devices=devices,
            thread_count=-1,
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        return r2(arr["y_val"], model.predict(arr["x_val"]))

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
        task_type=task,
        devices=devices,
        thread_count=-1,
        verbose=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best,
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.cbm"))
    meta = dict(best)
    meta["task_type_used"] = task
    meta["n_features"] = n_features
    (model_dir / "best_params.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model


def predict_catboost(model: CatBoostRegressor, x: np.ndarray) -> np.ndarray:
    return clip_nonneg(model.predict(x))


def load_catboost(model_dir: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(model_dir / "model.cbm"))
    return model
