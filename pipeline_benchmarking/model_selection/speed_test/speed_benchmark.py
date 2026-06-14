#!/usr/bin/env python3
"""Speed benchmark: train + inference time for 9 model candidates on 5 miRNA."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

STAGE = Path(__file__).resolve().parent
PILOT = STAGE.parent.parent / "pilot_borderline"
sys.path.insert(0, str(STAGE))

from constants import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    OPTUNA_TRIALS_SPEED,
    RESULTS,
    SEED,
    SPEED_N_TARGETS,
    XGB_DEFAULT,
)
from data import build_modality_bundle, select_features  # noqa: E402
from io_splits import load_features, load_pilot_targets  # noqa: E402

sys.path.insert(0, str(PILOT))
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

DEVICE = os.environ.get("STAGE03_DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
BATCH_SIZE = int(os.environ.get("STAGE03_BATCH", "512"))
OUT_DIR = RESULTS / "speed_benchmark"
TABNET_EPOCHS = int(os.environ.get("STAGE03_TABNET_EPOCHS", "100"))
TABNET_PATIENCE = int(os.environ.get("STAGE03_TABNET_PATIENCE", "20"))


def pick_speed_targets(n: int = SPEED_N_TARGETS) -> list[str]:
    all_t = load_pilot_targets()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_t), size=min(n, len(all_t)), replace=False)
    return sorted(all_t[i] for i in idx)


def suggest_xgb(trial: optuna.Trial) -> dict:
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


def train_xgb_default(x_tr, y_tr, sw, x_va, y_va):
    model = xgb.XGBRegressor(
        **XGB_DEFAULT,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        x_tr,
        y_tr,
        sample_weight=sw,
        eval_set=[(x_va, y_va)],
        verbose=False,
    )
    return model


def train_xgb_optuna(x_tr, y_tr, sw, x_va, y_va, n_trials: int = OPTUNA_TRIALS_SPEED):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb(trial)
        model = xgb.XGBRegressor(
            **XGB_DEFAULT,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )
        model.fit(x_tr, y_tr, sample_weight=sw, eval_set=[(x_va, y_va)], verbose=False)
        from sklearn.metrics import r2_score

        return float(r2_score(y_va, model.predict(x_va)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    model = xgb.XGBRegressor(
        **XGB_DEFAULT,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best,
    )
    model.fit(x_tr, y_tr, sample_weight=sw, eval_set=[(x_va, y_va)], verbose=False)
    return model


def train_tabnet_wrapped(x_tr, y_tr, x_va, y_va, model_dir, device, batch_size):
    return train_tabnet(
        x_tr,
        y_tr,
        x_va,
        y_va,
        model_dir,
        device,
        batch_size,
        TABNET_EPOCHS,
        TABNET_PATIENCE,
    )


def train_tabpfn(x_tr, y_tr, x_va, y_va):
    from tabpfn import TabPFNRegressor

    max_train = int(os.environ.get("TABPFN_MAX_TRAIN", "1024"))
    rng = np.random.default_rng(SEED)
    if len(x_tr) > max_train:
        idx = rng.choice(len(x_tr), size=max_train, replace=False)
        x_fit = x_tr[idx]
        y_fit = y_tr[idx]
    else:
        x_fit, y_fit = x_tr, y_tr

    model = TabPFNRegressor(device=DEVICE if DEVICE == "cuda" else "cpu")
    model.fit(x_fit.astype(np.float32), y_fit.astype(np.float32))
    return model


def predict_xgb(model, *xs):
    return [model.predict(x) for x in xs]


def predict_tabpfn(model, *xs):
    return [model.predict(x.astype(np.float32)) for x in xs]


def run_one_model(
    name: str,
    train_fn,
    predict_fn,
    target: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    sw: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te_bulk: np.ndarray,
    x_te_k1: np.ndarray,
    x_te_pb: np.ndarray,
    model_dir: Path,
) -> dict:
    row = {"model": name, "target": target, "status": "ok", "error": ""}
    try:
        t0 = time.perf_counter()
        if name.startswith("xgb"):
            artifact = train_fn(x_tr, y_tr, sw, x_va, y_va)
        elif name == "tabpfn":
            artifact = train_fn(x_tr, y_tr, x_va, y_va)
        elif name in ("tabnet", "tabm"):
            artifact = train_fn(x_tr, y_tr, x_va, y_va, model_dir, DEVICE, BATCH_SIZE)
        else:
            artifact = train_fn(x_tr, y_tr, x_va, y_va, model_dir, DEVICE, BATCH_SIZE)
        row["train_sec"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        if name.startswith("xgb"):
            preds = predict_fn(artifact, x_te_bulk, x_te_k1, x_te_pb)
        elif name == "tabpfn":
            preds = predict_fn(artifact, x_te_bulk, x_te_k1, x_te_pb)
        elif name == "tabnet":
            preds = [
                predict_tabnet(model_dir, x)
                for x in (x_te_bulk, x_te_k1, x_te_pb)
            ]
        elif name == "tabm":
            preds = [
                predict_tabm(model_dir, x, DEVICE)
                for x in (x_te_bulk, x_te_k1, x_te_pb)
            ]
        else:
            preds = [
                predict_torch_model(model_dir, x)
                for x in (x_te_bulk, x_te_k1, x_te_pb)
            ]
        row["infer_sec"] = round(time.perf_counter() - t1, 3)
        row["n_train"] = len(x_tr)
        row["n_infer"] = len(x_te_bulk) + len(x_te_k1) + len(x_te_pb)
        row["pred_shapes"] = [len(p) for p in preds]
    except Exception as exc:
        row["status"] = "fail"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


MODELS = [
    ("xgb_default", train_xgb_default, predict_xgb),
    ("xgb_optuna", train_xgb_optuna, predict_xgb),
    ("tabm", train_tabm, None),
    ("tabnet", train_tabnet_wrapped, None),
    ("realmlp", train_realmlp, None),
    ("resnet", train_resnet, None),
    ("dcnv2", train_dcnv2, None),
    ("fttransformer", train_fttransformer, None),
    ("tabpfn", train_tabpfn, predict_tabpfn),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_map = load_features()
    targets = pick_speed_targets()
    bundle = build_modality_bundle()

    meta = {
        "targets": targets,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "optuna_trials": OPTUNA_TRIALS_SPEED,
        "impute_stats": bundle.impute_stats,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for target in targets:
        genes = feature_map[target]
        x_tr = select_features(bundle.x_train, genes).to_numpy(dtype=np.float32)
        y_tr = bundle.y_train[target].to_numpy(dtype=np.float64)
        sw = bundle.sample_weight
        x_va = select_features(bundle.x_val_k1, genes).to_numpy(dtype=np.float32)
        y_va = bundle.y_val_k1[target].to_numpy(dtype=np.float64)
        x_te_bulk = select_features(bundle.x_test_bulk, genes).to_numpy(dtype=np.float32)
        x_te_k1 = select_features(bundle.x_test_k1, genes).to_numpy(dtype=np.float32)
        x_te_pb = select_features(bundle.x_test_pb, genes).to_numpy(dtype=np.float32)

        for name, train_fn, predict_fn in MODELS:
            model_dir = OUT_DIR / "checkpoints" / name / target
            print(f"[speed] {name} / {target}", flush=True)
            row = run_one_model(
                name,
                train_fn,
                predict_fn,
                target,
                x_tr,
                y_tr,
                sw,
                x_va,
                y_va,
                x_te_bulk,
                x_te_k1,
                x_te_pb,
                model_dir,
            )
            rows.append(row)
            if row["status"] == "fail":
                print(f"  FAIL: {row['error']}", flush=True)
            else:
                print(
                    f"  train={row['train_sec']}s infer={row['infer_sec']}s",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "speed_results.csv", index=False)

    ok = df[df["status"] == "ok"].copy()
    summary = []
    for name, grp in ok.groupby("model"):
        summary.append(
            {
                "model": name,
                "n_ok": int(len(grp)),
                "n_fail": int((df["model"] == name).sum() - len(grp)),
                "mean_train_sec": round(float(grp["train_sec"].mean()), 3),
                "median_train_sec": round(float(grp["train_sec"].median()), 3),
                "mean_infer_sec": round(float(grp["infer_sec"].mean()), 3),
                "median_infer_sec": round(float(grp["infer_sec"].median()), 3),
                "total_train_50mirna_h": round(float(grp["train_sec"].mean()) * 50 / 3600, 2),
                "total_train_327mirna_h": round(float(grp["train_sec"].mean()) * 327 / 3600, 2),
            }
        )
    summary_df = pd.DataFrame(summary).sort_values("mean_train_sec")
    summary_df.to_csv(OUT_DIR / "speed_summary.csv", index=False)
    print("\n=== Speed summary (mean over targets) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
