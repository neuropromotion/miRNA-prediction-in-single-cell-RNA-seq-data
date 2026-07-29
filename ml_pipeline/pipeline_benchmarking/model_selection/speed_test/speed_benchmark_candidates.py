#!/usr/bin/env python3
"""Speed benchmark for candidate DL models (LassoNet, GANDALF) on 5 miRNA."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_STAGE03 = Path(__file__).resolve().parents[1]  # model_selection/
_ML_PIPELINE = _STAGE03.parents[1]  # ml_pipeline/
for _p in (_STAGE03, _ML_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.data import build_modality_bundle, concat_pb_outer_val, select_features
from shared.io_splits import load_features
from speed_test.constants import CANDIDATE_MODELS, RESULTS, SEED, SPEED_TARGETS

OUT_DIR = RESULTS / "speed_candidates"
DEVICE = os.environ.get("STAGE03_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("STAGE03_BATCH", "512"))
GANDALF_EPOCHS = int(os.environ.get("GANDALF_MAX_EPOCHS", "50"))
GANDALF_PATIENCE = int(os.environ.get("GANDALF_PATIENCE", "10"))
GANDALF_STAGES = int(os.environ.get("GANDALF_STAGES", "6"))
LASSONET_PATIENCE = int(os.environ.get("LASSONET_PATIENCE", "15"))


def _fit_scaler(x_train: np.ndarray) -> StandardScaler:
    noise = np.random.default_rng(SEED).normal(0.0, 1e-5, x_train.shape).astype(np.float32)
    return StandardScaler().fit(x_train + noise)


def train_lassonet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
) -> tuple[object, StandardScaler, list]:
    from lassonet import LassoNetRegressor

    scaler = _fit_scaler(x_train)
    x_tr = scaler.transform(x_train).astype(np.float32)
    x_va = scaler.transform(x_val).astype(np.float32)
    model = LassoNetRegressor(
        hidden_dims=(128, 64),
        verbose=0,
        patience=LASSONET_PATIENCE,
        backtrack=True,
    )
    path = model.path(
        x_tr,
        y_train,
        X_val=x_va,
        y_val=y_val,
        return_state_dicts=True,
    )
    best = min(path, key=lambda s: s.val_loss)
    model.load(best.state_dict)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    meta = {
        "n_selected": int(len(best.selected)),
        "val_loss": float(best.val_loss),
        "lambda": float(best.lambda_),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    torch.save(best.state_dict, model_dir / "state_dict.pt")
    return model, scaler, best.selected


def predict_lassonet(model, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    x_s = scaler.transform(x).astype(np.float32)
    pred = model.predict(x_s)
    return np.clip(np.asarray(pred, dtype=np.float64).reshape(-1), 0.0, None)


def train_gandalf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
) -> tuple[object, StandardScaler]:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig

    scaler = _fit_scaler(x_train)
    x_tr = scaler.transform(x_train).astype(np.float32)
    x_va = scaler.transform(x_val).astype(np.float32)
    cols = [f"f{i}" for i in range(x_tr.shape[1])]
    train_df = pd.DataFrame(x_tr, columns=cols)
    train_df["target"] = y_train
    val_df = pd.DataFrame(x_va, columns=cols)
    val_df["target"] = y_val

    data_config = DataConfig(
        target=["target"],
        continuous_cols=cols,
        categorical_cols=[],
    )
    trainer_config = TrainerConfig(
        batch_size=min(BATCH_SIZE, 512),
        max_epochs=GANDALF_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        early_stopping="valid_loss",
        early_stopping_patience=GANDALF_PATIENCE,
        progress_bar="none",
    )
    optimizer_config = OptimizerConfig()
    model_config = GANDALFConfig(
        task="regression",
        gflu_stages=GANDALF_STAGES,
        learning_rate=1e-3,
    )
    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=None,
    )
    model.fit(train=train_df, validation=val_df)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "gandalf"))
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(cols, model_dir / "columns.joblib")
    return model, scaler


def predict_gandalf(model, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    x_s = scaler.transform(x).astype(np.float32)
    cols = [f"f{i}" for i in range(x_s.shape[1])]
    df = pd.DataFrame(x_s, columns=cols)
    pred = model.predict(df)
    col = "target_prediction" if "target_prediction" in pred.columns else pred.columns[-1]
    return np.clip(pred[col].to_numpy(dtype=np.float64), 0.0, None)


def run_one(
    model_name: str,
    target: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te_bulk: np.ndarray,
    x_te_k1: np.ndarray,
    x_te_pb: np.ndarray,
    model_dir: Path,
) -> dict:
    row = {"model": model_name, "target": target, "status": "ok", "error": ""}
    try:
        t0 = time.perf_counter()
        if model_name == "lassonet":
            model, scaler, _ = train_lassonet(x_tr, y_tr, x_va, y_va, model_dir)
            row["train_sec"] = round(time.perf_counter() - t0, 3)
            t1 = time.perf_counter()
            preds = [
                predict_lassonet(model, scaler, x)
                for x in (x_te_bulk, x_te_k1, x_te_pb)
            ]
        elif model_name == "gandalf":
            model, scaler = train_gandalf(x_tr, y_tr, x_va, y_va, model_dir)
            row["train_sec"] = round(time.perf_counter() - t0, 3)
            t1 = time.perf_counter()
            preds = [
                predict_gandalf(model, scaler, x)
                for x in (x_te_bulk, x_te_k1, x_te_pb)
            ]
        else:
            raise ValueError(f"Unknown model {model_name}")
        row["infer_sec"] = round(time.perf_counter() - t1, 3)
        row["n_train"] = len(x_tr)
        row["n_infer"] = len(x_te_bulk) + len(x_te_k1) + len(x_te_pb)
        row["pred_shapes"] = [len(p) for p in preds]
    except Exception as exc:
        row["status"] = "fail"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


def load_done() -> set[tuple[str, str]]:
    path = OUT_DIR / "speed_results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"]
    return set(zip(ok["model"].astype(str), ok["target"].astype(str)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    bundle = build_modality_bundle()
    x_pb_all = concat_pb_outer_val(bundle.x_outer_val_pb)

    meta = {
        "targets": list(SPEED_TARGETS),
        "models": list(CANDIDATE_MODELS),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "gandalf_epochs": GANDALF_EPOCHS,
        "gandalf_stages": GANDALF_STAGES,
        "lassonet_patience": LASSONET_PATIENCE,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = load_done()
    rows: list[dict] = []
    if (OUT_DIR / "speed_results.csv").exists():
        rows = pd.read_csv(OUT_DIR / "speed_results.csv").to_dict("records")

    models = list(CANDIDATE_MODELS)
    if os.environ.get("SPEED_MODELS"):
        models = [m.strip() for m in os.environ["SPEED_MODELS"].split(",") if m.strip()]

    for target in SPEED_TARGETS:
        genes = features[target]
        x_tr = select_features(bundle.x_train, genes).to_numpy(dtype=np.float32)
        y_tr = bundle.y_train[target].to_numpy(dtype=np.float64)
        x_va = select_features(bundle.x_val_inner, genes).to_numpy(dtype=np.float32)
        y_va = bundle.y_val_inner[target].to_numpy(dtype=np.float64)
        x_te_bulk = select_features(bundle.x_outer_val_bulk, genes).to_numpy(dtype=np.float32)
        x_te_k1 = select_features(bundle.x_outer_val_k1, genes).to_numpy(dtype=np.float32)
        x_te_pb = select_features(x_pb_all, genes).to_numpy(dtype=np.float32)

        for model_name in models:
            if (model_name, target) in done:
                print(f"[speed] skip {model_name}/{target}", flush=True)
                continue
            print(f"[speed] {model_name} / {target} | n_feat={len(genes)}", flush=True)
            model_dir = OUT_DIR / "checkpoints" / model_name / target
            row = run_one(
                model_name,
                target,
                x_tr,
                y_tr,
                x_va,
                y_va,
                x_te_bulk,
                x_te_k1,
                x_te_pb,
                model_dir,
            )
            rows = [r for r in rows if not (r["model"] == model_name and r["target"] == target)]
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_DIR / "speed_results.csv", index=False)
            if row["status"] == "fail":
                print(f"  FAIL: {row['error']}", flush=True)
            else:
                print(f"  train={row['train_sec']}s infer={row['infer_sec']}s", flush=True)

    df = pd.DataFrame(rows)
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

    prev_path = RESULTS / "speed_benchmark" / "speed_summary.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev = prev[~prev["model"].isin(summary_df["model"])]
        combined = pd.concat([prev, summary_df], ignore_index=True).sort_values("mean_train_sec")
        combined.to_csv(OUT_DIR / "speed_comparison_all.csv", index=False)

    print("\n=== Candidate speed summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
