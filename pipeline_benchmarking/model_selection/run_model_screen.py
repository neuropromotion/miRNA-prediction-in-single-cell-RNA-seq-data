#!/usr/bin/env python3
"""Stage03: model screen on 50 miRNA (7 models, full protocol)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_STAGE03 = Path(__file__).resolve().parents[1]
if str(_STAGE03) not in sys.path:
    sys.path.insert(0, str(_STAGE03))

import numpy as np
import pandas as pd

from model_screen.constants import (
    FEATURES,
    MODEL_LABELS,
    OPTUNA_TRIALS,
    PILOT_TARGETS,
    RESULTS,
    SCREEN_MODELS,
    SEED,
    STAGE,
    TEST_METRIC_COLS,
)
from model_screen.metrics import r2
from model_screen.model_trainers import load_artifact, predict_model, train_one
from model_screen.screen_journal import log
from shared.data import build_modality_bundle, select_features
from shared.io_splits import PB_COHORTS, load_features, load_pilot_targets

DEVICE = os.environ.get("STAGE03_DEVICE", "cuda")
MODELS_RAW = os.environ.get("STAGE03_MODELS", ",".join(SCREEN_MODELS)).strip()


def parse_models() -> list[str]:
    if MODELS_RAW.lower() in ("all", "*", ""):
        return list(SCREEN_MODELS)
    models = [m.strip() for m in MODELS_RAW.split(",") if m.strip()]
    bad = [m for m in models if m not in SCREEN_MODELS]
    if bad:
        raise ValueError(f"Unknown models: {bad}; allowed={SCREEN_MODELS}")
    return models


def test_sets(bundle, target: str, genes: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    sets: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "bulk",
            select_features(bundle.x_test_bulk, genes).to_numpy(dtype=np.float32),
            bundle.y_test_bulk[target].to_numpy(dtype=np.float64),
        ),
        (
            "k1",
            select_features(bundle.x_test_k1, genes).to_numpy(dtype=np.float32),
            bundle.y_test_k1[target].to_numpy(dtype=np.float64),
        ),
    ]
    for cohort in PB_COHORTS:
        sets.append(
            (
                f"pb_{cohort}",
                select_features(bundle.x_test_pb[cohort], genes).to_numpy(dtype=np.float32),
                bundle.y_test_pb[cohort][target].to_numpy(dtype=np.float64),
            )
        )
    return sets


def eval_target(
    model_name: str,
    artifact,
    bundle,
    target: str,
    genes: list[str],
) -> dict:
    x_val = select_features(bundle.x_val_inner, genes).to_numpy(dtype=np.float32)
    y_val = bundle.y_val_inner[target].to_numpy(dtype=np.float64)
    pred_val = predict_model(model_name, artifact, x_val)

    row: dict = {
        "target": target,
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n_features": len(genes),
        "inner_val_r2": r2(y_val, pred_val),
        "status": "ok",
    }
    for name, x_te, y_te in test_sets(bundle, target, genes):
        pred = predict_model(model_name, artifact, x_te)
        row[f"test_{name}_r2"] = r2(y_te, pred)
    return row


def metrics_path(model_name: str) -> Path:
    return RESULTS / model_name / "test_metrics.csv"


def done_targets(model_name: str) -> set[str]:
    path = metrics_path(model_name)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "target" not in df.columns:
        return set()
    ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    return set(ok["target"].astype(str))


def append_metric(model_name: str, row: dict) -> None:
    path = metrics_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev["target"] != row["target"]]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)


def summarize_model(model_name: str) -> dict:
    path = metrics_path(model_name)
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    summary = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n_targets_ok": int(len(ok)),
        "n_targets_fail": int(len(df) - len(ok)),
    }
    for col in TEST_METRIC_COLS:
        if col in ok.columns:
            summary[f"mean_{col}"] = float(ok[col].mean()) if len(ok) else None
            summary[f"median_{col}"] = float(ok[col].median()) if len(ok) else None
    return summary


def run_model(model_name: str, targets: list[str], features: dict[str, list[str]], bundle) -> None:
    log(f"=== model: {model_name} ===", model_name)
    out_dir = RESULTS / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    finished = done_targets(model_name)
    t0 = time.time()

    for i, target in enumerate(targets, 1):
        if target in finished:
            log(f"skip ({i}/{len(targets)}) {target}", model_name)
            continue
        genes = features.get(target, [])
        if not genes:
            append_metric(model_name, {"target": target, "model": model_name, "status": "no_features"})
            continue

        model_dir = out_dir / "models" / target
        log(f"({i}/{len(targets)}) {target} | n_feat={len(genes)}", model_name)
        try:
            t1 = time.time()
            artifact = train_one(model_name, bundle, target, genes, model_dir)
            train_sec = round(time.time() - t1, 2)
            artifact = load_artifact(model_name, model_dir)
            row = eval_target(model_name, artifact, bundle, target, genes)
            row["train_sec"] = train_sec
            append_metric(model_name, row)
            log(
                f"{target}: inner_r2={row['inner_val_r2']:.4f} "
                f"k1_test={row['test_k1_r2']:.4f} bulk_test={row['test_bulk_r2']:.4f} "
                f"train={train_sec}s",
                model_name,
            )
        except Exception as exc:
            log(f"{target} FAILED: {exc}", model_name)
            log(traceback.format_exc(), model_name)
            append_metric(
                model_name,
                {
                    "target": target,
                    "model": model_name,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    summary = summarize_model(model_name)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"summary: {summary}", model_name)


def write_config(targets: list[str], models: list[str]) -> None:
    cfg = {
        "seed": SEED,
        "device": DEVICE,
        "models": models,
        "optuna_trials": OPTUNA_TRIALS,
        "features": str(FEATURES),
        "targets_file": str(PILOT_TARGETS),
        "n_targets": len(targets),
        "protocol": {
            "train": "stage00 train: bulk + K1_imp + PB all cohorts",
            "inner_split": "85/15 stratified by modality (bulk/k1/pb)",
            "inner_val_metric": "mixed inner val R2",
            "test": "stage00 val: bulk, k1, pb K2/K3/K4/K5/K10",
            "impute": "KNN k=5 on K1 only",
            "sample_weights": "inverse modality frequency",
        },
    }
    (STAGE / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    models = parse_models()
    targets = load_pilot_targets()
    features = load_features()
    write_config(targets, models)

    log(f"=== Stage03 model screen | targets={len(targets)} models={models} ===")
    log("Building data bundle (KNN k=5, inner split)...")
    t0 = time.time()
    bundle = build_modality_bundle()
    log(f"Bundle ready in {time.time() - t0:.1f}s | train={len(bundle.x_train)} inner_val={len(bundle.x_val_inner)}")

    summaries = []
    for model_name in models:
        run_model(model_name, targets, features, bundle)
        summaries.append(summarize_model(model_name))

    all_metrics = []
    for model_name in models:
        p = metrics_path(model_name)
        if p.exists():
            all_metrics.append(pd.read_csv(p))
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(RESULTS / "test_metrics_all.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "summary_by_model.csv", index=False)
    log("=== summary_by_model ===")
    log(summary_df.to_string(index=False))
    log("=== done ===")


if __name__ == "__main__":
    main()
