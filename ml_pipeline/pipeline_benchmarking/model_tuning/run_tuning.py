#!/usr/bin/env python3
"""Model tuning runner: TabM Optuna / DCNv2 Optuna / TabPack Muon (paper)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

_STAGE = Path(__file__).resolve().parent
_ML_PIPELINE = _STAGE.parents[1]
for _p in (_STAGE, _ML_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=FutureWarning)
for _name in ("pytorch_lightning", "lightning", "lightning_fabric", "torch.distributed"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from constants import (
    FEATURES,
    K_COHORT_METRIC_COLS,
    MODEL_LABELS,
    OPTUNA_TRIALS,
    OUTER_VAL_METRIC_COLS,
    PILOT_TARGETS,
    RESULTS,
    SEED,
    STAGE,
    TRAIN_MODELS,
    TUNING_MODELS,
)
from metrics import avg_k_cohort_scores, r2
from model_trainers import load_artifact, predict_model, train_one
from screen_journal import log
from shared.data import build_modality_bundle, select_features
from shared.io_splits import PB_COHORTS, load_features, load_pilot_targets

DEVICE = os.environ.get("STAGE03_DEVICE", os.environ.get("TUNING_DEVICE", "cuda"))
MODELS_RAW = os.environ.get("TUNING_MODELS", os.environ.get("STAGE03_MODELS", "")).strip()
METRICS_SUFFIX = os.environ.get("TUNING_METRICS_SUFFIX", os.environ.get("STAGE03_METRICS_SUFFIX", ""))


def parse_models() -> list[str]:
    if not MODELS_RAW or MODELS_RAW.lower() in ("all", "*"):
        return list(TRAIN_MODELS)
    models = [m.strip() for m in MODELS_RAW.split(",") if m.strip()]
    bad = [m for m in models if m not in TRAIN_MODELS]
    if bad:
        raise ValueError(f"Unknown models: {bad}; allowed={TRAIN_MODELS}")
    return models


def filter_targets(targets: list[str]) -> list[str]:
    if os.environ.get("TUNING_TARGETS") or os.environ.get("STAGE03_TARGETS"):
        raw = os.environ.get("TUNING_TARGETS") or os.environ["STAGE03_TARGETS"]
        wanted = {t.strip() for t in raw.split(",") if t.strip()}
        return [t for t in targets if t in wanted]
    shard = os.environ.get("TUNING_TARGET_SHARD") or os.environ.get("STAGE03_TARGET_SHARD")
    if shard:
        idx_s, n_s = shard.split("/")
        idx, n = int(idx_s), int(n_s)
        return [t for j, t in enumerate(targets) if j % n == idx]
    return targets


def outer_val_sets(bundle, target: str, genes: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    sets: list[tuple[str, np.ndarray, np.ndarray]] = [
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
        sets.append(
            (
                f"pb_{cohort}",
                select_features(bundle.x_outer_val_pb[cohort], genes).to_numpy(dtype=np.float32),
                bundle.y_outer_val_pb[cohort][target].to_numpy(dtype=np.float64),
            )
        )
    return sets


def eval_target(model_name: str, artifact, bundle, target: str, genes: list[str]) -> dict:
    x_val = select_features(bundle.x_val_inner, genes).to_numpy(dtype=np.float32)
    y_val = bundle.y_val_inner[target].to_numpy(dtype=np.float64)

    if model_name == "tabpack" or (isinstance(artifact, dict) and artifact.get("kind") == "tabpack"):
        row: dict = {
            "target": target,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "n_features": len(genes),
            "inner_val_r2": r2(y_val, artifact["val_pred"]),
            "status": "ok",
        }
        for name, _x_te, y_te in outer_val_sets(bundle, target, genes):
            pred = artifact["outer_preds"][name]
            row[f"outer_val_{name}_r2"] = r2(y_te, pred)
        return row

    pred_val = predict_model(model_name, artifact, x_val)
    row = {
        "target": target,
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n_features": len(genes),
        "inner_val_r2": r2(y_val, pred_val),
        "status": "ok",
    }
    for name, x_te, y_te in outer_val_sets(bundle, target, genes):
        pred = predict_model(model_name, artifact, x_te)
        row[f"outer_val_{name}_r2"] = r2(y_te, pred)
    return row


def metrics_path(model_name: str) -> Path:
    return RESULTS / model_name / f"outer_val_metrics{METRICS_SUFFIX}.csv"


def done_targets(model_name: str) -> set[str]:
    done: set[str] = set()
    paths = [metrics_path(model_name)]
    main = RESULTS / model_name / "outer_val_metrics.csv"
    if main not in paths:
        paths.append(main)
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "target" not in df.columns:
            continue
        ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
        done |= set(ok["target"].astype(str))
    return done


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
    if not path.exists():
        return {"model": model_name, "model_label": MODEL_LABELS[model_name], "n_targets_ok": 0}
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    summary = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n_targets_ok": int(len(ok)),
        "n_targets_fail": int(len(df) - len(ok)),
    }
    for col in OUTER_VAL_METRIC_COLS:
        if col in ok.columns:
            summary[f"mean_{col}"] = float(ok[col].mean()) if len(ok) else None
            summary[f"median_{col}"] = float(ok[col].median()) if len(ok) else None
    summary.update(avg_k_cohort_scores(summary, K_COHORT_METRIC_COLS))
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
            if model_name != "tabpack" and not isinstance(artifact, dict):
                artifact = load_artifact(model_name, model_dir)
            row = eval_target(model_name, artifact, bundle, target, genes)
            row["train_sec"] = train_sec
            append_metric(model_name, row)
            log(
                f"{target}: inner_r2={row['inner_val_r2']:.4f} "
                f"k1_outer_val={row['outer_val_k1_r2']:.4f} bulk_outer_val={row['outer_val_bulk_r2']:.4f} "
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
    summary_name = f"summary{METRICS_SUFFIX}.json"
    (out_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        "metrics_suffix": METRICS_SUFFIX,
        "protocol": {
            "train": "stage00 train: bulk + K1_imp + PB all cohorts",
            "inner_split": "85/15 stratified by modality",
            "inner_val_metric": "mixed inner val R2 (Optuna objective)",
            "outer_val": "stage00 val fold: bulk, k1, pb K2/K3/K4/K5/K10",
            "tabpack": "paper MuonAdamWPack (n_models=32), no Optuna",
            "tabm_dcnv2": "fixed arch defaults; A/B optimizers (muon / adamw_ema)",
            "baseline": "xgb_optuna imported from model_selection (no retrain)",
        },
    }
    name = f"config{METRICS_SUFFIX or ''}.json"
    path = STAGE / name
    try:
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except PermissionError:
        alt = RESULTS / name
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        log(f"config write to {path} denied; wrote {alt}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    models = parse_models()
    targets = filter_targets(load_pilot_targets())
    features = load_features()
    write_config(targets, models)

    log(f"=== model_tuning | targets={len(targets)} models={models} suffix={METRICS_SUFFIX!r} ===")
    log("Building data bundle (KNN k=5, inner split)...")
    t0 = time.time()
    bundle = build_modality_bundle()
    log(f"Bundle ready in {time.time() - t0:.1f}s | train={len(bundle.x_train)} inner_val={len(bundle.x_val_inner)}")

    for model_name in models:
        run_model(model_name, targets, features, bundle)

    if os.environ.get("TUNING_MERGE_RESULTS", os.environ.get("STAGE03_MERGE_RESULTS", "1")) != "0":
        from merge_results import write_combined_outputs

        write_combined_outputs()


if __name__ == "__main__":
    main()
