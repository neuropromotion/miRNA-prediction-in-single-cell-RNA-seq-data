#!/usr/bin/env python3
"""Fit Ridge stack on SC+PB validation; report metrics on all val splits."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from constants import ENSEMBLE_RESULTS, PB_COHORTS, RESULTS, STACK_MODELS
from stack import TUNE_SPLITS, apply_fit, fit_stack, fit_to_dict
from data import build_train_bundle, select_features
from io_splits import load_features, load_targets
from metrics import r2
from model_trainers import load_artifact, model_exists, predict_one

ENSEMBLE_ID = "catboost_tabm_resnet_stack"


def _predict_splits(bundle, target: str, genes: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    preds_by_model: dict[str, dict[str, np.ndarray]] = {m: {} for m in STACK_MODELS}
    y_true: dict[str, np.ndarray] = {}

    def add_split(name: str, x_df, y_df):
        y_true[name] = y_df[target].to_numpy(dtype=np.float64)
        x = select_features(x_df, genes).to_numpy(dtype=np.float32)
        for m in STACK_MODELS:
            art = load_artifact(m, target)
            preds_by_model[m][name] = predict_one(m, art, x)

    add_split("val_bulk", bundle.x_val_bulk, bundle.y_val_bulk)
    add_split("val_k1", bundle.x_val_k1, bundle.y_val_k1)
    add_split("val_pb", bundle.x_val_pb, bundle.y_val_pb)
    for cohort in PB_COHORTS:
        x_c, y_c = bundle.x_val_pb_by_cohort[cohort], bundle.y_val_pb_by_cohort[cohort]
        if len(x_c):
            add_split(f"val_pb_{cohort}", x_c, y_c)

    return preds_by_model, y_true


def filter_targets(targets: list[str]) -> list[str]:
    if os.environ.get("FINAL_SHARD"):
        idx_s, n_s = os.environ["FINAL_SHARD"].split("/")
        idx, n = int(idx_s), int(n_s)
        return [t for j, t in enumerate(targets) if j % n == idx]
    return targets


def main() -> None:
    out_dir = ENSEMBLE_RESULTS / ENSEMBLE_ID
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "journal.log"

    def jlog(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with journal.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    jlog("=== final_train ridge stack ensemble ===")
    features = load_features()
    targets = filter_targets(load_targets())
    bundle = build_train_bundle()

    rows: list[dict] = []
    metrics_path = out_dir / "val_metrics.csv"
    if metrics_path.exists():
        rows = pd.read_csv(metrics_path).to_dict("records")

    ok = fail = 0
    for i, target in enumerate(targets, 1):
        for m in STACK_MODELS:
            if not model_exists(m, target):
                raise SystemExit(f"Missing model {m} for {target}; train base models first.")
        genes = features[target]
        jlog(f"({i}/{len(targets)}) {target}")
        try:
            preds_by_model, y_true = _predict_splits(bundle, target, genes)
            y_pool, pred_matrix = _pool_from_preds(preds_by_model, y_true, TUNE_SPLITS)
            fit = fit_stack(y_pool, pred_matrix, STACK_MODELS, preds_by_model, y_true)

            wpath = weights_dir / f"{target}.json"
            wpath.write_text(json.dumps(fit_to_dict(fit), indent=2), encoding="utf-8")

            row = {"target": target, "status": "ok", "tune_r2": fit.tune_r2, "fallback": fit.fallback_best_solo}
            for split in y_true:
                mat = np.column_stack([preds_by_model[m][split] for m in STACK_MODELS])
                pred = apply_fit(fit, mat, STACK_MODELS)
                row[f"{split}_r2"] = r2(y_true[split], pred)
            ok += 1
            jlog(f"  tune_r2={fit.tune_r2:.4f} fallback={fit.fallback_best_solo}")
        except Exception as exc:
            fail += 1
            row = {"target": target, "status": "fail", "error": str(exc)}
            jlog(f"  FAIL: {exc}")

        rows = [r for r in rows if r.get("target") != target]
        rows.append(row)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary = {"ensemble": ENSEMBLE_ID, "n_ok": ok, "n_fail": fail, "n_targets": len(targets)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    jlog(f"done ok={ok} fail={fail}")


def _pool_from_preds(preds_by_model, y_true, splits):
    from final_train.ensemble.stack import pool_tune_data
    return pool_tune_data(preds_by_model, y_true, STACK_MODELS, splits)


if __name__ == "__main__":
    main()
