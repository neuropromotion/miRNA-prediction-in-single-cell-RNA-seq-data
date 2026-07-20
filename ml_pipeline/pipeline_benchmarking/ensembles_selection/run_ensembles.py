#!/usr/bin/env python3
"""Stage04 v2: tune and evaluate ensembles of 4 stage03 winners."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_STAGE = Path(__file__).resolve().parent
_ML_PIPELINE = _STAGE.parents[1]
if str(_ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE))

import numpy as np
import pandas as pd

from base_models import model_exists, predict_all_splits, true_all_splits
from constants import (
    BASE_MODELS,
    ENSEMBLE_METHODS,
    ENSEMBLE_SETS,
    EVAL_SPLITS,
    METHOD_LABELS,
    R2_THRESHOLD,
    RESULTS,
    STAGE,
    TEST_METRIC_COLS,
    TUNE_SPLITS,
)
from ensemble import apply_fit, ensemble_id, fit_ensemble, fit_to_dict
from metrics import r2
from shared.data import build_modality_bundle
from shared.io_splits import load_features, load_pilot_targets

METHODS_RAW = os.environ.get("STAGE04_METHODS", ",".join(ENSEMBLE_METHODS)).strip()
SETS_RAW = os.environ.get("STAGE04_SETS", "all").strip()


def parse_list(raw: str, allowed) -> list[str]:
    keys = allowed if isinstance(allowed, dict) else allowed
    if raw.lower() in ("all", "*", ""):
        return list(keys)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in keys]
    if bad:
        raise ValueError(f"Unknown items {bad}; allowed={list(keys)}")
    return items


def all_ensemble_ids() -> list[str]:
    ids: list[str] = []
    for model_set in ENSEMBLE_SETS:
        for method in ENSEMBLE_METHODS:
            eid = ensemble_id(model_set, method)
            if metrics_path(eid).exists():
                ids.append(eid)
    return sorted(ids)


def rebuild_global_outputs() -> None:
    eids = all_ensemble_ids()
    if not eids:
        return
    parts = [pd.read_csv(metrics_path(eid)) for eid in eids]
    pd.concat(parts, ignore_index=True).to_csv(RESULTS / "test_metrics_all.csv", index=False)

    summaries: list[dict] = []
    for eid in eids:
        summary = summarize_ensemble(eid)
        sj = RESULTS / eid / "summary.json"
        if sj.exists():
            saved = json.loads(sj.read_text(encoding="utf-8"))
            if "elapsed_sec" in saved:
                summary["elapsed_sec"] = saved["elapsed_sec"]
        summaries.append(summary)
    write_integral_summary(summaries)


def metrics_path(eid: str) -> Path:
    return RESULTS / eid / "test_metrics.csv"


def done_targets(eid: str) -> set[str]:
    path = metrics_path(eid)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    ok = df[df.get("status", "ok") == "ok"]
    return set(ok["target"].astype(str))


def append_metric(eid: str, row: dict) -> None:
    path = metrics_path(eid)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev["target"] != row["target"]]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)


def build_pred_matrix(preds_by_model: dict[str, dict[str, np.ndarray]], models: tuple[str, ...], split: str) -> np.ndarray:
    return np.column_stack([preds_by_model[m][split] for m in models])


def run_one_target(
    target: str,
    genes: list[str],
    bundle,
    model_set: str,
    models: tuple[str, ...],
    method: str,
    eid: str,
) -> None:
    if not all(model_exists(m, target) for m in models):
        missing = [m for m in models if not model_exists(m, target)]
        append_metric(
            eid,
            {
                "target": target,
                "ensemble": eid,
                "model_set": model_set,
                "method": method,
                "status": "missing_base_model",
                "error": f"missing: {missing}",
            },
        )
        return

    preds_by_model = {m: predict_all_splits(bundle, target, genes, m) for m in models}
    y_true = true_all_splits(bundle, target)
    fit = fit_ensemble(method, preds_by_model, y_true, models)

    row = {
        "target": target,
        "ensemble": eid,
        "model_set": model_set,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "models": ",".join(models),
        "tune_splits": ",".join(TUNE_SPLITS),
        "status": "ok",
        "fallback_best_solo": fit.fallback_best_solo,
        "best_solo_model": fit.best_solo_model,
        "tune_r2": fit.tune_r2,
        "tune_r2_mean_splits": fit.tune_r2_mean_splits,
    }

    for split in EVAL_SPLITS:
        mat = build_pred_matrix(preds_by_model, models, split)
        pred = apply_fit(fit, mat, models)
        row[f"{split}_r2"] = r2(y_true[split], pred)

    append_metric(eid, row)

    wpath = RESULTS / eid / "weights" / f"{target}.json"
    wpath.parent.mkdir(parents=True, exist_ok=True)
    wpath.write_text(json.dumps(fit_to_dict(fit), indent=2), encoding="utf-8")


def _threshold_stats(df_ok: pd.DataFrame) -> dict:
    out: dict = {}
    if "test_k1_r2" not in df_ok.columns or df_ok.empty:
        return {"n_targets_k1_gt_0_4": 0, "n_exclusive_k1_gt_0_4": 0}
    out["n_targets_k1_gt_0_4"] = int((df_ok["test_k1_r2"] > R2_THRESHOLD).sum())
    out["n_exclusive_k1_gt_0_4"] = 0
    return out


def summarize_ensemble(eid: str) -> dict:
    path = metrics_path(eid)
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"]
    row = {
        "ensemble": eid,
        "n_targets_ok": int(len(ok)),
        "n_targets_fail": int(len(df) - len(ok)),
        "n_fallback_solo": int(ok["fallback_best_solo"].sum()) if "fallback_best_solo" in ok.columns else 0,
    }
    if "tune_r2" in ok.columns and len(ok):
        row["mean_tune_r2"] = float(ok["tune_r2"].mean())
        row["median_tune_r2"] = float(ok["tune_r2"].median())
    row.update(_threshold_stats(ok))
    for col in TEST_METRIC_COLS:
        if col in ok.columns:
            row[f"mean_{col}"] = float(ok[col].mean()) if len(ok) else None
            row[f"median_{col}"] = float(ok[col].median()) if len(ok) else None
    return row


def write_integral_summary(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    cols = [
        "ensemble",
        "n_targets_ok",
        "n_targets_fail",
        "n_fallback_solo",
        "mean_tune_r2",
        "median_tune_r2",
        "mean_inner_val_r2",
        "median_inner_val_r2",
        "mean_test_bulk_r2",
        "median_test_bulk_r2",
        "mean_test_k1_r2",
        "median_test_k1_r2",
        "n_targets_k1_gt_0_4",
        "n_exclusive_k1_gt_0_4",
    ]
    for col in TEST_METRIC_COLS:
        if col.startswith("test_pb"):
            cols.extend([f"mean_{col}", f"median_{col}"])
    cols.append("elapsed_sec")
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("median_test_k1_r2", ascending=False, na_position="last")
    df.to_csv(RESULTS / "summary_by_ensemble.csv", index=False)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    methods = parse_list(METHODS_RAW, ENSEMBLE_METHODS)
    model_sets = parse_list(SETS_RAW, ENSEMBLE_SETS)
    targets = load_pilot_targets()
    features = load_features()

    cfg = {
        "base_models": list(BASE_MODELS),
        "model_sets": {k: list(ENSEMBLE_SETS[k]) for k in model_sets},
        "methods": methods,
        "n_targets": len(targets),
        "tune_splits": list(TUNE_SPLITS),
        "protocol": {
            "tune": "pooled K1 + pseudo-bulk PB (K2-K10), no real bulk",
            "tune_metric": "R2 on pooled tune samples (+ mean per-split for reporting)",
            "test": "inner val, bulk (held-out), K1, PB K2-K10",
            "base_artifacts": "pipeline_benchmarking/model_selection/results/{model}/models/{target}/",
        },
    }
    (STAGE / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("Building data bundle...", flush=True)
    t0 = time.time()
    bundle = build_modality_bundle()
    print(f"Bundle ready in {time.time() - t0:.1f}s", flush=True)

    summaries: list[dict] = []
    all_eids: list[str] = []

    for model_set in model_sets:
        models = ENSEMBLE_SETS[model_set]
        for method in methods:
            eid = ensemble_id(model_set, method)
            all_eids.append(eid)
            finished = done_targets(eid)
            print(f"\n=== {eid} ===", flush=True)
            t1 = time.time()

            for i, target in enumerate(targets, 1):
                if target in finished:
                    continue
                genes = features.get(target, [])
                if not genes:
                    append_metric(eid, {"target": target, "ensemble": eid, "status": "no_features"})
                    continue
                print(f"  ({i}/{len(targets)}) {target}", flush=True)
                try:
                    run_one_target(target, genes, bundle, model_set, models, method, eid)
                except Exception as exc:
                    print(f"    FAIL: {exc}", flush=True)
                    traceback.print_exc()
                    append_metric(
                        eid,
                        {
                            "target": target,
                            "ensemble": eid,
                            "status": "fail",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

            summary = summarize_ensemble(eid)
            summary["elapsed_sec"] = round(time.time() - t1, 1)
            (RESULTS / eid / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)
            print(f"  median_k1={summary.get('median_test_k1_r2')}", flush=True)

    parts = [pd.read_csv(metrics_path(eid)) for eid in all_eids if metrics_path(eid).exists()]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(RESULTS / "test_metrics_all.csv", index=False)

    rebuild_global_outputs()
    print("\n=== summary_by_ensemble.csv ===")
    print(pd.read_csv(RESULTS / "summary_by_ensemble.csv").to_string(index=False))
    print("=== done ===")


if __name__ == "__main__":
    main()
