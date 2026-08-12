#!/usr/bin/env python3
"""Train TabPack / DCNv2 / TabM on 327 miRNA targets (final_train)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from constants import MODELS, RESULTS, ROOT
from data import build_train_bundle, select_features
from io_splits import load_features, load_targets, load_zero_expressed_mirs
from metrics import r2
from journal import log
from model_trainers import (
    eval_tabpack_row,
    load_artifact,
    model_dir,
    model_exists,
    predict_one,
    train_one,
)

DEVICE = os.environ.get("FINAL_DEVICE", "cuda")


def parse_model() -> str:
    model = os.environ.get("FINAL_MODEL", "").strip()
    if model not in MODELS:
        raise SystemExit(f"Set FINAL_MODEL to one of {MODELS}")
    return model


def filter_targets(targets: list[str]) -> list[str]:
    if os.environ.get("FINAL_TARGETS"):
        wanted = {t.strip() for t in os.environ["FINAL_TARGETS"].split(",") if t.strip()}
        return [t for t in targets if t in wanted]
    shard = os.environ.get("FINAL_SHARD")
    if shard:
        idx_s, n_s = shard.split("/")
        idx, n = int(idx_s), int(n_s)
        return [t for j, t in enumerate(targets) if j % n == idx]
    return targets


def metrics_path_for(model_name: str) -> Path:
    """Shard-safe metrics file (avoid races when FINAL_METRICS_SUFFIX is set)."""
    suffix = os.environ.get("FINAL_METRICS_SUFFIX", "").strip()
    name = f"val_metrics{suffix}.csv" if suffix else "val_metrics.csv"
    return RESULTS / model_name / name


def val_eval_sets(bundle, target: str, genes: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    from constants import PB_COHORTS

    sets = [
        (
            "val_bulk",
            select_features(bundle.x_val_bulk, genes).to_numpy(dtype=np.float32),
            bundle.y_val_bulk[target].to_numpy(dtype=np.float64),
        ),
        (
            "val_k1",
            select_features(bundle.x_val_k1, genes).to_numpy(dtype=np.float32),
            bundle.y_val_k1[target].to_numpy(dtype=np.float64),
        ),
        (
            "val_pb",
            select_features(bundle.x_val_pb, genes).to_numpy(dtype=np.float32),
            bundle.y_val_pb[target].to_numpy(dtype=np.float64),
        ),
    ]
    for cohort in PB_COHORTS:
        x_c, y_c = bundle.x_val_pb_by_cohort[cohort], bundle.y_val_pb_by_cohort[cohort]
        if len(x_c) == 0:
            continue
        sets.append(
            (
                f"val_pb_{cohort}",
                select_features(x_c, genes).to_numpy(dtype=np.float32),
                y_c[target].to_numpy(dtype=np.float64),
            )
        )
    return sets


def eval_target(model_name: str, artifact, bundle, target: str, genes: list[str]) -> dict:
    if model_name == "tabpack":
        row = eval_tabpack_row(artifact, bundle, target)
        row["n_features"] = len(genes)
        return row

    x_val = select_features(bundle.x_val, genes).to_numpy(dtype=np.float32)
    y_val = bundle.y_val[target].to_numpy(dtype=np.float64)
    pred_val = predict_one(model_name, artifact, x_val)

    row: dict = {
        "target": target,
        "model": model_name,
        "n_features": len(genes),
        "val_mix_r2": r2(y_val, pred_val),
        "status": "ok",
        "train_sec": np.nan,
        "error": "",
    }
    for split_name, x_te, y_te in val_eval_sets(bundle, target, genes):
        pred = predict_one(model_name, artifact, x_te)
        row[f"{split_name}_r2"] = r2(y_te, pred)
    return row


def main() -> None:
    model_name = parse_model()
    out_root = RESULTS / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    log("=== final_train model training ===", model_name)
    log(f"device={DEVICE} winner_stack={MODELS}", model_name)
    features = load_features()
    excluded = load_zero_expressed_mirs()
    targets = filter_targets(load_targets())
    log(f"targets={len(targets)} (excluded_zero_expressed={len(excluded)})", model_name)

    bundle = build_train_bundle()
    log(
        f"train={len(bundle.x_train)} val={len(bundle.x_val)} impute={bundle.impute_stats}",
        model_name,
    )

    metrics_path = metrics_path_for(model_name)
    done: set[str] = set()
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        done = set(prev.loc[prev["status"] == "ok", "target"].astype(str))

    rows: list[dict] = []
    if metrics_path.exists():
        rows = pd.read_csv(metrics_path).to_dict("records")

    ok = fail = skip = 0
    for i, target in enumerate(targets, 1):
        if target in done and model_exists(model_name, target):
            skip += 1
            continue
        genes = features[target]
        log(f"({i}/{len(targets)}) {target} n_feat={len(genes)}", model_name)
        t0 = time.perf_counter()
        try:
            artifact = train_one(model_name, bundle, target, genes)
            row = eval_target(model_name, artifact, bundle, target, genes)
            row["train_sec"] = round(time.perf_counter() - t0, 2)
            ok += 1
            log(f"  val_mix_r2={row['val_mix_r2']:.4f}", model_name)
        except Exception as exc:
            fail += 1
            row = {
                "target": target,
                "model": model_name,
                "n_features": len(genes),
                "status": "fail",
                "train_sec": round(time.perf_counter() - t0, 2),
                "error": str(exc),
            }
            log(f"  FAIL: {exc}", model_name)
            traceback.print_exc()

        rows = [r for r in rows if r.get("target") != target]
        rows.append(row)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary = {
        "model": model_name,
        "n_targets": len(targets),
        "n_ok": ok,
        "n_fail": fail,
        "n_skip": skip,
        "device": DEVICE,
        "metrics_path": str(metrics_path),
        "model_dir_pattern": str(model_dir(model_name, "{target}")),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done ok={ok} fail={fail} skip={skip}", model_name)


if __name__ == "__main__":
    main()
