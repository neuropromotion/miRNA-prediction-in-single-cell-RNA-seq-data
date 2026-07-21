#!/usr/bin/env python3
"""Merge per-model metrics into combined stage03 result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_STAGE03 = Path(__file__).resolve().parents[1]
if str(_STAGE03) not in sys.path:
    sys.path.insert(0, str(_STAGE03))

import pandas as pd

from model_screen_final_11.constants import MODEL_LABELS, RESULTS, SCREEN_MODELS, OUTER_VAL_METRIC_COLS
from model_screen_final_11.screen_journal import log


def summarize_from_metrics(model_name: str, df: pd.DataFrame) -> dict:
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    summary = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n_targets_ok": int(len(ok)),
        "n_targets_fail": int(len(df) - len(ok)),
    }
    for col in OUTER_VAL_METRIC_COLS:
        if col in ok.columns and len(ok):
            summary[f"mean_{col}"] = float(ok[col].mean())
            summary[f"median_{col}"] = float(ok[col].median())
    if "train_sec" in ok.columns and len(ok):
        summary["elapsed_sec"] = round(float(ok["train_sec"].sum()), 1)
    return summary


def merge_ft_shards() -> int:
    model_name = "fttransformer"
    out_dir = RESULTS / model_name
    parts = sorted(out_dir.glob("outer_val_metrics_shard*.csv"))
    if not parts:
        log("no FT shards to merge", "merge")
        return 0

    merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    merged = merged.drop_duplicates(subset=["target"], keep="last")
    merged.to_csv(out_dir / "outer_val_metrics.csv", index=False)

    summary = summarize_from_metrics(model_name, merged)
    shard_summaries = sorted(out_dir.glob("summary_shard*.json"))
    if shard_summaries:
        walls = [json.loads(p.read_text(encoding="utf-8")).get("elapsed_sec", 0) for p in shard_summaries]
        summary["elapsed_wall_sec"] = round(max(walls), 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"merged FT: {len(merged)} targets from {len(parts)} shards", "merge")
    return len(merged)


def refresh_model_summaries(models: tuple[str, ...] = SCREEN_MODELS) -> None:
    for model_name in models:
        metrics_path = RESULTS / model_name / "outer_val_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        summary = summarize_from_metrics(model_name, df)
        (RESULTS / model_name / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


def write_combined_outputs() -> tuple[int, int]:
    merge_ft_shards()
    refresh_model_summaries()

    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for model_name in SCREEN_MODELS:
        metrics_path = RESULTS / model_name / "outer_val_metrics.csv"
        if metrics_path.exists():
            all_metrics.append(pd.read_csv(metrics_path))
        summary_path = RESULTS / model_name / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    n_metrics = 0
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(RESULTS / "outer_val_metrics_all.csv", index=False)
        n_metrics = len(combined)
        log(f"wrote outer_val_metrics_all.csv ({n_metrics} rows)", "merge")

    n_models = 0
    if summaries:
        df_sum = pd.DataFrame(summaries)
        std_cols = ["model", "model_label", "n_targets_ok", "n_targets_fail"]
        for col in OUTER_VAL_METRIC_COLS:
            std_cols.extend([f"mean_{col}", f"median_{col}"])
        std_cols.append("elapsed_sec")
        if "elapsed_wall_sec" in df_sum.columns:
            std_cols.append("elapsed_wall_sec")
        df_sum = df_sum[[c for c in std_cols if c in df_sum.columns]]
        df_sum.to_csv(RESULTS / "summary_by_model.csv", index=False)
        n_models = len(df_sum)
        log(f"wrote summary_by_model.csv ({n_models} models)", "merge")

    return n_metrics, n_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stage03 model screen outputs")
    parser.add_argument("--ft-only", action="store_true", help="Only merge FT shards")
    args = parser.parse_args()

    if args.ft_only:
        merge_ft_shards()
        return

    n_metrics, n_models = write_combined_outputs()
    if n_models < len(SCREEN_MODELS):
        missing = [m for m in SCREEN_MODELS if not (RESULTS / m / "outer_val_metrics.csv").exists()]
        log(f"missing models ({len(missing)}): {', '.join(missing)}", "merge")


if __name__ == "__main__":
    main()
