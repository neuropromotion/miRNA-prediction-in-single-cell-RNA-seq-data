#!/usr/bin/env python3
"""Merge per-model metrics into combined model_tuning result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_STAGE = Path(__file__).resolve().parent
_ML_PIPELINE = _STAGE.parents[1]
for _p in (_STAGE, _ML_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd

from constants import (
    K_COHORT_METRIC_COLS,
    MODEL_LABELS,
    OUTER_VAL_METRIC_COLS,
    RESULTS,
    TUNING_MODELS,
)
from metrics import avg_k_cohort_scores
from screen_journal import log


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
    summary.update(avg_k_cohort_scores(summary, K_COHORT_METRIC_COLS))
    if "train_sec" in ok.columns and len(ok):
        summary["elapsed_sec"] = round(float(ok["train_sec"].sum()), 1)
    return summary


def merge_model_shards(model_name: str) -> int:
    out_dir = RESULTS / model_name
    parts = sorted(out_dir.glob("outer_val_metrics_shard*.csv"))
    frames: list[pd.DataFrame] = []
    main_path = out_dir / "outer_val_metrics.csv"
    if main_path.exists():
        frames.append(pd.read_csv(main_path))
    frames.extend(pd.read_csv(p) for p in parts)
    if not frames:
        log(f"no shards/metrics to merge for {model_name}", "merge")
        return 0

    merged = pd.concat(frames, ignore_index=True)
    if "status" in merged.columns:
        merged["_ok"] = (merged["status"] == "ok").astype(int)
        merged = merged.sort_values("_ok").drop_duplicates(subset=["target"], keep="last")
        merged = merged.drop(columns=["_ok"])
    else:
        merged = merged.drop_duplicates(subset=["target"], keep="last")
    merged.to_csv(main_path, index=False)

    summary = summarize_from_metrics(model_name, merged)
    shard_summaries = sorted(out_dir.glob("summary_shard*.json"))
    if shard_summaries:
        walls = [json.loads(p.read_text(encoding="utf-8")).get("elapsed_sec", 0) for p in shard_summaries]
        summary["elapsed_wall_sec"] = round(max(walls), 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"merged {model_name}: {len(merged)} targets from {len(parts)} shards (+main)", "merge")
    return len(merged)


def merge_all_pending_shards() -> None:
    for model_name in TUNING_MODELS:
        out_dir = RESULTS / model_name
        if list(out_dir.glob("outer_val_metrics_shard*.csv")):
            merge_model_shards(model_name)


def refresh_model_summaries(models: tuple[str, ...] = TUNING_MODELS) -> None:
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


def add_best_target_counts(df_sum: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    ok = combined[combined["status"] == "ok"] if "status" in combined.columns else combined
    out = df_sum.copy()
    for metric, short in (("outer_val_k1_r2", "outer_val_k1"), ("outer_val_bulk_r2", "outer_val_bulk")):
        if metric not in ok.columns:
            continue
        pivot = ok.pivot_table(index="target", columns="model", values=metric, aggfunc="first")
        n_best = {m: 0 for m in out["model"]}
        n_unique = {m: 0 for m in out["model"]}
        for _, row in pivot.iterrows():
            vals = row.dropna()
            if vals.empty:
                continue
            best_val = float(vals.max())
            winners = [m for m, v in vals.items() if abs(float(v) - best_val) <= 1e-12]
            for m in winners:
                if m in n_best:
                    n_best[m] += 1
            if len(winners) == 1 and winners[0] in n_unique:
                n_unique[winners[0]] += 1
        out[f"n_best_{short}"] = out["model"].map(n_best).fillna(0).astype(int)
        out[f"n_best_unique_{short}"] = out["model"].map(n_unique).fillna(0).astype(int)
    return out


def write_combined_outputs() -> tuple[int, int]:
    merge_all_pending_shards()
    refresh_model_summaries()

    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for model_name in TUNING_MODELS:
        metrics_path = RESULTS / model_name / "outer_val_metrics.csv"
        if metrics_path.exists():
            all_metrics.append(pd.read_csv(metrics_path))
        summary_path = RESULTS / model_name / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    n_metrics = 0
    combined = None
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(RESULTS / "outer_val_metrics_all.csv", index=False)
        n_metrics = len(combined)
        log(f"wrote outer_val_metrics_all.csv ({n_metrics} rows)", "merge")

    n_models = 0
    if summaries:
        df_sum = pd.DataFrame(summaries)
        if combined is not None:
            df_sum = add_best_target_counts(df_sum, combined)
        std_cols = [
            "model",
            "model_label",
            "n_targets_ok",
            "n_targets_fail",
            "n_best_outer_val_k1",
            "n_best_unique_outer_val_k1",
            "n_best_outer_val_bulk",
            "n_best_unique_outer_val_bulk",
            "avg_of_medians_K",
            "avg_of_means_K",
        ]
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
    parser = argparse.ArgumentParser(description="Merge model_tuning outputs")
    parser.add_argument("--model", action="append", default=[], help="Merge shards for a model")
    args = parser.parse_args()

    if args.model:
        for m in args.model:
            merge_model_shards(m)
        return

    n_metrics, n_models = write_combined_outputs()
    if n_models < len(TUNING_MODELS):
        missing = [m for m in TUNING_MODELS if not (RESULTS / m / "outer_val_metrics.csv").exists()]
        log(f"missing models ({len(missing)}): {', '.join(missing)}", "merge")


if __name__ == "__main__":
    main()
