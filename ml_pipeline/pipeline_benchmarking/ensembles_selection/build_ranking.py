#!/usr/bin/env python3
"""Rank ensembles_selection_v4 by median outer_val K1."""

from __future__ import annotations

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
    BASE_MODEL_LABELS,
    BASE_MODELS,
    ENSEMBLE_METHODS,
    ENSEMBLE_SETS,
    MODEL_ARTIFACT_ROOTS,
    MODEL_SHORT,
    MODEL_TUNING_RESULTS,
    PRIMARY_RANK_COL,
    RESULTS,
    STAGE03,
    STAGE03_RESULTS,
    TABLES
)
from ensemble import ensemble_id
from run_ensembles import attach_best_counts, rebuild_global_outputs, summarize_ensemble

OUT = RESULTS / "ranking"


def _read_summary_csv(candidates: list[Path]) -> pd.DataFrame | None:
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def load_solo_baselines() -> pd.DataFrame:
    """Pull solos from model_selection + model_tuning summaries (mixed roots)."""
    parts: list[pd.DataFrame] = []

    sel = _read_summary_csv(
        [
            STAGE03_RESULTS / "summary_by_model.csv",
            STAGE03 / "tables" / "summary_by_model.csv",
        ]
    )
    if sel is not None:
        parts.append(sel[sel["model"].isin(("xgb_optuna", "dcnv2", "tabm"))].copy())

    tun = _read_summary_csv(
        [
            MODEL_TUNING_RESULTS / "summary_by_model.csv",
            MODEL_TUNING_RESULTS.parent / "tables" / "summary_by_model.csv",
        ]
    )
    if tun is not None:
        parts.append(tun[tun["model"] == "tabpack"].copy())

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True, sort=False)
    df = df.drop_duplicates(subset=["model"], keep="first")
    df["kind"] = "solo"
    df["ensemble"] = df["model"]
    df["model_set"] = df["model"].map(MODEL_SHORT)
    df["method"] = "solo"
    df["n_models"] = 1
    df["models"] = df["model"]
    df["models_label"] = df["model"].map(BASE_MODEL_LABELS)
    df["artifact_root"] = df["model"].map(lambda m: str(MODEL_ARTIFACT_ROOTS[m]))
    return df


def load_ensemble_ranking() -> pd.DataFrame:
    rebuild_global_outputs()
    # Prefer the rebuilt summary (already has n_best_* win counts).
    summary_path = TABLES / "summary_by_ensemble.csv"
    if summary_path.exists():
        base = pd.read_csv(summary_path)
        rows: list[dict] = []
        for model_set, models in ENSEMBLE_SETS.items():
            for method in ENSEMBLE_METHODS:
                eid = ensemble_id(model_set, method)
                sub = base[base["ensemble"] == eid]
                if sub.empty:
                    continue
                row = sub.iloc[0].to_dict()
                row["kind"] = "ensemble"
                row["model_set"] = model_set
                row["method"] = method
                row["n_models"] = len(models)
                row["models"] = ",".join(models)
                row["models_label"] = "+".join(BASE_MODEL_LABELS[m] for m in models)
                rows.append(row)
        return pd.DataFrame(rows)

    # Fallback if summary missing: recompute wins from metrics_all.
    metrics_all = pd.read_csv(TABLES / "outer_val_metrics_all.csv")
    rows = []
    for model_set, models in ENSEMBLE_SETS.items():
        for method in ENSEMBLE_METHODS:
            eid = ensemble_id(model_set, method)
            if not (TABLES / eid / "outer_val_metrics.csv").exists():
                continue
            row = summarize_ensemble(eid)
            row["kind"] = "ensemble"
            row["model_set"] = model_set
            row["method"] = method
            row["n_models"] = len(models)
            row["models"] = ",".join(models)
            row["models_label"] = "+".join(BASE_MODEL_LABELS[m] for m in models)
            rows.append(row)
    return pd.DataFrame(attach_best_counts(rows, metrics_all))


def pick_best(df: pd.DataFrame) -> dict:
    ens = df[df["kind"] == "ensemble"].sort_values(PRIMARY_RANK_COL, ascending=False)
    solo = df[df["kind"] == "solo"].sort_values(PRIMARY_RANK_COL, ascending=False)
    best_ens = ens.iloc[0].to_dict() if len(ens) else {}
    best_solo = solo.iloc[0].to_dict() if len(solo) else {}
    best_stack = ens[ens["method"] == "stack"].head(1)
    best_by_method: dict[str, dict] = {}
    for method in ENSEMBLE_METHODS:
        sub = ens[ens["method"] == method].head(1)
        if len(sub):
            best_by_method[method] = sub.iloc[0].to_dict()
    best_by_size: dict[str, dict] = {}
    for n in (2, 3):
        sub = ens[ens["n_models"] == n].head(1)
        if len(sub):
            best_by_size[str(n)] = sub.iloc[0].to_dict()
    return {
        "best_ensemble_overall": best_ens,
        "best_solo": best_solo,
        "best_stack": best_stack.iloc[0].to_dict() if len(best_stack) else {},
        "best_by_method": best_by_method,
        "best_by_n_models": best_by_size,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ens_df = load_ensemble_ranking()
    solo_df = load_solo_baselines()
    combined = pd.concat([solo_df, ens_df], ignore_index=True, sort=False)

    rank_cols = [
        "kind",
        "ensemble",
        "model_set",
        "method",
        "n_models",
        "models_label",
        "n_best_outer_val_k1",
        "n_best_unique_outer_val_k1",
        "n_best_outer_val_bulk",
        "n_best_unique_outer_val_bulk",
        "avg_of_medians_K",
        "avg_of_means_K",
        "median_outer_val_k1_r2",
        "mean_outer_val_k1_r2",
        "median_outer_val_bulk_r2",
        "mean_outer_val_bulk_r2",
        "n_targets_k1_gt_0_4",
        "n_fallback_solo",
        "median_tune_r2",
        "elapsed_sec",
    ]
    rank_cols = [c for c in rank_cols if c in combined.columns]
    ranked = combined.sort_values(PRIMARY_RANK_COL, ascending=False, na_position="last")
    ranked[rank_cols].to_csv(OUT / "ranking_all.csv", index=False)

    ens_only = ens_df.sort_values(PRIMARY_RANK_COL, ascending=False)
    ens_only.to_csv(OUT / "ranking_ensembles_only.csv", index=False)

    picks = pick_best(combined)
    (OUT / "best_pick.json").write_text(json.dumps(picks, indent=2, default=str), encoding="utf-8")

    print("=== Top (solo + ensemble) by median outer_val K1 ===")
    print(ranked[rank_cols].head(10).to_string(index=False))
    print()
    if picks.get("best_ensemble_overall"):
        b = picks["best_ensemble_overall"]
        print(f"BEST ENSEMBLE: {b.get('ensemble')} | median K1={b.get('median_outer_val_k1_r2'):.4f}")
    if picks.get("best_stack"):
        s = picks["best_stack"]
        print(f"BEST STACK:    {s.get('ensemble')} | median K1={s.get('median_outer_val_k1_r2'):.4f}")
    print(f"\nWrote {OUT}/ranking_all.csv")


if __name__ == "__main__":
    main()
