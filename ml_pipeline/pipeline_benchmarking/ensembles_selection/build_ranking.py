#!/usr/bin/env python3
"""Rank all pair/triple/quad ensemble combinations and pick the best."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_STAGE = Path(__file__).resolve().parent
_ML_PIPELINE = _STAGE.parents[1]
if str(_ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE))

import pandas as pd

from constants import (
    BASE_MODEL_LABELS,
    BASE_MODELS,
    ENSEMBLE_METHODS,
    ENSEMBLE_SETS,
    MODEL_SHORT,
    RESULTS,
    STAGE03_RESULTS,
)
from ensemble import ensemble_id
from run_ensembles import rebuild_global_outputs, summarize_ensemble

OUT = RESULTS / "ranking"


def _parse_ensemble_id(eid: str) -> tuple[str, str]:
    for method in ENSEMBLE_METHODS:
        suffix = f"_{method}"
        if eid.endswith(suffix):
            return eid[: -len(suffix)], method
    return eid, ""


def load_solo_baselines() -> pd.DataFrame:
    p = STAGE03_RESULTS / "summary_by_model.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["model"].isin(BASE_MODELS)].copy()
    df["kind"] = "solo"
    df["ensemble"] = df["model"]
    df["model_set"] = df["model"].map(MODEL_SHORT)
    df["method"] = "solo"
    df["n_models"] = 1
    df["models"] = df["model"]
    df["models_label"] = df["model"].map(BASE_MODEL_LABELS)
    return df


def load_ensemble_ranking() -> pd.DataFrame:
    rebuild_global_outputs()
    rows: list[dict] = []
    for model_set, models in ENSEMBLE_SETS.items():
        for method in ENSEMBLE_METHODS:
            eid = ensemble_id(model_set, method)
            if not (RESULTS / eid / "test_metrics.csv").exists():
                continue
            row = summarize_ensemble(eid)
            row["kind"] = "ensemble"
            row["model_set"] = model_set
            row["method"] = method
            row["n_models"] = len(models)
            row["models"] = ",".join(models)
            row["models_label"] = "+".join(BASE_MODEL_LABELS[m] for m in models)
            rows.append(row)
    return pd.DataFrame(rows)


def pick_best(df: pd.DataFrame) -> dict:
    ens = df[df["kind"] == "ensemble"].sort_values("median_test_k1_r2", ascending=False)
    solo = df[df["kind"] == "solo"].sort_values("median_test_k1_r2", ascending=False)
    best_ens = ens.iloc[0].to_dict() if len(ens) else {}
    best_solo = solo.iloc[0].to_dict() if len(solo) else {}
    best_stack = ens[ens["method"] == "stack"].head(1)
    best_by_size: dict[str, dict] = {}
    for n in (2, 3, 4):
        sub = ens[ens["n_models"] == n].head(1)
        if len(sub):
            best_by_size[str(n)] = sub.iloc[0].to_dict()
    return {
        "best_ensemble_overall": best_ens,
        "best_solo": best_solo,
        "best_stack": best_stack.iloc[0].to_dict() if len(best_stack) else {},
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
        "median_test_k1_r2",
        "mean_test_k1_r2",
        "n_targets_k1_gt_0_4",
        "n_fallback_solo",
        "median_test_bulk_r2",
        "median_tune_r2",
        "elapsed_sec",
    ]
    rank_cols = [c for c in rank_cols if c in combined.columns]
    ranked = combined.sort_values("median_test_k1_r2", ascending=False, na_position="last")
    ranked[rank_cols].to_csv(OUT / "ranking_all.csv", index=False)

    ens_only = ens_df.sort_values("median_test_k1_r2", ascending=False)
    ens_only.to_csv(OUT / "ranking_ensembles_only.csv", index=False)

    picks = pick_best(combined)
    (OUT / "best_pick.json").write_text(json.dumps(picks, indent=2, default=str), encoding="utf-8")

    print("=== Top 10 (solo + ensemble) by median K1 ===")
    print(ranked[rank_cols].head(10).to_string(index=False))
    print()
    if picks.get("best_ensemble_overall"):
        b = picks["best_ensemble_overall"]
        print(f"BEST ENSEMBLE: {b.get('ensemble')} | median K1={b.get('median_test_k1_r2'):.4f}")
    if picks.get("best_stack"):
        s = picks["best_stack"]
        print(f"BEST STACK:    {s.get('ensemble')} | median K1={s.get('median_test_k1_r2'):.4f}")
    print(f"\nWrote {OUT}/ranking_all.csv")


if __name__ == "__main__":
    main()
