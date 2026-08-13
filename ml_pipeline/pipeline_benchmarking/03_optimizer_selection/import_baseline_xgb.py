#!/usr/bin/env python3
"""Import XGB Optuna baseline metrics from model_selection (no retrain)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_STAGE = Path(__file__).resolve().parent
_ML = _STAGE.parents[1]
sys.path.insert(0, str(_STAGE))
sys.path.insert(0, str(_ML))

from constants import BASELINE_SCREEN, MODEL_LABELS, RESULTS
from merge_results import summarize_from_metrics
from screen_journal import log

MODEL = "xgb_optuna"


def main() -> None:
    src = BASELINE_SCREEN / MODEL
    dst = RESULTS / MODEL
    src_metrics = src / "outer_val_metrics.csv"
    if not src_metrics.exists():
        raise FileNotFoundError(f"Baseline missing: {src_metrics}")

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_metrics, dst / "outer_val_metrics.csv")

    # Optional: copy trained boosters for later ensembles (best-effort).
    src_models = src / "models"
    if src_models.is_dir():
        dst_models = dst / "models"
        if dst_models.exists():
            shutil.rmtree(dst_models)
        shutil.copytree(src_models, dst_models)
        log(f"copied models/ from screen ({MODEL})", "baseline")

    import pandas as pd

    df = pd.read_csv(dst / "outer_val_metrics.csv")
    # Ensure model_label matches tuning naming
    if "model_label" in df.columns:
        df["model_label"] = MODEL_LABELS[MODEL]
        df["model"] = MODEL
        df.to_csv(dst / "outer_val_metrics.csv", index=False)

    summary = summarize_from_metrics(MODEL, df)
    src_sum = src / "summary.json"
    if src_sum.exists():
        old = json.loads(src_sum.read_text(encoding="utf-8"))
        if "elapsed_sec" in old:
            summary["elapsed_sec"] = old["elapsed_sec"]
    summary["baseline_source"] = str(src_metrics)
    summary["retrained"] = False
    (dst / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        f"imported {MODEL}: n_ok={summary['n_targets_ok']} "
        f"avg_med_K={summary.get('avg_of_medians_K')} avg_mean_K={summary.get('avg_of_means_K')}",
        "baseline",
    )


if __name__ == "__main__":
    main()
