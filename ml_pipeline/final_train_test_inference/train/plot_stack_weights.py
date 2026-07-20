#!/usr/bin/env python3
"""Analyze Ridge stack coefficients and model contributions (final_train)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPT = Path(__file__).resolve()
# ml_pipeline: .../final_train_test_inference/train/plot_stack_weights.py
TRAIN_DIR = _SCRIPT.parent
BASE = TRAIN_DIR.parent  # final_train_test_inference/

ENSEMBLE = "catboost_tabm_resnet_stack"
STACK_DIR = TRAIN_DIR / "results" / "ensemble" / ENSEMBLE
WEIGHTS_DIR = STACK_DIR / "weights"
OUT = STACK_DIR / "weight_analysis"
SORT_METRICS = TRAIN_DIR / "results" / "catboost_optuna" / "val_metrics.csv"
SORT_COL = "val_k1_r2"
SORT_LABEL = "CatBoost Optuna val K1 R²"

MODELS = ["catboost_optuna", "tabm", "resnet"]
MODEL_LABELS = {
    "catboost_optuna": "CatBoost Optuna",
    "tabm": "TabM",
    "resnet": "ResNet",
}
MODEL_COLORS = {
    "catboost_optuna": "#ff7f0e",
    "tabm": "#2ca02c",
    "resnet": "#8c564b",
}
FALLBACK_COLOR = "#FF1493"
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_weights_table() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(WEIGHTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        target = path.stem
        row: dict = {
            "target": target,
            "fallback_best_solo": bool(data.get("fallback_best_solo", False)),
            "best_solo_model": data.get("best_solo_model") or "",
            "ridge_intercept": float(data.get("ridge_intercept") or 0.0),
            "ridge_alpha": float(data.get("ridge_alpha") or np.nan),
            "tune_r2": float(data.get("tune_r2") or np.nan),
        }
        coef = data.get("ridge_coef") or {}
        eff = data.get("weights") or {}
        for model in MODELS:
            c = float(coef.get(model, 0.0))
            row[f"coef_{model}"] = c
            row[f"effective_{model}"] = float(eff.get(model, 0.0)) if row["fallback_best_solo"] else c
        abs_sum = sum(abs(row[f"coef_{m}"]) for m in MODELS)
        for model in MODELS:
            row[f"abs_share_{model}"] = (
                abs(row[f"coef_{model}"]) / abs_sum if abs_sum > 1e-12 else np.nan
            )
            if row["fallback_best_solo"]:
                row[f"effective_share_{model}"] = row[f"effective_{model}"]
            else:
                row[f"effective_share_{model}"] = row[f"abs_share_{model}"]
        rows.append(row)
    return pd.DataFrame(rows)


def _target_order(df: pd.DataFrame) -> list[str]:
    if SORT_METRICS.exists():
        solo = pd.read_csv(SORT_METRICS)
        solo = solo[(solo["status"] == "ok") & solo[SORT_COL].notna()]
        order = solo.set_index("target")[SORT_COL].sort_values(ascending=True).index.tolist()
        return [t for t in order if t in set(df["target"])]
    return sorted(df["target"])


def _plot_coef_violin(df: pd.DataFrame) -> None:
    stack_df = df[~df["fallback_best_solo"]].copy()
    long = stack_df.melt(
        id_vars="target",
        value_vars=[f"coef_{m}" for m in MODELS],
        var_name="model",
        value_name="ridge_coef",
    )
    long["model"] = long["model"].str.replace("coef_", "", regex=False)
    long["model_label"] = long["model"].map(MODEL_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sns.violinplot(
        data=long,
        x="model_label",
        y="ridge_coef",
        order=[MODEL_LABELS[m] for m in MODELS],
        palette=[MODEL_COLORS[m] for m in MODELS],
        cut=0,
        inner="quartile",
        ax=axes[0],
    )
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Ridge coefficient")
    axes[0].set_title(f"Raw Ridge coefficients ({len(stack_df)} targets, no fallback)")

    share_long = stack_df.melt(
        id_vars="target",
        value_vars=[f"abs_share_{m}" for m in MODELS],
        var_name="model",
        value_name="abs_share",
    )
    share_long["model"] = share_long["model"].str.replace("abs_share_", "", regex=False)
    share_long["model_label"] = share_long["model"].map(MODEL_LABELS)
    sns.boxplot(
        data=share_long,
        x="model_label",
        y="abs_share",
        order=[MODEL_LABELS[m] for m in MODELS],
        palette=[MODEL_COLORS[m] for m in MODELS],
        ax=axes[1],
    )
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("|coef| / sum(|coef|)")
    axes[1].set_title("Relative absolute contribution")
    axes[1].set_ylim(0, 1)

    fig.suptitle("Stack weight distributions (CatBoost+TabM+ResNet)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "stack_coef_distributions")


def _plot_abs_share_stacked(df: pd.DataFrame, order: list[str]) -> None:
    sorted_df = df.set_index("target").reindex(order).reset_index()
    x = np.arange(len(order))
    bottom = np.zeros(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=(16, 6.5))
    for model in MODELS:
        vals = sorted_df[f"effective_share_{model}"].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=1.0,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.2,
        )
        bottom += vals

    fallback_idx = sorted_df["fallback_best_solo"].to_numpy()
    for i in np.where(fallback_idx)[0]:
        ax.axvline(i, color=FALLBACK_COLOR, linestyle=":", linewidth=1.2, alpha=0.7)

    tick_step = 10 if len(order) > 40 else 5
    ticks = x[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([order[i] for i in ticks], rotation=60, ha="right", fontsize=6)
    ax.set_xlabel(f"Targets (sorted by {SORT_LABEL}); dotted line = fallback to solo")
    ax.set_ylabel("Effective model share")
    ax.set_title("Per-target stack contribution (|coef| share; fallback = 100% solo)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "stack_abs_share_by_target")


def _plot_coef_lines(df: pd.DataFrame, order: list[str]) -> None:
    sorted_df = df.set_index("target").reindex(order).reset_index()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(16, 6.5))

    for model in MODELS:
        y = sorted_df[f"coef_{model}"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            linewidth=1.6,
            alpha=0.9,
        )

    for i, row in sorted_df.iterrows():
        if row["fallback_best_solo"]:
            ax.axvline(i, color=FALLBACK_COLOR, linestyle=":", linewidth=1.0, alpha=0.5)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    tick_step = 10 if len(order) > 40 else 5
    ticks = x[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([order[i] for i in ticks], rotation=60, ha="right", fontsize=6)
    ax.set_xlabel(f"Targets (sorted by {SORT_LABEL})")
    ax.set_ylabel("Ridge coefficient")
    ax.set_title("Per-target Ridge coefficients (dotted = fallback)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "stack_coef_by_target")


def _plot_heatmap(df: pd.DataFrame, order: list[str]) -> None:
    matrix = df.set_index("target").reindex(order)[[f"coef_{m}" for m in MODELS]]
    matrix.columns = [MODEL_LABELS[m] for m in MODELS]
    n_rows, _ = matrix.shape
    fig_h = max(12.0, n_rows * 0.12 + 2.0)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-0.5,
        vmax=1.0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Ridge coef", "shrink": 0.55},
        yticklabels=True,
        xticklabels=True,
    )
    ax.set_title("Ridge coefficients by target")
    ax.set_ylabel("Target miRNA")
    ax.set_xlabel("Model")
    ax.tick_params(axis="y", labelsize=4)
    fig.tight_layout()
    _save(fig, "stack_coef_heatmap")


def _save_summary(df: pd.DataFrame) -> pd.DataFrame:
    stack_df = df[~df["fallback_best_solo"]]
    rows = []
    for model in MODELS:
        coef = stack_df[f"coef_{model}"]
        share = stack_df[f"abs_share_{model}"]
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_targets": len(stack_df),
                "mean_ridge_coef": coef.mean(),
                "median_ridge_coef": coef.median(),
                "std_ridge_coef": coef.std(),
                "pct_negative_coef": 100.0 * (coef < 0).mean(),
                "mean_abs_share": share.mean(),
                "median_abs_share": share.median(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "stack_weight_summary.csv", index=False)
    df.to_csv(OUT / "stack_weights_per_target.csv", index=False)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    mpl.rcParams["pdf.fonttype"] = 42
    sns.set_theme(style="white")

    df = _load_weights_table()
    order = _target_order(df)
    summary = _save_summary(df)

    _plot_coef_violin(df)
    _plot_abs_share_stacked(df, order)
    _plot_coef_lines(df, order)
    _plot_heatmap(df, order)

    n_fallback = int(df["fallback_best_solo"].sum())
    print(f"Wrote stack weight analysis to {OUT}")
    print(f"targets: {len(df)}, fallback: {n_fallback}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
