#!/usr/bin/env python3
"""Compare stage03 model-screen results: K1 threshold table + R² visualizations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_COMP = Path(__file__).resolve().parent
_STAGE03 = _COMP.parents[1]
if str(_STAGE03) not in sys.path:
    sys.path.insert(0, str(_STAGE03))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model_screen.comparison.constants import (
    BASELINE,
    COLORS,
    COMPARISON,
    METRICS_PATH,
    MODEL_LABELS,
    MODEL_ORDER,
    OUT,
    PLOT_METRICS,
    R2_THRESHOLD,
)

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics: {METRICS_PATH}")
    df = pd.read_csv(METRICS_PATH)
    ok = df[df["status"] == "ok"].copy()
    missing = [m for m in MODEL_ORDER if m not in set(ok["model"])]
    if missing:
        raise ValueError(f"Models missing from metrics: {missing}")
    return ok


def k1_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="target", columns="model", values="test_k1_r2")
    return wide.reindex(columns=list(MODEL_ORDER))


def build_k1_threshold_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    above = wide > R2_THRESHOLD

    for model in MODEL_ORDER:
        vals = wide[model].dropna()
        others = [m for m in MODEL_ORDER if m != model]
        exclusive_mask = above[model] & (~above[others].any(axis=1))
        exclusive_targets = wide.index[exclusive_mask].tolist()

        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_targets": int(len(vals)),
                "mean_test_k1_r2": round(float(vals.mean()), 4),
                "median_test_k1_r2": round(float(vals.median()), 4),
                f"n_targets_gt_{R2_THRESHOLD}": int((vals > R2_THRESHOLD).sum()),
                f"pct_targets_gt_{R2_THRESHOLD}": round(float((vals > R2_THRESHOLD).mean()) * 100, 1),
                f"n_exclusive_gt_{R2_THRESHOLD}": int(exclusive_mask.sum()),
                f"exclusive_targets_gt_{R2_THRESHOLD}": ";".join(exclusive_targets),
            }
        )

    return pd.DataFrame(rows).sort_values("median_test_k1_r2", ascending=False).reset_index(drop=True)


def save_exclusive_detail(wide: pd.DataFrame, table: pd.DataFrame) -> Path:
    rows: list[dict] = []
    col = f"exclusive_targets_gt_{R2_THRESHOLD}"
    for _, row in table.iterrows():
        model = row["model"]
        for target in [t for t in str(row[col]).split(";") if t]:
            rec = {"model": model, "model_label": row["model_label"], "target": target}
            for m in MODEL_ORDER:
                rec[f"{m}_k1_r2"] = round(float(wide.loc[target, m]), 4)
            rows.append(rec)
    path = OUT / f"exclusive_targets_k1_gt_{R2_THRESHOLD}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def order_for_metric(df: pd.DataFrame, metric: str) -> list[str]:
    return (
        df[df["model"] == BASELINE][["target", metric]]
        .sort_values(metric, ascending=True)["target"]
        .tolist()
    )


def plot_r2_by_target(df: pd.DataFrame, metric: str, order: list[str]) -> Path:
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(14, 6))

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].set_index("target").reindex(order)
        y = sub[metric].to_numpy()
        if model == BASELINE:
            lw, alpha, z = 2.5, 1.0, 5
        elif model == "tabm":
            lw, alpha, z = 2.2, 1.0, 4
        else:
            lw, alpha, z = 1.5, 0.85, 3
        ax.plot(
            x,
            y,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            linewidth=lw,
            alpha=alpha,
            zorder=z,
        )

    title = PLOT_METRICS[metric]
    ax.axhline(R2_THRESHOLD, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"R²={R2_THRESHOLD}")
    slug = metric.replace("test_", "").replace("_r2", "")
    ax.set_xlabel(f"Targets (sorted by {MODEL_LABELS[BASELINE]} {title} R²)")
    ax.set_ylabel(f"{title} R²")
    ax.set_title(f"Per-target {title} R² by model (50 miRNA pilot)")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([order[i] for i in x[::5]], rotation=60, ha="right", fontsize=7)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = OUT / f"r2_by_target_{slug}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_r2_matrix(df: pd.DataFrame, metric: str, order: list[str]) -> Path:
    wide = df.pivot(index="target", columns="model", values=metric).reindex(order)
    wide = wide.reindex(columns=list(MODEL_ORDER))
    slug = metric.replace("test_", "").replace("_r2", "")
    path = OUT / f"r2_matrix_{slug}.csv"
    wide.to_csv(path)
    return path


def plot_mean_median_bars(df: pd.DataFrame) -> Path:
    summary = pd.DataFrame(
        [
            {
                "model": MODEL_LABELS[m],
                "mean_k1": df[df["model"] == m]["test_k1_r2"].mean(),
                "median_k1": df[df["model"] == m]["test_k1_r2"].median(),
                "mean_bulk": df[df["model"] == m]["test_bulk_r2"].mean(),
                "median_bulk": df[df["model"] == m]["test_bulk_r2"].median(),
            }
            for m in MODEL_ORDER
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(summary))
    w = 0.35
    for ax, mean_col, med_col, title in [
        (axes[0], "mean_k1", "median_k1", "SC K1 test"),
        (axes[1], "mean_bulk", "median_bulk", "Bulk test"),
    ]:
        ax.bar(x - w / 2, summary[mean_col], width=w, label="Mean", color="#2ca02c", alpha=0.85)
        ax.bar(x + w / 2, summary[med_col], width=w, label="Median", color="#1f77b4", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["model"], rotation=30, ha="right")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.axhline(R2_THRESHOLD, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Mean vs median R² by model (50 miRNA pilot)", y=1.02)
    fig.tight_layout()
    out = OUT / "mean_median_r2_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(OUT / "mean_median_r2_by_model.csv", index=False)
    return out


def plot_k1_density(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODEL_ORDER:
        vals = df[df["model"] == model]["test_k1_r2"].dropna()
        sns.kdeplot(
            vals,
            ax=ax,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            linewidth=2 if model in (BASELINE, "tabm") else 1.5,
        )
    ax.axvline(R2_THRESHOLD, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("SC K1 test R²")
    ax.set_ylabel("Density")
    ax.set_title("K1 test R² distribution by model (KDE)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = OUT / "k1_r2_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_threshold_counts(table: pd.DataFrame) -> Path:
    col_n = f"n_targets_gt_{R2_THRESHOLD}"
    col_ex = f"n_exclusive_gt_{R2_THRESHOLD}"
    plot_df = table.sort_values("median_test_k1_r2", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(plot_df))
    h = 0.35
    ax.barh(y - h / 2, plot_df[col_n], height=h, label=f"Total > {R2_THRESHOLD}", color="#2ca02c")
    ax.barh(y + h / 2, plot_df[col_ex], height=h, label=f"Exclusive > {R2_THRESHOLD}", color="#9467bd")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_label"])
    ax.set_xlabel("Number of targets")
    ax.set_title(f"K1 test: targets with R² > {R2_THRESHOLD}")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out = OUT / f"k1_threshold_counts_{R2_THRESHOLD}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_k1_heatmap(df: pd.DataFrame, order: list[str]) -> Path:
    wide = df.pivot(index="target", columns="model", values="test_k1_r2")
    wide = wide.reindex(order).reindex(columns=list(MODEL_ORDER))
    wide.columns = [MODEL_LABELS[c] for c in wide.columns]

    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(
        wide,
        ax=ax,
        cmap="RdYlGn",
        center=R2_THRESHOLD,
        vmin=-0.5,
        vmax=1.0,
        linewidths=0.2,
        cbar_kws={"label": "K1 test R²"},
    )
    ax.set_title("K1 test R² heatmap (targets × models)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    out = OUT / "k1_r2_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_model_rank_violin(df: pd.DataFrame) -> Path:
    wide = df.pivot(index="target", columns="model", values="test_k1_r2")
    ranks = wide.rank(axis=1, ascending=False, method="min")
    long = ranks.reset_index().melt(id_vars="target", var_name="model", value_name="rank")
    long["model_label"] = long["model"].map(MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    order_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    sns.violinplot(
        data=long,
        x="model_label",
        y="rank",
        order=order_labels,
        hue="model_label",
        palette=[COLORS[m] for m in MODEL_ORDER],
        cut=0,
        inner="quartile",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Rank on K1 test (1 = best)")
    ax.set_title("Per-target K1 rank distribution by model")
    ax.invert_yaxis()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    out = OUT / "k1_rank_violin.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    wide_k1 = k1_wide(df)

    k1_table = build_k1_threshold_table(wide_k1)
    k1_path = OUT / f"k1_threshold_summary_{R2_THRESHOLD}.csv"
    k1_table.to_csv(k1_path, index=False)
    exclusive_path = save_exclusive_detail(wide_k1, k1_table)

    order_k1 = order_for_metric(df, "test_k1_r2")
    plots: list[str] = []

    for metric in PLOT_METRICS:
        order = order_for_metric(df, metric)
        plots.append(str(plot_r2_by_target(df, metric, order).relative_to(COMPARISON)))
        save_r2_matrix(df, metric, order)

    plots.extend(
        str(p.relative_to(COMPARISON))
        for p in [
            plot_mean_median_bars(df),
            plot_k1_density(df),
            plot_threshold_counts(k1_table),
            plot_k1_heatmap(df, order_k1),
            plot_model_rank_violin(df),
        ]
    )

    meta = {
        "r2_threshold": R2_THRESHOLD,
        "baseline": BASELINE,
        "sort_by": f"{BASELINE} metric ascending",
        "metrics_file": str(METRICS_PATH.relative_to(_STAGE03)),
        "n_targets": int(df["target"].nunique()),
        "models": list(MODEL_ORDER),
        "outputs": plots,
        "tables": [
            str(k1_path.relative_to(COMPARISON)),
            str(exclusive_path.relative_to(COMPARISON)),
        ],
    }
    (OUT / "plot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {k1_path}")
    print(f"Wrote {exclusive_path}")
    print("\n=== K1 threshold summary ===")
    show_cols = [
        "model_label",
        "mean_test_k1_r2",
        "median_test_k1_r2",
        f"n_targets_gt_{R2_THRESHOLD}",
        f"n_exclusive_gt_{R2_THRESHOLD}",
    ]
    print(k1_table[show_cols].to_string(index=False))
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
