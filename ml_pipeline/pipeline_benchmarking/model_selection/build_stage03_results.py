#!/usr/bin/env python3
"""Build consolidated stage03 comparison tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

_SCRIPT = Path(__file__).resolve()
# ml_pipeline: .../pipeline_benchmarking/model_selection/build_stage03_results.py
BASE = _SCRIPT.parent
SRC = BASE / "results"
OUT = BASE / "tables"

R2_COLS = {
    "inner_val_r2": "Inner val R2",
    "outer_val_bulk_r2": "Bulk outer_val R2",
    "outer_val_k1_r2": "K1 outer_val R2",
}

CELL_W = 0.72
CELL_H = 0.34
DPI = 300
BASELINE_MODEL = "xgb_default"
R2_THRESHOLD = 0.4

LINE_PLOTS = {
    "inner_val_r2": ("Inner val R²", "inner_val"),
    "outer_val_bulk_r2": ("Bulk outer_val R²", "bulk"),
    "outer_val_k1_r2": ("K1 outer_val R²", "k1"),
}

MODEL_COLORS = {
    "xgb_default": "#1f77b4",
    "xgb_optuna": "#d62728",
    "catboost_optuna": "#ff7f0e",
    "tabm": "#2ca02c",
    "fttransformer": "#9467bd",
    "resnet": "#8c564b",
    "gandalf": "#e377c2",
    "realmlp": "#17becf",
    "tabnet": "#bcbd22",
    "lassonet": "#7f7f7f",
    "dcnv2": "#aec7e8",
}


def _ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def _load_metrics() -> pd.DataFrame:
    df = pd.read_csv(SRC / "outer_val_metrics_all.csv")
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    return ok


def _model_order(summary: pd.DataFrame) -> list[str]:
    return summary.sort_values("median_outer_val_k1_r2", ascending=False)["model"].tolist()


def _label_matrix(matrix: pd.DataFrame, model_labels: dict[str, str]) -> pd.DataFrame:
    out = matrix.copy()
    out.columns = [model_labels.get(c, c) for c in out.columns]
    return out


def _figsize_for_matrix(n_rows: int, n_cols: int, dendro_w: float = 0.0, dendro_h: float = 0.0) -> tuple[float, float]:
    w = max(8.0, n_cols * CELL_W + 2.8 + dendro_w)
    h = max(10.0, n_rows * CELL_H + 2.2 + dendro_h)
    return w, h


def _set_nearest(ax) -> None:
    for im in ax.collections:
        if hasattr(im, "set_interpolation"):
            im.set_interpolation("nearest")
        if hasattr(im, "set_antialiased"):
            im.set_antialiased(False)


def _save_summary_table(summary: pd.DataFrame) -> None:
    cols = [
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
        "mean_inner_val_r2",
        "median_inner_val_r2",
        "mean_outer_val_bulk_r2",
        "median_outer_val_bulk_r2",
        "mean_outer_val_k1_r2",
        "median_outer_val_k1_r2",
        "n_targets_k1_gt_0_4",
        "n_exclusive_k1_gt_0_4",
        "elapsed_sec",
    ]
    keep = [c for c in cols if c in summary.columns]
    summary[keep].to_csv(OUT / "summary_by_model.csv", index=False)

    mean_med = summary.set_index("model_label")[
        [c for c in summary.columns if c.startswith("mean_") or c.startswith("median_")]
    ]
    mean_med.to_csv(OUT / "mean_median_r2_by_model.csv")


def _compute_best_counts(
    df: pd.DataFrame,
    model_order: list[str],
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-target argmax wins for `metric`.

    Ties: every model at the max counts toward n_best_*; only sole winners
    toward n_best_unique_*.
    """
    short = metric.removesuffix("_r2")
    pivot = df.pivot(index="target", columns="model", values=metric).reindex(columns=model_order)
    n_best = {m: 0 for m in model_order}
    n_unique = {m: 0 for m in model_order}
    rows: list[dict] = []

    for target, row in pivot.iterrows():
        vals = row.dropna()
        if vals.empty:
            continue
        best_val = float(vals.max())
        winners = vals[np.isclose(vals.astype(float), best_val, rtol=0.0, atol=1e-12)].index.tolist()
        if not winners:
            winners = vals[vals == best_val].index.tolist()
        for m in winners:
            n_best[m] += 1
        sole = winners[0] if len(winners) == 1 else ""
        if sole:
            n_unique[sole] += 1
        rows.append(
            {
                "target": target,
                "metric": metric,
                "best_value": best_val,
                "n_tied": len(winners),
                "best_models": ",".join(winners),
                "best_model_sole": sole,
            }
        )

    stats = pd.DataFrame(
        {
            "model": model_order,
            f"n_best_{short}": [n_best[m] for m in model_order],
            f"n_best_unique_{short}": [n_unique[m] for m in model_order],
        }
    )
    return stats, pd.DataFrame(rows)


def _compute_k1_threshold_stats(df: pd.DataFrame, model_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    k1 = df.pivot(index="target", columns="model", values="outer_val_k1_r2").reindex(columns=model_order)
    above = k1 > R2_THRESHOLD

    n_gt = above.sum(axis=0).astype(int)
    exclusive_rows: list[dict] = []
    for target, row in above.iterrows():
        winners = row[row].index.tolist()
        if len(winners) != 1:
            continue
        model = winners[0]
        exclusive_rows.append(
            {
                "model": model,
                "target": target,
                "outer_val_k1_r2": float(k1.loc[target, model]),
            }
        )

    exclusive_df = pd.DataFrame(exclusive_rows)
    if exclusive_df.empty:
        n_exclusive = pd.Series(0, index=model_order)
    else:
        n_exclusive = exclusive_df.groupby("model").size()

    stats = pd.DataFrame(
        {
            "model": model_order,
            "n_targets_k1_gt_0_4": [int(n_gt.get(m, 0)) for m in model_order],
            "n_exclusive_k1_gt_0_4": [int(n_exclusive.get(m, 0)) for m in model_order],
        }
    )
    return stats, exclusive_df


def _save_r2_matrix(df: pd.DataFrame, model_order: list[str], metric: str, out_name: str) -> pd.DataFrame:
    matrix = (
        df.pivot(index="target", columns="model", values=metric)
        .reindex(columns=model_order)
        .sort_index()
    )
    matrix.to_csv(OUT / out_name)
    return matrix


def _cluster_column_order(matrix: pd.DataFrame) -> list[str]:
    if matrix.shape[1] <= 1:
        return matrix.columns.tolist()
    filled = matrix.copy()
    for col in filled.columns:
        col_mean = filled[col].mean(skipna=True)
        filled[col] = filled[col].fillna(col_mean if np.isfinite(col_mean) else 0.0)
    dist = pdist(filled.T.values, metric="correlation")
    if not np.all(np.isfinite(dist)):
        dist = pdist(filled.T.values, metric="euclidean")
    link = hierarchy.linkage(dist, method="average")
    order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link, dist))
    return [matrix.columns[i] for i in order]


def _plot_heatmap(matrix: pd.DataFrame, title: str, path: Path, model_labels: dict[str, str]) -> None:
    labeled = _label_matrix(matrix, model_labels)
    n_rows, n_cols = labeled.shape
    fig_w, fig_h = _figsize_for_matrix(n_rows, n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        labeled,
        ax=ax,
        cmap="viridis",
        center=0.0,
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "R2", "shrink": 0.55},
        xticklabels=True,
        yticklabels=True,
        square=False,
    )
    _set_nearest(ax)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("Target miRNA")
    ax.tick_params(axis="x", labelrotation=40, labelsize=9)
    ax.tick_params(axis="y", labelsize=7)
    plt.subplots_adjust(left=0.22, bottom=0.18, right=0.92, top=0.95)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_heatmap_clustered_columns(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    model_labels: dict[str, str],
) -> None:
    filled = matrix.copy()
    for col in filled.columns:
        col_mean = filled[col].mean(skipna=True)
        filled[col] = filled[col].fillna(col_mean if np.isfinite(col_mean) else 0.0)
    dist = pdist(filled.T.values, metric="correlation")
    if not np.all(np.isfinite(dist)):
        dist = pdist(filled.T.values, metric="euclidean")
    link = hierarchy.linkage(dist, method="average")
    leaves = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link, dist))
    col_order = [matrix.columns[i] for i in leaves]
    ordered = matrix[col_order]
    labeled = _label_matrix(ordered, model_labels)

    n_rows, n_cols = labeled.shape
    fig_w, fig_h = _figsize_for_matrix(n_rows, n_cols, dendro_w=0.0, dendro_h=1.4)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, n_rows * CELL_H + 1.8], hspace=0.02)
    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])

    hierarchy.dendrogram(
        link,
        ax=ax_dendro,
        orientation="top",
        labels=labeled.columns.tolist(),
        leaf_rotation=40,
        leaf_font_size=9,
        color_threshold=0,
        above_threshold_color="#444444",
    )
    ax_dendro.set_ylabel("")
    ax_dendro.set_xticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    sns.heatmap(
        labeled,
        ax=ax_heat,
        cmap="viridis",
        center=0.0,
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "R2", "shrink": 0.55},
        xticklabels=True,
        yticklabels=True,
        square=False,
    )
    _set_nearest(ax_heat)
    ax_heat.set_title(title, fontsize=13, pad=12)
    ax_heat.set_xlabel("Model (clustered)")
    ax_heat.set_ylabel("Target miRNA")
    ax_heat.tick_params(axis="x", labelrotation=40, labelsize=9)
    ax_heat.tick_params(axis="y", labelsize=7)
    plt.subplots_adjust(left=0.22, bottom=0.08, right=0.92, top=0.98)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_r2_by_target_lines(
    matrix: pd.DataFrame,
    metric_label: str,
    slug: str,
    model_labels: dict[str, str],
    model_order: list[str],
    baseline: str = BASELINE_MODEL,
) -> Path:
    if baseline not in matrix.columns:
        baseline = matrix.columns[0]
    order = matrix[baseline].sort_values(ascending=True).index.tolist()
    sorted_matrix = matrix.reindex(order)
    sorted_matrix.to_csv(OUT / f"r2_matrix_{slug}_sorted.csv")

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(16, 7))
    palette = plt.cm.tab20.colors
    baseline_label = model_labels.get(baseline, baseline)

    for i, model in enumerate(model_order):
        if model not in sorted_matrix.columns:
            continue
        y = sorted_matrix[model].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, palette[i % len(palette)])
        label = model_labels.get(model, model)
        if model == baseline:
            lw, alpha, zorder = 2.6, 1.0, 3
        else:
            lw, alpha, zorder = 1.5, 0.85, 2
        ax.plot(x, y, label=label, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

    ax.axhline(
        R2_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"R²={R2_THRESHOLD}",
    )
    ax.set_xlabel(f"Targets (sorted by {baseline_label} {metric_label})")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Per-target {metric_label} by model ({baseline_label} baseline)")
    tick_step = 5 if len(order) > 20 else 2
    ticks = x[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([order[i] for i in ticks], rotation=60, ha="right", fontsize=7)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = OUT / f"r2_by_target_{slug}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _plot_model_mean_median(summary: pd.DataFrame, model_order: list[str]) -> None:
    plot_df = summary.set_index("model").reindex(model_order).reset_index()
    plot_df = plot_df[["model_label", "mean_outer_val_k1_r2", "median_outer_val_k1_r2"]].melt(
        id_vars="model_label",
        var_name="stat",
        value_name="r2",
    )
    plot_df["stat"] = plot_df["stat"].map(
        {"mean_outer_val_k1_r2": "mean", "median_outer_val_k1_r2": "median"}
    )
    plt.figure(figsize=(max(12, len(model_order) * 0.95), 5.8))
    sns.barplot(data=plot_df, x="model_label", y="r2", hue="stat")
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xticks(rotation=40, ha="right")
    plt.xlabel("Model")
    plt.ylabel("K1 R2")
    plt.title("Mean/Median K1 R2 by model (all 11 models)")
    plt.tight_layout()
    plt.savefig(OUT / "mean_median_r2_by_model.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_k1_density(df: pd.DataFrame, model_order: list[str], model_labels: dict[str, str]) -> None:
    plt.figure(figsize=(11, 6.5))
    for model in model_order:
        data = df.loc[df["model"] == model, "outer_val_k1_r2"].dropna()
        if len(data) < 2:
            continue
        sns.kdeplot(data=data, label=model_labels.get(model, model), fill=False, linewidth=2)
    plt.axvline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("K1 R2")
    plt.ylabel("Density")
    plt.title("K1 R2 density across models")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "k1_r2_density.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_k1_violin(df: pd.DataFrame, model_order: list[str], model_labels: dict[str, str]) -> None:
    p = df.copy()
    p["model_label"] = p["model"].map(model_labels).fillna(p["model"])
    label_order = [model_labels.get(m, m) for m in model_order]
    plt.figure(figsize=(max(12, len(model_order) * 0.95), 5.8))
    sns.violinplot(data=p, x="model_label", y="outer_val_k1_r2", order=label_order, cut=0, inner="quartile")
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xticks(rotation=40, ha="right")
    plt.xlabel("Model")
    plt.ylabel("K1 R2")
    plt.title("K1 R2 distribution by model")
    plt.tight_layout()
    plt.savefig(OUT / "k1_rank_violin.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    _ensure_out()
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    sns.set_theme(style="white")

    summary = pd.read_csv(SRC / "summary_by_model.csv")
    model_labels = dict(zip(summary["model"], summary["model_label"]))
    model_order = _model_order(summary)

    df = _load_metrics()
    df.to_csv(OUT / "outer_val_metrics_all_ok.csv", index=False)

    k1_stats, exclusive_df = _compute_k1_threshold_stats(df, model_order)
    best_k1_stats, best_k1_detail = _compute_best_counts(df, model_order, "outer_val_k1_r2")
    best_bulk_stats, best_bulk_detail = _compute_best_counts(df, model_order, "outer_val_bulk_r2")

    drop_cols = [
        "n_targets_k1_gt_0_4",
        "n_exclusive_k1_gt_0_4",
        "n_best_outer_val_k1",
        "n_best_unique_outer_val_k1",
        "n_best_outer_val_bulk",
        "n_best_unique_outer_val_bulk",
    ]
    summary = summary.drop(columns=drop_cols, errors="ignore")
    summary = summary.merge(k1_stats, on="model", how="left")
    summary = summary.merge(best_k1_stats, on="model", how="left")
    summary = summary.merge(best_bulk_stats, on="model", how="left")
    _save_summary_table(summary)

    best_k1_out = best_k1_detail.copy()
    best_k1_out["best_model_labels"] = best_k1_out["best_models"].apply(
        lambda s: ",".join(model_labels.get(m, m) for m in s.split(",") if m)
    )
    best_k1_out.to_csv(OUT / "best_model_per_target_k1.csv", index=False)
    best_bulk_detail.to_csv(OUT / "best_model_per_target_bulk.csv", index=False)

    if not exclusive_df.empty:
        exclusive_out = exclusive_df.copy()
        exclusive_out["model_label"] = exclusive_out["model"].map(model_labels)
        exclusive_out = exclusive_out[["model", "model_label", "target", "outer_val_k1_r2"]]
        exclusive_out.to_csv(OUT / "exclusive_targets_k1_gt_0.4.csv", index=False)
    else:
        pd.DataFrame(columns=["model", "model_label", "target", "outer_val_k1_r2"]).to_csv(
            OUT / "exclusive_targets_k1_gt_0.4.csv",
            index=False,
        )

    matrices: dict[str, pd.DataFrame] = {}
    for metric, label in R2_COLS.items():
        matrix = _save_r2_matrix(df, model_order, metric, f"r2_matrix_{metric.replace('_r2', '')}.csv")
        matrices[metric] = matrix
        _plot_heatmap(
            matrix,
            f"{label} by target and model",
            OUT / f"r2_by_target_{metric.replace('_r2', '')}.png",
            model_labels,
        )

    _plot_heatmap_clustered_columns(
        matrices["outer_val_k1_r2"],
        "K1 outer_val R2 heatmap (column-clustered by model)",
        OUT / "k1_r2_heatmap_clustered_columns.png",
        model_labels,
    )

    line_plots: list[str] = []
    for metric, (label, slug) in LINE_PLOTS.items():
        path = _plot_r2_by_target_lines(
            matrices[metric],
            label,
            slug,
            model_labels,
            model_order,
        )
        line_plots.append(path.name)

    _plot_model_mean_median(summary, model_order)
    _plot_k1_density(df, model_order, model_labels)
    _plot_k1_violin(df, model_order, model_labels)

    threshold_df = summary[
        [
            "model",
            "model_label",
            "n_best_outer_val_k1",
            "n_best_unique_outer_val_k1",
            "n_targets_k1_gt_0_4",
            "n_exclusive_k1_gt_0_4",
        ]
    ].copy()
    if "n_targets_ok" in summary.columns:
        threshold_df["pct_targets_k1_gt_0_4"] = (
            100.0 * threshold_df["n_targets_k1_gt_0_4"] / summary.set_index("model").loc[threshold_df["model"], "n_targets_ok"].values
        ).round(1)
        threshold_df["pct_best_outer_val_k1"] = (
            100.0 * threshold_df["n_best_outer_val_k1"] / summary.set_index("model").loc[threshold_df["model"], "n_targets_ok"].values
        ).round(1)
    threshold_df.to_csv(OUT / "k1_threshold_summary_0.4.csv", index=False)

    with (OUT / "plot_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_metrics": str(SRC / "outer_val_metrics_all.csv"),
                "source_summary": str(SRC / "summary_by_model.csv"),
                "models": model_order,
                "model_labels": model_labels,
                "models_count": int(summary.shape[0]),
                "targets_count": int(df["target"].nunique()),
                "r2_metrics": list(R2_COLS.keys()),
                "baseline_model": BASELINE_MODEL,
                "r2_threshold": R2_THRESHOLD,
                "line_plots": line_plots,
                "heatmap_dpi": DPI,
                "heatmap_cell_size_in": [CELL_W, CELL_H],
            },
            f,
            indent=2,
        )

    print(f"stage03 consolidated report is ready: {OUT}")
    print(f"models: {len(model_order)} -> {', '.join(model_labels[m] for m in model_order)}")


if __name__ == "__main__":
    main()
