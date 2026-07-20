"""Line plots: per-target R2 by method, sorted by Spearman within each modality."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from constants import RESULTS, STAGE

COMPARISON = RESULTS / "comparison"
R2_THRESHOLD = 0.4

PLOT_ORDER = [
    "spearman",
    "lasso",
    "elasticnet",
    "xgb",
    "mi",
    "lasso_xgb",
    "lasso_mi",
    "elasticnet_xgb",
    "elasticnet_mi",
]

LABELS = {
    "spearman": "Spearman",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "xgb": "XGB",
    "mi": "MI",
    "lasso_xgb": "Lasso+XGB",
    "lasso_mi": "Lasso+MI",
    "elasticnet_xgb": "ElasticNet+XGB",
    "elasticnet_mi": "ElasticNet+MI",
}

COLORS = {
    "spearman": "#1f77b4",
    "lasso": "#ff7f0e",
    "elasticnet": "#2ca02c",
    "xgb": "#d62728",
    "mi": "#9467bd",
    "lasso_xgb": "#8c564b",
    "lasso_mi": "#e377c2",
    "elasticnet_xgb": "#7f7f7f",
    "elasticnet_mi": "#bcbd22",
}


def load_metrics() -> pd.DataFrame:
    path = COMPARISON / "val_metrics_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run train_eval_comparison.py first: {path}")
    return pd.read_csv(path)


def plot_modality(df: pd.DataFrame, modality: str) -> Path:
    col = f"{modality}_val_r2"
    baseline = "spearman"
    base_df = df[df["config"] == baseline][["target", col]].rename(columns={col: "baseline_r2"})
    order = base_df.sort_values("baseline_r2", ascending=True)["target"].tolist()

    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(14, 6))

    for config in PLOT_ORDER:
        sub = df[df["config"] == config].set_index("target").reindex(order)
        y = sub[col].to_numpy()
        lw = 2.5 if config == baseline else 1.5
        alpha = 1.0 if config == baseline else 0.85
        ax.plot(x, y, label=LABELS[config], color=COLORS[config], linewidth=lw, alpha=alpha)

    ax.axhline(R2_THRESHOLD, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"R²={R2_THRESHOLD}")
    ax.set_xlabel("Targets (sorted by Spearman R²)")
    ax.set_ylabel(f"{modality.upper()} validation R²")
    ax.set_title(f"Per-target {modality.upper()} R² by feature-selection method (XGB baseline)")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([order[i] for i in x[::5]], rotation=60, ha="right", fontsize=7)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = COMPARISON / f"r2_by_target_{modality}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_sorted_tables(df: pd.DataFrame) -> None:
    for modality in ("bulk", "sc"):
        col = f"{modality}_val_r2"
        baseline = "spearman"
        order = (
            df[df["config"] == baseline][["target", col]]
            .sort_values(col, ascending=True)["target"]
            .tolist()
        )
        wide = df.pivot(index="target", columns="config", values=col).reindex(order)
        wide.index.name = "target"
        wide.to_csv(COMPARISON / f"r2_matrix_{modality}.csv")


def main() -> None:
    df = load_metrics()
    bulk_path = plot_modality(df, "bulk")
    sc_path = plot_modality(df, "sc")
    save_sorted_tables(df)

    meta = {
        "r2_threshold": R2_THRESHOLD,
        "sort_by": "spearman ascending within each modality plot",
        "plots": [str(bulk_path.relative_to(STAGE)), str(sc_path.relative_to(STAGE))],
        "configs": PLOT_ORDER,
    }
    with (COMPARISON / "plot_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {bulk_path}")
    print(f"Saved {sc_path}")


if __name__ == "__main__":
    main()
