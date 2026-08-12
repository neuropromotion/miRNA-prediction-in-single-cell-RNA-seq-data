"""Figures for final TEST metrics (png + pdf)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import COHORTS, FIGURES

DPI = 300
K_COLORS = {
    "K1": "#1f77b4",
    "K2": "#2ca02c",
    "K3": "#ff7f0e",
    "K4": "#d62728",
    "K5": "#9467bd",
    "K10": "#FF1493",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        FIGURES / f"{name}.png",
        format="png",
        bbox_inches="tight",
        facecolor="white",
        dpi=DPI,
    )
    plt.close(fig)


def plot_all(df: pd.DataFrame) -> None:
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    mpl.rcParams["pdf.fonttype"] = 42
    sns.set_theme(style="whitegrid")
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    _plot_r2_box(ok)
    _plot_mse_box(ok)
    _plot_sc_vs_bulk_r2(ok)
    _plot_by_assigned_k(ok)


def _plot_r2_box(ok: pd.DataFrame) -> None:
    long = pd.DataFrame(
        {
            "target": np.concatenate([ok["target"], ok["target"]]),
            "modality": ["SC @ assigned K"] * len(ok) + ["bulk"] * len(ok),
            "median_r2": np.concatenate([ok["sc_r2_median"], ok["bulk_r2_median"]]),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=long, x="modality", y="median_r2", ax=ax, showfliers=False)
    sns.stripplot(data=long, x="modality", y="median_r2", ax=ax, color="0.3", size=3, alpha=0.45)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel("Bootstrap median R² (eval)")
    ax.set_xlabel("")
    ax.set_title(f"Final TEST R² (n={len(ok)} eligible)")
    fig.tight_layout()
    _save(fig, "eval_r2_boxplot")


def _plot_mse_box(ok: pd.DataFrame) -> None:
    long = pd.DataFrame(
        {
            "target": np.concatenate([ok["target"], ok["target"]]),
            "modality": ["SC @ assigned K"] * len(ok) + ["bulk"] * len(ok),
            "median_mse": np.concatenate([ok["sc_mse_median"], ok["bulk_mse_median"]]),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=long, x="modality", y="median_mse", ax=ax, showfliers=False)
    sns.stripplot(data=long, x="modality", y="median_mse", ax=ax, color="0.3", size=3, alpha=0.45)
    ax.set_ylabel("Bootstrap median MSE (eval)")
    ax.set_xlabel("")
    ax.set_title(f"Final TEST MSE (n={len(ok)} eligible)")
    fig.tight_layout()
    _save(fig, "eval_mse_boxplot")


def _plot_sc_vs_bulk_r2(ok: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k in COHORTS:
        sub = ok[ok["assigned_k"] == k]
        if sub.empty:
            continue
        ax.scatter(
            sub["bulk_r2_median"],
            sub["sc_r2_median"],
            s=36,
            alpha=0.75,
            color=K_COLORS.get(k, "gray"),
            label=f"{k} (n={len(sub)})",
        )
    lims = [
        min(ok["bulk_r2_median"].min(), ok["sc_r2_median"].min(), -0.1),
        max(ok["bulk_r2_median"].max(), ok["sc_r2_median"].max(), 1.0),
    ]
    ax.plot(lims, lims, color="black", linewidth=1, alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("bulk median R² (eval)")
    ax.set_ylabel("SC@K median R² (eval)")
    ax.set_title("Eval bootstrap medians: SC vs bulk")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, "eval_sc_vs_bulk_r2")


def _plot_by_assigned_k(ok: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = [k for k in COHORTS if (ok["assigned_k"] == k).any()]
    sns.boxplot(
        data=ok,
        x="assigned_k",
        y="sc_r2_median",
        order=order,
        ax=axes[0],
        showfliers=False,
        palette=[K_COLORS[k] for k in order],
    )
    axes[0].set_title("SC median R² by assigned K")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Bootstrap median R²")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)

    sns.boxplot(
        data=ok,
        x="assigned_k",
        y="sc_mse_median",
        order=order,
        ax=axes[1],
        showfliers=False,
        palette=[K_COLORS[k] for k in order],
    )
    axes[1].set_title("SC median MSE by assigned K")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Bootstrap median MSE")
    fig.suptitle("Eval metrics stratified by Optimal_K assignment", y=1.02)
    fig.tight_layout()
    _save(fig, "eval_metrics_by_assigned_k")
