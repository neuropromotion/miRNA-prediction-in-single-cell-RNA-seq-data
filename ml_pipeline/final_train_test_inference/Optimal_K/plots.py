"""Figures for Optimal_K (png + pdf → figures/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import COHORTS, DELTA, FIGURES, MEDIAN_THRESHOLD

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


def plot_all(decisions: pd.DataFrame) -> None:
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    mpl.rcParams["pdf.fonttype"] = 42
    sns.set_theme(style="whitegrid")

    _plot_cohort_counts(decisions)
    _plot_median_box(decisions)
    _plot_bulk_vs_best_sc(decisions)
    _plot_eligible_scatter(decisions)


def _plot_cohort_counts(decisions: pd.DataFrame) -> None:
    elig = decisions[decisions["eligible"]]
    counts = elig["optimal_k"].value_counts().reindex(list(COHORTS), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [K_COLORS[k] for k in counts.index]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="white")
    ax.set_ylabel("n targets")
    ax.set_title(
        f"Optimal_K assignments (eligible={len(elig)} / {len(decisions)}; "
        f"thr={MEDIAN_THRESHOLD}, δ={DELTA})"
    )
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(int(v)), ha="center", va="bottom")
    ax.set_ylim(0, max(counts.values.max() * 1.15, 1))
    fig.tight_layout()
    _save(fig, "optimal_k_cohort_counts")


def _plot_median_box(decisions: pd.DataFrame) -> None:
    cols = [f"m_{c}" for c in COHORTS] + ["m_bulk"]
    long = decisions.melt(
        id_vars=["target", "eligible"],
        value_vars=cols,
        var_name="cohort",
        value_name="median_r2",
    )
    long["cohort"] = long["cohort"].str.replace("m_", "", regex=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=long,
        x="cohort",
        y="median_r2",
        order=[*COHORTS, "bulk"],
        hue="eligible",
        ax=ax,
        showfliers=False,
    )
    ax.axhline(MEDIAN_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"thr={MEDIAN_THRESHOLD}")
    ax.set_xlabel("")
    ax.set_ylabel("Bootstrap median R² (tune)")
    ax.set_title("Tune-half bootstrap median R² by cohort")
    ax.legend(title="eligible")
    fig.tight_layout()
    _save(fig, "bootstrap_median_r2_box")


def _plot_bulk_vs_best_sc(decisions: pd.DataFrame) -> None:
    m_sc = decisions[[f"m_{c}" for c in COHORTS]].max(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for flag, marker, label in ((True, "o", "eligible"), (False, "x", "ineligible")):
        mask = decisions["eligible"] == flag
        ax.scatter(
            decisions.loc[mask, "m_bulk"],
            m_sc[mask],
            s=28,
            alpha=0.7,
            marker=marker,
            label=label,
        )
    ax.axvline(MEDIAN_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.axhline(MEDIAN_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("median R² bulk (tune)")
    ax.set_ylabel("max_K median R² (tune)")
    ax.set_title("Eligibility gate")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, "eligibility_bulk_vs_maxK")


def _plot_eligible_scatter(decisions: pd.DataFrame) -> None:
    elig = decisions[decisions["eligible"]].copy()
    if elig.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for k in COHORTS:
        sub = elig[elig["optimal_k"] == k]
        if sub.empty:
            continue
        ax.scatter(
            sub["m_bulk"],
            sub[f"m_{k}"],
            s=36,
            alpha=0.75,
            color=K_COLORS[k],
            label=f"{k} (n={len(sub)})",
        )
    ax.axhline(MEDIAN_THRESHOLD, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("median R² bulk")
    ax.set_ylabel("median R² at assigned K")
    ax.set_title("Eligible targets: bulk vs assigned-K median R²")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "eligible_bulk_vs_assigned_k")
