#!/usr/bin/env python3
"""Tables + figures for final_train bases and Ridge stack (Stage00 val)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TRAIN_DIR = Path(__file__).resolve().parent
RESULTS = TRAIN_DIR / "results"
ENSEMBLE = "tabpack_dcnv2_tabm_stack"
STACK_DIR = RESULTS / "ensemble" / ENSEMBLE
# Canonical deliverables live under train/{figures,tables} (not nested under results/).
FIGS = TRAIN_DIR / "figures"
TABLES = TRAIN_DIR / "tables"

MODELS = ["tabpack", "dcnv2", "tabm", "stack"]
MODEL_LABELS = {
    "tabpack": "TabPack Muon",
    "dcnv2": "DCNv2 AdamW",
    "tabm": "TabM AdamW",
    "stack": "Ridge stack",
}
MODEL_COLORS = {
    "tabpack": "#1f77b4",
    "dcnv2": "#ff7f0e",
    "tabm": "#2ca02c",
    "stack": "#d62728",
}

SPLIT_COLS = [
    "val_k1_r2",
    "val_pb_r2",
    "val_bulk_r2",
    "val_pb_K2_r2",
    "val_pb_K3_r2",
    "val_pb_K4_r2",
    "val_pb_K5_r2",
    "val_pb_K10_r2",
]
SPLIT_LABELS = {
    "val_k1_r2": "val K1",
    "val_pb_r2": "val PB (all)",
    "val_bulk_r2": "val bulk",
    "val_pb_K2_r2": "val PB K2",
    "val_pb_K3_r2": "val PB K3",
    "val_pb_K4_r2": "val PB K4",
    "val_pb_K5_r2": "val PB K5",
    "val_pb_K10_r2": "val PB K10",
}
PRIMARY = ["val_k1_r2", "val_pb_r2", "val_bulk_r2"]
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGS / f"{name}.png", format="png", bbox_inches="tight", facecolor="white", dpi=DPI)
    plt.close(fig)


def load_all() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model in ("tabpack", "dcnv2", "tabm"):
        df = pd.read_csv(RESULTS / model / "val_metrics.csv")
        df = df[df["status"] == "ok"].copy()
        df["model"] = model
        parts.append(df)
    stack = pd.read_csv(STACK_DIR / "val_metrics.csv")
    stack = stack[stack["status"] == "ok"].copy()
    stack["model"] = "stack"
    if "val_mix_r2" not in stack.columns and "tune_r2" in stack.columns:
        stack["val_mix_r2"] = stack["tune_r2"]
    parts.append(stack)
    return pd.concat(parts, ignore_index=True, sort=False)


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model]
        row: dict = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n_targets": len(sub),
        }
        for col in SPLIT_COLS:
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            row[f"mean_{col}"] = float(s.mean())
            row[f"median_{col}"] = float(s.median())
            row[f"std_{col}"] = float(s.std())
            row[f"q25_{col}"] = float(s.quantile(0.25))
            row[f"q75_{col}"] = float(s.quantile(0.75))
        if model == "stack" and "fallback" in sub.columns:
            row["n_fallback"] = int(sub["fallback"].fillna(False).astype(bool).sum())  # noqa: PD901
        rows.append(row)
    return pd.DataFrame(rows)


def compact_summary(full: pd.DataFrame) -> pd.DataFrame:
    """Readable mean/median table for primary splits."""
    rows = []
    for model in MODELS:
        sub = full[full["model"] == model].iloc[0]
        row = {"model": MODEL_LABELS[model], "n": int(sub["n_targets"])}
        for col in PRIMARY:
            row[f"median {SPLIT_LABELS[col]}"] = round(float(sub[f"median_{col}"]), 4)
            row[f"mean {SPLIT_LABELS[col]}"] = round(float(sub[f"mean_{col}"]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def per_target_wide(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["target"] + [c for c in SPLIT_COLS if c in df.columns]
    pieces = []
    for model in MODELS:
        sub = df[df["model"] == model][keep].copy()
        rename = {c: f"{model}__{c}" for c in keep if c != "target"}
        pieces.append(sub.rename(columns=rename))
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="target", how="outer")
    # deltas vs best solo on K1 / PB / bulk
    for split in PRIMARY:
        solo_cols = [f"{m}__{split}" for m in ("tabpack", "dcnv2", "tabm")]
        out[f"best_solo__{split}"] = out[solo_cols].max(axis=1)
        out[f"stack_minus_best_solo__{split}"] = out[f"stack__{split}"] - out[f"best_solo__{split}"]
        out[f"best_solo_model__{split}"] = out[solo_cols].idxmax(axis=1).str.replace(f"__{split}", "", regex=False)
    if "fallback" in df.columns:
        fb = df[df["model"] == "stack"][["target", "fallback", "tune_r2"]].copy()
        out = out.merge(fb, on="target", how="left")
    return out.sort_values("stack__val_k1_r2", ascending=False)


def n_best_counts(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in PRIMARY:
        cols = {m: f"{m}__{split}" for m in MODELS}
        mat = wide[[cols[m] for m in MODELS]].copy()
        mat.columns = MODELS
        best = mat.idxmax(axis=1)
        for model in MODELS:
            rows.append(
                {
                    "split": SPLIT_LABELS[split],
                    "model": MODEL_LABELS[model],
                    "n_best": int((best == model).sum()),
                    "frac_best": float((best == model).mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_primary_box(df: pd.DataFrame) -> None:
    long = df.melt(
        id_vars=["target", "model"],
        value_vars=PRIMARY,
        var_name="split",
        value_name="r2",
    )
    long["model_label"] = long["model"].map(MODEL_LABELS)
    long["split_label"] = long["split"].map(SPLIT_LABELS)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=long,
        x="split_label",
        y="r2",
        hue="model_label",
        hue_order=[MODEL_LABELS[m] for m in MODELS],
        palette=[MODEL_COLORS[m] for m in MODELS],
        ax=ax,
        showfliers=False,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("")
    ax.set_ylabel("R²")
    ax.set_title("Final train — Stage00 val R² (312 targets)")
    ax.legend(title="", loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "r2_boxplot_primary_splits")


def plot_median_bars(full: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, col in zip(axes, PRIMARY):
        vals = [float(full[full["model"] == m].iloc[0][f"median_{col}"]) for m in MODELS]
        bars = ax.bar(
            [MODEL_LABELS[m] for m in MODELS],
            vals,
            color=[MODEL_COLORS[m] for m in MODELS],
            edgecolor="white",
        )
        ax.set_title(f"Median {SPLIT_LABELS[col]}")
        ax.set_ylabel("R²")
        ax.set_ylim(0, max(0.05, max(vals) * 1.15))
        ax.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Median R² by model (Stage00 val)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "median_r2_bars")


def plot_k1_scatter(wide: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharex=True, sharey=True)
    for ax, model in zip(axes, ("tabpack", "dcnv2", "tabm")):
        x = wide[f"{model}__val_k1_r2"]
        y = wide["stack__val_k1_r2"]
        ax.scatter(x, y, s=14, alpha=0.55, color=MODEL_COLORS[model], edgecolors="none")
        lims = [
            min(x.min(), y.min(), -0.2),
            max(x.max(), y.max(), 1.0),
        ]
        ax.plot(lims, lims, color="black", linewidth=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"{MODEL_LABELS[model]} val K1 R²")
        ax.set_ylabel("Stack val K1 R²")
        ax.set_title(MODEL_LABELS[model])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        win = float((y > x).mean())
        ax.text(0.05, 0.95, f"stack > solo: {100*win:.0f}%", transform=ax.transAxes, va="top", fontsize=9)
    fig.suptitle("Stack vs each base on val K1", fontsize=12, y=1.03)
    fig.tight_layout()
    _save(fig, "stack_vs_solo_k1_scatter")


def plot_delta_hist(wide: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, split in zip(axes, PRIMARY):
        d = wide[f"stack_minus_best_solo__{split}"].dropna()
        ax.hist(d, bins=30, color=MODEL_COLORS["stack"], edgecolor="white", alpha=0.9)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.axvline(d.median(), color="#333333", linestyle="--", linewidth=1.2, label=f"median={d.median():.3f}")
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("stack − best solo R²")
        ax.set_ylabel("n targets")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Stack gain over best base model", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "stack_minus_best_solo_hist")


def plot_pb_cohorts(df: pd.DataFrame) -> None:
    cols = [c for c in SPLIT_COLS if c.startswith("val_pb_K")]
    long = df.melt(
        id_vars=["target", "model"],
        value_vars=cols,
        var_name="split",
        value_name="r2",
    )
    long["model_label"] = long["model"].map(MODEL_LABELS)
    long["split_label"] = long["split"].map(SPLIT_LABELS)
    med = (
        long.groupby(["split_label", "model_label"], as_index=False)["r2"]
        .median()
        .rename(columns={"r2": "median_r2"})
    )
    order = [SPLIT_LABELS[c] for c in cols]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(
        data=med,
        x="split_label",
        y="median_r2",
        hue="model_label",
        order=order,
        hue_order=[MODEL_LABELS[m] for m in MODELS],
        palette=[MODEL_COLORS[m] for m in MODELS],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median R²")
    ax.set_title("Median R² by PB cohort (Stage00 val)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="", fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "median_r2_pb_cohorts")


def _df_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *body])


def write_readme(full: pd.DataFrame, wide: pd.DataFrame, n_best: pd.DataFrame) -> None:
    compact = compact_summary(full)
    fb_raw = full[full["model"] == "stack"].iloc[0].get("n_fallback")
    fb = int(fb_raw) if pd.notna(fb_raw) else 0
    k1_gain = wide["stack_minus_best_solo__val_k1_r2"]
    lines = [
        "# Final train report — Stage00 validation",
        "",
        f"Ensemble: `{ENSEMBLE}`  ",
        "Bases: TabPack Muon + DCNv2 AdamW + TabM AdamW → Ridge stack  ",
        f"Targets: **312** (15 zero-expressed excluded)  ",
        f"Stack fallbacks to best solo: **{fb}** / 312  ",
        "Tune splits: `val_k1` + `val_pb_K*` (no bulk)",
        "",
        "## Median / mean R²",
        "",
        _df_md(compact),
        "",
        f"Stack − best solo on val K1: median **{k1_gain.median():.4f}**, "
        f"mean **{k1_gain.mean():.4f}**, "
        f"stack wins on **{(k1_gain > 0).mean()*100:.1f}%** targets",
        "",
        "## n_best (argmax R² per target)",
        "",
        _df_md(n_best),
        "",
        "## Files",
        "",
        f"- `{TABLES.relative_to(TRAIN_DIR)}/summary_by_model.csv` — full mean/median/std/quantiles",
        f"- `{TABLES.relative_to(TRAIN_DIR)}/summary_compact.csv` — primary splits only",
        f"- `{TABLES.relative_to(TRAIN_DIR)}/per_target_all_models.csv` — wide join + deltas",
        f"- `{TABLES.relative_to(TRAIN_DIR)}/n_best_by_split.csv`",
        f"- `{FIGS.relative_to(TRAIN_DIR)}/*.pdf` + `*.png`",
        f"- weight tables/figures from `plot_stack_weights.py` in the same dirs",
        "",
        "True holdout (`sc_TEST` / `bulk_TEST`) is not in this report.",
        "",
    ]
    (TABLES / "README_final_train_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    mpl.rcParams["pdf.fonttype"] = 42
    sns.set_theme(style="whitegrid")

    df = load_all()
    full = summary_by_model(df)
    compact = compact_summary(full)
    wide = per_target_wide(df)
    n_best = n_best_counts(wide)

    full.to_csv(TABLES / "summary_by_model.csv", index=False)
    compact.to_csv(TABLES / "summary_compact.csv", index=False)
    wide.to_csv(TABLES / "per_target_all_models.csv", index=False)
    n_best.to_csv(TABLES / "n_best_by_split.csv", index=False)

    # also park summary copies next to stack root for quick access
    compact.to_csv(STACK_DIR / "summary_compact.csv", index=False)
    full.to_csv(STACK_DIR / "summary_by_model.csv", index=False)

    plot_primary_box(df)
    plot_median_bars(full)
    plot_k1_scatter(wide)
    plot_delta_hist(wide)
    plot_pb_cohorts(df)
    write_readme(full, wide, n_best)

    print(f"Wrote figures → {FIGS}")
    print(f"Wrote tables  → {TABLES}")
    print(compact.to_string(index=False))
    print()
    print(n_best.to_string(index=False))


if __name__ == "__main__":
    main()
