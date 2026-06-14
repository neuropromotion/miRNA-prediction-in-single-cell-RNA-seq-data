#!/usr/bin/env python3
"""Prepare and visualize stage03 train / inner-val / test splits for review."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import INNER_VAL_FRAC, RESULTS, SEED, STAGE
from data import build_modality_bundle, modality_fractions
from io_splits import PB_COHORTS

OUT = RESULTS / "data_prep"


def counts_table(bundle) -> pd.DataFrame:
    rows = []
    stats = bundle.impute_stats

    def add_row(split: str, modality: str, n: int, extra: str = "") -> None:
        rows.append({"split": split, "modality": modality, "cohort": extra, "n_samples": n})

    for m, n in stats["pool_modality_counts"].items():
        add_row("pool_train_stage00", m, n)
    for m, n in stats["train_modality_counts"].items():
        add_row("inner_train_85pct", m, n)
    for m, n in stats["inner_val_modality_counts"].items():
        add_row("inner_val_15pct", m, n)
    add_row("test_stage00_val", "bulk", stats["n_test_bulk"])
    add_row("test_stage00_val", "k1", stats["n_test_k1"])
    for c, n in stats["n_test_pb_by_cohort"].items():
        add_row("test_stage00_val", "pb", n, c)
    for c, n in stats["n_train_pb_by_cohort"].items():
        add_row("train_pb_cohort_detail", "pb", n, c)

    return pd.DataFrame(rows)


def stratification_table(bundle) -> pd.DataFrame:
    rows = []
    for split_name, fracs in [
        ("pool", bundle.impute_stats["pool_modality_fractions"]),
        ("inner_train", bundle.impute_stats["train_modality_fractions"]),
        ("inner_val", bundle.impute_stats["inner_val_modality_fractions"]),
    ]:
        for mod in ("bulk", "k1", "pb"):
            rows.append(
                {
                    "split": split_name,
                    "modality": mod,
                    "fraction": fracs.get(mod, 0.0),
                    "pct": round(100.0 * fracs.get(mod, 0.0), 2),
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="modality", columns="split", values="pct")
    pivot["train_minus_val_pp"] = pivot["inner_train"] - pivot["inner_val"]
    pivot.to_csv(OUT / "stratification_pct.csv")
    return df


def sample_weight_table(bundle) -> pd.DataFrame:
    rows = []
    for mod in np.unique(bundle.train_modality):
        mask = bundle.train_modality == mod
        w = bundle.sample_weight[mask]
        rows.append(
            {
                "modality": str(mod),
                "n_samples": int(mask.sum()),
                "weight_mean": float(w.mean()),
                "weight_min": float(w.min()),
                "weight_max": float(w.max()),
                "effective_n": float(1.0 / np.mean(w**2)) if len(w) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def index_audit(bundle, n_per_mod: int = 5) -> pd.DataFrame:
    rows = []
    for split_name, x, mods in [
        ("inner_train", bundle.x_train, bundle.train_modality),
        ("inner_val", bundle.x_val_inner, bundle.val_modality),
    ]:
        for mod in ("bulk", "k1", "pb"):
            idx = x.index[mods == mod][:n_per_mod]
            for sample_id in idx:
                rows.append({"split": split_name, "modality": mod, "sample_id": str(sample_id)})
    for cohort in PB_COHORTS:
        x = bundle.x_test_pb[cohort]
        for sample_id in x.index[:n_per_mod]:
            rows.append(
                {
                    "split": "test_pb",
                    "modality": "pb",
                    "cohort": cohort,
                    "sample_id": str(sample_id),
                }
            )
    return pd.DataFrame(rows)


def plot_modality_counts(counts: pd.DataFrame) -> Path:
    focus = counts[counts["split"].isin(["inner_train_85pct", "inner_val_15pct"])].copy()
    focus["label"] = focus["split"].str.replace("_85pct", "").str.replace("_15pct", "")

    mods = ["bulk", "k1", "pb"]
    x = np.arange(len(mods))
    w = 0.35
    train = focus[focus["split"] == "inner_train_85pct"].set_index("modality").reindex(mods)["n_samples"]
    val = focus[focus["split"] == "inner_val_15pct"].set_index("modality").reindex(mods)["n_samples"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, train, width=w, label="inner train (85%)", color="#2ca02c")
    ax.bar(x + w / 2, val, width=w, label="inner val (15%)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel("Sample count")
    ax.set_title(f"Inner split stratification (seed={SEED}, val_frac={INNER_VAL_FRAC})")
    for i, (a, b) in enumerate(zip(train, val)):
        ax.text(i - w / 2, a + 50, str(int(a)), ha="center", fontsize=8)
        ax.text(i + w / 2, b + 50, str(int(b)), ha="center", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "inner_split_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stratification_pct(strat: pd.DataFrame) -> Path:
    pivot = strat.pivot(index="modality", columns="split", values="pct")
    cols = [c for c in ("pool", "inner_train", "inner_val") if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Modality share (%)")
    ax.set_title("Modality composition: pool vs inner train vs inner val")
    ax.legend(title="split")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "stratification_pct.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_test_overview(bundle) -> Path:
    labels = ["bulk", "k1"] + [f"pb_{c}" for c in PB_COHORTS]
    values = [
        len(bundle.x_test_bulk),
        len(bundle.x_test_k1),
    ] + [len(bundle.x_test_pb[c]) for c in PB_COHORTS]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#1f77b4")
    ax.set_ylabel("Sample count")
    ax.set_title("Test set (stage00 val) — per modality / PB cohort")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(values):
        ax.text(i, v + 10, str(v), ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "test_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_train_pool_overview(bundle) -> Path:
    stats = bundle.impute_stats
    labels = ["bulk", "k1", "pb_all"] + [f"pb_{c}" for c in PB_COHORTS]
    values = [
        stats["pool_modality_counts"]["bulk"],
        stats["pool_modality_counts"]["k1"],
        stats["pool_modality_counts"]["pb"],
    ] + [stats["n_train_pb_by_cohort"][c] for c in PB_COHORTS]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c"] * 3 + ["#98df8a"] * len(PB_COHORTS)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Sample count")
    ax.set_title("Train pool (stage00 train) — modalities and PB cohorts")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(values):
        ax.text(i, v + 50, str(v), ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "train_pool_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_protocol(bundle) -> None:
    protocol = {
        "description": "Stage03 data protocol",
        "train_source": "stage00 train: bulk + K1 (KNN k=5 imputed) + PB all cohorts",
        "inner_split": {
            "method": "train_test_split stratified by modality tag",
            "train_frac": 1.0 - INNER_VAL_FRAC,
            "val_frac": INNER_VAL_FRAC,
            "seed": SEED,
            "inner_val_scope": "mixed bulk+k1+pb (same composition as train)",
        },
        "test_source": "stage00 val (frozen)",
        "test_reporting": ["bulk", "k1", "pb_K2", "pb_K3", "pb_K4", "pb_K5", "pb_K10"],
        "imputation": "K1 only; ref = K1 train",
        "sample_weights": "inverse modality frequency on inner train",
        "impute_stats": bundle.impute_stats,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building modality bundle...", flush=True)
    bundle = build_modality_bundle()

    counts = counts_table(bundle)
    counts.to_csv(OUT / "split_counts.csv", index=False)

    strat = stratification_table(bundle)
    strat.to_csv(OUT / "stratification_long.csv", index=False)

    weights = sample_weight_table(bundle)
    weights.to_csv(OUT / "sample_weights.csv", index=False)

    audit = index_audit(bundle)
    audit.to_csv(OUT / "index_audit_sample.csv", index=False)

    write_protocol(bundle)

    plots = [
        plot_train_pool_overview(bundle),
        plot_modality_counts(counts),
        plot_stratification_pct(strat),
        plot_test_overview(bundle),
    ]

    meta = {
        "output_dir": str(OUT.relative_to(STAGE)),
        "plots": [str(p.relative_to(STAGE)) for p in plots],
        "tables": [
            "split_counts.csv",
            "stratification_pct.csv",
            "stratification_long.csv",
            "sample_weights.csv",
            "index_audit_sample.csv",
            "protocol.json",
        ],
    }
    (OUT / "prep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Split counts ===")
    print(counts.to_string(index=False))
    print("\n=== Stratification (pct) ===")
    print(pd.read_csv(OUT / "stratification_pct.csv").to_string())
    print("\n=== Sample weights ===")
    print(weights.to_string(index=False))
    print(f"\nSaved review artifacts to {OUT}")


if __name__ == "__main__":
    main()
