"""Leave-one-out SC gene ablation on eval TEST."""

from __future__ import annotations

import pandas as pd

import masking as mk


def compute_loo_importance(
    bundle,
    x: pd.DataFrame,
    y: pd.Series,
    sc_genes: list[str],
    *,
    common_mod,
    k1_ref: pd.DataFrame | None,
    progress_every: int = 10,
    on_progress=None,
) -> tuple[float, pd.DataFrame]:
    """Return baseline R² and table gene, delta_r2, weight, rank."""
    y_true = y.to_numpy()
    pred0 = common_mod.predict_with_bundle(bundle, x, k1_ref)
    r2_0 = common_mod.r2(y_true, pred0)

    rows: list[dict] = []
    n_genes = len(sc_genes)
    for i, g in enumerate(sc_genes, 1):
        if on_progress and (i == 1 or i % progress_every == 0 or i == n_genes):
            on_progress(i, n_genes)
        if g not in x.columns:
            rows.append({"gene": g, "delta_r2": float("nan"), "in_matrix": False})
            continue
        x_m = mk.zero_genes(x, [g])
        pred = common_mod.predict_with_bundle(bundle, x_m, k1_ref)
        r2_g = common_mod.r2(y_true, pred)
        rows.append({"gene": g, "delta_r2": max(0.0, r2_0 - r2_g), "in_matrix": True})

    df = pd.DataFrame(rows)
    in_mat = df["in_matrix"] & df["delta_r2"].notna()
    total = float(df.loc[in_mat, "delta_r2"].sum())
    if total > 0:
        df["weight"] = df["delta_r2"].fillna(0.0) / total
    else:
        n = max(int(in_mat.sum()), 1)
        df["weight"] = df["in_matrix"].astype(float) / n
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["r2_baseline"] = r2_0
    return r2_0, df
