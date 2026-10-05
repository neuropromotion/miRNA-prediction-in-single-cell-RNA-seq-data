"""Global SC-gene dropout masks and R² sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as cm


def apply_global_dropout(
    x: pd.DataFrame,
    genes: list[str],
    absent_frac: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    panel = [g for g in genes if g in x.columns]
    if not panel or absent_frac <= 0:
        return x
    n_drop = int(round(absent_frac * len(panel)))
    n_drop = min(max(n_drop, 0), len(panel))
    if n_drop <= 0:
        return x
    chosen = list(rng.choice(panel, size=n_drop, replace=False))
    out = x.copy()
    out.loc[:, chosen] = 0.0
    return out


def sample_r2_at_absent_frac(
    bundle: cm.TargetBundle,
    x: pd.DataFrame,
    y: pd.Series,
    sc_genes: list[str],
    absent_frac: float,
    *,
    n_masks: int,
    rng: np.random.Generator,
    k1_ref: pd.DataFrame | None,
) -> np.ndarray:
    """Return ``n_masks`` R² values (unmasked baseline if ``absent_frac==0``)."""
    y_true = y.to_numpy()
    out = np.empty(n_masks, dtype=np.float64)
    for i in range(n_masks):
        if absent_frac <= 0:
            x_m = x
        else:
            x_m = apply_global_dropout(x, sc_genes, absent_frac, rng)
        pred = cm.predict_with_bundle(bundle, x_m, k1_ref)
        out[i] = cm.r2(y_true, pred)
    return out


def retention_statistic(r2_samples: np.ndarray, stat: str) -> float:
    stat = stat.lower().strip()
    if stat == "median":
        return float(np.median(r2_samples))
    if stat in ("q25", "p25", "quantile25"):
        return float(np.quantile(r2_samples, 0.25))
    raise ValueError(f"unknown retention stat: {stat!r} (use median or q25)")
