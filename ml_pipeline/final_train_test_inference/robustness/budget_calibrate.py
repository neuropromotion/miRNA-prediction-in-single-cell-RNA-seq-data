"""Bisect max importance budget B on eval (joint masks, weighted subsets)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import masking as mk


def retention_statistic(r2_samples: np.ndarray, stat: str) -> float:
    stat = stat.lower().strip()
    if stat == "median":
        return float(np.median(r2_samples))
    if stat in ("q25", "p25", "quantile25"):
        return float(np.quantile(r2_samples, 0.25))
    raise ValueError(f"unknown retention stat: {stat!r}")


def sample_r2_at_budget(
    bundle,
    x: pd.DataFrame,
    y: pd.Series,
    sc_genes: list[str],
    weights: dict[str, float],
    budget: float,
    *,
    n_masks: int,
    rng: np.random.Generator,
    common_mod,
    k1_ref: pd.DataFrame | None,
) -> np.ndarray:
    y_true = y.to_numpy()
    panel = [g for g in sc_genes if g in x.columns]
    out = np.empty(n_masks, dtype=np.float64)
    for i in range(n_masks):
        if budget <= 0:
            x_m = x
        else:
            drop = mk.random_subset_at_budget(panel, weights, budget, rng)
            x_m = mk.zero_genes(x, drop) if drop else x
        pred = common_mod.predict_with_bundle(bundle, x_m, k1_ref)
        out[i] = common_mod.r2(y_true, pred)
    return out


@dataclass
class BudgetResult:
    importance_budget: float
    r2_reference: float
    r2_threshold: float
    r2_baseline_loo: float
    stat_at_budget: float
    median_at_budget: float
    meets_threshold: bool
    n_sc_features: int
    probes: list[dict]


def calibrate_importance_budget(
    bundle,
    x: pd.DataFrame,
    y: pd.Series,
    sc_genes: list[str],
    weights: dict[str, float],
    *,
    r2_reference: float,
    retention_ratio: float,
    retention_stat: str,
    search_n_masks: int,
    confirm_n_masks: int,
    budget_eps: float,
    b_max: float,
    common_mod,
    k1_ref: pd.DataFrame | None,
    seed: int,
) -> BudgetResult:
    threshold = float(retention_ratio * r2_reference)
    rng = np.random.default_rng(seed)
    probes: list[dict] = []

    base_samples = sample_r2_at_budget(
        bundle,
        x,
        y,
        sc_genes,
        weights,
        0.0,
        n_masks=search_n_masks,
        rng=rng,
        common_mod=common_mod,
        k1_ref=k1_ref,
    )
    stat0 = retention_statistic(base_samples, retention_stat)
    r2_baseline = float(np.median(base_samples))
    probes.append({"budget": 0.0, "phase": "search", "stat": stat0, "ok": stat0 >= threshold})

    if stat0 < threshold:
        return BudgetResult(
            importance_budget=0.0,
            r2_reference=float(r2_reference),
            r2_threshold=threshold,
            r2_baseline_loo=r2_baseline,
            stat_at_budget=stat0,
            median_at_budget=float(np.median(base_samples)),
            meets_threshold=False,
            n_sc_features=len(sc_genes),
            probes=probes,
        )

    lo, hi = 0.0, float(b_max)
    while hi - lo > budget_eps:
        mid = round((lo + hi) / 2.0, 4)
        samples = sample_r2_at_budget(
            bundle,
            x,
            y,
            sc_genes,
            weights,
            mid,
            n_masks=search_n_masks,
            rng=rng,
            common_mod=common_mod,
            k1_ref=k1_ref,
        )
        stat = retention_statistic(samples, retention_stat)
        ok = stat >= threshold
        probes.append({"budget": mid, "phase": "search", "stat": stat, "ok": ok})
        if ok:
            lo = mid
        else:
            hi = mid

    b_star = float(lo)
    confirm = sample_r2_at_budget(
        bundle,
        x,
        y,
        sc_genes,
        weights,
        b_star,
        n_masks=confirm_n_masks,
        rng=rng,
        common_mod=common_mod,
        k1_ref=k1_ref,
    )
    stat_final = retention_statistic(confirm, retention_stat)
    probes.append(
        {
            "budget": b_star,
            "phase": "confirm",
            "stat": stat_final,
            "ok": stat_final >= threshold,
        }
    )

    return BudgetResult(
        importance_budget=b_star,
        r2_reference=float(r2_reference),
        r2_threshold=threshold,
        r2_baseline_loo=r2_baseline,
        stat_at_budget=float(stat_final),
        median_at_budget=float(np.median(confirm)),
        meets_threshold=bool(stat_final >= threshold),
        n_sc_features=len(sc_genes),
        probes=probes,
    )
