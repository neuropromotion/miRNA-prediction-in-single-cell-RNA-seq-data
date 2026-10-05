"""Binary search for max global SC dropout fraction (v3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import common as cm
import masking as mk
import robustness_paths as C


@dataclass
class BisectResult:
    max_absent_frac: float
    min_present_frac: float
    r2_reference: float
    r2_threshold: float
    retention_stat: str
    stat_at_limit: float
    median_at_limit: float
    q25_at_limit: float
    median_baseline: float
    n_sc_features: int
    n_search_masks: int
    n_confirm_masks: int
    bisect_probes: list[dict]
    meets_threshold: bool


def _meets(
    bundle: cm.TargetBundle,
    x: pd.DataFrame,
    y: pd.Series,
    sc_genes: list[str],
    absent_frac: float,
    *,
    threshold: float,
    retention_stat: str,
    n_masks: int,
    rng: np.random.Generator,
    k1_ref: pd.DataFrame | None,
) -> tuple[bool, float, np.ndarray]:
    samples = mk.sample_r2_at_absent_frac(
        bundle,
        x,
        y,
        sc_genes,
        absent_frac,
        n_masks=n_masks,
        rng=rng,
        k1_ref=k1_ref,
    )
    value = mk.retention_statistic(samples, retention_stat)
    return value >= threshold, value, samples


def estimate_max_absent_bisect(
    bundle: cm.TargetBundle,
    x: pd.DataFrame,
    y: pd.DataFrame,
    sc_genes: list[str],
    *,
    r2_reference: float,
    retention_ratio: float = C.RETENTION_RATIO,
    retention_stat: str = C.RETENTION_STAT,
    p_max: float = C.P_MAX,
    bisect_eps: float = C.BISECT_EPS,
    search_n_masks: int = C.SEARCH_N_MASKS,
    confirm_n_masks: int = C.CONFIRM_N_MASKS,
    k1_ref: pd.DataFrame | None = None,
    seed: int = C.SEED,
) -> BisectResult:
    target = bundle.target
    if target not in y.columns:
        raise KeyError(f"{target} not in y columns")
    y_s = y[target]
    threshold = float(retention_ratio * r2_reference)
    rng = np.random.default_rng(seed)
    probes: list[dict] = []

    ok0, stat0, base_samples = _meets(
        bundle,
        x,
        y_s,
        sc_genes,
        0.0,
        threshold=threshold,
        retention_stat=retention_stat,
        n_masks=search_n_masks,
        rng=rng,
        k1_ref=k1_ref,
    )
    median_baseline = float(np.median(base_samples))
    probes.append(
        {
            "absent_frac": 0.0,
            "phase": "search",
            "stat": stat0,
            "ok": ok0,
            "n_masks": search_n_masks,
        }
    )
    if not ok0:
        return BisectResult(
            max_absent_frac=0.0,
            min_present_frac=1.0,
            r2_reference=float(r2_reference),
            r2_threshold=threshold,
            retention_stat=retention_stat,
            stat_at_limit=stat0,
            median_at_limit=float(np.median(base_samples)),
            q25_at_limit=float(np.quantile(base_samples, 0.25)),
            median_baseline=median_baseline,
            n_sc_features=len(sc_genes),
            n_search_masks=search_n_masks,
            n_confirm_masks=confirm_n_masks,
            bisect_probes=probes,
            meets_threshold=False,
        )

    lo, hi = 0.0, float(p_max)
    while hi - lo > bisect_eps:
        mid = round((lo + hi) / 2.0, 4)
        ok, stat, _ = _meets(
            bundle,
            x,
            y_s,
            sc_genes,
            mid,
            threshold=threshold,
            retention_stat=retention_stat,
            n_masks=search_n_masks,
            rng=rng,
            k1_ref=k1_ref,
        )
        probes.append(
            {
                "absent_frac": mid,
                "phase": "search",
                "stat": stat,
                "ok": ok,
                "n_masks": search_n_masks,
            }
        )
        if ok:
            lo = mid
        else:
            hi = mid

    max_af = float(lo)
    _, _, confirm = _meets(
        bundle,
        x,
        y_s,
        sc_genes,
        max_af,
        threshold=threshold,
        retention_stat=retention_stat,
        n_masks=confirm_n_masks,
        rng=rng,
        k1_ref=k1_ref,
    )
    stat_final = mk.retention_statistic(confirm, retention_stat)
    probes.append(
        {
            "absent_frac": max_af,
            "phase": "confirm",
            "stat": stat_final,
            "ok": stat_final >= threshold,
            "n_masks": confirm_n_masks,
        }
    )

    return BisectResult(
        max_absent_frac=max_af,
        min_present_frac=float(1.0 - max_af),
        r2_reference=float(r2_reference),
        r2_threshold=threshold,
        retention_stat=retention_stat,
        stat_at_limit=float(stat_final),
        median_at_limit=float(np.median(confirm)),
        q25_at_limit=float(np.quantile(confirm, 0.25)),
        median_baseline=median_baseline,
        n_sc_features=len(sc_genes),
        n_search_masks=search_n_masks,
        n_confirm_masks=confirm_n_masks,
        bisect_probes=probes,
        meets_threshold=bool(stat_final >= threshold),
    )


def probes_to_dataframe(target: str, cohort: str, probes: list[dict]) -> pd.DataFrame:
    rows = [{"target": target, "cohort": cohort, **p} for p in probes]
    return pd.DataFrame(rows)
