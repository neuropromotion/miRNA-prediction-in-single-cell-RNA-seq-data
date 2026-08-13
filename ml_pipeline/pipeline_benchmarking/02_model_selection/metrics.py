"""Metrics helpers for stage03 model screen."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(r2_score(y_true, y_pred))


def clip_nonneg(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)


def avg_k_cohort_scores(summary_row: dict, k_metric_cols: tuple[str, ...]) -> dict:
    """Average of per-cohort means/medians over K1 + PB K2–K10 (exclude bulk)."""
    means: list[float] = []
    medians: list[float] = []
    for col in k_metric_cols:
        m = summary_row.get(f"mean_{col}")
        d = summary_row.get(f"median_{col}")
        if m is not None and m == m:  # not None / NaN
            means.append(float(m))
        if d is not None and d == d:
            medians.append(float(d))
    out: dict = {}
    if means:
        out["avg_of_means_K"] = float(sum(means) / len(means))
    if medians:
        out["avg_of_medians_K"] = float(sum(medians) / len(medians))
    return out
