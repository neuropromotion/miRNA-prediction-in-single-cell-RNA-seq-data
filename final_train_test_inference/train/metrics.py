"""Regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def clip_nonneg(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(r2_score(y_true, y_pred))


def weighted_r2(y_true, y_pred, sample_weight) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.sum() <= 0:
        return r2(y_true, y_pred)
    y_mean = np.average(y_true, weights=w)
    ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def weighted_rmse(y_true, y_pred, sample_weight) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.sum() <= 0:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))
