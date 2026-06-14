"""Metrics helpers for stage03."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(r2_score(y_true, y_pred))


def clip_nonneg(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
