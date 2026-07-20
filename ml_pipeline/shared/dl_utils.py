"""Shared metrics and torch training helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def metrics(y_true, y_pred) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rho = spearmanr(y_true, y_pred).statistic
    rho = float(rho) if not math.isnan(rho) else -1.0
    return {"r2": r2, "spearman": rho, "score": 0.5 * (r2 + rho)}


@dataclass
class LabelStats:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std + self.mean).astype(np.float64)


def val_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
