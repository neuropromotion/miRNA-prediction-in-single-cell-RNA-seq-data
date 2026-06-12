"""Adaptive Spearman screening."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from constants import (
    MAX_SPEARMAN_FEATURES,
    MIN_SPEARMAN_FEATURES,
    SPEARMAN_CHUNK,
    SPEARMAN_THR_HIGH,
    SPEARMAN_THR_LOW,
)


def spearman_abs_all(X: np.ndarray, y: np.ndarray, chunk: int = SPEARMAN_CHUNK) -> np.ndarray:
    y_rank = rankdata(y, method="average").astype(np.float64)
    y_rank = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-12)
    n_features = X.shape[1]
    rhos = np.empty(n_features, dtype=np.float64)
    for start in range(0, n_features, chunk):
        end = min(start + chunk, n_features)
        Xc = X[:, start:end]
        Xr = np.apply_along_axis(rankdata, 0, Xc, method="average").astype(np.float64)
        Xr = Xr - Xr.mean(axis=0)
        Xr = Xr / (Xr.std(axis=0) + 1e-12)
        rhos[start:end] = np.abs((Xr * y_rank[:, None]).mean(axis=0))
    return rhos


def adaptive_spearman_indices(
    X: np.ndarray,
    y: np.ndarray,
    thr_high: float = SPEARMAN_THR_HIGH,
    thr_low: float = SPEARMAN_THR_LOW,
    min_features: int = MIN_SPEARMAN_FEATURES,
    max_features: int = MAX_SPEARMAN_FEATURES,
    rhos: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if rhos is None:
        rhos = spearman_abs_all(X, y)
    order = np.argsort(-rhos)

    def pick(thr: float) -> np.ndarray:
        return np.where(rhos >= thr)[0]

    used_thr = thr_high
    idx = pick(thr_high)
    if idx.size < min_features:
        used_thr = thr_low
        idx = pick(thr_low)
    if idx.size < min_features:
        idx = order[:min_features]
        used_thr = float(rhos[idx[-1]]) if idx.size else 0.0
    if idx.size > max_features:
        idx = order[:max_features]

    meta = {
        "n_after_spearman": int(idx.size),
        "spearman_thr_used": used_thr,
        "max_abs_rho": float(rhos[idx].max()) if idx.size else 0.0,
        "min_abs_rho": float(rhos[idx].min()) if idx.size else 0.0,
    }
    return idx.astype(int), meta
