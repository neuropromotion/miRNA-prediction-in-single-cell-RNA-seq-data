"""Bootstrap R² and MSE on eval indices."""

from __future__ import annotations

import numpy as np

from config import N_BOOTSTRAP, SEED


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def mse_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def bootstrap_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Return bootstrap distributions for R² and MSE (length n_boot)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    n = len(y)
    r2s = np.empty(n_boot, dtype=np.float64)
    mses = np.empty(n_boot, dtype=np.float64)
    if n < 2:
        r2s[:] = np.nan
        mses[:] = np.nan
        return {"r2": r2s, "mse": mses}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], pred[idx]
        r2s[b] = r2_safe(yb, pb)
        mses[b] = mse_safe(yb, pb)
    return {"r2": r2s, "mse": mses}


def summarize_dist(arr: np.ndarray, prefix: str) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q95": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        f"{prefix}_q05": float(np.quantile(a, 0.05)),
        f"{prefix}_q25": float(np.quantile(a, 0.25)),
        f"{prefix}_q75": float(np.quantile(a, 0.75)),
        f"{prefix}_q95": float(np.quantile(a, 0.95)),
    }
