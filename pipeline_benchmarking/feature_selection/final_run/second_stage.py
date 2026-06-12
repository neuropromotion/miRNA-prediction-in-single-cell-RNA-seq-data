"""ElasticNet second-stage selector."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants import (
    ENET_L1_RATIOS,
    LINEAR_ALPHAS,
    LINEAR_CV,
    LINEAR_MAX_ITER,
    LINEAR_MAX_POOL,
    LINEAR_MAX_SAMPLES,
    MAX_MODALITY_FEATURES,
    SEED,
)


def _maybe_subsample(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = X.shape[0]
    if n <= max_samples:
        return X, y, n
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx], y[idx], n


def _cap_pool(pool_idx: np.ndarray, rhos: np.ndarray | None) -> np.ndarray:
    if pool_idx.size <= LINEAR_MAX_POOL or rhos is None:
        return pool_idx
    keep = np.argsort(-rhos[pool_idx])[:LINEAR_MAX_POOL]
    return pool_idx[keep]


def _adaptive_fallback(scores: np.ndarray, pool_idx: np.ndarray, max_k: int, min_k: int) -> np.ndarray:
    order = np.argsort(-scores)
    if order.size == 0:
        return pool_idx
    n = min(max(min_k, 1), pool_idx.size, max_k)
    return pool_idx[order[:n]]


def select_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    pool_idx: np.ndarray,
    rhos: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"fallback": "empty_pool", "n_selected": 0, "model": "elasticnet"}

    pool_idx = _cap_pool(pool_idx, rhos)
    Xp = X[:, pool_idx]
    Xs, ys, n_orig = _maybe_subsample(Xp, y, LINEAR_MAX_SAMPLES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        reg = ElasticNetCV(
            cv=LINEAR_CV,
            n_alphas=LINEAR_ALPHAS,
            l1_ratio=ENET_L1_RATIOS,
            max_iter=LINEAR_MAX_ITER,
            random_state=SEED,
            n_jobs=-1,
        )
        model = make_pipeline(StandardScaler(), reg)
        model.fit(Xs, ys)

    coef = model.named_steps["elasticnetcv"].coef_
    chosen = pool_idx[np.where(np.abs(coef) > 1e-10)[0]]
    meta = {
        "model": "elasticnet",
        "n_nonzero": int(chosen.size),
        "fallback": None,
        "n_samples_used": int(Xs.shape[0]),
        "n_samples_total": int(n_orig),
        "pool_size": int(pool_idx.size),
        "l1_ratio": float(model.named_steps["elasticnetcv"].l1_ratio_),
        "alpha": float(model.named_steps["elasticnetcv"].alpha_),
    }

    if chosen.size == 0:
        scores = rhos[pool_idx] if rhos is not None else np.abs(coef)
        chosen = _adaptive_fallback(scores, pool_idx, max_k=min(50, pool_idx.size), min_k=min(20, pool_idx.size))
        meta["fallback"] = "adaptive_spearman_50"
        meta["n_selected"] = int(chosen.size)
        return chosen, meta

    if chosen.size > MAX_MODALITY_FEATURES:
        imp = np.abs(coef[np.where(np.abs(coef) > 1e-10)[0]])
        order = np.argsort(-imp)[:MAX_MODALITY_FEATURES]
        chosen = chosen[order]
        meta["capped_at"] = MAX_MODALITY_FEATURES

    meta["n_selected"] = int(chosen.size)
    return chosen, meta
