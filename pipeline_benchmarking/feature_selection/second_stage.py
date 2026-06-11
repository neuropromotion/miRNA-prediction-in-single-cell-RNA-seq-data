"""Second-stage feature selectors."""

from __future__ import annotations

import warnings

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants import (
    LASSO_ALPHAS,
    LASSO_CV,
    LASSO_MAX_ITER,
    LASSO_MAX_POOL,
    LASSO_MAX_SAMPLES,
    SECOND_STAGE_TOP_K,
    SEED,
    XGB_SHALLOW,
)


def _maybe_subsample(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = X.shape[0]
    if n <= max_samples:
        return X, y, n
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx], y[idx], n


def _top_k_indices(scores: np.ndarray, pool_idx: np.ndarray, k: int) -> np.ndarray:
    k = min(k, pool_idx.size)
    if k <= 0:
        return np.array([], dtype=int)
    local_order = np.argsort(-scores)[:k]
    return pool_idx[local_order]


def select_lasso(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray, rhos: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"fallback": "empty_pool", "n_selected": 0}

    if pool_idx.size > LASSO_MAX_POOL and rhos is not None:
        keep = np.argsort(-rhos[pool_idx])[:LASSO_MAX_POOL]
        pool_idx = pool_idx[keep]

    Xp = X[:, pool_idx]
    Xs, ys, n_orig = _maybe_subsample(Xp, y, LASSO_MAX_SAMPLES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = make_pipeline(
            StandardScaler(),
            LassoCV(
                cv=LASSO_CV,
                n_alphas=LASSO_ALPHAS,
                max_iter=LASSO_MAX_ITER,
                random_state=SEED,
                n_jobs=-1,
            ),
        )
        model.fit(Xs, ys)
    coef = model.named_steps["lassocv"].coef_
    chosen = pool_idx[np.where(np.abs(coef) > 1e-10)[0]]

    meta = {
        "n_lasso_nonzero": int(chosen.size),
        "fallback": None,
        "lasso_n_samples_used": int(Xs.shape[0]),
        "lasso_n_samples_total": int(n_orig),
        "lasso_pool_size": int(pool_idx.size),
    }
    if chosen.size == 0:
        scores = rhos[pool_idx] if rhos is not None else np.abs(coef)
        chosen = _top_k_indices(scores, pool_idx, min(50, pool_idx.size))
        meta["fallback"] = "top_spearman_50"
    meta["n_selected"] = int(chosen.size)
    return chosen, meta


def select_xgb_importance(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray, top_k: int = SECOND_STAGE_TOP_K) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"n_selected": 0}

    Xp = X[:, pool_idx]
    model = xgb.XGBRegressor(**XGB_SHALLOW)
    model.fit(Xp, y, verbose=False)
    chosen = _top_k_indices(model.feature_importances_, pool_idx, top_k)
    return chosen, {"n_selected": int(chosen.size), "top_k": top_k}


def select_mi(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray, top_k: int = SECOND_STAGE_TOP_K) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"n_selected": 0}

    Xp = X[:, pool_idx]
    mi = mutual_info_regression(Xp, y, random_state=SEED, n_jobs=-1)
    chosen = _top_k_indices(mi, pool_idx, top_k)
    return chosen, {"n_selected": int(chosen.size), "top_k": top_k}
