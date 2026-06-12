"""Second-stage feature selectors."""

from __future__ import annotations

import warnings

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants import (
    ADAPTIVE_MIN_FEATURES,
    ADAPTIVE_REL_FRAC,
    ENET_L1_RATIOS,
    LINEAR_ALPHAS,
    LINEAR_CV,
    LINEAR_MAX_ITER,
    LINEAR_MAX_POOL,
    LINEAR_MAX_SAMPLES,
    MAX_MODALITY_FEATURES,
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


def _cap_pool(pool_idx: np.ndarray, rhos: np.ndarray | None) -> np.ndarray:
    if pool_idx.size <= LINEAR_MAX_POOL or rhos is None:
        return pool_idx
    keep = np.argsort(-rhos[pool_idx])[:LINEAR_MAX_POOL]
    return pool_idx[keep]


def adaptive_from_scores(
    scores: np.ndarray,
    pool_idx: np.ndarray,
    max_k: int = MAX_MODALITY_FEATURES,
    min_k: int = ADAPTIVE_MIN_FEATURES,
    rel_frac: float = ADAPTIVE_REL_FRAC,
) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"n_selected": 0, "selection": "adaptive", "cutoff": None}

    order = np.argsort(-scores)
    max_s = float(scores[order[0]])
    if max_s <= 1e-12:
        n = min(min_k, pool_idx.size, max_k)
        chosen = order[:n]
        return pool_idx[chosen], {"n_selected": int(n), "selection": "adaptive_min_fallback", "cutoff": 0.0}

    cutoff = max_s * rel_frac
    keep = order[scores[order] >= cutoff]
    if keep.size < min_k:
        keep = order[: min(min_k, pool_idx.size)]
    if keep.size > max_k:
        keep = order[:max_k]

    return pool_idx[keep], {
        "n_selected": int(keep.size),
        "selection": "adaptive",
        "cutoff": cutoff,
        "max_score": max_s,
    }


def _linear_sparse_select(
    X: np.ndarray,
    y: np.ndarray,
    pool_idx: np.ndarray,
    rhos: np.ndarray | None,
    model_name: str,
) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"fallback": "empty_pool", "n_selected": 0, "model": model_name}

    pool_idx = _cap_pool(pool_idx, rhos)
    Xp = X[:, pool_idx]
    Xs, ys, n_orig = _maybe_subsample(Xp, y, LINEAR_MAX_SAMPLES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if model_name == "lasso":
            reg = LassoCV(
                cv=LINEAR_CV,
                n_alphas=LINEAR_ALPHAS,
                max_iter=LINEAR_MAX_ITER,
                random_state=SEED,
                n_jobs=-1,
            )
        else:
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

    step = "lassocv" if model_name == "lasso" else "elasticnetcv"
    coef = model.named_steps[step].coef_
    chosen = pool_idx[np.where(np.abs(coef) > 1e-10)[0]]
    meta = {
        "model": model_name,
        "n_nonzero": int(chosen.size),
        "fallback": None,
        "n_samples_used": int(Xs.shape[0]),
        "n_samples_total": int(n_orig),
        "pool_size": int(pool_idx.size),
    }
    if model_name == "elasticnet":
        meta["l1_ratio"] = float(model.named_steps[step].l1_ratio_)
        meta["alpha"] = float(model.named_steps[step].alpha_)

    if chosen.size == 0:
        scores = rhos[pool_idx] if rhos is not None else np.abs(coef)
        chosen, am = adaptive_from_scores(scores, pool_idx, max_k=min(50, pool_idx.size), min_k=min(20, pool_idx.size))
        meta["fallback"] = "adaptive_spearman_50"
        meta["n_selected"] = int(chosen.size)
        meta.update({f"fallback_{k}": v for k, v in am.items()})
        return chosen, meta

    if chosen.size > MAX_MODALITY_FEATURES:
        imp = np.abs(coef[np.where(np.abs(coef) > 1e-10)[0]])
        order = np.argsort(-imp)[:MAX_MODALITY_FEATURES]
        chosen = chosen[order]
        meta["capped_at"] = MAX_MODALITY_FEATURES

    meta["n_selected"] = int(chosen.size)
    return chosen, meta


def select_lasso(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray, rhos: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    return _linear_sparse_select(X, y, pool_idx, rhos, "lasso")


def select_elasticnet(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray, rhos: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    return _linear_sparse_select(X, y, pool_idx, rhos, "elasticnet")


def select_spearman_only(pool_idx: np.ndarray, rhos: np.ndarray) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"n_selected": 0, "selection": "spearman_only"}
    scores = rhos[pool_idx]
    order = np.argsort(-scores)
    if pool_idx.size > MAX_MODALITY_FEATURES:
        order = order[:MAX_MODALITY_FEATURES]
    chosen = pool_idx[order]
    return chosen, {"n_selected": int(chosen.size), "selection": "spearman_only"}


def select_xgb_importance(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"n_selected": 0}
    Xp = X[:, pool_idx]
    model = xgb.XGBRegressor(**XGB_SHALLOW)
    model.fit(Xp, y, verbose=False)
    return adaptive_from_scores(model.feature_importances_, pool_idx)


def select_mi(X: np.ndarray, y: np.ndarray, pool_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    if pool_idx.size == 0:
        return pool_idx, {"n_selected": 0}
    Xp = X[:, pool_idx]
    mi = mutual_info_regression(Xp, y, random_state=SEED, n_jobs=-1)
    return adaptive_from_scores(mi, pool_idx)
