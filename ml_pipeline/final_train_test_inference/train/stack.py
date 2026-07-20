"""Ridge stack ensemble with safety gate (SC+PB val tune only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import RidgeCV

from constants import PB_COHORTS, STACK_MODELS
from metrics import clip_nonneg, r2

RIDGE_ALPHAS = tuple(float(x) for x in np.logspace(-2, 4, 40))
SAFETY_GATE = True

# Ensemble tuning: K1 val + all PB val cohorts (no bulk).
TUNE_SPLITS = ("val_k1", "val_pb", *(f"val_pb_{c}" for c in PB_COHORTS))


@dataclass
class FitResult:
    method: str
    models: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)
    active_models: list[str] = field(default_factory=list)
    ridge_intercept: float = 0.0
    ridge_coef: dict[str, float] = field(default_factory=dict)
    ridge_alpha: float | None = None
    tune_r2: float = 0.0
    tune_r2_mean_splits: float = 0.0
    fallback_best_solo: bool = False
    best_solo_model: str = ""


def pool_tune_data(
    preds_by_model: dict[str, dict[str, np.ndarray]],
    y_true: dict[str, np.ndarray],
    models: tuple[str, ...],
    splits: tuple[str, ...] = TUNE_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for split in splits:
        if split not in y_true:
            continue
        y_parts.append(y_true[split])
        cols.append(np.column_stack([preds_by_model[m][split] for m in models]))
    if not y_parts:
        raise ValueError("No tune splits available")
    return np.concatenate(y_parts), np.concatenate(cols, axis=0)


def mean_r2_on_splits(
    y_true: dict[str, np.ndarray],
    pred_by_split: dict[str, np.ndarray],
    splits: tuple[str, ...] = TUNE_SPLITS,
) -> float:
    scores = [r2(y_true[s], pred_by_split[s]) for s in splits if s in y_true]
    return float(np.mean(scores)) if scores else 0.0


def _solo_scores(y: np.ndarray, pred_matrix: np.ndarray, models: tuple[str, ...]) -> dict[str, float]:
    return {m: r2(y, pred_matrix[:, i]) for i, m in enumerate(models)}


def apply_fit(fit: FitResult, pred_matrix: np.ndarray, models: tuple[str, ...]) -> np.ndarray:
    if fit.fallback_best_solo:
        out = np.zeros(pred_matrix.shape[0], dtype=np.float64)
        j = models.index(fit.best_solo_model)
        out += pred_matrix[:, j]
        return clip_nonneg(out)

    pred = np.full(pred_matrix.shape[0], fit.ridge_intercept, dtype=np.float64)
    for i, m in enumerate(models):
        pred += fit.ridge_coef.get(m, 0.0) * pred_matrix[:, i]
    return clip_nonneg(pred)


def split_predictions_from_pooled(
    preds_by_model: dict[str, dict[str, np.ndarray]],
    models: tuple[str, ...],
    fit: FitResult,
    splits: tuple[str, ...] = TUNE_SPLITS,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for split in splits:
        if split not in preds_by_model[models[0]]:
            continue
        mat = np.column_stack([preds_by_model[m][split] for m in models])
        out[split] = apply_fit(fit, mat, models)
    return out


def fit_stack(
    y_pool: np.ndarray,
    pred_matrix: np.ndarray,
    models: tuple[str, ...],
    preds_by_model: dict[str, dict[str, np.ndarray]] | None = None,
    y_true: dict[str, np.ndarray] | None = None,
) -> FitResult:
    n = len(y_pool)
    cv = min(5, max(2, n // 10))
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=cv)
    ridge.fit(pred_matrix, y_pool)
    coef = {models[i]: float(ridge.coef_[i]) for i in range(len(models))}
    fit = FitResult(
        method="stack",
        models=models,
        ridge_intercept=float(ridge.intercept_),
        ridge_coef=coef,
        ridge_alpha=float(ridge.alpha_),
        active_models=list(models),
    )
    ens_pred = apply_fit(fit, pred_matrix, models)
    fit.tune_r2 = r2(y_pool, ens_pred)
    if preds_by_model is not None and y_true is not None:
        split_preds = split_predictions_from_pooled(preds_by_model, models, fit)
        fit.tune_r2_mean_splits = mean_r2_on_splits(y_true, split_preds)

    if not SAFETY_GATE:
        return fit

    solo = _solo_scores(y_pool, pred_matrix, models)
    best_model = max(solo, key=solo.get)
    best_r2 = solo[best_model]
    if fit.tune_r2 <= best_r2:
        fit.fallback_best_solo = True
        fit.best_solo_model = best_model
        fit.weights = {m: (1.0 if m == best_model else 0.0) for m in models}
        fit.active_models = [best_model]
        fit.tune_r2 = best_r2
        if preds_by_model is not None and y_true is not None:
            fit.tune_r2_mean_splits = mean_r2_on_splits(
                y_true, {s: preds_by_model[best_model][s] for s in TUNE_SPLITS if s in y_true}
            )
    return fit


def fit_to_dict(fit: FitResult) -> dict:
    return {
        "method": fit.method,
        "models": list(fit.models),
        "weights": fit.weights,
        "active_models": fit.active_models,
        "ridge_intercept": fit.ridge_intercept,
        "ridge_coef": fit.ridge_coef,
        "ridge_alpha": fit.ridge_alpha,
        "tune_r2": fit.tune_r2,
        "tune_r2_mean_splits": fit.tune_r2_mean_splits,
        "fallback_best_solo": fit.fallback_best_solo,
        "best_solo_model": fit.best_solo_model,
        "tune_splits": [s for s in TUNE_SPLITS],
    }


def fit_from_dict(data: dict) -> FitResult:
    models = tuple(data.get("models") or STACK_MODELS)
    return FitResult(
        method=str(data.get("method") or "stack"),
        models=models,
        weights=dict(data.get("weights") or {}),
        active_models=list(data.get("active_models") or []),
        ridge_intercept=float(data.get("ridge_intercept") or 0.0),
        ridge_coef={k: float(v) for k, v in (data.get("ridge_coef") or {}).items()},
        ridge_alpha=(float(data["ridge_alpha"]) if data.get("ridge_alpha") is not None else None),
        tune_r2=float(data.get("tune_r2") or 0.0),
        tune_r2_mean_splits=float(data.get("tune_r2_mean_splits") or 0.0),
        fallback_best_solo=bool(data.get("fallback_best_solo", False)),
        best_solo_model=str(data.get("best_solo_model") or ""),
    )
