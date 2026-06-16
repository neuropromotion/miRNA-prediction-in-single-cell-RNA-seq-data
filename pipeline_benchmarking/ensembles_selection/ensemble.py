"""Ensemble fit/apply: blend, soup variants, Ridge stack."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import RidgeCV

from stage04_ensembles_v2.constants import (
    BLEND_GRID_STEP,
    RIDGE_ALPHAS,
    SAFETY_GATE,
    SOUP_MIX_STEP,
    SOUP_PRUNE_PASSES,
    TUNE_SPLITS,
)
from stage04_ensembles_v2.metrics import clip_nonneg, r2


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
        y_parts.append(y_true[split])
        cols.append(np.column_stack([preds_by_model[m][split] for m in models]))
    y = np.concatenate(y_parts)
    matrix = np.concatenate(cols, axis=0)
    return y, matrix


def mean_r2_on_splits(
    y_true: dict[str, np.ndarray],
    pred_by_split: dict[str, np.ndarray],
    splits: tuple[str, ...] = TUNE_SPLITS,
) -> float:
    scores = [r2(y_true[s], pred_by_split[s]) for s in splits]
    return float(np.mean(scores))


def split_predictions_from_pooled(
    preds_by_model: dict[str, dict[str, np.ndarray]],
    models: tuple[str, ...],
    fit: FitResult,
    splits: tuple[str, ...] = TUNE_SPLITS,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for split in splits:
        mat = np.column_stack([preds_by_model[m][split] for m in models])
        out[split] = apply_fit(fit, mat, models)
    return out


def _simplex_weight_grid(n_models: int, step: float = BLEND_GRID_STEP) -> list[tuple[float, ...]]:
    grid = np.arange(0.0, 1.0 + step / 2, step)
    if n_models == 2:
        return [(float(a), float(1.0 - a)) for a in grid]
    if n_models == 3:
        out: list[tuple[float, ...]] = []
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 >= -1e-9:
                    out.append((float(w0), float(w1), float(max(0.0, w2))))
        return out
    if n_models == 4:
        out = []
        for w0 in grid:
            for w1 in grid:
                for w2 in grid:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 >= -1e-9:
                        out.append((float(w0), float(w1), float(w2), float(max(0.0, w3))))
        return out
    raise ValueError(f"simplex grid not implemented for n={n_models}")


def _combine(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    out = np.zeros_like(preds[0], dtype=np.float64)
    for p, w in zip(preds, weights):
        out += w * np.asarray(p, dtype=np.float64)
    return clip_nonneg(out)


def _solo_scores(y: np.ndarray, pred_matrix: np.ndarray, models: tuple[str, ...]) -> dict[str, float]:
    return {m: r2(y, pred_matrix[:, i]) for i, m in enumerate(models)}


def apply_fit(fit: FitResult, pred_matrix: np.ndarray, models: tuple[str, ...]) -> np.ndarray:
    if fit.fallback_best_solo:
        weights = [1.0 if m == fit.best_solo_model else 0.0 for m in models]
        return _combine([pred_matrix[:, i] for i in range(len(models))], weights)

    if fit.method == "stack":
        pred = np.full(pred_matrix.shape[0], fit.ridge_intercept, dtype=np.float64)
        for i, m in enumerate(models):
            pred += fit.ridge_coef.get(m, 0.0) * pred_matrix[:, i]
        return clip_nonneg(pred)

    weights = [fit.weights.get(m, 0.0) for m in models]
    preds = [pred_matrix[:, i] for i in range(len(models))]
    return _combine(preds, weights)


def _finalize_fit(
    y_pool: np.ndarray,
    pred_matrix: np.ndarray,
    models: tuple[str, ...],
    fit: FitResult,
    ens_pred: np.ndarray,
    preds_by_model: dict[str, dict[str, np.ndarray]] | None = None,
    y_true: dict[str, np.ndarray] | None = None,
) -> FitResult:
    fit.tune_r2 = r2(y_pool, ens_pred)
    if preds_by_model is not None and y_true is not None:
        split_preds = split_predictions_from_pooled(preds_by_model, models, fit)
        fit.tune_r2_mean_splits = mean_r2_on_splits(y_true, split_preds)

    if not SAFETY_GATE:
        return fit

    solo = _solo_scores(y_pool, pred_matrix, models)
    best_model = max(solo, key=solo.get)
    best_r2 = solo[best_model]
    ens_r2 = fit.tune_r2

    if ens_r2 <= best_r2:
        fit.fallback_best_solo = True
        fit.best_solo_model = best_model
        fit.weights = {m: (1.0 if m == best_model else 0.0) for m in models}
        fit.active_models = [best_model]
        fit.tune_r2 = best_r2
        if preds_by_model is not None and y_true is not None:
            split_preds = {s: preds_by_model[best_model][s] for s in TUNE_SPLITS}
            fit.tune_r2_mean_splits = mean_r2_on_splits(y_true, split_preds)
    return fit


def fit_blend(
    y_pool: np.ndarray,
    pred_matrix: np.ndarray,
    models: tuple[str, ...],
    preds_by_model: dict[str, dict[str, np.ndarray]] | None = None,
    y_true: dict[str, np.ndarray] | None = None,
) -> FitResult:
    best_r2 = -np.inf
    best_weights: dict[str, float] = {}

    for w_tuple in _simplex_weight_grid(len(models)):
        pred = _combine([pred_matrix[:, i] for i in range(len(models))], list(w_tuple))
        score = r2(y_pool, pred)
        if score > best_r2:
            best_r2 = score
            best_weights = {models[i]: w_tuple[i] for i in range(len(models))}

    fit = FitResult(
        method="blend",
        models=models,
        weights=best_weights,
        active_models=[m for m in models if best_weights.get(m, 0) > 1e-9],
    )
    ens_pred = apply_fit(fit, pred_matrix, models)
    return _finalize_fit(y_pool, pred_matrix, models, fit, ens_pred, preds_by_model, y_true)


def fit_soup_uniform(
    y_pool: np.ndarray,
    pred_matrix: np.ndarray,
    models: tuple[str, ...],
    preds_by_model: dict[str, dict[str, np.ndarray]] | None = None,
    y_true: dict[str, np.ndarray] | None = None,
) -> FitResult:
    w = 1.0 / len(models)
    weights = {m: w for m in models}
    fit = FitResult(method="soup_uniform", models=models, weights=weights, active_models=list(models))
    ens_pred = apply_fit(fit, pred_matrix, models)
    return _finalize_fit(y_pool, pred_matrix, models, fit, ens_pred, preds_by_model, y_true)


def fit_soup_greedy(
    y_pool: np.ndarray,
    pred_matrix: np.ndarray,
    models: tuple[str, ...],
    preds_by_model: dict[str, dict[str, np.ndarray]] | None = None,
    y_true: dict[str, np.ndarray] | None = None,
) -> FitResult:
    solo = _solo_scores(y_pool, pred_matrix, models)
    order = sorted(models, key=lambda m: solo[m], reverse=True)

    current = pred_matrix[:, models.index(order[0])].astype(np.float64)
    active = [order[0]]
    best_r2 = solo[order[0]]

    for cand in order[1:]:
        j = models.index(cand)
        best_lam = 0.0
        best_local = best_r2
        best_mix = current
        for lam in np.arange(0.0, 1.0 + SOUP_MIX_STEP / 2, SOUP_MIX_STEP):
            if lam <= 0.0:
                continue
            mix = (1.0 - lam) * current + lam * pred_matrix[:, j]
            score = r2(y_pool, mix)
            if score > best_local:
                best_local = score
                best_lam = float(lam)
                best_mix = mix
        if best_lam > 0.0:
            active.append(cand)
            current = best_mix
            best_r2 = best_local

    if len(active) == 1:
        weights = {m: (1.0 if m == active[0] else 0.0) for m in models}
    else:
        idx = [models.index(m) for m in active]
        sub_fit = fit_blend(y_pool, pred_matrix[:, idx], tuple(active))
        weights = {m: sub_fit.weights.get(m, 0.0) for m in models}

    fit = FitResult(method="soup_greedy", models=models, weights=weights, active_models=active)
    ens_pred = apply_fit(fit, pred_matrix, models)
    return _finalize_fit(y_pool, pred_matrix, models, fit, ens_pred, preds_by_model, y_true)


def fit_soup_pruned(
    y_pool: np.ndarray,
    pred_matrix: np.ndarray,
    models: tuple[str, ...],
    preds_by_model: dict[str, dict[str, np.ndarray]] | None = None,
    y_true: dict[str, np.ndarray] | None = None,
) -> FitResult:
    solo = _solo_scores(y_pool, pred_matrix, models)
    active = list(models)
    weights = {m: 1.0 / len(models) for m in models}

    for _ in range(SOUP_PRUNE_PASSES):
        improved = False
        baseline = apply_fit(
            FitResult(method="soup_pruned", models=models, weights=weights, active_models=active),
            pred_matrix,
            models,
        )
        base_score = r2(y_pool, baseline)
        for cand in sorted(active, key=lambda m: solo[m]):
            if len(active) <= 1:
                break
            trial = [m for m in active if m != cand]
            w = 1.0 / len(trial)
            trial_w = [w] * len(trial)
            pred = _combine([pred_matrix[:, models.index(m)] for m in trial], trial_w)
            if r2(y_pool, pred) >= base_score:
                active = trial
                weights = {m: (1.0 / len(active) if m in active else 0.0) for m in models}
                improved = True
        if not improved:
            break

    fit = FitResult(method="soup_pruned", models=models, weights=weights, active_models=active)
    ens_pred = apply_fit(fit, pred_matrix, models)
    return _finalize_fit(y_pool, pred_matrix, models, fit, ens_pred, preds_by_model, y_true)


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
    return _finalize_fit(y_pool, pred_matrix, models, fit, ens_pred, preds_by_model, y_true)


def fit_ensemble(
    method: str,
    preds_by_model: dict[str, dict[str, np.ndarray]],
    y_true: dict[str, np.ndarray],
    models: tuple[str, ...],
) -> FitResult:
    y_pool, pred_matrix = pool_tune_data(preds_by_model, y_true, models)
    fn = {
        "blend": fit_blend,
        "soup_uniform": fit_soup_uniform,
        "soup_greedy": fit_soup_greedy,
        "soup_pruned": fit_soup_pruned,
        "stack": fit_stack,
    }[method]
    return fn(y_pool, pred_matrix, models, preds_by_model, y_true)


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
        "tune_splits": list(TUNE_SPLITS),
    }


def ensemble_id(model_set: str, method: str) -> str:
    return f"{model_set}_{method}"
