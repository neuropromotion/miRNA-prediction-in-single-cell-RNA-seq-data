"""Bootstrap R² on tune half + eligibility / Optimal_K selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import COHORTS, DELTA, MEDIAN_THRESHOLD, N_BOOTSTRAP, SEED


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def bootstrap_r2(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    n = len(y)
    if n < 2:
        return np.full(n_boot, np.nan)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = r2_safe(y[idx], pred[idx])
    return out


@dataclass(frozen=True)
class TargetDecision:
    target: str
    eligible: bool
    optimal_k: str | None
    reason: str
    medians: dict[str, float]  # bulk + each K


def decide_from_medians(target: str, medians: dict[str, float]) -> TargetDecision:
    m_bulk = float(medians["bulk"])
    k_med = {k: float(medians[k]) for k in COHORTS}
    best_k_m = max(k_med.values())

    if m_bulk < MEDIAN_THRESHOLD or best_k_m < MEDIAN_THRESHOLD:
        reasons = []
        if m_bulk < MEDIAN_THRESHOLD:
            reasons.append(f"m_bulk={m_bulk:.4f} < {MEDIAN_THRESHOLD}")
        if best_k_m < MEDIAN_THRESHOLD:
            reasons.append(f"max_K m={best_k_m:.4f} < {MEDIAN_THRESHOLD}")
        return TargetDecision(
            target=target,
            eligible=False,
            optimal_k=None,
            reason="; ".join(reasons),
            medians=medians,
        )

    above = {k: m for k, m in k_med.items() if m >= MEDIAN_THRESHOLD}
    m_star = max(above.values())
    candidates = [k for k in COHORTS if k in above and above[k] >= m_star - DELTA]
    opt = candidates[0]  # COHORTS already small→large
    return TargetDecision(
        target=target,
        eligible=True,
        optimal_k=opt,
        reason=(
            f"m*={m_star:.4f}; candidates={[c for c in candidates]} "
            f"(δ={DELTA}); chose smallest {opt} (m={above[opt]:.4f})"
        ),
        medians=medians,
    )


def medians_on_tune(
    preds: dict[str, tuple[np.ndarray, np.ndarray]],
    tune_sc_idx: np.ndarray,
    tune_bulk_idx: np.ndarray,
    *,
    target_seed: int,
) -> dict[str, float]:
    """Bootstrap medians for bulk + each SC cohort on tune indices."""
    out: dict[str, float] = {}
    for i, name in enumerate((*COHORTS, "bulk")):
        y_full, pred_full = preds[name]
        idx = tune_bulk_idx if name == "bulk" else tune_sc_idx
        y = y_full[idx]
        pred = pred_full[idx]
        boots = bootstrap_r2(y, pred, seed=target_seed + 17 * (i + 1))
        out[name] = float(np.nanmedian(boots))
    return out
