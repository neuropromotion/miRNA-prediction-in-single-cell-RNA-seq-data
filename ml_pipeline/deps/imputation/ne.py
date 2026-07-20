"""Network Enhancement imputation for scRNA (log2 TPM+1 input).

Adapted from BMC Bioinformatics 2023 benchmark (Joye9285/Imputation-benchmark)
and Stanford NE core (jipq6175/network_enhancement_pytorch).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

EPS = 2e-16


@dataclass(frozen=True)
class NEConfig:
    n_pca: int = 50
    k: int = 20
    alpha: float = 0.9
    order: int = 2
    self_weight: float = 1.5


def _dn(w: np.ndarray, tp: str = "ave") -> np.ndarray:
    n = w.shape[0]
    wn = w.copy()
    wn *= n
    d = np.abs(wn).sum(axis=1) + EPS
    if tp == "ave":
        wn = (1.0 / d)[:, None] * wn
    else:
        inv = 1.0 / np.sqrt(d)
        wn = inv[:, None] * wn * inv[None, :]
    return wn


def _transition_fields(w: np.ndarray) -> np.ndarray:
    n = w.shape[0]
    wn = w.copy()
    zero_rows = np.where(wn.sum(axis=1) == 0.0)[0]
    wn *= n
    wn = _dn(wn, tp="ave")
    d = np.sqrt(np.abs(wn).sum(axis=0) + EPS)
    wn = wn / d[None, :]
    wn = wn @ wn.T
    wn[zero_rows, :] = 0.0
    wn[:, zero_rows] = 0.0
    return wn


def _dominate_set(maff: np.ndarray, nr_knn: int) -> np.ndarray:
    n = maff.shape[0]
    nr_knn = min(nr_knn, n)
    out = np.zeros_like(maff)
    top_idx = np.argsort(-maff, axis=1)[:, :nr_knn]
    rows = np.arange(n)[:, None]
    vals = maff[rows, top_idx]
    for i in range(n):
        for j, v in zip(top_idx[i], vals[i]):
            out[i, j] = v
            out[j, i] = v
    return 0.5 * out


def network_enhancement(
    w: np.ndarray,
    *,
    order: int = 2,
    k: int = 20,
    alpha: float = 0.9,
) -> np.ndarray:
    """Denoise a symmetric cell-cell affinity matrix."""
    n = w.shape[0]
    k = int(min(k, max(1, np.ceil(n / 10))))
    w = np.asarray(w, dtype=np.float64)
    w = 0.5 * (w + w.T)
    w = w * (1.0 - np.eye(n))

    active = np.where(np.abs(w).sum(axis=0) > 0.0)[0]
    if active.size == 0:
        return np.zeros((n, n), dtype=np.float64)

    w0 = w[np.ix_(active, active)]
    n0 = active.size
    w_sub = _dn(w0, tp="ave")
    w_sub = 0.5 * (w_sub + w_sub.T)
    dd = np.abs(w0).sum(axis=0)

    if np.unique(w_sub).size == 2:
        p = w_sub
    else:
        p = _dominate_set(np.abs(w_sub), min(k, n0 - 1)) * np.sign(w_sub)

    p = p + np.eye(n0) + np.diag(np.abs(p).sum(axis=0))
    p = _transition_fields(p)

    lambdas, evectors = np.linalg.eigh(p)
    d = lambdas - EPS
    d = (1.0 - alpha) * d / (1.0 - alpha * d**order)
    w_out = evectors @ np.diag(d) @ np.linalg.inv(evectors)

    diag = np.diag(w_out)
    denom = (1.0 - diag).clip(min=EPS)
    w_out = (w_out * (1.0 - np.eye(n0))) / denom[:, None]
    w_out = np.diag(dd) @ w_out
    w_out[w_out < 0] = 0.0
    w_out = 0.5 * (w_out + w_out.T)

    result = np.zeros((n, n), dtype=np.float64)
    result[np.ix_(active, active)] = w_out
    return result


def build_affinity(data: np.ndarray, n_pca: int) -> np.ndarray:
    """Pearson correlation between cells; optional PCA for speed."""
    x = np.asarray(data, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)

    n_comp = min(n_pca, x.shape[0] - 1, x.shape[1]) if n_pca > 0 else 0
    if n_comp >= 2:
        coords = PCA(n_components=n_comp, random_state=42).fit_transform(x)
        aff = np.corrcoef(coords)
    else:
        aff = np.corrcoef(x)

    aff = np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(aff, 0.0)
    return aff


def build_weight_matrix(affinity: np.ndarray, self_weight: float = 1.5) -> np.ndarray:
    w = network_enhancement(affinity)
    for i in range(w.shape[0]):
        row_max = float(w[i].max()) if w.shape[0] > 1 else 0.0
        w[i, i] = self_weight * row_max
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > EPS, row_sums, 1.0)
    return w / row_sums


def impute_array(data: np.ndarray, cfg: NEConfig | None = None) -> tuple[np.ndarray, dict]:
    """Impute cells x genes matrix in log2 space."""
    cfg = cfg or NEConfig()
    t0 = time.perf_counter()
    x = np.asarray(data, dtype=np.float64)

    aff = build_affinity(x, cfg.n_pca)
    t_aff = time.perf_counter()
    w = build_weight_matrix(aff, cfg.self_weight)
    t_ne = time.perf_counter()
    out = w @ x
    t_done = time.perf_counter()

    stats = {
        "n_cells": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "n_pca": cfg.n_pca,
        "ne_k": cfg.k,
        "ne_alpha": cfg.alpha,
        "ne_order": cfg.order,
        "self_weight": cfg.self_weight,
        "seconds_affinity": t_aff - t0,
        "seconds_ne": t_ne - t_aff,
        "seconds_total": t_done - t0,
    }
    return out, stats


def impute_matrix(x: pd.DataFrame, cfg: NEConfig | None = None) -> pd.DataFrame:
    """Impute DataFrame (cells x genes, log2 TPM+1)."""
    out, _ = impute_array(x.values, cfg)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def zero_fraction(x: pd.DataFrame) -> float:
    arr = x.values
    if arr.size == 0:
        return 0.0
    return float((arr == 0).sum() / arr.size)


def ne_config_from_env() -> NEConfig:
    return NEConfig(
        n_pca=int(os.environ.get("NE_PCA", "50")),
        k=int(os.environ.get("NE_K", "20")),
        alpha=float(os.environ.get("NE_ALPHA", "0.9")),
        order=int(os.environ.get("NE_ORDER", "2")),
        self_weight=float(os.environ.get("NE_SELF_WEIGHT", "1.5")),
    )
