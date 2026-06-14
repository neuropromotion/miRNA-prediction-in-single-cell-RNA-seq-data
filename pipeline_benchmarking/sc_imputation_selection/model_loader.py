#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


PathLike = Union[str, Path]

BULK_SIZE_SC = 1.0
DEFAULT_KNN_NEIGHBORS = 5

# ---------------------------------------------------------------------------
# Matrix orientation / KNN impute
# ---------------------------------------------------------------------------
def auto_orient_X(df: pd.DataFrame, gene_prefix: Sequence[str] = ("ENSG",)) -> pd.DataFrame:
    cols_ok = any(str(c).startswith(gene_prefix) for c in df.columns[:50])
    rows_ok = any(str(r).startswith(gene_prefix) for r in df.index[:50])
    if cols_ok and not rows_ok:
        return df
    if rows_ok and not cols_ok:
        return df.T
    return df


def knn_impute_cpu(
    X_df: pd.DataFrame,
    zero_mask: pd.DataFrame,
    indices: np.ndarray,
    donor_df: pd.DataFrame,
) -> pd.DataFrame:
    X = X_df.values.astype(np.float32).copy()
    mask = zero_mask.values
    donor = donor_df.values.astype(np.float32)
    _, n_features = X.shape
    for j in range(n_features):
        missing_idx = np.where(mask[:, j])[0]
        if len(missing_idx) == 0:
            continue
        neigh_vals = donor[indices[missing_idx], j]
        neigh_vals = np.where(neigh_vals == 0, np.nan, neigh_vals)
        with np.errstate(all="ignore"):
            imputed = np.nanmean(neigh_vals, axis=1)
        imputed = np.where(np.isnan(imputed), 0.0, imputed)
        X[missing_idx, j] = imputed
    return pd.DataFrame(X, index=X_df.index, columns=X_df.columns)


def run_imputer(
    X_ref: pd.DataFrame,
    X_query: pd.DataFrame,
    n_neighbors: int = DEFAULT_KNN_NEIGHBORS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(X_ref.columns) & set(X_query.columns))
    Xr = X_ref[common].copy()
    Xq = X_query[common].copy()
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(Xr.values.astype(np.float32))
    _, ind_ref = nn.kneighbors(Xr.values.astype(np.float32))
    _, ind_q = nn.kneighbors(Xq.values.astype(np.float32))
    Xr_filled = knn_impute_cpu(Xr, Xr == 0, ind_ref, Xr)
    Xq_filled = knn_impute_cpu(Xq, Xq == 0, ind_q, Xr_filled)
    return Xr_filled, Xq_filled


def ensure_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        miss_df = pd.DataFrame(fill_value, index=out.index, columns=missing, dtype=np.float32)
        out = pd.concat([out, miss_df], axis=1)
    return out[list(cols)]


def align_and_impute_for_inference(
    X_query: pd.DataFrame,
    required_cols: Sequence[str],
    X_ref_knn: pd.DataFrame,
    n_neighbors: int = DEFAULT_KNN_NEIGHBORS,
) -> pd.DataFrame:
    Xq = ensure_columns(X_query, required_cols, fill_value=0.0)
    Xref = ensure_columns(X_ref_knn, required_cols, fill_value=0.0)
    _, Xq_imp = run_imputer(Xref, Xq, n_neighbors=n_neighbors)
    return Xq_imp
