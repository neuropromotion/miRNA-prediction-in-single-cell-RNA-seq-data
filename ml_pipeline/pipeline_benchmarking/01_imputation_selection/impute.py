"""K1-only imputation strategies (log2 TPM+1, zeros = dropout)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from constants import INFERENCE_DIR, MAGIC_KNN, MAGIC_T, NE_MODULE, SOFTIMPUTE_MAX_ITERS

sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(NE_MODULE))

from model_loader import align_and_impute_for_inference, run_imputer  # noqa: E402
from ne import NEConfig, impute_matrix, zero_fraction  # noqa: E402

from constants import NE_CONFIG as NE_CFG


def impute_raw(x: pd.DataFrame) -> pd.DataFrame:
    return x.copy()


def impute_knn_train(x: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    _, filled = run_imputer(x, x, n_neighbors=n_neighbors)
    return filled


def impute_knn_val(x: pd.DataFrame, ref: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    required = sorted(set(ref.columns) | set(x.columns))
    return align_and_impute_for_inference(
        X_query=x,
        required_cols=required,
        X_ref_knn=ref,
        n_neighbors=n_neighbors,
    )


def impute_softimpute_train(x: pd.DataFrame, max_iters: int = SOFTIMPUTE_MAX_ITERS) -> pd.DataFrame:
    from fancyimpute import SoftImpute

    arr = x.values.astype(np.float64, copy=True)
    arr[arr == 0.0] = np.nan
    filled = SoftImpute(max_iters=max_iters, convergence_threshold=1e-2).fit_transform(arr)
    filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(filled, index=x.index, columns=x.columns)


def impute_softimpute_val(x: pd.DataFrame, train_filled: pd.DataFrame) -> pd.DataFrame:
    gene_means = train_filled.mean(axis=0)
    out = x.values.astype(np.float32, copy=True)
    zero_mask = out == 0.0
    for j, col in enumerate(x.columns):
        if zero_mask[:, j].any():
            out[zero_mask[:, j], j] = np.float32(gene_means[col])
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def impute_magic(x: pd.DataFrame) -> pd.DataFrame:
    import magic

    op = magic.MAGIC(
        knn=MAGIC_KNN,
        t=MAGIC_T,
        n_pca=min(100, x.shape[0] - 1, x.shape[1]),
        solver="approximate",
        random_state=42,
        verbose=0,
    )
    out = op.fit_transform(x.values.astype(np.float64))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def impute_ne(x: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = NEConfig(**NE_CFG)
    out = impute_matrix(x, cfg)
    stats = {
        "n_pca": cfg.n_pca,
        "ne_k": cfg.k,
        "ne_alpha": cfg.alpha,
        "ne_order": cfg.order,
        "self_weight": cfg.self_weight,
    }
    return out, stats


def apply_k1_imputation(
    k1_train: pd.DataFrame,
    k1_val: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stats = {
        "method": method,
        "train_zero_before": zero_fraction(k1_train),
        "val_zero_before": zero_fraction(k1_val),
    }

    if method == "raw":
        train_imp = impute_raw(k1_train)
        val_imp = impute_raw(k1_val)
    elif method.startswith("knn_k"):
        k = int(method.split("_k", 1)[1])
        train_imp = impute_knn_train(k1_train, n_neighbors=k)
        val_imp = impute_knn_val(k1_val, k1_train, n_neighbors=k)
        stats["knn_k"] = k
        stats["knn_ref"] = "k1_train"
    elif method == "ne":
        train_imp, ne_stats = impute_ne(k1_train)
        val_imp, val_ne_stats = impute_ne(k1_val)
        stats.update(ne_stats)
        stats["val_ne"] = val_ne_stats
    elif method == "softimpute":
        train_imp = impute_softimpute_train(k1_train)
        val_imp = impute_softimpute_val(k1_val, train_imp)
        stats["softimpute_max_iters"] = SOFTIMPUTE_MAX_ITERS
        stats["val_policy"] = "gene_mean_from_train"
    elif method == "magic":
        train_imp = impute_magic(k1_train)
        val_imp = impute_magic(k1_val)
        stats["magic_knn"] = MAGIC_KNN
        stats["magic_t"] = MAGIC_T
        stats["magic_solver"] = "approximate"
        stats["val_policy"] = "separate_fit_transform_no_leakage"
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    stats["train_zero_after"] = zero_fraction(train_imp)
    stats["val_zero_after"] = zero_fraction(val_imp)
    return train_imp, val_imp, stats
