"""KNN k=5 imputation for SC K1 only."""

from __future__ import annotations

import sys

import pandas as pd

from constants import INFERENCE_DIR, KNN_K, NE_MODULE

sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(NE_MODULE))

from model_loader import align_and_impute_for_inference, run_imputer  # noqa: E402
from ne import zero_fraction  # noqa: E402


def impute_k1_train(x: pd.DataFrame) -> pd.DataFrame:
    _, filled = run_imputer(x, x, n_neighbors=KNN_K)
    return filled


def impute_k1_query(x: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    required = sorted(set(ref.columns) | set(x.columns))
    return align_and_impute_for_inference(
        X_query=x,
        required_cols=required,
        X_ref_knn=ref,
        n_neighbors=KNN_K,
    )


def apply_k1_imputation(
    k1_train: pd.DataFrame,
    k1_query: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_imp = impute_k1_train(k1_train)
    query_imp = impute_k1_query(k1_query, k1_train)
    stats = {
        "method": f"knn_k{KNN_K}",
        "train_zero_before": zero_fraction(k1_train),
        "query_zero_before": zero_fraction(k1_query),
        "train_zero_after": zero_fraction(train_imp),
        "query_zero_after": zero_fraction(query_imp),
    }
    return train_imp, query_imp, stats
