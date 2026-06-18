"""KNN k=5 imputation for SC K1 (reference = K1 train)."""

from __future__ import annotations

import sys

import pandas as pd

from final_train.constants import INFERENCE_DIR, SEED

KNN_K = 5

sys.path.insert(0, str(INFERENCE_DIR))
from model_loader import align_and_impute_for_inference, run_imputer  # noqa: E402


def zero_fraction(x: pd.DataFrame) -> float:
    arr = x.values
    if arr.size == 0:
        return 0.0
    return float((arr == 0).sum() / arr.size)


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
        "seed": SEED,
        "train_zero_before": zero_fraction(k1_train),
        "query_zero_before": zero_fraction(k1_query),
        "train_zero_after": zero_fraction(train_imp),
        "query_zero_after": zero_fraction(query_imp),
    }
    return train_imp, query_imp, stats
