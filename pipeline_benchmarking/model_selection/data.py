"""Train / inner-val / test bundles (shared by model_screen and speed_test)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared.impute import apply_k1_imputation
from shared.io_splits import (
    PB_COHORTS,
    count_pb_cohorts,
    load_bulk_test,
    load_bulk_train,
    load_k1_test,
    load_k1_train,
    load_pb_test,
    load_pb_train,
    split_pb_by_cohort,
)
from shared.paths import INNER_VAL_FRAC, SEED


@dataclass
class ModalityBundle:
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    sample_weight: np.ndarray
    train_modality: np.ndarray
    x_val_inner: pd.DataFrame
    y_val_inner: pd.DataFrame
    val_modality: np.ndarray
    x_test_bulk: pd.DataFrame
    y_test_bulk: pd.DataFrame
    x_test_k1: pd.DataFrame
    y_test_k1: pd.DataFrame
    x_test_pb: dict[str, pd.DataFrame] = field(default_factory=dict)
    y_test_pb: dict[str, pd.DataFrame] = field(default_factory=dict)
    impute_stats: dict = field(default_factory=dict)
    pool_modality: np.ndarray | None = None


def _tagged_concat(
    parts: list[tuple[pd.DataFrame, pd.DataFrame, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    xs, ys, mods = [], [], []
    for x, y, mod in parts:
        xs.append(x)
        ys.append(y)
        mods.extend([mod] * len(x))
    x_all = pd.concat(xs, axis=0)
    y_all = pd.concat(ys, axis=0)
    mod_arr = np.array(mods, dtype=object)
    return x_all, y_all, mod_arr


def modality_sample_weights(modality: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(modality), dtype=np.float64)
    for mod in np.unique(modality):
        n = int((modality == mod).sum())
        weights[modality == mod] = 1.0 / max(n, 1)
    weights *= len(weights) / weights.sum()
    return weights


def modality_fractions(modality: np.ndarray) -> dict[str, float]:
    n = len(modality)
    if n == 0:
        return {}
    return {str(m): float((modality == m).sum()) / n for m in np.unique(modality)}


def build_modality_bundle() -> ModalityBundle:
    bulk_tr_x, bulk_tr_y = load_bulk_train()
    bulk_te_x, bulk_te_y = load_bulk_test()
    k1_tr_x, k1_tr_y = load_k1_train()
    k1_te_x, k1_te_y = load_k1_test()
    pb_tr_x, pb_tr_y = load_pb_train()
    pb_te_x, pb_te_y = load_pb_test()

    k1_tr_imp, k1_te_imp, impute_stats = apply_k1_imputation(k1_tr_x, k1_te_x)
    pb_test = split_pb_by_cohort(pb_te_x, pb_te_y)

    x_pool, y_pool, mod_pool = _tagged_concat(
        [
            (bulk_tr_x, bulk_tr_y, "bulk"),
            (k1_tr_imp, k1_tr_y, "k1"),
            (pb_tr_x, pb_tr_y, "pb"),
        ]
    )
    idx = np.arange(len(x_pool))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=INNER_VAL_FRAC,
        random_state=SEED,
        stratify=mod_pool,
    )

    x_train = x_pool.iloc[tr_idx]
    y_train = y_pool.iloc[tr_idx]
    mod_train = mod_pool[tr_idx]
    sample_weight = modality_sample_weights(mod_train)

    x_val_inner = x_pool.iloc[va_idx]
    y_val_inner = y_pool.iloc[va_idx]
    mod_val = mod_pool[va_idx]

    impute_stats.update(
        {
            "inner_val_frac": INNER_VAL_FRAC,
            "seed": SEED,
            "n_pool": int(len(x_pool)),
            "n_train": int(len(x_train)),
            "n_inner_val": int(len(x_val_inner)),
            "n_test_bulk": int(len(bulk_te_x)),
            "n_test_k1": int(len(k1_te_imp)),
            "n_test_pb_total": int(len(pb_te_x)),
            "n_test_pb_by_cohort": count_pb_cohorts(pb_te_x.index),
            "n_train_pb_by_cohort": count_pb_cohorts(pb_tr_x.index),
            "pool_modality_counts": {m: int((mod_pool == m).sum()) for m in np.unique(mod_pool)},
            "train_modality_counts": {m: int((mod_train == m).sum()) for m in np.unique(mod_train)},
            "inner_val_modality_counts": {m: int((mod_val == m).sum()) for m in np.unique(mod_val)},
            "pool_modality_fractions": modality_fractions(mod_pool),
            "train_modality_fractions": modality_fractions(mod_train),
            "inner_val_modality_fractions": modality_fractions(mod_val),
        }
    )

    return ModalityBundle(
        x_train=x_train,
        y_train=y_train,
        sample_weight=sample_weight,
        train_modality=mod_train,
        x_val_inner=x_val_inner,
        y_val_inner=y_val_inner,
        val_modality=mod_val,
        x_test_bulk=bulk_te_x,
        y_test_bulk=bulk_te_y,
        x_test_k1=k1_te_imp,
        y_test_k1=k1_te_y,
        x_test_pb={c: pb_test[c][0] for c in PB_COHORTS},
        y_test_pb={c: pb_test[c][1] for c in PB_COHORTS},
        impute_stats=impute_stats,
        pool_modality=mod_pool,
    )


def select_features(x: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    missing = [g for g in genes if g not in x.columns]
    if missing:
        raise KeyError(f"missing {len(missing)} features, e.g. {missing[:3]}")
    return x[genes]


def concat_pb_test_x(x_test_pb: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all PB test cohorts (K2–K10) for speed-benchmark inference."""
    parts = [x_test_pb[c] for c in PB_COHORTS if c in x_test_pb and len(x_test_pb[c])]
    if not parts:
        raise ValueError("no PB test cohorts in bundle")
    return pd.concat(parts, axis=0)
