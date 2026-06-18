"""Pooled train/val bundles with modality sample weights."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from final_train.shared.impute import apply_k1_imputation
from final_train.shared.io_splits import load_bulk, load_k1, load_pb, split_pb_by_cohort


@dataclass
class TrainBundle:
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    sw_train: np.ndarray
    mod_train: np.ndarray

    x_val: pd.DataFrame
    y_val: pd.DataFrame
    sw_val: np.ndarray
    mod_val: np.ndarray

    x_val_bulk: pd.DataFrame
    y_val_bulk: pd.DataFrame
    x_val_k1: pd.DataFrame
    y_val_k1: pd.DataFrame
    x_val_pb: pd.DataFrame
    y_val_pb: pd.DataFrame
    x_val_pb_by_cohort: dict[str, pd.DataFrame] = field(default_factory=dict)
    y_val_pb_by_cohort: dict[str, pd.DataFrame] = field(default_factory=dict)

    impute_stats: dict = field(default_factory=dict)


def modality_sample_weights(modality: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(modality), dtype=np.float64)
    for mod in np.unique(modality):
        n = int((modality == mod).sum())
        weights[modality == mod] = 1.0 / max(n, 1)
    weights *= len(weights) / weights.sum()
    return weights


def _tagged_concat(
    parts: list[tuple[pd.DataFrame, pd.DataFrame, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    xs, ys, mods = [], [], []
    for x, y, mod in parts:
        xs.append(x)
        ys.append(y)
        mods.extend([mod] * len(x))
    return pd.concat(xs, axis=0), pd.concat(ys, axis=0), np.array(mods, dtype=object)


def select_features(x: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    missing = [g for g in genes if g not in x.columns]
    if missing:
        raise KeyError(f"missing {len(missing)} features, e.g. {missing[:3]}")
    return x[genes]


def build_train_bundle() -> TrainBundle:
    bulk_tr_x, bulk_tr_y, bulk_va_x, bulk_va_y = load_bulk()
    k1_tr_x, k1_tr_y, k1_va_x, k1_va_y = load_k1()
    pb_tr_x, pb_tr_y, pb_va_x, pb_va_y = load_pb()

    k1_tr_imp, k1_va_imp, impute_stats = apply_k1_imputation(k1_tr_x, k1_va_x)

    x_train, y_train, mod_train = _tagged_concat(
        [
            (bulk_tr_x, bulk_tr_y, "bulk"),
            (k1_tr_imp, k1_tr_y, "k1"),
            (pb_tr_x, pb_tr_y, "pb"),
        ]
    )
    x_val, y_val, mod_val = _tagged_concat(
        [
            (bulk_va_x, bulk_va_y, "bulk"),
            (k1_va_imp, k1_va_y, "k1"),
            (pb_va_x, pb_va_y, "pb"),
        ]
    )

    pb_val_by_cohort = split_pb_by_cohort(pb_va_x, pb_va_y)

    return TrainBundle(
        x_train=x_train,
        y_train=y_train,
        sw_train=modality_sample_weights(mod_train),
        mod_train=mod_train,
        x_val=x_val,
        y_val=y_val,
        sw_val=modality_sample_weights(mod_val),
        mod_val=mod_val,
        x_val_bulk=bulk_va_x,
        y_val_bulk=bulk_va_y,
        x_val_k1=k1_va_imp,
        y_val_k1=k1_va_y,
        x_val_pb=pb_va_x,
        y_val_pb=pb_va_y,
        x_val_pb_by_cohort={c: pb_val_by_cohort[c][0] for c in pb_val_by_cohort},
        y_val_pb_by_cohort={c: pb_val_by_cohort[c][1] for c in pb_val_by_cohort},
        impute_stats=impute_stats,
    )
