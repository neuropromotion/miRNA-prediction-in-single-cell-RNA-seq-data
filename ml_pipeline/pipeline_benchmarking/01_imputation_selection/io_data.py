"""Load stage00 splits and feature lists."""

from __future__ import annotations

import json

import pandas as pd

from constants import FEATURES, PILOT_TARGETS, SPLITS


def load_pilot_targets() -> list[str]:
    return [t.strip() for t in PILOT_TARGETS.read_text().splitlines() if t.strip()]


def load_features() -> dict[str, list[str]]:
    return json.loads(FEATURES.read_text(encoding="utf-8"))


def load_bulk_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "bulk" / "X_train.parquet")
    y = pd.read_parquet(SPLITS / "bulk" / "Y_train.parquet")
    return x, y


def load_bulk_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "bulk" / "X_val.parquet")
    y = pd.read_parquet(SPLITS / "bulk" / "Y_val.parquet")
    return x, y


def load_k1_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "sc_k1" / "X_train.parquet")
    y = pd.read_parquet(SPLITS / "sc_k1" / "Y_train.parquet")
    return x, y


def load_k1_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "sc_k1" / "X_val.parquet")
    y = pd.read_parquet(SPLITS / "sc_k1" / "Y_val.parquet")
    return x, y


def _k2_mask(index: pd.Index) -> pd.Index:
    return index.astype(str).str.contains("K2")


def load_pb_train_k2() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "sc_pb" / "X_train.parquet")
    y = pd.read_parquet(SPLITS / "sc_pb" / "Y_train.parquet")
    m = _k2_mask(x.index)
    return x.loc[m], y.loc[m]


def load_pb_val_k2() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "sc_pb" / "X_val.parquet")
    y = pd.read_parquet(SPLITS / "sc_pb" / "Y_val.parquet")
    m = _k2_mask(x.index)
    return x.loc[m], y.loc[m]
