"""Load stage00 split matrices."""

from __future__ import annotations

import json

import pandas as pd

from constants import SPLITS


def load_target_list() -> list[str]:
    summary = json.loads((SPLITS / "split_summary.json").read_text(encoding="utf-8"))
    return list(summary["targets"])


def load_bulk_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "bulk" / "X_train.parquet")
    y = pd.read_parquet(SPLITS / "bulk" / "Y_train.parquet")
    return x, y


def _k2_mask(index: pd.Index) -> pd.Index:
    return index.astype(str).str.contains("K2")


def load_sc_train_combo() -> tuple[pd.DataFrame, pd.DataFrame]:
    x_k1 = pd.read_parquet(SPLITS / "sc_k1" / "X_train.parquet")
    y_k1 = pd.read_parquet(SPLITS / "sc_k1" / "Y_train.parquet")
    x_pb = pd.read_parquet(SPLITS / "sc_pb" / "X_train.parquet")
    y_pb = pd.read_parquet(SPLITS / "sc_pb" / "Y_train.parquet")
    m = _k2_mask(x_pb.index)
    x = pd.concat([x_k1, x_pb.loc[m]], axis=0)
    y = pd.concat([y_k1, y_pb.loc[m]], axis=0)
    return x, y
