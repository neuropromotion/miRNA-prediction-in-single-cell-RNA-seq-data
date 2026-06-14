"""Load stage00 splits."""

from __future__ import annotations

import json
import re

import pandas as pd

from constants import FEATURES, PILOT_TARGETS, SPLITS

PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")


def load_pilot_targets() -> list[str]:
    return [t.strip() for t in PILOT_TARGETS.read_text().splitlines() if t.strip()]


def load_features() -> dict[str, list[str]]:
    return json.loads(FEATURES.read_text(encoding="utf-8"))


def load_bulk_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(SPLITS / "bulk" / "X_train.parquet"),
        pd.read_parquet(SPLITS / "bulk" / "Y_train.parquet"),
    )


def load_bulk_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(SPLITS / "bulk" / "X_val.parquet"),
        pd.read_parquet(SPLITS / "bulk" / "Y_val.parquet"),
    )


def load_k1_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(SPLITS / "sc_k1" / "X_train.parquet"),
        pd.read_parquet(SPLITS / "sc_k1" / "Y_train.parquet"),
    )


def load_k1_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(SPLITS / "sc_k1" / "X_val.parquet"),
        pd.read_parquet(SPLITS / "sc_k1" / "Y_val.parquet"),
    )


def load_pb_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "sc_pb" / "X_train.parquet")
    y = pd.read_parquet(SPLITS / "sc_pb" / "Y_train.parquet")
    return x, y


def load_pb_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(SPLITS / "sc_pb" / "X_val.parquet")
    y = pd.read_parquet(SPLITS / "sc_pb" / "Y_val.parquet")
    return x, y


def pb_cohort_mask(index: pd.Index, cohort: str) -> pd.Series:
    """Match pseudobulk cohort from sample id, e.g. boot_K5_A549-MS_25."""
    if cohort not in PB_COHORTS:
        raise ValueError(f"Unknown PB cohort {cohort!r}; expected one of {PB_COHORTS}")
    pattern = rf"boot_{re.escape(cohort)}_"
    return index.astype(str).str.contains(pattern, regex=True)


def split_pb_by_cohort(
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for cohort in PB_COHORTS:
        m = pb_cohort_mask(x.index, cohort)
        out[cohort] = (x.loc[m], y.loc[m])
    return out


def count_pb_cohorts(index: pd.Index) -> dict[str, int]:
    return {c: int(pb_cohort_mask(index, c).sum()) for c in PB_COHORTS}
