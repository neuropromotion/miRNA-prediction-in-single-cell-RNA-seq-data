"""Load prepared splits and feature lists."""

from __future__ import annotations

import json
import re

import pandas as pd

from final_train.constants import FEATURES, PB_COHORTS, SPLITS


def load_features() -> dict[str, list[str]]:
    return json.loads(FEATURES.read_text(encoding="utf-8"))


def load_targets() -> list[str]:
    y = pd.read_parquet(SPLITS / "bulk" / "Y_train.parquet")
    return list(y.columns)


def _load_xy(split_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = SPLITS / split_dir
    return (
        pd.read_parquet(base / "X_train.parquet"),
        pd.read_parquet(base / "Y_train.parquet"),
        pd.read_parquet(base / "X_val.parquet"),
        pd.read_parquet(base / "Y_val.parquet"),
    )


def load_bulk() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_xy("bulk")


def load_k1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_xy("sc_k1")


def load_pb() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_xy("sc_pb")


def pb_cohort_mask(index: pd.Index, cohort: str) -> pd.Series:
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
