"""Load prepared splits and feature lists."""

from __future__ import annotations

import json
import re

import pandas as pd

from constants import FEATURES, PB_COHORTS, SPLITS, ZERO_EXPRESSED_MIRS


def load_features() -> dict[str, list[str]]:
    return json.loads(FEATURES.read_text(encoding="utf-8"))


def load_zero_expressed_mirs() -> set[str]:
    """Targets that must not be trained (from zero_expressed_mirs.txt)."""
    if not ZERO_EXPRESSED_MIRS.is_file():
        return set()
    return {
        line.strip()
        for line in ZERO_EXPRESSED_MIRS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_targets() -> list[str]:
    """All Y columns minus zero-expressed exclude list (327 → 312 by default)."""
    y = pd.read_parquet(SPLITS / "bulk" / "Y_train.parquet")
    exclude = load_zero_expressed_mirs()
    return [t for t in y.columns if t not in exclude]


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
