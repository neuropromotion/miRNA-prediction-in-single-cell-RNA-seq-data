"""Load sc_TEST / bulk_TEST for Optimal_K."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import pandas as pd

from config import (
    BULK_TEST,
    COHORTS,
    K1_REF,
    ML_PIPELINE,
    PB_COHORTS,
    SC_TEST,
    SELECTED_FEATURES,
    TRAIN_DIR,
    ZERO_EXPRESSED,
)

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from impute import impute_k1_query  # noqa: E402
from transforms import log2p1  # noqa: E402


@dataclass
class SplitData:
    name: str
    x: pd.DataFrame
    y: pd.DataFrame


def _align(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


def load_sc_splits() -> dict[str, SplitData]:
    if not SC_TEST.is_dir():
        raise FileNotFoundError(f"sc_TEST missing: {SC_TEST}")
    if not K1_REF.is_file():
        raise FileNotFoundError(f"K1 ref missing: {K1_REF}")
    k1_ref = pd.read_parquet(K1_REF)
    out: dict[str, SplitData] = {}

    x = log2p1(pd.read_parquet(SC_TEST / "X_TEST_K1.parquet"))
    y = log2p1(pd.read_parquet(SC_TEST / "Y_TEST_K1.parquet"))
    x, y = _align(x, y)
    x = impute_k1_query(x, k1_ref)
    out["K1"] = SplitData("K1", x, y)

    for cohort in PB_COHORTS:
        x = log2p1(pd.read_parquet(SC_TEST / f"X_TEST_PB_{cohort}.parquet"))
        y = log2p1(pd.read_parquet(SC_TEST / f"Y_TEST_PB_{cohort}.parquet"))
        x, y = _align(x, y)
        out[cohort] = SplitData(cohort, x, y)

    ns = {k: len(v.x) for k, v in out.items()}
    if len(set(ns.values())) != 1:
        raise ValueError(f"SC cohort lengths differ: {ns}")
    return out


def load_bulk_split() -> SplitData:
    x = log2p1(pd.read_parquet(BULK_TEST / "X_BULK_TEST.parquet"))
    y = log2p1(pd.read_parquet(BULK_TEST / "Y_BULK_TEST.parquet"))
    x, y = _align(x, y)
    return SplitData("bulk", x, y)


def load_features() -> dict[str, list[str]]:
    return json.loads(SELECTED_FEATURES.read_text(encoding="utf-8"))


def load_targets() -> list[str]:
    excluded = {
        ln.strip()
        for ln in ZERO_EXPRESSED.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    feats = load_features()
    # Prefer targets that have selected features and are not zero-expressed.
    return sorted(t for t in feats if t not in excluded)
