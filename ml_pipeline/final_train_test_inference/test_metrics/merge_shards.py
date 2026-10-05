#!/usr/bin/env python3
"""Merge test_metrics shard CSVs → final tables + figures."""

from __future__ import annotations

import pandas as pd

from config import PER_TARGET_PATH, TABLES
from evaluate import finalize


def main() -> None:
    parts = sorted(TABLES.glob("per_target_bootstrap_summary_shard*.csv"))
    if not parts:
        raise SystemExit(f"No shard CSVs under {TABLES}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["target"], keep="last").sort_values("target")
    df.to_csv(PER_TARGET_PATH, index=False)
    print(f"merged n={len(df)} from {len(parts)} shards → {PER_TARGET_PATH}")
    finalize(df)


if __name__ == "__main__":
    main()
