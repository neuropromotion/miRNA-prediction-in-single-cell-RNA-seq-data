#!/usr/bin/env python3
"""Merge Optimal_K shard CSVs → decisions/bootstrap + proto_prediction_config + figures."""

from __future__ import annotations

import pandas as pd

from config import BOOT_SUMMARY_PATH, DECISIONS_PATH, TABLES
from data_loading import load_features
from run_optimal_k import finalize


def main() -> None:
    dec_parts = sorted(TABLES.glob("decisions_shard*.csv"))
    boot_parts = sorted(TABLES.glob("bootstrap_medians_shard*.csv"))
    if not dec_parts:
        raise SystemExit(f"No decisions_shard*.csv under {TABLES}")
    decisions = pd.concat([pd.read_csv(p) for p in dec_parts], ignore_index=True)
    decisions = decisions.drop_duplicates(subset=["target"], keep="last")
    decisions = decisions.sort_values("target").reset_index(drop=True)
    decisions.to_csv(DECISIONS_PATH, index=False)
    print(f"merged decisions n={len(decisions)} from {len(dec_parts)} shards → {DECISIONS_PATH}")

    if boot_parts:
        boot = pd.concat([pd.read_csv(p) for p in boot_parts], ignore_index=True)
        boot = boot.drop_duplicates(subset=["target"], keep="last").sort_values("target")
        boot.to_csv(BOOT_SUMMARY_PATH, index=False)
        print(f"merged bootstrap medians → {BOOT_SUMMARY_PATH}")

    finalize(decisions, load_features())


if __name__ == "__main__":
    main()
