#!/usr/bin/env python3
"""Merge shard val_metrics_shard*.csv → val_metrics.csv for one model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from constants import MODELS, RESULTS


def merge_model(model: str) -> Path:
    root = RESULTS / model
    shards = sorted(root.glob("val_metrics_shard*.csv"))
    if not shards:
        # Already a single file or nothing yet.
        out = root / "val_metrics.csv"
        if out.exists():
            print(f"{model}: using existing {out}")
            return out
        raise SystemExit(f"No shard metrics under {root}")

    parts = [pd.read_csv(p) for p in shards]
    df = pd.concat(parts, ignore_index=True)
    # Prefer ok rows; drop duplicates by target.
    df = df.sort_values(["target", "status"], key=lambda s: s.map({"ok": 0}).fillna(1))
    df = df.drop_duplicates(subset=["target"], keep="first")
    out = root / "val_metrics.csv"
    df.to_csv(out, index=False)
    n_ok = int((df["status"] == "ok").sum()) if "status" in df.columns else len(df)
    print(f"{model}: merged {len(shards)} shards → {out} (n={len(df)} ok={n_ok})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    for m in args.models:
        if m not in MODELS and m != "all":
            # allow ensemble path? skip unknown
            pass
        merge_model(m)


if __name__ == "__main__":
    main()
