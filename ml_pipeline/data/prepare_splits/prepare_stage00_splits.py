#!/usr/bin/env python3
"""Stage00: train/outer_val splits + log2(x+1) from raw TRAIN sources.

Creates parquet matrices under data/splits/ used by all benchmarking stages.
See ../SPLIT_PROTOCOL.md for how this fold relates to inner_val and sc_TEST.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ML_PIPELINE = Path(__file__).resolve().parents[2]
if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.paths import DATA, SPLITS  # noqa: E402

BULK_SOURCE = DATA / "raw" / "bulk_TRAIN"
SC_SOURCE = DATA / "raw" / "sc_TRAIN"

SEED = 42
TRANSFORM = "log2(x+1)"

# Benchmarking protocol (see stage00_splits/journal.log)
VAL_FRAC = {
    "bulk": 0.15,
    "sc_pb": 0.15,
    "sc_k1": 0.20,
}

LOG_PATH = Path(__file__).resolve().parent / "journal_prepare.log"


def log2p1(df: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df.astype(np.float64) + 1.0)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pb_cohort(index: pd.Index) -> np.ndarray:
    out = []
    for s in index.astype(str):
        m = re.search(r"boot_(K\d+)_", s)
        out.append(m.group(1) if m else "other")
    return np.array(out)


def _split_xy(
    x: pd.DataFrame,
    y: pd.DataFrame,
    name: str,
    val_frac: float,
    stratify: np.ndarray | None = None,
) -> dict:
    idx = np.arange(len(x))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_frac,
        random_state=SEED,
        shuffle=True,
        stratify=stratify,
    )
    x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
    x_va, y_va = x.iloc[va_idx], y.iloc[va_idx]

    x_tr = log2p1(x_tr)
    y_tr = log2p1(y_tr)
    x_va = log2p1(x_va)
    y_va = log2p1(y_va)

    out_dir = SPLITS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    x_tr.to_parquet(out_dir / "X_train.parquet")
    y_tr.to_parquet(out_dir / "Y_train.parquet")
    x_va.to_parquet(out_dir / "X_val.parquet")
    y_va.to_parquet(out_dir / "Y_val.parquet")
    pd.Series(x_tr.index, name="id").to_csv(out_dir / "train_ids.txt", index=False, header=False)
    pd.Series(x_va.index, name="id").to_csv(out_dir / "val_ids.txt", index=False, header=False)

    meta = {
        "dataset": name,
        "n_total": int(len(x)),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
        "val_frac": val_frac,
        "n_features": int(x.shape[1]),
        "n_targets": int(y.shape[1]),
        "transform": TRANSFORM,
        "seed": SEED,
        "role_val_fold": "outer_val (benchmarking holdout; not sc_TEST)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"{name}: total={meta['n_total']} train={meta['n_train']} outer_val={meta['n_val']}")
    return meta


def main() -> None:
    _log("=== Stage00: splits + log2(x+1) ===")
    _log(f"seed={SEED} bulk/pb val={VAL_FRAC['bulk']} k1 val={VAL_FRAC['sc_k1']}")
    SPLITS.mkdir(parents=True, exist_ok=True)

    bulk_x = pd.read_parquet(BULK_SOURCE / "X_BULK_TRAIN.parquet")
    bulk_y = pd.read_parquet(BULK_SOURCE / "Y_BULK_TRAIN.parquet").loc[bulk_x.index]

    k1_x = pd.read_parquet(SC_SOURCE / "X_TRAIN_K1.parquet")
    k1_y = pd.read_parquet(SC_SOURCE / "Y_TRAIN_K1.parquet").loc[k1_x.index]

    pb_x = pd.read_parquet(SC_SOURCE / "X_TRAIN_PB.parquet")
    pb_y = pd.read_parquet(SC_SOURCE / "Y_TRAIN_PB.parquet").loc[pb_x.index]

    targets = list(bulk_y.columns)
    _log(f"targets={len(targets)}")

    summary = {
        "seed": SEED,
        "transform": TRANSFORM,
        "splits": {
            "bulk": {"train_frac": 1 - VAL_FRAC["bulk"], "val_frac": VAL_FRAC["bulk"]},
            "sc_pb": {"train_frac": 1 - VAL_FRAC["sc_pb"], "val_frac": VAL_FRAC["sc_pb"]},
            "sc_k1": {"train_frac": 1 - VAL_FRAC["sc_k1"], "val_frac": VAL_FRAC["sc_k1"]},
        },
        "n_targets": len(targets),
        "targets": targets,
        "datasets": {},
        "naming": {
            "X_train_Y_train": "training pool for benchmarking (may be sub-split into inner train/inner_val at runtime)",
            "X_val_Y_val": "outer_val holdout for benchmarking (loaded as outer_val in code; NOT sc_TEST)",
        },
    }
    summary["datasets"]["bulk"] = _split_xy(bulk_x, bulk_y, "bulk", VAL_FRAC["bulk"])
    summary["datasets"]["sc_k1"] = _split_xy(k1_x, k1_y, "sc_k1", VAL_FRAC["sc_k1"])
    summary["datasets"]["sc_pb"] = _split_xy(
        pb_x, pb_y, "sc_pb", VAL_FRAC["sc_pb"], stratify=_pb_cohort(pb_x.index)
    )

    (SPLITS / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"saved {SPLITS / 'split_summary.json'}")
    _log("=== done ===")


if __name__ == "__main__":
    main()
