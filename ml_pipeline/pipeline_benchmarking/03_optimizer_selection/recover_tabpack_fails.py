#!/usr/bin/env python3
"""Recover TabPack targets that failed only at ensemble-pred extraction.

Reuses existing experiments under deps/tabpack/experiments/mirna_tuning/
(no retrain). Updates shard CSVs in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STAGE = Path(__file__).resolve().parent
_ML = _STAGE.parents[1]
for _p in (_STAGE, _ML):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from constants import MODEL_LABELS, RESULTS
from metrics import r2
from screen_journal import log
from shared.data import build_modality_bundle, select_features
from shared.io_splits import PB_COHORTS, load_features
from shared.tabpack_trainer import _ensemble_preds, _tabpack_root

EXP_NS = "mirna_tuning"


def _outer_val_sets(bundle, target: str, genes: list[str]):
    sets = [
        ("bulk", select_features(bundle.x_outer_val_bulk, genes).to_numpy(dtype=np.float32),
         bundle.y_outer_val_bulk[target].to_numpy(dtype=np.float64)),
        ("k1", select_features(bundle.x_outer_val_k1, genes).to_numpy(dtype=np.float32),
         bundle.y_outer_val_k1[target].to_numpy(dtype=np.float64)),
    ]
    for cohort in PB_COHORTS:
        sets.append(
            (
                f"pb_{cohort}",
                select_features(bundle.x_outer_val_pb[cohort], genes).to_numpy(dtype=np.float32),
                bundle.y_outer_val_pb[cohort][target].to_numpy(dtype=np.float64),
            )
        )
    return sets


def recover_one(target: str, bundle, features: dict[str, list[str]], model_dir: Path) -> dict:
    root = _tabpack_root()
    safe = target.replace("/", "_")
    exp_dir = root / "experiments" / EXP_NS / safe / "main"
    if not (exp_dir / "report.json").exists() or not (exp_dir / "predictions.npz").exists():
        raise FileNotFoundError(f"missing experiment artifacts: {exp_dir}")

    slices_path = model_dir / "dataset" / "outer_slices.json"
    if not slices_path.exists():
        raise FileNotFoundError(slices_path)
    slices = {k: tuple(v) for k, v in json.loads(slices_path.read_text()).items()}

    val_pred, test_pred, ids = _ensemble_preds(exp_dir)
    genes = features.get(target, [])
    y_val = bundle.y_val_inner[target].to_numpy(dtype=np.float64)
    row = {
        "target": target,
        "model": "tabpack",
        "model_label": MODEL_LABELS["tabpack"],
        "n_features": len(genes),
        "inner_val_r2": r2(y_val, val_pred),
        "status": "ok",
        "error": "",
        "ensemble_ids": json.dumps(ids),
    }
    for name, _x, y_te in _outer_val_sets(bundle, target, genes):
        a, b = slices[name]
        row[f"outer_val_{name}_r2"] = r2(y_te, test_pred[a:b])

    # Persist preds like trainer
    np.savez_compressed(
        model_dir / "preds.npz",
        val_pred=val_pred,
        **{f"outer_{k}": test_pred[a:b].copy() for k, (a, b) in slices.items()},
    )
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"kind": "tabpack", "target": target}
    meta.update(
        {
            "ensemble_ids": ids,
            "exp_dir": str(exp_dir),
            "recovered": True,
            "outer_keys": list(slices.keys()),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return row


def main() -> None:
    out = RESULTS / "tabpack"
    fails: list[tuple[str, Path]] = []
    for csv_path in sorted(out.glob("outer_val_metrics_shard*.csv")):
        df = pd.read_csv(csv_path)
        if "status" not in df.columns:
            continue
        for _, r in df[df["status"] != "ok"].iterrows():
            t = str(r["target"])
            fails.append((t, csv_path))

    if not fails:
        log("no failed tabpack rows to recover", "recover")
        return

    log(f"recovering {len(fails)} failed tabpack targets (no retrain)...", "recover")
    features = load_features()
    bundle = build_modality_bundle()

    # Dedup targets (same target shouldn't appear in multiple shards)
    seen: set[str] = set()
    for target, csv_path in fails:
        if target in seen:
            continue
        seen.add(target)
        model_dir = out / "models" / target
        try:
            row = recover_one(target, bundle, features, model_dir)
            df = pd.read_csv(csv_path)
            df = df[df["target"] != target]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv_path, index=False)
            log(
                f"recovered {target} -> {csv_path.name} "
                f"inner={row['inner_val_r2']:.4f} k1={row['outer_val_k1_r2']:.4f}",
                "recover",
            )
        except Exception as exc:
            log(f"recover FAILED {target}: {exc}", "recover")


if __name__ == "__main__":
    main()
