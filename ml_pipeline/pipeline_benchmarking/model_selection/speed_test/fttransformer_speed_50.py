#!/usr/bin/env python3
"""FT-Transformer speed benchmark on all 50 pilot miRNA targets (train time only)."""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

_STAGE03 = Path(__file__).resolve().parents[1]  # model_selection/
_ML_PIPELINE = _STAGE03.parents[1]  # ml_pipeline/
for _p in (_STAGE03, _ML_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.data import build_modality_bundle, select_features  # noqa: E402
from shared.dl_trainers import train_fttransformer  # noqa: E402
from shared.io_splits import load_features, load_pilot_targets  # noqa: E402
from speed_test.constants import RESULTS  # noqa: E402

OUT_DIR = RESULTS / "fttransformer_speed_50"
RESULTS_CSV = OUT_DIR / "timing_results.csv"
SUMMARY_TXT = OUT_DIR / "summary.txt"
CHECKPOINT_ROOT = OUT_DIR / "checkpoints"

BATCH_SIZE = int(os.environ.get("STAGE03_BATCH", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIELDNAMES = [
    "target",
    "status",
    "n_features",
    "train_sec",
    "epochs_ran",
    "best_epoch",
    "val_rmse",
    "batch_size",
    "error",
    "finished_at",
]


def load_done() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(df.loc[df["status"] == "ok", "target"].astype(str))


def append_row(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def train_one(
    target: str,
    genes: list[str],
    bundle,
) -> dict:
    x_tr = select_features(bundle.x_train, genes).values
    y_tr = bundle.y_train[target].values
    x_va = select_features(bundle.x_val_inner, genes).values
    y_va = bundle.y_val_inner[target].values
    model_dir = CHECKPOINT_ROOT / target

    batch = BATCH_SIZE
    last_err = ""
    for attempt_batch in (min(batch, 64), 32, 16):
        try:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            t0 = time.perf_counter()
            info = train_fttransformer(
                x_tr,
                y_tr,
                x_va,
                y_va,
                model_dir=model_dir,
                device=DEVICE,
                batch_size=attempt_batch,
            )
            train_sec = time.perf_counter() - t0
            return {
                "target": target,
                "status": "ok",
                "n_features": len(genes),
                "train_sec": round(train_sec, 2),
                "epochs_ran": info.get("epochs_ran", ""),
                "best_epoch": info.get("best_epoch", ""),
                "val_rmse": round(info.get("val_rmse", float("nan")), 6),
                "batch_size": attempt_batch,
                "error": "",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        except torch.cuda.OutOfMemoryError:
            last_err = f"OOM batch={attempt_batch}"
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            break

    return {
        "target": target,
        "status": "fail",
        "n_features": len(genes),
        "train_sec": "",
        "epochs_ran": "",
        "best_epoch": "",
        "val_rmse": "",
        "batch_size": "",
        "error": last_err,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_summary() -> None:
    if not RESULTS_CSV.exists():
        return
    df = pd.read_csv(RESULTS_CSV)
    ok = df[df["status"] == "ok"]
    fail = df[df["status"] != "ok"]
    lines = [
        f"device={DEVICE}",
        f"batch_env={BATCH_SIZE}",
        f"n_targets_total={len(df)}",
        f"n_ok={len(ok)}",
        f"n_fail={len(fail)}",
    ]
    if len(ok):
        mean_s = ok["train_sec"].astype(float).mean()
        med_s = ok["train_sec"].astype(float).median()
        total_h = ok["train_sec"].astype(float).sum() / 3600
        lines += [
            f"mean_train_sec={mean_s:.1f}",
            f"median_train_sec={med_s:.1f}",
            f"total_train_h={total_h:.2f}",
            f"extrapolate_50_h={mean_s * 50 / 3600:.2f}",
            f"extrapolate_327_h={mean_s * 327 / 3600:.1f}",
        ]
    if len(fail):
        lines.append("failed_targets=" + ",".join(fail["target"].astype(str).tolist()))
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    print(f"[ft_speed_50] device={DEVICE} batch={BATCH_SIZE}", flush=True)
    if DEVICE == "cuda":
        print(f"[ft_speed_50] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    features = load_features()
    targets = load_pilot_targets()
    done = load_done()
    print(f"[ft_speed_50] targets={len(targets)} already_done={len(done)}", flush=True)

    bundle = build_modality_bundle()
    print(f"[ft_speed_50] train pool n={len(bundle.x_train)}", flush=True)

    for i, target in enumerate(targets, 1):
        if target in done:
            print(f"[{i}/{len(targets)}] skip {target}", flush=True)
            continue
        genes = features[target]
        print(f"[{i}/{len(targets)}] {target} n_feat={len(genes)}", flush=True)
        row = train_one(target, genes, bundle)
        append_row(row)
        write_summary()
        print(
            f"  -> {row['status']} train_sec={row.get('train_sec', '?')} "
            f"val_rmse={row.get('val_rmse', '?')}",
            flush=True,
        )

    write_summary()


if __name__ == "__main__":
    main()
