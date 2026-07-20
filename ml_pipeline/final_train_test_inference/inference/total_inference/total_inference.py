#!/usr/bin/env python3
"""Batch predict_all on every scRNA-seq dataset (parquet/csv)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# .../inference/total_inference/total_inference.py -> add .../inference to path
INFERENCE_DIR = Path(__file__).resolve().parents[1]
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from preprocessor import SingleCell  # noqa: E402
from constants import INFERENCE_INPUT_DIR, INFERENCE_OUTPUT_DIR  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run predict_all on all parquet/csv datasets.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=INFERENCE_INPUT_DIR,
        help=(
            "Folder with input .parquet / .csv files "
            f"(default: {INFERENCE_INPUT_DIR}). "
            "Download/copy scRNA matrices here — not shipped in git."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=INFERENCE_OUTPUT_DIR,
        help=f"Folder for output .csv predictions (default: {INFERENCE_OUTPUT_DIR})",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--mapping-path",
        default=None,
        help="Override HGNC→ENSG map (auto-detected by default)",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.parquet")) + sorted(args.input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No .parquet/.csv files in {args.input_dir}\n"
            "Place scRNA count matrices there (see ml_pipeline/data/README.md)."
        )

    sc = SingleCell(device=args.device, catboost_task="CPU", log=False)
    print(f"Datasets: {len(files)} | miRNAs: {len(sc.available_mirnas)} | out: {args.output_dir}")

    ok, skipped, failed = 0, 0, 0
    t_all = time.time()

    for i, path in enumerate(files, 1):
        out = args.output_dir / f"{path.stem}.csv"
        if out.exists() and not args.force:
            print(f"[{i}/{len(files)}] skip {path.name}")
            skipped += 1
            continue

        print(f"[{i}/{len(files)}] {path.name}")
        t0 = time.time()
        try:
            pred = sc.predict_all(path, mapping_path=args.mapping_path)
            pred.to_csv(out)
            print(f"  -> {out.name} {pred.shape} ({time.time() - t0:.1f}s)")
            ok += 1
        except Exception as exc:
            print(f"  !! failed: {exc}")
            failed += 1

    print(f"Done in {time.time() - t_all:.1f}s | ok={ok} skip={skipped} fail={failed}")


if __name__ == "__main__":
    main()
