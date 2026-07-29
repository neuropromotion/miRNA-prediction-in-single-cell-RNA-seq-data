#!/usr/bin/env python3
"""Non-destructive smoke test for model_selection.

Writes ONLY under model_selection/smoke_tmp/. Does not touch results/, tables/,
figures/, config.json, or journal.log.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_STAGE03 = Path(__file__).resolve().parent
_ML_PIPELINE = _STAGE03.parents[1]
for _p in (_STAGE03, _ML_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SMOKE_ROOT = _STAGE03 / "smoke_tmp"
SMOKE_RESULTS = SMOKE_ROOT / "results"
SMOKE_JOURNAL = SMOKE_ROOT / "journal.log"

# Force CPU-friendly defaults unless caller overrides.
os.environ.setdefault("STAGE03_DEVICE", "cpu")
os.environ.setdefault("STAGE03_BATCH", "256")


def _assert_inside_ml_pipeline(path: Path) -> None:
    path = path.resolve()
    root = _ML_PIPELINE.resolve()
    if root not in path.parents and path != root:
        raise RuntimeError(f"Path escapes ml_pipeline: {path}")


def main() -> int:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    SMOKE_RESULTS.mkdir(parents=True, exist_ok=True)

    import model_screen_final_11.constants as constants
    import model_screen_final_11.screen_journal as journal

    # Redirect all outputs away from real results/.
    constants.RESULTS = SMOKE_RESULTS
    journal.LOG_PATH = SMOKE_JOURNAL

    from model_screen_final_11.model_trainers import load_artifact, predict_model, train_one
    from model_screen_final_11.metrics import r2
    from shared.data import build_modality_bundle, select_features
    from shared.io_splits import load_features, load_pilot_targets
    from shared.paths import FEATURES, PILOT_TARGETS, SPLITS

    print("=== model_selection smoke test ===", flush=True)
    print(f"ml_pipeline={_ML_PIPELINE}", flush=True)
    print(f"smoke_out={SMOKE_ROOT}", flush=True)

    # Self-containment checks
    for label, p in [
        ("FEATURES", FEATURES),
        ("PILOT_TARGETS", PILOT_TARGETS),
        ("SPLITS", SPLITS),
        ("bulk X_train", SPLITS / "bulk" / "X_train.parquet"),
        ("shared.dl_trainers", _ML_PIPELINE / "shared" / "dl_trainers.py"),
        ("deps.imputation", _ML_PIPELINE / "deps" / "imputation" / "model_loader.py"),
    ]:
        _assert_inside_ml_pipeline(Path(p))
        ok = Path(p).exists()
        print(f"  [{('OK' if ok else 'MISSING')}] {label}: {p}", flush=True)
        if not ok:
            return 1

    targets = load_pilot_targets()
    features = load_features()
    target = targets[0]
    genes = features[target]
    model_name = "xgb_default"
    print(f"target={target} n_feat={len(genes)} model={model_name}", flush=True)

    t0 = time.time()
    print("Building modality bundle...", flush=True)
    bundle = build_modality_bundle()
    print(
        f"Bundle ready in {time.time() - t0:.1f}s | "
        f"train={len(bundle.x_train)} inner_val={len(bundle.x_val_inner)}",
        flush=True,
    )

    model_dir = SMOKE_RESULTS / model_name / "models" / target
    model_dir.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    train_one(model_name, bundle, target, genes, model_dir)
    train_sec = round(time.time() - t1, 2)
    artifact = load_artifact(model_name, model_dir)

    x_val = select_features(bundle.x_val_inner, genes).to_numpy(dtype="float32")
    y_val = bundle.y_val_inner[target].to_numpy(dtype="float64")
    pred = predict_model(model_name, artifact, x_val)
    inner_r2 = float(r2(y_val, pred))

    x_k1 = select_features(bundle.x_outer_val_k1, genes).to_numpy(dtype="float32")
    y_k1 = bundle.y_outer_val_k1[target].to_numpy(dtype="float64")
    k1_r2 = float(r2(y_k1, predict_model(model_name, artifact, x_k1)))

    summary = {
        "status": "ok",
        "model": model_name,
        "target": target,
        "n_features": len(genes),
        "train_sec": train_sec,
        "inner_val_r2": inner_r2,
        "outer_val_k1_r2": k1_r2,
        "smoke_root": str(SMOKE_ROOT),
        "device": os.environ.get("STAGE03_DEVICE"),
    }
    (SMOKE_ROOT / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("=== smoke OK ===", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
