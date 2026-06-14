#!/usr/bin/env python3
"""TabPFN-3 speed benchmark on 5 miRNA targets (full train, no subsample)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE))

from constants import RESULTS, SEED, SPEED_N_TARGETS  # noqa: E402
from data import build_modality_bundle, select_features  # noqa: E402
from io_splits import load_features, load_pilot_targets  # noqa: E402

OUT_DIR = RESULTS / "tabpfn3_speed"
DEVICE = os.environ.get("STAGE03_DEVICE", "cuda")
MAX_TRAIN = int(os.environ.get("TABPFN_MAX_TRAIN", "0"))  # 0 = full train pool
V3_CKPT = "tabpfn-v3-regressor-v3_default.ckpt"
V3_REPO = "Prior-Labs/tabpfn_3"


def resolve_v3_model_path() -> str:
    """Download TabPFN-3 regressor checkpoint (HF token) for local model_path."""
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf:
        raise RuntimeError(
            "HF_TOKEN required to download TabPFN-3 weights "
            f"from {V3_REPO} (accept license on HuggingFace first)."
        )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=V3_REPO, filename=V3_CKPT, token=hf)
    print(f"Model checkpoint ready: {V3_CKPT}", flush=True)
    return str(path)


def make_model(model_path: str):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version = os.environ.get("TABPFN_VERSION", "v3").lower()
    if version in ("v3", "3"):
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            device=DEVICE,
            model_path=model_path,
        )
    if version in ("auto", "default", ""):
        return TabPFNRegressor(device=DEVICE, model_path=model_path)
    raise ValueError(f"Unknown TABPFN_VERSION={version!r}")


def run_one(
    target: str,
    bundle,
    feature_map: dict[str, list[str]],
    model_path: str,
) -> dict:
    row = {"target": target, "status": "ok", "error": ""}
    try:
        genes = feature_map[target]
        x_tr = select_features(bundle.x_train, genes).to_numpy(dtype=np.float32)
        y_tr = bundle.y_train[target].to_numpy(dtype=np.float32)
        x_te_bulk = select_features(bundle.x_test_bulk, genes).to_numpy(dtype=np.float32)
        x_te_k1 = select_features(bundle.x_test_k1, genes).to_numpy(dtype=np.float32)
        x_te_pb = select_features(bundle.x_test_pb, genes).to_numpy(dtype=np.float32)

        if MAX_TRAIN > 0 and len(x_tr) > MAX_TRAIN:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(x_tr), size=MAX_TRAIN, replace=False)
            x_fit, y_fit = x_tr[idx], y_tr[idx]
        else:
            x_fit, y_fit = x_tr, y_tr

        row.update(
            {
                "n_features": len(genes),
                "n_train_pool": len(x_tr),
                "n_fit": len(x_fit),
                "n_infer": len(x_te_bulk) + len(x_te_k1) + len(x_te_pb),
            }
        )

        model = make_model(model_path)
        row["model_path"] = model_path

        t0 = time.perf_counter()
        model.fit(x_fit, y_fit)
        row["train_sec"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        _ = model.predict(x_te_bulk)
        _ = model.predict(x_te_k1)
        _ = model.predict(x_te_pb)
        row["infer_sec"] = round(time.perf_counter() - t1, 3)
    except Exception as exc:
        row["status"] = "fail"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


def pick_targets(n: int = SPEED_N_TARGETS) -> list[str]:
    all_t = load_pilot_targets()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_t), size=min(n, len(all_t)), replace=False)
    return sorted(all_t[i] for i in idx)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tabpfn_token = os.environ.get("TABPFN_TOKEN") or os.environ.get("CASCADE_TABPFN_TOKEN")
    print(f"HF_TOKEN set: {bool(hf_token)}", flush=True)
    print(f"TABPFN_TOKEN set: {bool(tabpfn_token)}", flush=True)
    if not hf_token:
        print("ERROR: set HF_TOKEN (HuggingFace read token with tabpfn_3 access).", flush=True)
        sys.exit(1)

    import tabpfn

    print(f"tabpfn package: {tabpfn.__version__}", flush=True)
    print(f"device={DEVICE} | max_train={MAX_TRAIN or 'full'}", flush=True)

    model_path = resolve_v3_model_path()

    targets = pick_targets()
    print(f"Targets ({len(targets)}): {targets}", flush=True)

    bundle = build_modality_bundle()
    feature_map = load_features()

    meta = {
        "tabpfn_version": tabpfn.__version__,
        "tabpfn_model_version": os.environ.get("TABPFN_VERSION", "v3"),
        "model_checkpoint": V3_CKPT,
        "device": DEVICE,
        "max_train": MAX_TRAIN,
        "targets": targets,
        "hf_token_set": bool(hf_token),
        "tabpfn_token_set": bool(tabpfn_token),
        "impute_stats": bundle.impute_stats,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for i, target in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {target}", flush=True)
        row = run_one(target, bundle, feature_map, model_path)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"  train={row['train_sec']}s infer={row['infer_sec']}s "
                f"(fit={row['n_fit']}, feat={row['n_features']})",
                flush=True,
            )
        else:
            print(f"  FAIL: {row['error']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "speed_results.csv", index=False)

    ok = df[df["status"] == "ok"]
    if len(ok):
        summary = {
            "n_ok": int(len(ok)),
            "n_fail": int(len(df) - len(ok)),
            "mean_train_sec": round(float(ok["train_sec"].mean()), 3),
            "median_train_sec": round(float(ok["train_sec"].median()), 3),
            "mean_infer_sec": round(float(ok["infer_sec"].mean()), 3),
            "median_infer_sec": round(float(ok["infer_sec"].median()), 3),
            "total_train_50mirna_h": round(float(ok["train_sec"].mean()) * 50 / 3600, 2),
            "total_train_327mirna_h": round(float(ok["train_sec"].mean()) * 327 / 3600, 2),
        }
    else:
        summary = {"n_ok": 0, "n_fail": int(len(df))}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== TabPFN-3 speed summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
