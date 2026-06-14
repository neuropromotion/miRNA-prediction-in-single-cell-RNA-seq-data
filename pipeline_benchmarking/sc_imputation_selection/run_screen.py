"""Stage02: imputation method screen on 50 miRNA (XGB default)."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from constants import FEATURES, IMPUTE_METHODS, RESULTS, SEED, STAGE, XGB_DEFAULT
from impute import apply_k1_imputation
from io_data import (
    load_bulk_train,
    load_bulk_val,
    load_features,
    load_k1_train,
    load_k1_val,
    load_pb_train_k2,
    load_pb_val_k2,
    load_pilot_targets,
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (STAGE / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class DataBundle:
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_val_mixed: pd.DataFrame
    y_val_mixed: pd.DataFrame
    x_val_k1: pd.DataFrame
    y_val_k1: pd.DataFrame
    x_val_bulk: pd.DataFrame
    y_val_bulk: pd.DataFrame
    impute_stats: dict


def build_bundle(method: str) -> DataBundle:
    bulk_x_tr, bulk_y_tr = load_bulk_train()
    bulk_x_va, bulk_y_va = load_bulk_val()
    k1_x_tr, k1_y_tr = load_k1_train()
    k1_x_va, k1_y_va = load_k1_val()
    pb_x_tr, pb_y_tr = load_pb_train_k2()
    pb_x_va, pb_y_va = load_pb_val_k2()

    k1_tr_imp, k1_va_imp, impute_stats = apply_k1_imputation(k1_x_tr, k1_x_va, method)

    x_train = pd.concat([bulk_x_tr, k1_tr_imp, pb_x_tr], axis=0)
    y_train = pd.concat([bulk_y_tr, k1_y_tr, pb_y_tr], axis=0)
    x_val_mixed = pd.concat([bulk_x_va, k1_va_imp, pb_x_va], axis=0)
    y_val_mixed = pd.concat([bulk_y_va, k1_y_va, pb_y_va], axis=0)

    impute_stats.update(
        {
            "n_bulk_train": int(len(bulk_x_tr)),
            "n_k1_train": int(len(k1_tr_imp)),
            "n_pb_train": int(len(pb_x_tr)),
            "n_train_total": int(len(x_train)),
            "n_bulk_val": int(len(bulk_x_va)),
            "n_k1_val": int(len(k1_va_imp)),
            "n_pb_val": int(len(pb_x_va)),
            "n_val_mixed_total": int(len(x_val_mixed)),
        }
    )

    return DataBundle(
        x_train=x_train,
        y_train=y_train,
        x_val_mixed=x_val_mixed,
        y_val_mixed=y_val_mixed,
        x_val_k1=k1_va_imp,
        y_val_k1=k1_y_va,
        x_val_bulk=bulk_x_va,
        y_val_bulk=bulk_y_va,
        impute_stats=impute_stats,
    )


def eval_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def train_eval(
    flist: list[str],
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    y_va: pd.Series,
) -> float:
    model = xgb.XGBRegressor(**XGB_DEFAULT)
    x_train = x_tr[flist].to_numpy(dtype=np.float64)
    x_val = x_va[flist].to_numpy(dtype=np.float64)
    model.fit(x_train, y_tr.to_numpy(dtype=np.float64), verbose=False)
    pred = np.clip(model.predict(x_val), 0.0, None)
    return eval_r2(y_va.to_numpy(dtype=np.float64), pred)


def run_method(method: str, targets: list[str], features: dict[str, list[str]]) -> pd.DataFrame:
    log(f"--- impute method: {method} ---")
    t0 = time.time()
    bundle = build_bundle(method)
    log(
        f"bundle: train={bundle.x_train.shape} val_mixed={bundle.x_val_mixed.shape} "
        f"k1_val={bundle.x_val_k1.shape} train_zero {bundle.impute_stats['train_zero_before']:.4f}"
        f"->{bundle.impute_stats['train_zero_after']:.4f}"
    )

    out_dir = RESULTS / method
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "impute_stats.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.impute_stats, f, indent=2)

    rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        log(f"[{method}] ({i}/{len(targets)}) {target}")
        flist = features.get(target, [])
        if not flist:
            rows.append({"target": target, "method": method, "status": "no_features"})
            continue
        try:
            k1_r2 = train_eval(flist, bundle.x_train, bundle.y_train[target], bundle.x_val_k1, bundle.y_val_k1[target])
            bulk_r2 = train_eval(
                flist, bundle.x_train, bundle.y_train[target], bundle.x_val_bulk, bundle.y_val_bulk[target]
            )
            mixed_r2 = train_eval(
                flist, bundle.x_train, bundle.y_train[target], bundle.x_val_mixed, bundle.y_val_mixed[target]
            )
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "n_features": len(flist),
                    "k1_val_r2": k1_r2,
                    "bulk_val_r2": bulk_r2,
                    "val_mixed_r2": mixed_r2,
                    "n_train": int(len(bundle.x_train)),
                    "n_k1_val": int(len(bundle.x_val_k1)),
                }
            )
            log(f"[{method}] {target}: k1_r2={k1_r2:.4f} bulk_r2={bulk_r2:.4f} mixed_r2={mixed_r2:.4f}")
        except Exception as exc:
            log(f"[{method}] {target} FAILED: {exc}")
            log(traceback.format_exc())
            rows.append({"target": target, "method": method, "status": "error", "error": str(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "val_metrics.csv", index=False)

    ok = df[df["k1_val_r2"].notna()] if "k1_val_r2" in df.columns else df.iloc[0:0]
    summary = {
        "method": method,
        "n_targets": int(len(targets)),
        "mean_k1_val_r2": float(ok["k1_val_r2"].mean()) if len(ok) else None,
        "mean_bulk_val_r2": float(ok["bulk_val_r2"].mean()) if len(ok) else None,
        "mean_val_mixed_r2": float(ok["val_mixed_r2"].mean()) if len(ok) else None,
        "median_k1_val_r2": float(ok["k1_val_r2"].median()) if len(ok) else None,
        "n_k1_r2_gt_0.4": int((ok["k1_val_r2"] > 0.4).sum()) if len(ok) else 0,
        "mean_n_features": float(ok["n_features"].mean()) if len(ok) and "n_features" in ok else None,
        "bundle_seconds": round(time.time() - t0, 2),
        "impute_stats": bundle.impute_stats,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return df


def main() -> None:
    if (STAGE / "journal.log").exists():
        (STAGE / "journal.log").unlink()
    RESULTS.mkdir(parents=True, exist_ok=True)

    log("=== Stage02: imputation screen (50 miRNA) ===")
    targets = load_pilot_targets()
    features = load_features()
    log(f"targets={len(targets)} methods={IMPUTE_METHODS}")

    config = {
        "seed": SEED,
        "impute_methods": list(IMPUTE_METHODS),
        "impute_scope": "sc_k1_only",
        "train_mix": "bulk_train + k1_train_imp + pb_train_k2",
        "val_mixed": "bulk_val + k1_val_imp + pb_val_k2",
        "primary_metric": "k1_val_r2",
        "features": str(FEATURES),
        "targets": targets,
    }
    with (STAGE / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    summaries = []
    all_rows = []
    for method in IMPUTE_METHODS:
        if (RESULTS / method / "summary.json").exists():
            log(f"skip {method}: already done")
            summaries.append(json.loads((RESULTS / method / "summary.json").read_text()))
            prev = RESULTS / method / "val_metrics.csv"
            if prev.exists():
                all_rows.append(pd.read_csv(prev))
            continue
        df = run_method(method, targets, features)
        all_rows.append(df)
        summaries.append(json.loads((RESULTS / method / "summary.json").read_text()))

    pd.concat(all_rows, ignore_index=True).to_csv(RESULTS / "val_metrics_all.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "summary_by_method.csv", index=False)

    log("=== summary ===")
    log(summary_df.to_string(index=False))
    log("=== done ===")


if __name__ == "__main__":
    main()
