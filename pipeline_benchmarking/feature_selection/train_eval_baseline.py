"""Stage01: XGB default train/val after feature selection."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from constants import METHODS, RESULTS, STAGE, XGB_DEFAULT
from io_data import load_bulk_train, load_bulk_val, load_sc_train_combo, load_sc_val_combo


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (STAGE / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def eval_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def train_eval_target(
    flist: list[str],
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    y_va: pd.Series,
) -> dict:
    missing = [g for g in flist if g not in x_tr.columns]
    if missing:
        raise KeyError(f"missing {len(missing)} features, e.g. {missing[:3]}")
    model = xgb.XGBRegressor(**XGB_DEFAULT)
    x_train = x_tr[flist].to_numpy(dtype=np.float64)
    x_val = x_va[flist].to_numpy(dtype=np.float64)
    model.fit(x_train, y_tr.to_numpy(dtype=np.float64), verbose=False)
    pred = model.predict(x_val)
    pred = np.clip(pred, 0.0, None)
    return {
        "val_r2": eval_r2(y_va.to_numpy(dtype=np.float64), pred),
        "n_features": len(flist),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
    }


def run_method(method: str, targets: list[str]) -> pd.DataFrame:
    feat_path = RESULTS / method / "selected_features.json"
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    features = json.loads(feat_path.read_text(encoding="utf-8"))

    bulk_x_tr, bulk_y_tr = load_bulk_train()
    bulk_x_va, bulk_y_va = load_bulk_val()
    sc_x_tr, sc_y_tr = load_sc_train_combo()
    sc_x_va, sc_y_va = load_sc_val_combo()

    rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        log(f"[eval {method}] ({i}/{len(targets)}) {target}")
        flist = features.get(target, [])
        if not flist:
            rows.append({"target": target, "method": method, "status": "no_features"})
            continue
        try:
            bulk_m = train_eval_target(flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
            sc_m = train_eval_target(flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])
            row = {
                "target": target,
                "method": method,
                "n_features": bulk_m["n_features"],
                "bulk_val_r2": bulk_m["val_r2"],
                "sc_val_r2": sc_m["val_r2"],
                "bulk_n_train": bulk_m["n_train"],
                "bulk_n_val": bulk_m["n_val"],
                "sc_n_train": sc_m["n_train"],
                "sc_n_val": sc_m["n_val"],
            }
            rows.append(row)
            log(
                f"[eval {method}] {target}: bulk_r2={bulk_m['val_r2']:.4f} "
                f"sc_r2={sc_m['val_r2']:.4f} n_feat={bulk_m['n_features']}"
            )
        except Exception as exc:
            log(f"[eval {method}] {target} FAILED: {exc}")
            log(traceback.format_exc())
            rows.append({"target": target, "method": method, "status": "error", "error": str(exc)})

    df = pd.DataFrame(rows)
    out_dir = RESULTS / method
    df.to_csv(out_dir / "val_metrics.csv", index=False)
    ok = df[df["bulk_val_r2"].notna()] if "bulk_val_r2" in df.columns else df.iloc[0:0]
    summary = {
        "method": method,
        "n_targets": int(len(targets)),
        "mean_bulk_val_r2": float(ok["bulk_val_r2"].mean()) if len(ok) else None,
        "mean_sc_val_r2": float(ok["sc_val_r2"].mean()) if len(ok) else None,
        "median_bulk_val_r2": float(ok["bulk_val_r2"].median()) if len(ok) else None,
        "median_sc_val_r2": float(ok["sc_val_r2"].median()) if len(ok) else None,
        "mean_n_features": float(ok["n_features"].mean()) if len(ok) and "n_features" in ok else None,
    }
    with (out_dir / "val_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return df


def main() -> None:
    log("=== Stage01: XGB default train/val eval ===")
    config = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    targets = config["targets"]

    summaries = []
    for method in METHODS:
        log(f"--- eval {method} ---")
        run_method(method, targets)
        summaries.append(json.loads((RESULTS / method / "val_summary.json").read_text()))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "summary_by_method.csv", index=False)
    log("=== summary ===")
    log(summary_df.to_string(index=False))
    log("=== eval done ===")


if __name__ == "__main__":
    main()
