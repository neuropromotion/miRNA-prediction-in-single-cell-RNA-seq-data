"""Stage01: eval on saved feature lists + method combos (no re-screen)."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from constants import RESULTS, STAGE, XGB_DEFAULT
from io_data import load_bulk_train, load_bulk_val, load_sc_train_combo, load_sc_val_combo

COMPARISON = RESULTS / "comparison"
R2_THRESHOLD = 0.4

BASE_METHODS: dict[str, str] = {
    "spearman": "method_e_spearman",
    "lasso": "method_a_lasso",
    "elasticnet": "method_d_elasticnet",
    "xgb": "method_b_xgb_imp",
    "mi": "method_c_mi",
}

COMBOS: dict[str, tuple[str, str]] = {
    "lasso_xgb": ("lasso", "xgb"),
    "lasso_mi": ("lasso", "mi"),
    "elasticnet_xgb": ("elasticnet", "xgb"),
    "elasticnet_mi": ("elasticnet", "mi"),
}

ALL_CONFIGS: dict[str, str | tuple[str, str]] = {**BASE_METHODS, **COMBOS}
BASE_CONFIGS = set(BASE_METHODS)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (STAGE / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_method_features(method_key: str) -> dict[str, list[str]]:
    path = RESULTS / BASE_METHODS[method_key] / "selected_features.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def union_features(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys(a + b))


def features_for_config(
    config: str,
    spec: str | tuple[str, str],
    loaded: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    if isinstance(spec, str):
        return loaded[spec]
    left, right = spec
    left_feats = loaded[left]
    right_feats = loaded[right]
    targets = set(left_feats) | set(right_feats)
    return {t: union_features(left_feats.get(t, []), right_feats.get(t, [])) for t in targets}


def eval_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def train_eval_target(
    flist: list[str],
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    y_va: pd.Series,
) -> dict:
    model = xgb.XGBRegressor(**XGB_DEFAULT)
    x_train = x_tr[flist].to_numpy(dtype=np.float64)
    x_val = x_va[flist].to_numpy(dtype=np.float64)
    model.fit(x_train, y_tr.to_numpy(dtype=np.float64), verbose=False)
    pred = np.clip(model.predict(x_val), 0.0, None)
    return {"val_r2": eval_r2(y_va.to_numpy(dtype=np.float64), pred), "n_features": len(flist)}


def run_config(
    config: str,
    spec: str | tuple[str, str],
    targets: list[str],
    features: dict[str, list[str]],
    bulk_x_tr,
    bulk_y_tr,
    bulk_x_va,
    bulk_y_va,
    sc_x_tr,
    sc_y_tr,
    sc_x_va,
    sc_y_va,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        log(f"[comparison {config}] ({i}/{len(targets)}) {target}")
        flist = features.get(target, [])
        if not flist:
            rows.append({"target": target, "config": config, "status": "no_features"})
            continue
        try:
            bulk_m = train_eval_target(flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
            sc_m = train_eval_target(flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])
            rows.append(
                {
                    "target": target,
                    "config": config,
                    "n_features": bulk_m["n_features"],
                    "bulk_val_r2": bulk_m["val_r2"],
                    "sc_val_r2": sc_m["val_r2"],
                }
            )
            log(
                f"[comparison {config}] {target}: bulk_r2={bulk_m['val_r2']:.4f} "
                f"sc_r2={sc_m['val_r2']:.4f} n_feat={bulk_m['n_features']}"
            )
        except Exception as exc:
            log(f"[comparison {config}] {target} FAILED: {exc}")
            log(traceback.format_exc())
            rows.append({"target": target, "config": config, "status": "error", "error": str(exc)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df["bulk_val_r2"].notna()] if "bulk_val_r2" in df.columns else df.iloc[0:0]
    return {
        "n_targets": int(len(ok)),
        "mean_bulk_val_r2": float(ok["bulk_val_r2"].mean()) if len(ok) else None,
        "mean_sc_val_r2": float(ok["sc_val_r2"].mean()) if len(ok) else None,
        "median_bulk_val_r2": float(ok["bulk_val_r2"].median()) if len(ok) else None,
        "median_sc_val_r2": float(ok["sc_val_r2"].median()) if len(ok) else None,
        "n_bulk_r2_gt_0.4": int((ok["bulk_val_r2"] > R2_THRESHOLD).sum()) if len(ok) else 0,
        "n_sc_r2_gt_0.4": int((ok["sc_val_r2"] > R2_THRESHOLD).sum()) if len(ok) else 0,
        "mean_n_features": float(ok["n_features"].mean()) if len(ok) and "n_features" in ok else None,
    }


def load_existing_base_metrics(config: str, method_dir: str) -> pd.DataFrame | None:
    path = RESULTS / method_dir / "val_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "bulk_val_r2" not in df.columns:
        return None
    out = df[["target", "n_features", "bulk_val_r2", "sc_val_r2"]].copy()
    out["config"] = config
    return out


def main() -> None:
    log("=== Stage01 comparison eval (saved features, no re-screen) ===")
    config_data = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    targets = config_data["targets"]
    COMPARISON.mkdir(parents=True, exist_ok=True)

    loaded = {key: load_method_features(key) for key in BASE_METHODS}
    bulk_x_tr, bulk_y_tr = load_bulk_train()
    bulk_x_va, bulk_y_va = load_bulk_val()
    sc_x_tr, sc_y_tr = load_sc_train_combo()
    sc_x_va, sc_y_va = load_sc_val_combo()

    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for config, spec in ALL_CONFIGS.items():
        log(f"--- comparison {config} ---")
        if config in BASE_CONFIGS and isinstance(spec, str):
            existing = load_existing_base_metrics(config, spec)
            if existing is not None:
                log(f"[comparison {config}] reuse existing val_metrics from {spec}")
                df = existing
            else:
                features = features_for_config(config, spec, loaded)
                df = run_config(
                    config,
                    spec,
                    targets,
                    features,
                    bulk_x_tr,
                    bulk_y_tr,
                    bulk_x_va,
                    bulk_y_va,
                    sc_x_tr,
                    sc_y_tr,
                    sc_x_va,
                    sc_y_va,
                )
        else:
            features = features_for_config(config, spec, loaded)
            df = run_config(
                config,
                spec,
                targets,
                features,
                bulk_x_tr,
                bulk_y_tr,
                bulk_x_va,
                bulk_y_va,
                sc_x_tr,
                sc_y_tr,
                sc_x_va,
                sc_y_va,
            )
        out_path = COMPARISON / f"{config}_val_metrics.csv"
        df.to_csv(out_path, index=False)
        summary = {"config": config, **summarize(df)}
        summaries.append(summary)
        all_metrics.append(df)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(COMPARISON / "val_metrics_all.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(COMPARISON / "summary_by_config.csv", index=False)
    with (COMPARISON / "summary_by_config.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    log("=== comparison summary ===")
    log(summary_df.to_string(index=False))
    log("=== comparison eval done ===")


if __name__ == "__main__":
    main()
