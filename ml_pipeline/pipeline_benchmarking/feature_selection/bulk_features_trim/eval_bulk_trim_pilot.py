"""Pilot: per-target bulk_only trim via XGB shallow (50 miRNA)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from constants import SEED, STAGE, XGB_DEFAULT, XGB_SHALLOW
from io_data import load_bulk_train, load_bulk_val, load_sc_train_combo, load_sc_val_combo

ROOT = STAGE.parent
FULL_RESULTS = ROOT / "stage01_full" / "results"
OUT = STAGE / "results" / "bulk_trim_pilot"

K_OPTIONS = (50, 100, 150, 200)
MIN_BULK_ONLY = 50
MIN_BASELINE_BULK_R2 = 0.4
MAX_REL_DROP = 0.10
MAX_ABS_DROP = 0.02
R2_THRESHOLD = 0.4


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (OUT / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_pilot_targets() -> list[str]:
    path = STAGE / "selected_targets.txt"
    return [t.strip() for t in path.read_text().splitlines() if t.strip()]


def load_full_features() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    bulk = json.loads((FULL_RESULTS / "bulk_features.json").read_text(encoding="utf-8"))
    sc = json.loads((FULL_RESULTS / "sc_features.json").read_text(encoding="utf-8"))
    return bulk, sc


def bulk_only_genes(bulk: list[str], sc: list[str]) -> list[str]:
    sc_set = set(sc)
    return [g for g in bulk if g not in sc_set]


def union_features(sc: list[str], bulk_part: list[str]) -> list[str]:
    sc_unique = list(dict.fromkeys(sc))
    sc_set = set(sc_unique)
    bulk_only = [g for g in bulk_part if g not in sc_set]
    return sc_unique + bulk_only


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


def rank_bulk_only(
    genes: list[str],
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
) -> list[str]:
    if not genes:
        return []
    model = xgb.XGBRegressor(**XGB_SHALLOW)
    model.fit(x_tr[genes].to_numpy(dtype=np.float64), y_tr.to_numpy(dtype=np.float64), verbose=False)
    imp = model.feature_importances_
    order = np.argsort(-imp)
    return [genes[i] for i in order]


def bulk_ok(reduced_r2: float, baseline_r2: float) -> bool:
    if baseline_r2 <= 0:
        return reduced_r2 >= baseline_r2 - MAX_ABS_DROP
    rel_drop = (baseline_r2 - reduced_r2) / baseline_r2
    abs_drop = baseline_r2 - reduced_r2
    return rel_drop <= MAX_REL_DROP and abs_drop <= MAX_ABS_DROP


def pick_k(
    ranked_bulk_only: list[str],
    sc_genes: list[str],
    baseline_bulk_r2: float,
    baseline_sc_r2: float,
    bulk_x_tr,
    bulk_y_tr,
    bulk_x_va,
    bulk_y_va,
    sc_x_tr,
    sc_y_tr,
    sc_x_va,
    sc_y_va,
    target: str,
) -> tuple[int | str, float, float, list[str]]:
    """Pick minimum K with bulk preserved and SC val not below baseline."""
    n_bo = len(ranked_bulk_only)
    candidates = [k for k in K_OPTIONS if k <= n_bo]
    for k in candidates:
        trimmed = ranked_bulk_only[:k]
        flist = union_features(sc_genes, trimmed)
        bulk_r2 = train_eval(flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
        if not bulk_ok(bulk_r2, baseline_bulk_r2):
            continue
        sc_r2 = train_eval(flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])
        if sc_r2 < baseline_sc_r2:
            continue
        return k, bulk_r2, sc_r2, flist

    flist = union_features(sc_genes, ranked_bulk_only)
    bulk_r2 = train_eval(flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
    sc_r2 = train_eval(flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])
    return "full", bulk_r2, sc_r2, flist


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "journal.log").exists():
        (OUT / "journal.log").unlink()

    log("=== bulk_only trim pilot v2 (50 miRNA, SC guard) ===")
    targets = load_pilot_targets()
    bulk_feats, sc_feats = load_full_features()
    log(f"targets={len(targets)} features from {FULL_RESULTS}")

    bulk_x_tr, bulk_y_tr = load_bulk_train()
    bulk_x_va, bulk_y_va = load_bulk_val()
    sc_x_tr, sc_y_tr = load_sc_train_combo()
    sc_x_va, sc_y_va = load_sc_val_combo()

    config = {
        "seed": SEED,
        "k_options": list(K_OPTIONS),
        "min_bulk_only": MIN_BULK_ONLY,
        "min_baseline_bulk_r2": MIN_BASELINE_BULK_R2,
        "max_rel_drop": MAX_REL_DROP,
        "max_abs_drop": MAX_ABS_DROP,
        "sc_guard": "raise_k_if_sc_val_below_baseline",
        "rank_on": "bulk_train_xgb_shallow",
        "eval_model": "xgb_default",
        "feature_source": str(FULL_RESULTS),
        "targets": targets,
    }
    with (OUT / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        log(f"({i}/{len(targets)}) {target}")
        bulk = bulk_feats.get(target, [])
        sc = sc_feats.get(target, [])
        bo = bulk_only_genes(bulk, sc)
        full_flist = union_features(sc, bo)

        baseline_bulk = train_eval(full_flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
        baseline_sc = train_eval(full_flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])

        if len(bo) < MIN_BULK_ONLY:
            k_sel = "skip_small_pool"
            final_flist = full_flist
            sel_bulk = baseline_bulk
            sel_sc = baseline_sc
        elif baseline_bulk < MIN_BASELINE_BULK_R2:
            k_sel = "skip_low_baseline"
            final_flist = full_flist
            sel_bulk = baseline_bulk
            sel_sc = baseline_sc
        else:
            ranked = rank_bulk_only(bo, bulk_x_tr, bulk_y_tr[target])
            k_sel, sel_bulk, sel_sc, final_flist = pick_k(
                ranked,
                sc,
                baseline_bulk,
                baseline_sc,
                bulk_x_tr,
                bulk_y_tr,
                bulk_x_va,
                bulk_y_va,
                sc_x_tr,
                sc_y_tr,
                sc_x_va,
                sc_y_va,
                target,
            )

        bulk_drop = baseline_bulk - sel_bulk
        bulk_rel_drop = bulk_drop / baseline_bulk if baseline_bulk > 0 else None

        row = {
            "target": target,
            "n_sc": len(sc),
            "n_bulk_only": len(bo),
            "n_full": len(full_flist),
            "n_final": len(final_flist),
            "k_selected": k_sel,
            "baseline_bulk_r2": baseline_bulk,
            "selected_bulk_r2": sel_bulk,
            "bulk_r2_drop": bulk_drop,
            "bulk_rel_drop": bulk_rel_drop,
            "baseline_sc_r2": baseline_sc,
            "selected_sc_r2": sel_sc,
            "sc_r2_delta": sel_sc - baseline_sc,
        }
        rows.append(row)
        log(
            f"{target}: k={k_sel} bulk {baseline_bulk:.4f}->{sel_bulk:.4f} "
            f"sc {baseline_sc:.4f}->{sel_sc:.4f} n_feat {len(full_flist)}->{len(final_flist)}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "val_metrics.csv", index=False)

    trimmed = df[~df["k_selected"].astype(str).str.startswith("skip")]
    numeric_k = trimmed[trimmed["k_selected"] != "full"]
    summary = {
        "n_targets": len(df),
        "n_trimmed": int(len(trimmed)),
        "n_skip_small": int((df["k_selected"] == "skip_small_pool").sum()),
        "n_skip_low_baseline": int((df["k_selected"] == "skip_low_baseline").sum()),
        "n_used_full_fallback": int((df["k_selected"] == "full").sum()),
        "k_counts": df["k_selected"].value_counts().to_dict(),
        "mean_baseline_bulk_r2": float(df["baseline_bulk_r2"].mean()),
        "mean_selected_bulk_r2": float(df["selected_bulk_r2"].mean()),
        "mean_bulk_r2_drop": float(df["bulk_r2_drop"].mean()),
        "pct_bulk_ok": float((df["bulk_rel_drop"].fillna(0) <= MAX_REL_DROP).mean()),
        "mean_baseline_sc_r2": float(df["baseline_sc_r2"].mean()),
        "mean_selected_sc_r2": float(df["selected_sc_r2"].mean()),
        "mean_sc_r2_delta": float(df["sc_r2_delta"].mean()),
        "pct_sc_non_degraded": float((df["sc_r2_delta"] >= 0).mean()),
        "n_baseline_bulk_gt_0.4": int((df["baseline_bulk_r2"] > R2_THRESHOLD).sum()),
        "n_selected_bulk_gt_0.4": int((df["selected_bulk_r2"] > R2_THRESHOLD).sum()),
        "n_baseline_sc_gt_0.4": int((df["baseline_sc_r2"] > R2_THRESHOLD).sum()),
        "n_selected_sc_gt_0.4": int((df["selected_sc_r2"] > R2_THRESHOLD).sum()),
        "mean_n_full": float(df["n_full"].mean()),
        "mean_n_final": float(df["n_final"].mean()),
    }
    if len(numeric_k):
        summary["mean_k_selected"] = float(pd.to_numeric(numeric_k["k_selected"], errors="coerce").mean())

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("=== summary ===")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log("=== done ===")


if __name__ == "__main__":
    main()
