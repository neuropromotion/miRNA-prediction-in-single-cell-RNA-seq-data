"""Stage01 bulk trim: per-target K on bulk_only for all 327 miRNA."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from constants import (
    FEATURE_SOURCE,
    K_OPTIONS,
    MAX_ABS_DROP,
    MAX_REL_DROP,
    MIN_BASELINE_BULK_R2,
    MIN_BULK_ONLY,
    R2_THRESHOLD,
    RESULTS,
    SEED,
    STAGE,
    XGB_DEFAULT,
    XGB_SHALLOW,
)
from io_data import load_bulk_train, load_bulk_val, load_sc_train_combo, load_sc_val_combo, load_target_list


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (STAGE / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_source_features() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    bulk = json.loads((FEATURE_SOURCE / "bulk_features.json").read_text(encoding="utf-8"))
    sc = json.loads((FEATURE_SOURCE / "sc_features.json").read_text(encoding="utf-8"))
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


def rank_bulk_only(genes: list[str], x_tr: pd.DataFrame, y_tr: pd.Series) -> list[str]:
    if not genes:
        return []
    model = xgb.XGBRegressor(**XGB_SHALLOW)
    model.fit(x_tr[genes].to_numpy(dtype=np.float64), y_tr.to_numpy(dtype=np.float64), verbose=False)
    order = np.argsort(-model.feature_importances_)
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
) -> tuple[int | str, float, float, list[str], list[str]]:
    """Return k, bulk_r2, sc_r2, final_features, kept_bulk_only."""
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
        return k, bulk_r2, sc_r2, flist, trimmed

    flist = union_features(sc_genes, ranked_bulk_only)
    bulk_r2 = train_eval(flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
    sc_r2 = train_eval(flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])
    return "full", bulk_r2, sc_r2, flist, ranked_bulk_only


def load_state() -> tuple[dict, dict, dict, dict]:
    selected: dict = {}
    sc_out: dict = {}
    bulk_trimmed: dict = {}
    details: dict = {}
    for name, store in [
        ("selected_features.json", selected),
        ("sc_features.json", sc_out),
        ("bulk_trimmed_features.json", bulk_trimmed),
        ("details.json", details),
    ]:
        path = RESULTS / name
        if path.exists():
            store.update(json.loads(path.read_text(encoding="utf-8")))
    return selected, sc_out, bulk_trimmed, details


def save_state(
    selected: dict,
    sc_out: dict,
    bulk_trimmed: dict,
    details: dict,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "selected_features.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    with (RESULTS / "sc_features.json").open("w", encoding="utf-8") as f:
        json.dump(sc_out, f, ensure_ascii=False, indent=2)
    with (RESULTS / "bulk_trimmed_features.json").open("w", encoding="utf-8") as f:
        json.dump(bulk_trimmed, f, ensure_ascii=False, indent=2)
    with (RESULTS / "details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    rows = [
        {
            "target": t,
            "k_selected": d.get("k_selected"),
            "n_sc": d.get("n_sc"),
            "n_bulk_only": d.get("n_bulk_only"),
            "n_bulk_trimmed": d.get("n_bulk_trimmed"),
            "n_full": d.get("n_full"),
            "n_final": d.get("n_final"),
            "baseline_bulk_r2": d.get("baseline_bulk_r2"),
            "selected_bulk_r2": d.get("selected_bulk_r2"),
            "baseline_sc_r2": d.get("baseline_sc_r2"),
            "selected_sc_r2": d.get("selected_sc_r2"),
            "elapsed_s": d.get("elapsed_s"),
            "status": d.get("status", "ok"),
        }
        for t, d in details.items()
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "val_metrics.csv", index=False)


def write_summary(details: dict) -> None:
    df = pd.DataFrame([d for d in details.values() if d.get("status") != "error"])
    if df.empty:
        return
    numeric_k = df[~df["k_selected"].astype(str).str.startswith("skip")]
    numeric_k = numeric_k[numeric_k["k_selected"] != "full"]
    summary = {
        "n_targets": int(len(df)),
        "n_skip_small": int((df["k_selected"] == "skip_small_pool").sum()),
        "n_skip_low_baseline": int((df["k_selected"] == "skip_low_baseline").sum()),
        "n_used_full_fallback": int((df["k_selected"] == "full").sum()),
        "k_counts": df["k_selected"].value_counts().to_dict(),
        "mean_baseline_bulk_r2": float(df["baseline_bulk_r2"].mean()),
        "mean_selected_bulk_r2": float(df["selected_bulk_r2"].mean()),
        "mean_baseline_sc_r2": float(df["baseline_sc_r2"].mean()),
        "mean_selected_sc_r2": float(df["selected_sc_r2"].mean()),
        "mean_sc_r2_delta": float((df["selected_sc_r2"] - df["baseline_sc_r2"]).mean()),
        "n_baseline_bulk_gt_0.4": int((df["baseline_bulk_r2"] > R2_THRESHOLD).sum()),
        "n_selected_bulk_gt_0.4": int((df["selected_bulk_r2"] > R2_THRESHOLD).sum()),
        "n_baseline_sc_gt_0.4": int((df["baseline_sc_r2"] > R2_THRESHOLD).sum()),
        "n_selected_sc_gt_0.4": int((df["selected_sc_r2"] > R2_THRESHOLD).sum()),
        "mean_n_full": float(df["n_full"].mean()),
        "mean_n_final": float(df["n_final"].mean()),
    }
    if len(numeric_k):
        summary["mean_k_selected"] = float(pd.to_numeric(numeric_k["k_selected"], errors="coerce").mean())
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log("=== Stage01 bulk trim full (327 miRNA) ===")

    targets = load_target_list()
    bulk_src, sc_src = load_source_features()
    log(f"targets={len(targets)} source={FEATURE_SOURCE}")

    bulk_x_tr, bulk_y_tr = load_bulk_train()
    bulk_x_va, bulk_y_va = load_bulk_val()
    sc_x_tr, sc_y_tr = load_sc_train_combo()
    sc_x_va, sc_y_va = load_sc_val_combo()

    config = {
        "seed": SEED,
        "n_targets": len(targets),
        "k_options": list(K_OPTIONS),
        "min_bulk_only": MIN_BULK_ONLY,
        "min_baseline_bulk_r2": MIN_BASELINE_BULK_R2,
        "max_rel_drop": MAX_REL_DROP,
        "max_abs_drop": MAX_ABS_DROP,
        "sc_guard": "raise_k_if_sc_val_below_baseline",
        "rank_on": "bulk_train_xgb_shallow",
        "feature_source": str(FEATURE_SOURCE),
        "targets": targets,
    }
    with (STAGE / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    selected, sc_out, bulk_trimmed, details = load_state()
    done = {t for t, d in details.items() if d.get("status") != "error"}
    if done:
        log(f"resume: {len(done)} targets already done")

    errors = 0
    for i, target in enumerate(targets, start=1):
        if target in done:
            continue
        log(f"({i}/{len(targets)}) {target}")
        t0 = time.time()
        try:
            bulk = bulk_src.get(target, [])
            sc = sc_src.get(target, [])
            bo = bulk_only_genes(bulk, sc)
            full_flist = union_features(sc, bo)

            baseline_bulk = train_eval(full_flist, bulk_x_tr, bulk_y_tr[target], bulk_x_va, bulk_y_va[target])
            baseline_sc = train_eval(full_flist, sc_x_tr, sc_y_tr[target], sc_x_va, sc_y_va[target])

            if len(bo) < MIN_BULK_ONLY:
                k_sel = "skip_small_pool"
                final_flist = full_flist
                kept_bo = bo
                sel_bulk, sel_sc = baseline_bulk, baseline_sc
            elif baseline_bulk < MIN_BASELINE_BULK_R2:
                k_sel = "skip_low_baseline"
                final_flist = full_flist
                kept_bo = bo
                sel_bulk, sel_sc = baseline_bulk, baseline_sc
            else:
                ranked = rank_bulk_only(bo, bulk_x_tr, bulk_y_tr[target])
                k_sel, sel_bulk, sel_sc, final_flist, kept_bo = pick_k(
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

            selected[target] = final_flist
            sc_out[target] = sc
            bulk_trimmed[target] = kept_bo
            details[target] = {
                "k_selected": k_sel,
                "n_sc": len(sc),
                "n_bulk_only": len(bo),
                "n_bulk_trimmed": len(kept_bo),
                "n_full": len(full_flist),
                "n_final": len(final_flist),
                "baseline_bulk_r2": baseline_bulk,
                "selected_bulk_r2": sel_bulk,
                "baseline_sc_r2": baseline_sc,
                "selected_sc_r2": sel_sc,
                "sc_r2_delta": sel_sc - baseline_sc,
                "elapsed_s": round(time.time() - t0, 2),
            }
            log(
                f"{target}: k={k_sel} bulk {baseline_bulk:.4f}->{sel_bulk:.4f} "
                f"sc {baseline_sc:.4f}->{sel_sc:.4f} n_feat {len(full_flist)}->{len(final_flist)} "
                f"({details[target]['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            errors += 1
            log(f"{target} FAILED: {exc}")
            log(traceback.format_exc())
            details[target] = {"status": "error", "error": str(exc), "elapsed_s": round(time.time() - t0, 2)}

        save_state(selected, sc_out, bulk_trimmed, details)

    write_summary(details)
    ok = sum(1 for d in details.values() if d.get("status") != "error")
    log(f"=== done: {ok}/{len(targets)} ok, {errors} errors ===")
    log(f"Saved under {RESULTS}")


if __name__ == "__main__":
    main()
