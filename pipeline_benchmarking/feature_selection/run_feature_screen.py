"""Stage01: feature selection screen (3 methods × 50 miRNA)."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    METHODS,
    N_PILOT_TARGETS,
    RESULTS,
    SEED,
    STAGE,
    SPEARMAN_THR_HIGH,
    SPEARMAN_THR_LOW,
    MIN_SPEARMAN_FEATURES,
    MAX_SPEARMAN_FEATURES,
    SECOND_STAGE_TOP_K,
)
from io_data import load_bulk_train, load_sc_train_combo, load_target_list
from second_stage import select_lasso, select_mi, select_xgb_importance
from spearman_screen import adaptive_spearman_indices, spearman_abs_all


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (STAGE / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pick_pilot_targets(all_targets: list[str]) -> list[str]:
    path = STAGE / "selected_targets.txt"
    if path.exists():
        return [t.strip() for t in path.read_text().splitlines() if t.strip()]
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(all_targets, size=N_PILOT_TARGETS, replace=False).tolist())
    path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return chosen


def genes_from_indices(feature_names: list[str], idx: np.ndarray) -> list[str]:
    return [feature_names[i] for i in idx.tolist()]


def screen_modality(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    method: str,
) -> tuple[list[str], dict]:
    Xn = X.to_numpy(dtype=np.float64)
    yn = y.to_numpy(dtype=np.float64)
    rhos = spearman_abs_all(Xn, yn)
    pool_idx, sp_meta = adaptive_spearman_indices(Xn, yn, rhos=rhos)

    if method == "method_a_lasso":
        chosen_idx, stage_meta = select_lasso(Xn, yn, pool_idx, rhos=rhos)
    elif method == "method_b_xgb_imp":
        chosen_idx, stage_meta = select_xgb_importance(Xn, yn, pool_idx)
    elif method == "method_c_mi":
        chosen_idx, stage_meta = select_mi(Xn, yn, pool_idx)
    else:
        raise ValueError(method)

    meta = {**sp_meta, **stage_meta, "method": method}
    return genes_from_indices(feature_names, chosen_idx), meta


def run_method(method: str, targets: list[str], bulk_x, bulk_y, sc_x, sc_y, feature_names: list[str]) -> None:
    out_dir = RESULTS / method
    out_dir.mkdir(parents=True, exist_ok=True)
    features_out: dict[str, list[str]] = {}
    details: dict[str, dict] = {}

    for i, target in enumerate(targets, start=1):
        log(f"[{method}] ({i}/{len(targets)}) {target}")
        t0 = time.time()
        try:
            bulk_genes, bulk_meta = screen_modality(bulk_x, bulk_y[target], feature_names, method)
            sc_genes, sc_meta = screen_modality(sc_x, sc_y[target], feature_names, method)
            union = sorted(set(bulk_genes) | set(sc_genes))
            features_out[target] = union
            details[target] = {
                "n_bulk": len(bulk_genes),
                "n_sc": len(sc_genes),
                "n_union": len(union),
                "bulk_meta": bulk_meta,
                "sc_meta": sc_meta,
                "elapsed_s": round(time.time() - t0, 2),
            }
            log(
                f"[{method}] {target}: bulk={len(bulk_genes)} sc={len(sc_genes)} "
                f"union={len(union)} ({details[target]['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            log(f"[{method}] {target} FAILED: {exc}")
            log(traceback.format_exc())
            details[target] = {"status": "error", "error": str(exc)}

    with (out_dir / "selected_features.json").open("w", encoding="utf-8") as f:
        json.dump(features_out, f, ensure_ascii=False, indent=2)
    with (out_dir / "details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    rows = [
        {
            "target": t,
            "n_bulk": d.get("n_bulk"),
            "n_sc": d.get("n_sc"),
            "n_union": d.get("n_union"),
            "elapsed_s": d.get("elapsed_s"),
        }
        for t, d in details.items()
        if d.get("status") != "error"
    ]
    pd.DataFrame(rows).to_csv(out_dir / "feature_counts.csv", index=False)


def main() -> None:
    if (STAGE / "journal.log").exists():
        (STAGE / "journal.log").unlink()
    RESULTS.mkdir(parents=True, exist_ok=True)

    log("=== Stage01: feature selection screen ===")
    all_targets = load_target_list()
    targets = pick_pilot_targets(all_targets)
    log(f"pilot targets: {len(targets)} seed={SEED}")

    bulk_x, bulk_y = load_bulk_train()
    sc_x, sc_y = load_sc_train_combo()
    feature_names = list(bulk_x.columns)
    log(f"bulk train {bulk_x.shape}, sc train combo {sc_x.shape}, features {len(feature_names)}")

    config = {
        "seed": SEED,
        "n_pilot_targets": N_PILOT_TARGETS,
        "spearman_thr_high": SPEARMAN_THR_HIGH,
        "spearman_thr_low": SPEARMAN_THR_LOW,
        "min_spearman_features": MIN_SPEARMAN_FEATURES,
        "max_spearman_features": MAX_SPEARMAN_FEATURES,
        "second_stage_top_k": SECOND_STAGE_TOP_K,
        "sc_train": "sc_k1_train + sc_pb_train(K2)",
        "methods": list(METHODS),
        "targets": targets,
    }
    with (STAGE / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    for method in METHODS:
        log(f"--- {method} ---")
        run_method(method, targets, bulk_x, bulk_y, sc_x, sc_y, feature_names)

    log(f"Saved under {RESULTS}")
    log("=== screen done ===")


if __name__ == "__main__":
    main()
