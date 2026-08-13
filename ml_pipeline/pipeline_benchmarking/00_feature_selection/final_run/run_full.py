"""Stage01 full: ElasticNet feature selection for all 327 miRNA."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import pandas as pd

from constants import (
    MAX_MODALITY_FEATURES,
    MAX_SPEARMAN_FEATURES,
    MIN_SPEARMAN_FEATURES,
    RESULTS,
    SEED,
    SPEARMAN_THR_HIGH,
    SPEARMAN_THR_LOW,
    STAGE,
)
from feature_union import union_full
from io_data import load_bulk_train, load_sc_train_combo, load_target_list
from second_stage import select_elasticnet
from spearman_screen import adaptive_spearman_indices, spearman_abs_all


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (STAGE / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def genes_from_indices(feature_names: list[str], idx) -> list[str]:
    return [feature_names[i] for i in idx.tolist()]


def screen_modality(X, y, feature_names: list[str]) -> tuple[list[str], dict]:
    import numpy as np

    Xn = X.to_numpy(dtype=np.float64)
    yn = y.to_numpy(dtype=np.float64)
    rhos = spearman_abs_all(Xn, yn)
    pool_idx, sp_meta = adaptive_spearman_indices(Xn, yn, rhos=rhos)
    chosen_idx, stage_meta = select_elasticnet(Xn, yn, pool_idx, rhos=rhos)
    meta = {**sp_meta, **stage_meta, "method": "elasticnet"}
    return genes_from_indices(feature_names, chosen_idx), meta


def save_state(
    bulk_out: dict,
    sc_out: dict,
    final_out: dict,
    details: dict,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "bulk_features.json").open("w", encoding="utf-8") as f:
        json.dump(bulk_out, f, ensure_ascii=False, indent=2)
    with (RESULTS / "sc_features.json").open("w", encoding="utf-8") as f:
        json.dump(sc_out, f, ensure_ascii=False, indent=2)
    with (RESULTS / "selected_features.json").open("w", encoding="utf-8") as f:
        json.dump(final_out, f, ensure_ascii=False, indent=2)
    with (RESULTS / "details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    rows = [
        {
            "target": t,
            "n_bulk": d.get("n_bulk"),
            "n_sc": d.get("n_sc"),
            "n_overlap": d.get("union_meta", {}).get("n_overlap"),
            "n_final": d.get("n_final"),
            "elapsed_s": d.get("elapsed_s"),
            "status": d.get("status", "ok"),
        }
        for t, d in details.items()
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "feature_counts.csv", index=False)


def load_state() -> tuple[dict, dict, dict, dict]:
    bulk_out: dict = {}
    sc_out: dict = {}
    final_out: dict = {}
    details: dict = {}
    for name, store in [
        ("bulk_features.json", bulk_out),
        ("sc_features.json", sc_out),
        ("selected_features.json", final_out),
        ("details.json", details),
    ]:
        path = RESULTS / name
        if path.exists():
            store.update(json.loads(path.read_text(encoding="utf-8")))
    return bulk_out, sc_out, final_out, details


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log("=== Stage01 full: ElasticNet feature selection (327 miRNA) ===")

    targets = load_target_list()
    log(f"targets: {len(targets)} seed={SEED}")

    bulk_x, bulk_y = load_bulk_train()
    sc_x, sc_y = load_sc_train_combo()
    feature_names = list(bulk_x.columns)
    log(f"bulk train {bulk_x.shape}, sc train combo {sc_x.shape}, features {len(feature_names)}")

    config = {
        "method": "elasticnet",
        "seed": SEED,
        "n_targets": len(targets),
        "spearman_thr_high": SPEARMAN_THR_HIGH,
        "spearman_thr_low": SPEARMAN_THR_LOW,
        "min_spearman_features": MIN_SPEARMAN_FEATURES,
        "max_spearman_features": MAX_SPEARMAN_FEATURES,
        "max_modality_features": MAX_MODALITY_FEATURES,
        "linear_max_pool": 1500,
        "union_policy": "sc_first_bulk_only_no_cap",
        "sc_train": "sc_k1_train + sc_pb_train(K2)",
        "targets": targets,
    }
    with (STAGE / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    bulk_out, sc_out, final_out, details = load_state()
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
            bulk_genes, bulk_meta = screen_modality(bulk_x, bulk_y[target], feature_names)
            sc_genes, sc_meta = screen_modality(sc_x, sc_y[target], feature_names)
            final_genes, union_meta = union_full(sc_genes, bulk_genes)
            bulk_out[target] = bulk_genes
            sc_out[target] = sc_genes
            final_out[target] = final_genes
            details[target] = {
                "n_bulk": len(bulk_genes),
                "n_sc": len(sc_genes),
                "n_final": len(final_genes),
                "union_meta": union_meta,
                "bulk_meta": bulk_meta,
                "sc_meta": sc_meta,
                "elapsed_s": round(time.time() - t0, 2),
            }
            log(
                f"{target}: bulk={len(bulk_genes)} sc={len(sc_genes)} "
                f"overlap={union_meta['n_overlap']} final={len(final_genes)} "
                f"({details[target]['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            errors += 1
            log(f"{target} FAILED: {exc}")
            log(traceback.format_exc())
            details[target] = {"status": "error", "error": str(exc), "elapsed_s": round(time.time() - t0, 2)}

        save_state(bulk_out, sc_out, final_out, details)

    ok = sum(1 for d in details.values() if d.get("status") != "error")
    log(f"=== done: {ok}/{len(targets)} ok, {errors} errors ===")
    log(f"Saved under {RESULTS}")


if __name__ == "__main__":
    main()
