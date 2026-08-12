#!/usr/bin/env python3
"""Run Optimal_K on TEST tune half: predict → bootstrap → select → tables/figures.

Supports FINAL_SHARD=i/N (writes tables/*_shard{i}.csv). After all shards finish,
run ``python merge_shards.py`` to merge tables, write proto_prediction_config, figures.

Production ``prediction_config.json`` (with TEST eval metrics) is built in test_metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_select import decide_from_medians, medians_on_tune
from config import (
    ASSIGNMENT_RULE,
    BOOT_SUMMARY_PATH,
    COHORT_COUNTS_PATH,
    COHORTS,
    DECISIONS_PATH,
    DELTA,
    FIGURES,
    JOURNAL,
    MEDIAN_THRESHOLD,
    MODELS_ROOT,
    N_BOOTSTRAP,
    PREDICTION_CONFIG_PATH,
    RESULTS,
    SEED,
    SELECTED_FEATURES,
    SPLIT_PATH,
    TABLES,
    WEIGHTS_DIR,
)
from data_loading import load_bulk_split, load_features, load_sc_splits, load_targets
from plots import plot_all
from predict_stack import predict_target_all_cohorts
from splits import ensure_split

os.environ.setdefault("FINAL_MODELS_ROOT", str(MODELS_ROOT))


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _shard_suffix() -> str:
    suffix = os.environ.get("FINAL_METRICS_SUFFIX", "").strip()
    if suffix:
        return suffix
    if os.environ.get("FINAL_SHARD"):
        idx_s, _ = os.environ["FINAL_SHARD"].split("/")
        return f"_shard{idx_s}"
    return ""


def _out_paths() -> tuple[Path, Path, bool]:
    """Return (decisions_path, boot_path, do_finalize)."""
    suffix = _shard_suffix()
    if not suffix:
        return DECISIONS_PATH, BOOT_SUMMARY_PATH, True
    return (
        DECISIONS_PATH.with_name(DECISIONS_PATH.stem + suffix + DECISIONS_PATH.suffix),
        BOOT_SUMMARY_PATH.with_name(BOOT_SUMMARY_PATH.stem + suffix + BOOT_SUMMARY_PATH.suffix),
        False,
    )


def filter_targets(targets: list[str]) -> list[str]:
    available = {p.stem for p in WEIGHTS_DIR.glob("*.json")}
    targets = [t for t in targets if t in available]
    if os.environ.get("FINAL_TARGETS"):
        wanted = {t.strip() for t in os.environ["FINAL_TARGETS"].split(",") if t.strip()}
        targets = [t for t in targets if t in wanted]
    shard = os.environ.get("FINAL_SHARD")
    if shard:
        idx_s, n_s = shard.split("/")
        idx, n = int(idx_s), int(n_s)
        targets = [t for j, t in enumerate(targets) if j % n == idx]
    return targets


def build_prediction_config(decisions: pd.DataFrame, features: dict[str, list[str]]) -> dict:
    elig = decisions[decisions["eligible"]].copy()
    by_k: dict[str, dict] = {k: {} for k in COHORTS}
    for _, row in elig.iterrows():
        t = str(row["target"])
        k = str(row["optimal_k"])
        by_k[k][t] = {
            "features": features.get(t, []),
            "m_bulk": float(row["m_bulk"]),
            "m_assigned": float(row[f"m_{k}"]),
        }
    return {
        "version": 2,
        "assignment_rule": ASSIGNMENT_RULE,
        "thresholds": {
            "median_r2": MEDIAN_THRESHOLD,
            "delta": DELTA,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "split_path": str(SPLIT_PATH),
        "features_source": str(SELECTED_FEATURES),
        "eligible_mirs": sorted(elig["target"].astype(str)),
        "n_eligible": int(len(elig)),
        "n_ineligible": int((~decisions["eligible"]).sum()),
        "cohorts": list(COHORTS),
        "cohort_counts": {k: len(by_k[k]) for k in COHORTS},
        **by_k,
    }


def finalize(decisions: pd.DataFrame, features: dict[str, list[str]]) -> None:
    ok = decisions[decisions.get("status", "ok") == "ok"].copy()
    elig = ok[ok["eligible"] == True]  # noqa: E712
    counts = (
        elig["optimal_k"].value_counts().reindex(list(COHORTS), fill_value=0).rename("n")
        if len(elig)
        else pd.Series(0, index=list(COHORTS), name="n")
    )
    counts.to_csv(COHORT_COUNTS_PATH, header=True)
    cfg = build_prediction_config(ok, features)
    PREDICTION_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    plot_all(ok)
    _log(f"done eligible={cfg['n_eligible']} ineligible={cfg['n_ineligible']}")
    _log(f"config → {PREDICTION_CONFIG_PATH}")
    _log(f"figures → {FIGURES}")
    _log(f"tables  → {TABLES}")
    print(counts.to_string())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    decisions_path, boot_path, do_finalize = _out_paths()

    _log("=== Optimal_K (TEST tune half, bootstrap) ===")
    _log(ASSIGNMENT_RULE)
    _log(f"models_root={MODELS_ROOT} out={decisions_path.name}")

    features = load_features()
    targets = filter_targets(load_targets())
    _log(f"targets={len(targets)}")

    sc_splits = load_sc_splits()
    bulk = load_bulk_split()
    n_sc = len(next(iter(sc_splits.values())).x)
    n_bulk = len(bulk.x)
    split = ensure_split(n_sc, n_bulk, force=os.environ.get("FORCE_SPLIT", "") == "1")
    _log(
        f"split={SPLIT_PATH} "
        f"sc_tune={len(split['sc']['tune_idx'])}/{n_sc} "
        f"bulk_tune={len(split['bulk']['tune_idx'])}/{n_bulk}"
    )
    tune_sc = np.asarray(split["sc"]["tune_idx"], dtype=np.int64)
    tune_bulk = np.asarray(split["bulk"]["tune_idx"], dtype=np.int64)

    force_pred = os.environ.get("FORCE_PRED", "") == "1"
    rows: list[dict] = []
    boot_rows: list[dict] = []

    for i, target in enumerate(targets, 1):
        genes = features[target]
        _log(f"({i}/{len(targets)}) {target} n_feat={len(genes)}")
        t0 = time.perf_counter()
        try:
            preds = predict_target_all_cohorts(
                target, genes, sc_splits, bulk, force=force_pred
            )
            tseed = SEED + int(hashlib.md5(target.encode()).hexdigest()[:8], 16) % 1_000_000
            med = medians_on_tune(preds, tune_sc, tune_bulk, target_seed=tseed)
            decision = decide_from_medians(target, med)
            row = {
                "target": target,
                "eligible": decision.eligible,
                "optimal_k": decision.optimal_k or "",
                "reason": decision.reason,
                "m_bulk": med["bulk"],
                **{f"m_{k}": med[k] for k in COHORTS},
                "sec": round(time.perf_counter() - t0, 2),
                "status": "ok",
            }
            rows.append(row)
            boot_rows.append({"target": target, **{k: med[k] for k in (*COHORTS, "bulk")}})
            _log(
                f"  eligible={decision.eligible} K={decision.optimal_k} "
                f"m_bulk={med['bulk']:.3f} maxK={max(med[k] for k in COHORTS):.3f} "
                f"({row['sec']}s)"
            )
        except Exception as exc:
            _log(f"  FAIL: {exc}")
            rows.append(
                {
                    "target": target,
                    "eligible": False,
                    "optimal_k": "",
                    "reason": f"error: {exc}",
                    "status": "fail",
                    "sec": round(time.perf_counter() - t0, 2),
                }
            )
        pd.DataFrame(rows).to_csv(decisions_path, index=False)

    decisions = pd.DataFrame(rows)
    decisions.to_csv(decisions_path, index=False)
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    _log(f"wrote {decisions_path} n={len(decisions)}")

    if do_finalize:
        finalize(decisions, features)
    else:
        _log("shard done — run merge_shards.py after all shards finish")


if __name__ == "__main__":
    main()
