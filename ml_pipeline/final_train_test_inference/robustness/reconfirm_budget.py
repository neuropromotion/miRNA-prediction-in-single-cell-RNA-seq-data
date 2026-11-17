#!/usr/bin/env python3
"""Re-run the budget confirmation step from saved results (no LOO, no new bisection).

For each selected miRNA, the budgets that passed the bisection search
(``tables/budget_probes_<mir>.csv``) are confirmed on fresh masks, largest
first, skipping budgets whose confirmation already failed. The first budget
that passes becomes ``importance_budget``; if none passes, the budget is 0
(only a complete SC panel is accepted). Gene weights are read from
``feature_weights/``.

Updates ``tables/calibration_summary.csv`` (one row per miRNA),
``tables/budget_probes_<mir>.csv`` (new confirm rows appended) and
``robustness_config.json``.

    ROBUSTNESS_ENTRY=reconfirm_budget.py bash run_robustness.sh            # failed confirmations only
    ROBUSTNESS_ENTRY=reconfirm_budget.py bash run_robustness.sh --dry-run  # list targets, change nothing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_ML = _THIS_DIR.parent.parent
_TABPACK = os.environ.get("TABPACK_ROOT", str(_ML / "deps" / "tabpack"))
_tab_src = str(Path(_TABPACK).resolve() / "src")
if _tab_src not in sys.path:
    sys.path.insert(0, _tab_src)
os.environ.setdefault("TABPACK_ROOT", str(Path(_TABPACK).resolve()))

import common as cm  # noqa: E402
import robustness_paths as C  # noqa: E402
from budget_calibrate import (  # noqa: E402
    confirm_with_fallback,
    confirmation_candidates,
    retention_statistic,
    target_seed,
)
from run_robustness import _journal, _safe_name, _write_config  # noqa: E402

CALIBRATION_PATH = C.TABLES_DIR / "calibration_summary.csv"


def _load_calibration() -> pd.DataFrame:
    cal = pd.read_csv(CALIBRATION_PATH)
    cal = cal.drop_duplicates(subset="mirna", keep="last").reset_index(drop=True)
    if "budget_search" not in cal.columns:
        cal["budget_search"] = np.nan
    if "n_confirm_rounds" not in cal.columns:
        cal["n_confirm_rounds"] = np.nan
    ok = cal["status"] == "ok"
    cal.loc[ok & cal["budget_search"].isna(), "budget_search"] = cal.loc[ok, "importance_budget"]
    cal.loc[ok & cal["n_confirm_rounds"].isna(), "n_confirm_rounds"] = 1
    return cal


def _select_targets(cal: pd.DataFrame, eligible: list[str], args: argparse.Namespace) -> list[str]:
    ok = cal[(cal["status"] == "ok") & cal["mirna"].isin(eligible)]
    if args.targets:
        wanted = {t.strip() for t in args.targets.split(",") if t.strip()}
        return [t for t in ok["mirna"] if t in wanted]
    failed = ~ok["meets_threshold"].astype(str).str.lower().isin(["true", "1"])
    return list(ok.loc[failed, "mirna"])


def _reconfirm_one(target, row, target_info, sc_features, test_splits, n_masks, device):
    probes_path = C.TABLES_DIR / f"budget_probes_{_safe_name(target)}.csv"
    probes = pd.read_csv(probes_path)
    probes["ok"] = probes["ok"].astype(str).str.lower().isin(["true", "1"])
    failed_confirm = set(probes.loc[(probes["phase"] == "confirm") & ~probes["ok"], "budget"].round(4))
    candidates = [
        b for b in confirmation_candidates(probes.to_dict("records")) if round(b, 4) not in failed_confirm
    ]

    cohort = str(target_info[target]["assigned_cohort"])
    split = test_splits[cohort]
    k1_ref = test_splits.get("_k1_ref") if cohort == "K1" else None
    bundle = cm.load_target_bundle(target, target_info, device=device)
    sc_genes = cm.sc_genes_for_target(target, bundle.genes, sc_features)
    w = pd.read_csv(C.WEIGHTS_DIR / f"{_safe_name(target)}.csv")
    weights = dict(zip(w["gene"].astype(str), w["weight"].astype(float)))

    # +1: masks independent of the original confirmation round
    rng = np.random.default_rng(target_seed(C.SEED, target) + 1)
    budget, samples, new_probes = confirm_with_fallback(
        bundle,
        split.x,
        split.y[target],
        sc_genes,
        weights,
        candidates,
        threshold=float(row["r2_threshold"]),
        retention_stat=C.RETENTION_STAT,
        confirm_n_masks=n_masks,
        rng=rng,
        common_mod=cm,
        k1_ref=k1_ref,
    )
    if budget is None:
        base = probes[(probes["phase"] == "search") & (probes["budget"] == 0)]
        budget, stat, median = 0.0, float(base["stat"].iloc[0]), float(base["stat"].iloc[0])
    else:
        stat = retention_statistic(samples, C.RETENTION_STAT)
        median = float(np.median(samples))

    pd.concat([probes, pd.DataFrame(new_probes)], ignore_index=True).to_csv(probes_path, index=False)
    n_prior = int((probes["phase"] == "confirm").sum())
    return {
        "importance_budget": float(budget),
        "stat_at_budget": float(stat),
        "median_at_budget": median,
        "meets_threshold": bool(stat >= float(row["r2_threshold"])),
        "n_confirm_rounds": n_prior + len(new_probes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=str, default=None,
                        help="comma-separated miRNAs (default: those whose confirmation failed)")
    parser.add_argument("--confirm-n-masks", type=int, default=C.CONFIRM_N_MASKS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = cm.load_target_config()
    eligible = list(config["eligible_mirs"])
    cal = _load_calibration()
    todo = _select_targets(cal, eligible, args)
    if args.dry_run:
        print("\n".join(todo))
        print(f"{len(todo)} targets")
        return
    _journal(f"reconfirm: {len(todo)} targets, confirm masks={args.confirm_n_masks}")

    target_info = config["targets"]
    sc_features = cm.load_sc_features()
    test_splits = cm.load_test_splits()
    try:
        test_splits["_k1_ref"] = cm.load_k1_reference()
    except FileNotFoundError as e:
        _journal(f"WARNING: {e}")
        test_splits["_k1_ref"] = None
    device = cm.resolve_device(args.device)

    idx = {m: i for i, m in enumerate(cal["mirna"])}
    for n, target in enumerate(todo, 1):
        t0 = time.time()
        row = cal.loc[idx[target]]
        try:
            upd = _reconfirm_one(target, row, target_info, sc_features, test_splits,
                                 args.confirm_n_masks, device)
        except Exception as exc:  # noqa: BLE001
            _journal(f"[FAILED] reconfirm {target}: {exc}\n{traceback.format_exc()}")
            continue
        for k, v in upd.items():
            cal.loc[idx[target], k] = v
        cal.to_csv(CALIBRATION_PATH, index=False)
        _journal(
            f"[reconfirm {n}/{len(todo)}] {target} ({time.time() - t0:.1f}s) "
            f"budget {row['budget_search']:.4f} -> {upd['importance_budget']:.4f} "
            f"({upd['n_confirm_rounds']} confirm rounds, meets={upd['meets_threshold']})"
        )
        cm.clear_model_cache()

    cal.to_csv(CALIBRATION_PATH, index=False)
    path = _write_config(eligible, target_info, C.RETENTION_RATIO)
    _journal(f"reconfirm done. Wrote {path}")


if __name__ == "__main__":
    main()
