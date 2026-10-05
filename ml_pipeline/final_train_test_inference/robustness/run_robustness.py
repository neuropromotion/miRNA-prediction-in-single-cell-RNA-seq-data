#!/usr/bin/env python3
"""Robustness v3: bisect max global SC dropout on Optimal_K eval half."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common as cm  # noqa: E402
import robustness_paths as C  # noqa: E402
from bisect_tolerance import BisectResult, estimate_max_absent_bisect, probes_to_dataframe  # noqa: E402
from config_builder import build_robustness_config, load_bootstrap_references  # noqa: E402

PROGRESS_PATH = C.TABLES_DIR / "_progress.csv"


def _journal(line: str) -> None:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with C.JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def _filter_targets(all_targets: list[str]) -> list[str]:
    targets = list(all_targets)
    manual = os.environ.get("FINAL_TARGETS")
    if manual:
        wanted = {t.strip() for t in manual.split(",") if t.strip()}
        targets = [t for t in targets if t in wanted]
    return targets


def _load_progress() -> dict[str, str]:
    if not PROGRESS_PATH.exists():
        return {}
    df = pd.read_csv(PROGRESS_PATH, on_bad_lines="skip")
    return dict(zip(df["target"].astype(str), df["status"].astype(str)))


def _append_progress(target: str, status: str, message: str = "") -> None:
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"target": target, "status": status, "message": message}])
    header = not PROGRESS_PATH.exists()
    row.to_csv(PROGRESS_PATH, mode="a", header=header, index=False)


def _r2_reference(target: str, cohort: str, target_info: dict, refs: pd.DataFrame) -> float:
    if target in refs.index:
        ref_row = refs.loc[target]
        if str(ref_row.get("assigned_k", cohort)) == cohort:
            r2 = float(ref_row["sc_r2_median"])
            if r2 == r2:
                return r2
    tsc = target_info.get(target, {}).get("test_sc_r2")
    if tsc is not None:
        return float(tsc)
    raise ValueError(f"no R² reference for {target}")


def _process_one(
    target: str,
    target_info: dict,
    sc_features: dict,
    test_splits: dict,
    refs: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    cohort = str(target_info[target]["assigned_cohort"])
    try:
        r2_ref = _r2_reference(target, cohort, target_info, refs)
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}

    split = test_splits.get(cohort)
    if split is None:
        return {"status": "failed", "error": f"no test split for {cohort}"}

    bundle = cm.load_target_bundle(target, target_info, device="cpu")
    sc_genes = cm.sc_genes_for_target(target, bundle.genes, sc_features)
    if not sc_genes:
        return {"status": "failed", "error": "empty SC panel"}

    k1_ref = test_splits.get("_k1_ref") if cohort == "K1" else None
    seed = C.SEED + (hash(target) % 10_000)

    result: BisectResult = estimate_max_absent_bisect(
        bundle,
        split.x,
        split.y,
        sc_genes,
        r2_reference=r2_ref,
        retention_ratio=args.retention_ratio,
        retention_stat=args.retention_stat,
        p_max=args.p_max,
        bisect_eps=args.bisect_eps,
        search_n_masks=args.search_n_masks,
        confirm_n_masks=args.confirm_n_masks,
        k1_ref=k1_ref,
        seed=seed,
    )
    cm.clear_model_cache()

    summary_row = {
        "target": target,
        "cohort": cohort,
        "n_sc_features": result.n_sc_features,
        "r2_reference": result.r2_reference,
        "r2_threshold": result.r2_threshold,
        "retention_stat": result.retention_stat,
        "max_absent_frac": result.max_absent_frac,
        "min_present_frac": result.min_present_frac,
        "stat_at_limit": result.stat_at_limit,
        "median_at_limit": result.median_at_limit,
        "q25_at_limit": result.q25_at_limit,
        "median_r2_baseline": result.median_baseline,
        "meets_threshold": result.meets_threshold,
    }
    probes_df = probes_to_dataframe(target, cohort, result.bisect_probes) if args.save_probes else None
    return {"status": "done", "summary_row": summary_row, "probes": probes_df}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-ratio", type=float, default=C.RETENTION_RATIO)
    parser.add_argument(
        "--retention-stat",
        choices=("median", "q25"),
        default=os.environ.get("ROBUSTNESS_RETENTION_STAT", C.RETENTION_STAT),
        help="aggregate R² over masks compared to threshold (default: median)",
    )
    parser.add_argument("--search-n-masks", type=int, default=C.SEARCH_N_MASKS)
    parser.add_argument("--confirm-n-masks", type=int, default=C.CONFIRM_N_MASKS)
    parser.add_argument("--bisect-eps", type=float, default=C.BISECT_EPS)
    parser.add_argument("--p-max", type=float, default=C.P_MAX)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--save-probes", action="store_true", help="append bisect_probes.csv")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    config = cm.load_target_config()
    eligible = list(config["eligible_mirs"])
    target_info = config["targets"]

    if args.figures_only:
        path = C.TABLES_DIR / "tolerance_summary.csv"
        summary = pd.read_csv(path) if path.exists() else pd.DataFrame()
        cfg = build_robustness_config(
            eligible,
            target_info,
            summary,
            retention_ratio=args.retention_ratio,
            retention_stat=args.retention_stat,
            search_n_masks=args.search_n_masks,
            confirm_n_masks=args.confirm_n_masks,
            bisect_eps=args.bisect_eps,
        )
        C.ROBUSTNESS_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {C.ROBUSTNESS_CONFIG_PATH}")
        return

    refs = load_bootstrap_references()
    sc_features = cm.load_sc_features()
    progress = _load_progress()
    todo = [t for t in _filter_targets(eligible) if progress.get(t) != "done"]
    todo.sort(key=lambda t: (str(target_info[t].get("assigned_cohort", "")) == "K1", t))
    if args.max_targets:
        todo = todo[: args.max_targets]

    print("Loading eval-half sc_TEST...", flush=True)
    _journal(f"run start n_todo={len(todo)} stat={args.retention_stat}")
    test_splits = cm.load_test_splits()
    try:
        test_splits["_k1_ref"] = cm.load_k1_reference()
    except FileNotFoundError:
        test_splits["_k1_ref"] = None

    n_ok = n_fail = 0
    for i, target in enumerate(todo, 1):
        t0 = time.time()
        _journal(f"[start] ({i}/{len(todo)}) {target}")
        try:
            out = _process_one(target, target_info, sc_features, test_splits, refs, args)
        except Exception as exc:  # noqa: BLE001
            out = {"status": "failed", "error": f"{exc}\n{traceback.format_exc()}"}

        if out["status"] != "done":
            _append_progress(target, "failed", out.get("error", ""))
            _journal(f"[FAILED] {target}: {str(out.get('error', ''))[:200]}")
            n_fail += 1
            continue

        row = out["summary_row"]
        summary_path = C.TABLES_DIR / "tolerance_summary.csv"
        pd.DataFrame([row]).to_csv(
            summary_path,
            mode="a",
            header=not summary_path.exists(),
            index=False,
        )
        if args.save_probes and out.get("probes") is not None:
            ppath = C.TABLES_DIR / "bisect_probes.csv"
            out["probes"].to_csv(ppath, mode="a", header=not ppath.exists(), index=False)

        _append_progress(target, "done")
        elapsed = time.time() - t0
        msg = (
            f"[done] {target} ({elapsed:.1f}s) max_absent={row['max_absent_frac']:.2f} "
            f"{row['retention_stat']}@limit={row['stat_at_limit']:.3f} τ={row['r2_threshold']:.3f}"
        )
        print(msg, flush=True)
        _journal(msg)
        n_ok += 1

    summary = pd.read_csv(C.TABLES_DIR / "tolerance_summary.csv") if (C.TABLES_DIR / "tolerance_summary.csv").exists() else pd.DataFrame()
    cfg = build_robustness_config(
        eligible,
        target_info,
        summary,
        retention_ratio=args.retention_ratio,
        retention_stat=args.retention_stat,
        search_n_masks=args.search_n_masks,
        confirm_n_masks=args.confirm_n_masks,
        bisect_eps=args.bisect_eps,
    )
    C.ROBUSTNESS_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Finished {n_ok} ok, {n_fail} failed. Wrote {C.ROBUSTNESS_CONFIG_PATH}")


if __name__ == "__main__":
    main()
