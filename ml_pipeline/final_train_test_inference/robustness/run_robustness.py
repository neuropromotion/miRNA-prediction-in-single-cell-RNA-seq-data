#!/usr/bin/env python3
"""Robustness v4: LOO SC feature weights + importance budget on eval TEST.

Writes ``results/robustness_config.json`` for ``miRPredictor`` (v4 gate).
"""

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
from joblib import Parallel, delayed

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
from budget_calibrate import calibrate_importance_budget  # noqa: E402
from config_builder import build_robustness_config, load_bootstrap_references  # noqa: E402
from loo_importance import compute_loo_importance  # noqa: E402

PROGRESS_PATH = C.TABLES_DIR / "_progress.csv"


def _journal(msg: str) -> None:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with C.JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _safe_name(mir: str) -> str:
    return mir.replace("/", "_")


def _filter_targets(all_targets: list[str]) -> list[str]:
    targets = list(all_targets)
    manual = os.environ.get("FINAL_TARGETS")
    if manual:
        wanted = {t.strip() for t in manual.split(",") if t.strip()}
        targets = [t for t in targets if t in wanted]
    shard = os.environ.get("FINAL_SHARD")
    if shard:
        idx_s, n_s = shard.split("/")
        idx, n = int(idx_s), int(n_s)
        targets = [t for i, t in enumerate(targets) if i % n == idx]
    return targets


def _load_progress() -> dict[str, str]:
    if PROGRESS_PATH.exists():
        df = pd.read_csv(PROGRESS_PATH)
        return dict(zip(df["target"], df["status"]))
    return {}


def _append_progress(target: str, status: str, message: str = "") -> None:
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"target": target, "status": status, "message": message}])
    header = not PROGRESS_PATH.exists()
    row.to_csv(PROGRESS_PATH, mode="a", header=header, index=False)


def _append_table(name: str, df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        return
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.TABLES_DIR / f"{name}.csv"
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)


def _read_table(name: str) -> pd.DataFrame:
    path = C.TABLES_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
    raise ValueError(f"no sc_r2_median reference for {target}")


def _parse_devices(gpus: str | None) -> list[str]:
    if not gpus:
        return cm.available_devices()
    items = [x.strip() for x in gpus.split(",") if x.strip()]
    if not items:
        return cm.available_devices()
    return [d if (d.startswith("cuda") or d == "cpu") else f"cuda:{d}" for d in items]


def _process_one_target(
    target: str,
    target_info: dict,
    sc_features: dict,
    test_splits: dict,
    refs: pd.DataFrame,
    args: argparse.Namespace,
    device: str | None = None,
) -> dict:
    if target not in target_info:
        return {"status": "failed", "error": "not in prediction_config"}

    cohort = str(target_info[target]["assigned_cohort"])
    split = test_splits.get(cohort)
    if split is None:
        return {"status": "failed", "error": f"no test split for cohort {cohort}"}

    try:
        r2_ref = _r2_reference(target, cohort, target_info, refs)
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}

    bundle = cm.load_target_bundle(target, target_info, device=device)
    sc_genes = cm.sc_genes_for_target(target, bundle.genes, sc_features)
    if not sc_genes:
        return {"status": "failed", "error": "empty SC panel"}

    y = split.y[target]
    k1_ref = test_splits.get("_k1_ref") if cohort == "K1" else None
    seed = C.SEED + (hash(target) % 10_000)

    def _loo_progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            _journal(f"  LOO {target}: {done}/{total} genes")

    r2_loo_base, imp_df = compute_loo_importance(
        bundle,
        split.x,
        y,
        sc_genes,
        common_mod=cm,
        k1_ref=k1_ref,
        on_progress=_loo_progress,
    )
    imp_df.insert(0, "mirna", target)
    imp_df.insert(1, "cohort", cohort)

    weights = dict(zip(imp_df["gene"].astype(str), imp_df["weight"].astype(float)))
    C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    imp_path = C.WEIGHTS_DIR / f"{_safe_name(target)}.csv"
    imp_df.to_csv(imp_path, index=False)

    cal = calibrate_importance_budget(
        bundle,
        split.x,
        y,
        sc_genes,
        weights,
        r2_reference=r2_ref,
        retention_ratio=args.retention_ratio,
        retention_stat=C.RETENTION_STAT,
        search_n_masks=args.search_n_masks,
        confirm_n_masks=args.confirm_n_masks,
        budget_eps=C.BUDGET_EPS,
        b_max=C.B_MAX,
        common_mod=cm,
        k1_ref=k1_ref,
        seed=seed,
    )

    probes_path = C.TABLES_DIR / f"budget_probes_{_safe_name(target)}.csv"
    pd.DataFrame(cal.probes).to_csv(probes_path, index=False)

    summary_row = {
        "mirna": target,
        "cohort": cohort,
        "status": "ok",
        "n_sc_features": cal.n_sc_features,
        "r2_reference": cal.r2_reference,
        "r2_threshold": cal.r2_threshold,
        "r2_baseline_loo": r2_loo_base,
        "r2_baseline_masks": cal.r2_baseline_loo,
        "importance_budget": cal.importance_budget,
        "stat_at_budget": cal.stat_at_budget,
        "median_at_budget": cal.median_at_budget,
        "meets_threshold": cal.meets_threshold,
        "weights_path": str(imp_path),
    }
    return {
        "status": "done",
        "importance": imp_df,
        "summary_row": summary_row,
    }


def _write_config(eligible: list[str], target_info: dict, retention_ratio: float) -> Path:
    cal = _read_table("calibration_summary")
    inf = _read_table("inference_datasets")
    cfg = build_robustness_config(
        eligible,
        target_info,
        cal,
        retention_ratio=retention_ratio,
        inference_datasets=inf if len(inf) else None,
    )
    C.ROBUSTNESS_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return C.ROBUSTNESS_CONFIG_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-ratio", type=float, default=C.RETENTION_RATIO)
    parser.add_argument("--search-n-masks", type=int, default=C.SEARCH_N_MASKS)
    parser.add_argument("--confirm-n-masks", type=int, default=C.CONFIRM_N_MASKS)
    parser.add_argument("--bootstrap-summary", type=Path, default=C.BOOTSTRAP_SUMMARY_PATH)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="rebuild robustness_config.json from tables/ and exit",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        metavar="N",
        help="process at most N pending targets then exit (resume loops)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

    if args.n_jobs is None:
        env_jobs = os.environ.get("FINAL_JOBS")
        args.n_jobs = int(env_jobs) if env_jobs else None
    if args.gpus is None:
        args.gpus = os.environ.get("FINAL_GPUS")

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    config = cm.load_target_config()
    target_info = config["targets"]
    eligible = list(config["eligible_mirs"])

    if args.config_only:
        path = _write_config(eligible, target_info, args.retention_ratio)
        print(f"Wrote {path}")
        return

    refs = load_bootstrap_references(args.bootstrap_summary)
    sc_features = cm.load_sc_features()
    targets = _filter_targets(eligible)
    progress = _load_progress()
    todo = [t for t in targets if progress.get(t) != "done"]
    todo.sort(key=lambda t: (str(target_info[t].get("assigned_cohort", "")) == "K1", t))
    if args.max_targets is not None and args.max_targets > 0:
        todo = todo[: args.max_targets]

    _journal("Loading sc_TEST (eval half) + K1 reference...")
    test_splits = cm.load_test_splits()
    try:
        test_splits["_k1_ref"] = cm.load_k1_reference()
    except FileNotFoundError as e:
        _journal(f"WARNING: {e}")
        test_splits["_k1_ref"] = None

    devices = _parse_devices(args.gpus)
    if args.n_jobs is None:
        args.n_jobs = len(devices) if any(d != "cpu" for d in devices) else 1
    device_for = {t: devices[i % len(devices)] for i, t in enumerate(todo)}

    header = (
        f"v4 start: {len(targets)} targets, {len(todo)} remaining; "
        f"retention={args.retention_ratio:.0%}×sc_r2_median; "
        f"masks search={args.search_n_masks} confirm={args.confirm_n_masks}; "
        f"n_jobs={args.n_jobs}"
    )
    _journal(header)

    def _run(target: str) -> tuple[str, dict]:
        t0 = time.time()
        _journal(f"[start] {target}")
        try:
            result = _process_one_target(
                target,
                target_info,
                sc_features,
                test_splits,
                refs,
                args,
                device=device_for.get(target),
            )
            result["_elapsed"] = time.time() - t0
            return target, result
        except Exception as exc:  # noqa: BLE001
            return target, {"status": "failed", "error": f"{exc}\n{traceback.format_exc()}"}

    n_done = n_failed = 0

    def _consume(target: str, result: dict) -> None:
        nonlocal n_done, n_failed
        if result["status"] != "done":
            _append_progress(target, "failed", result.get("error", ""))
            fail_row = {"mirna": target, "status": "failed", "error": result.get("error", "")}
            _append_table("calibration_summary", pd.DataFrame([fail_row]))
            msg = f"[FAILED] {target}: {str(result.get('error', ''))[:300]}"
            _journal(msg)
            n_failed += 1
            return
        _append_table("feature_importance", result["importance"])
        _append_table("calibration_summary", pd.DataFrame([result["summary_row"]]))
        _append_progress(target, "done")
        row = result["summary_row"]
        msg = (
            f"[done] {target} ({result['_elapsed']:.1f}s) "
            f"budget={row['importance_budget']:.3f} "
            f"meets={row['meets_threshold']} (n_sc={row['n_sc_features']})"
        )
        _journal(msg)
        cm.clear_model_cache()
        n_done += 1

    if args.n_jobs == 1:
        for t in todo:
            _consume(*_run(t))
    else:
        try:
            results_iter = Parallel(
                n_jobs=args.n_jobs, backend="loky", verbose=10, return_as="generator"
            )(delayed(_run)(t) for t in todo)
        except TypeError:
            results_iter = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
                delayed(_run)(t) for t in todo
            )
        for target, result in results_iter:
            _consume(target, result)

    path = _write_config(eligible, target_info, args.retention_ratio)
    _journal(f"Done. {n_done} ok, {n_failed} failed. Wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _journal(f"[FATAL] {exc}\n{traceback.format_exc()}")
        raise
