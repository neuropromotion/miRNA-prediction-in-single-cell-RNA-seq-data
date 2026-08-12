#!/usr/bin/env python3
"""Final TEST metrics: bootstrap R²+MSE on eval half for SC@assigned_K and bulk."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_metrics import bootstrap_metrics, mse_safe, r2_safe, summarize_dist
from build_prediction_config import write_prediction_config
from config import (
    BULK_SUMMARY_PATH,
    FIGURES,
    JOURNAL,
    N_BOOTSTRAP,
    OVERALL_PATH,
    PER_TARGET_PATH,
    PREDICTION_CONFIG_PATH,
    RESULTS,
    SC_SUMMARY_PATH,
    SEED,
    TABLES,
)
from io_artifacts import (
    eligible_assignments,
    ensure_cached_pair,
    load_prediction_config,
    load_split,
)
from plots import plot_all

os.environ.setdefault(
    "FINAL_MODELS_ROOT",
    str(Path(__file__).resolve().parents[1] / "train" / "results"),
)


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


def filter_assignments(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if os.environ.get("FINAL_TARGETS"):
        wanted = {t.strip() for t in os.environ["FINAL_TARGETS"].split(",") if t.strip()}
        pairs = [(t, k) for t, k in pairs if t in wanted]
    shard = os.environ.get("FINAL_SHARD")
    if shard:
        idx_s, n_s = shard.split("/")
        idx, n = int(idx_s), int(n_s)
        pairs = [(t, k) for j, (t, k) in enumerate(pairs) if j % n == idx]
    return pairs


def _point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"r2_full": r2_safe(y, pred), "mse_full": mse_safe(y, pred)}


def eval_one(
    target: str,
    assigned_k: str,
    eval_sc_idx: np.ndarray,
    eval_bulk_idx: np.ndarray,
) -> dict:
    tseed = SEED + int(hashlib.md5(target.encode()).hexdigest()[:8], 16) % 1_000_000
    row: dict = {
        "target": target,
        "assigned_k": assigned_k,
        "n_boot": N_BOOTSTRAP,
        "status": "ok",
        "error": "",
    }

    # SC @ assigned K
    y_sc, p_sc = ensure_cached_pair(target, assigned_k)
    y_sc_e, p_sc_e = y_sc[eval_sc_idx], p_sc[eval_sc_idx]
    row["n_sc_eval"] = int(len(y_sc_e))
    row.update({f"sc_{k}": v for k, v in _point_metrics(y_sc_e, p_sc_e).items()})
    boot_sc = bootstrap_metrics(y_sc_e, p_sc_e, seed=tseed + 1)
    row.update(summarize_dist(boot_sc["r2"], "sc_r2"))
    row.update(summarize_dist(boot_sc["mse"], "sc_mse"))

    # Bulk
    y_b, p_b = ensure_cached_pair(target, "bulk")
    y_b_e, p_b_e = y_b[eval_bulk_idx], p_b[eval_bulk_idx]
    row["n_bulk_eval"] = int(len(y_b_e))
    row.update({f"bulk_{k}": v for k, v in _point_metrics(y_b_e, p_b_e).items()})
    boot_b = bootstrap_metrics(y_b_e, p_b_e, seed=tseed + 2)
    row.update(summarize_dist(boot_b["r2"], "bulk_r2"))
    row.update(summarize_dist(boot_b["mse"], "bulk_mse"))
    return row


def _aggregate_modality(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """One-row summary across targets for a modality (sc|bulk)."""
    rows = []
    for metric in ("r2", "mse"):
        med_col = f"{prefix}_{metric}_median"
        full_col = f"{prefix}_{metric}_full"
        for col, label in ((med_col, "boot_median"), (full_col, "point_full")):
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            rows.append(
                {
                    "modality": prefix,
                    "metric": metric,
                    "stat_source": label,
                    "n_targets": int(len(s)),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def finalize(df: pd.DataFrame, proto: dict | None = None) -> None:
    ok = df[df["status"] == "ok"].copy()
    ok.to_csv(PER_TARGET_PATH, index=False)
    sc_sum = _aggregate_modality(ok, "sc")
    bulk_sum = _aggregate_modality(ok, "bulk")
    sc_sum.to_csv(SC_SUMMARY_PATH, index=False)
    bulk_sum.to_csv(BULK_SUMMARY_PATH, index=False)
    overall = pd.concat([sc_sum, bulk_sum], ignore_index=True)
    overall.to_csv(OVERALL_PATH, index=False)
    plot_all(ok)

    if proto is None:
        proto = load_prediction_config()
    cfg_path = write_prediction_config(proto, ok, PREDICTION_CONFIG_PATH)
    _log(f"wrote prediction_config → {cfg_path}")
    _log(f"wrote tables → {TABLES}")
    _log(f"wrote figures → {FIGURES}")
    print(overall.to_string(index=False))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    suffix = _shard_suffix()
    out_path = (
        PER_TARGET_PATH
        if not suffix
        else PER_TARGET_PATH.with_name(PER_TARGET_PATH.stem + suffix + PER_TARGET_PATH.suffix)
    )

    _log("=== TEST metrics (eval half, bootstrap R²+MSE) ===")
    cfg = load_prediction_config()  # Optimal_K proto
    split = load_split()
    pairs = filter_assignments(eligible_assignments(cfg))
    _log(
        f"eligible={cfg['n_eligible']} this_run={len(pairs)} "
        f"B={N_BOOTSTRAP} out={out_path.name}"
    )
    eval_sc = np.asarray(split["sc"]["eval_idx"], dtype=np.int64)
    eval_bulk = np.asarray(split["bulk"]["eval_idx"], dtype=np.int64)
    _log(f"eval n_sc={len(eval_sc)} n_bulk={len(eval_bulk)} split={split.get('seed')}")

    rows: list[dict] = []
    for i, (target, k) in enumerate(pairs, 1):
        _log(f"({i}/{len(pairs)}) {target} @ {k}")
        t0 = time.perf_counter()
        try:
            row = eval_one(target, k, eval_sc, eval_bulk)
            row["sec"] = round(time.perf_counter() - t0, 2)
            _log(
                f"  sc_r2_med={row['sc_r2_median']:.3f} bulk_r2_med={row['bulk_r2_median']:.3f} "
                f"sc_mse_med={row['sc_mse_median']:.3f} bulk_mse_med={row['bulk_mse_median']:.3f} "
                f"({row['sec']}s)"
            )
        except Exception as exc:
            row = {
                "target": target,
                "assigned_k": k,
                "status": "fail",
                "error": str(exc),
                "sec": round(time.perf_counter() - t0, 2),
            }
            _log(f"  FAIL: {exc}")
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    _log(f"wrote {out_path} n={len(df)}")

    if not suffix:
        finalize(df, proto=cfg)
    else:
        _log("shard done — run merge_shards.py after all shards")


if __name__ == "__main__":
    main()
