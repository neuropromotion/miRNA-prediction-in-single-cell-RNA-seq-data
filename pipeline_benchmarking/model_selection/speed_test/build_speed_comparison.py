#!/usr/bin/env python3
"""Merge all stage03 speed benchmarks into comparison CSV tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

STAGE = Path(__file__).resolve().parent
RESULTS = STAGE / "results"
OUT = RESULTS / "speed_comparison_all.csv"
OUT_DETAIL = RESULTS / "speed_comparison_per_target.csv"
OUT_SUMMARY = RESULTS / "speed_comparison_summary.csv"

N_INFER_BULK = 4537
N_INFER_K1 = 370
N_INFER_PB = 263
N_INFER_TOTAL = N_INFER_BULK + N_INFER_K1 + N_INFER_PB


def _infer_per_1k(infer_sec: float, n_infer: int) -> float | None:
    if pd.isna(infer_sec) or not n_infer:
        return None
    return round(float(infer_sec) / n_infer * 1000, 4)


def load_main_benchmark() -> pd.DataFrame:
    path = RESULTS / "speed_benchmark" / "speed_results.csv"
    df = pd.read_csv(path)
    df["benchmark"] = "stage03_speed"
    df["model_label"] = df["model"]
    df["n_fit"] = df["n_train"]
    df["n_infer_bulk"] = N_INFER_BULK
    df["n_infer_k1"] = N_INFER_K1
    df["n_infer_pb"] = N_INFER_PB
    df["n_infer_total"] = df["n_infer"].fillna(N_INFER_TOTAL).astype(int)
    df["infer_sec_per_1k_samples"] = df.apply(
        lambda r: _infer_per_1k(r.get("infer_sec"), int(r["n_infer_total"])),
        axis=1,
    )
    df["notes"] = ""
    return df


def load_tabpfn3() -> pd.DataFrame:
    path = RESULTS / "tabpfn3_speed" / "speed_results.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"n_train_pool": "n_train"})
    df["benchmark"] = "tabpfn3_speed"
    df["model"] = "tabpfn_v3"
    df["model_label"] = "TabPFN-3 (tabpfn 8.0.8, full train)"
    df["n_infer_bulk"] = N_INFER_BULK
    df["n_infer_k1"] = N_INFER_K1
    df["n_infer_pb"] = N_INFER_PB
    df["n_infer_total"] = df["n_infer"].astype(int)
    df["infer_sec_per_1k_samples"] = df.apply(
        lambda r: _infer_per_1k(r["infer_sec"], int(r["n_infer_total"])),
        axis=1,
    )
    df["notes"] = "full fit=24451; HF model download"
    return df


def load_tabpfn25_single() -> pd.DataFrame:
    path = RESULTS / "tabpfn_speed" / "tabpfn_speed.json"
    if not path.exists():
        return pd.DataFrame()
    meta = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "benchmark": "tabpfn25_speed",
        "model": "tabpfn_v2.5",
        "model_label": "TabPFN-2.5 (tabpfn 6.4.1, subsample 1024)",
        "target": meta["target"],
        "status": meta["status"],
        "error": "",
        "n_train": meta["n_train_pool"],
        "n_fit": meta["n_fit"],
        "n_infer_total": meta["n_infer"],
        "n_infer_bulk": meta["pred_shapes"]["bulk"],
        "n_infer_k1": meta["pred_shapes"]["k1"],
        "n_infer_pb": meta["pred_shapes"]["pb"],
        "train_sec": meta["train_sec"],
        "infer_sec": meta["infer_sec"],
        "infer_sec_per_1k_samples": _infer_per_1k(meta["infer_sec"], meta["n_infer"]),
        "notes": f"single target; max_train={meta['max_train']}",
    }
    return pd.DataFrame([row])


DETAIL_COLS = [
    "benchmark",
    "model",
    "model_label",
    "target",
    "status",
    "n_train",
    "n_fit",
    "n_infer_total",
    "n_infer_bulk",
    "n_infer_k1",
    "n_infer_pb",
    "train_sec",
    "infer_sec",
    "infer_sec_per_1k_samples",
    "notes",
    "error",
]


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (benchmark, model, model_label), grp in detail.groupby(
        ["benchmark", "model", "model_label"], sort=False
    ):
        ok = grp[grp["status"] == "ok"]
        row = {
            "benchmark": benchmark,
            "model": model,
            "model_label": model_label,
            "n_targets_ok": int(len(ok)),
            "n_targets_fail": int(len(grp) - len(ok)),
            "n_train_pool": int(ok["n_train"].median()) if len(ok) else None,
            "n_fit": int(ok["n_fit"].median()) if len(ok) else None,
            "n_infer_total": int(ok["n_infer_total"].median()) if len(ok) else N_INFER_TOTAL,
            "n_infer_bulk": N_INFER_BULK,
            "n_infer_k1": N_INFER_K1,
            "n_infer_pb": N_INFER_PB,
            "mean_train_sec": round(float(ok["train_sec"].mean()), 3) if len(ok) else None,
            "median_train_sec": round(float(ok["train_sec"].median()), 3) if len(ok) else None,
            "mean_infer_sec": round(float(ok["infer_sec"].mean()), 3) if len(ok) else None,
            "median_infer_sec": round(float(ok["infer_sec"].median()), 3) if len(ok) else None,
            "mean_infer_sec_per_1k_samples": round(
                float(ok["infer_sec_per_1k_samples"].mean()), 4
            )
            if len(ok)
            else None,
            "total_train_50mirna_h": round(float(ok["train_sec"].mean()) * 50 / 3600, 2)
            if len(ok)
            else None,
            "total_infer_50mirna_h": round(float(ok["infer_sec"].mean()) * 50 / 3600, 2)
            if len(ok)
            else None,
            "total_train_327mirna_h": round(float(ok["train_sec"].mean()) * 327 / 3600, 2)
            if len(ok)
            else None,
            "total_infer_327mirna_h": round(float(ok["infer_sec"].mean()) * 327 / 3600, 2)
            if len(ok)
            else None,
            "notes": ok["notes"].iloc[0] if len(ok) and ok["notes"].iloc[0] else "",
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["mean_train_sec", "mean_infer_sec"],
        na_position="last",
    ).reset_index(drop=True)
    return summary


def main() -> None:
    parts = [load_main_benchmark(), load_tabpfn3(), load_tabpfn25_single()]
    detail = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for col in DETAIL_COLS:
        if col not in detail.columns:
            detail[col] = ""
    detail = detail[DETAIL_COLS]
    detail.to_csv(OUT_DETAIL, index=False)

    summary = build_summary(detail)
    summary.to_csv(OUT_SUMMARY, index=False)
    # Main "all models" table = summary sorted for comparison
    summary.to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_DETAIL}")
    print(f"Wrote {OUT_SUMMARY}")
    print("\nSummary preview:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
