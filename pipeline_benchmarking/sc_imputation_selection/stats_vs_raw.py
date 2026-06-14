"""Paired tests: each imputation method vs raw (same 50 miRNA)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from constants import RESULTS, STAGE

BASELINE = "raw"
METRICS = ("k1_val_r2", "bulk_val_r2", "val_mixed_r2")
ALPHA = 0.05


def paired_tests(diff: np.ndarray) -> dict:
    diff = diff[np.isfinite(diff)]
    n = int(len(diff))
    if n < 3:
        return {"n_pairs": n, "p_paired_t": np.nan, "p_wilcoxon": np.nan}
    t_stat, p_t = stats.ttest_rel(np.zeros(n), diff)  # test mean(diff) == 0
    # ttest_rel(a, b) tests a-b; we pass 0 vs diff equivalent to one-sample on diff
    t_stat, p_t = stats.ttest_1samp(diff, popmean=0.0)
    try:
        w_stat, p_w = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        p_w = np.nan
    return {
        "n_pairs": n,
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
        "std_delta": float(diff.std(ddof=1)) if n > 1 else np.nan,
        "n_method_better": int((diff > 0).sum()),
        "n_raw_better": int((diff < 0).sum()),
        "p_paired_t": float(p_t),
        "p_wilcoxon": float(p_w),
    }


def main() -> None:
    df = pd.read_csv(RESULTS / "val_metrics_all.csv")
    methods = sorted(m for m in df["method"].unique() if m != BASELINE)

    rows = []
    for method in methods:
        for metric in METRICS:
            wide = df.pivot(index="target", columns="method", values=metric)
            if BASELINE not in wide.columns or method not in wide.columns:
                continue
            paired = wide[[BASELINE, method]].dropna()
            diff = (paired[method] - paired[BASELINE]).to_numpy()
            res = paired_tests(diff)
            rows.append(
                {
                    "baseline": BASELINE,
                    "method": method,
                    "metric": metric,
                    **res,
                }
            )

    out = pd.DataFrame(rows)
    n_comp = len(out)
    out["p_wilcoxon_bonferroni"] = np.minimum(out["p_wilcoxon"] * n_comp, 1.0)
    out["p_paired_t_bonferroni"] = np.minimum(out["p_paired_t"] * n_comp, 1.0)
    out["sig_wilcox_0.05"] = out["p_wilcoxon"] < ALPHA
    out["sig_wilcox_bonf_0.05"] = out["p_wilcoxon_bonferroni"] < ALPHA

    out_path = RESULTS / "stats_vs_raw.csv"
    out.to_csv(out_path, index=False)

    summary = {
        "baseline": BASELINE,
        "n_targets": int(df[df["method"] == BASELINE]["target"].nunique()),
        "alpha": ALPHA,
        "n_comparisons": n_comp,
        "note": "Paired per-target deltas (method - raw). Wilcoxon preferred for R2.",
    }
    with (RESULTS / "stats_vs_raw_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
