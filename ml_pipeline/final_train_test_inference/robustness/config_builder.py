"""Build ``robustness_config.json`` (v3)."""

from __future__ import annotations

import pandas as pd

import robustness_paths as C


def load_bootstrap_references(path=None) -> pd.DataFrame:
    path = path or C.BOOTSTRAP_SUMMARY_PATH
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df.loc[df["status"] == "ok"].copy()
    df["target"] = df["target"].astype(str)
    return df.set_index("target", drop=False)


def build_robustness_config(
    eligible_mirs: list[str],
    target_info: dict,
    tolerance_summary: pd.DataFrame | None,
    *,
    retention_ratio: float = C.RETENTION_RATIO,
    retention_stat: str = C.RETENTION_STAT,
    search_n_masks: int = C.SEARCH_N_MASKS,
    confirm_n_masks: int = C.CONFIRM_N_MASKS,
    bisect_eps: float = C.BISECT_EPS,
) -> dict:
    done: dict[str, dict] = {}
    if tolerance_summary is not None and len(tolerance_summary):
        for _, row in tolerance_summary.iterrows():
            mir = str(row["target"])
            done[mir] = {
                "status": "done",
                "cohort": str(row.get("cohort", target_info.get(mir, {}).get("assigned_cohort", ""))),
                "feature_scope": "sc_features",
                "n_sc_features": int(row["n_sc_features"]),
                "r2_reference": float(row["r2_reference"]),
                "r2_threshold": float(row["r2_threshold"]),
                "retention_stat": str(row.get("retention_stat", retention_stat)),
                "max_absent_frac": float(row["max_absent_frac"]),
                "min_present_frac": float(row["min_present_frac"]),
                "stat_at_limit": float(row["stat_at_limit"]),
                "median_at_limit": float(row["median_at_limit"]),
                "q25_at_limit": float(row["q25_at_limit"]),
                "median_r2_baseline": float(row["median_r2_baseline"]),
                "meets_threshold": bool(row.get("meets_threshold", True)),
            }

    targets_out: dict[str, dict] = {}
    for mir in eligible_mirs:
        if mir in done:
            targets_out[mir] = done[mir]
        else:
            info = target_info.get(mir) or {}
            targets_out[mir] = {
                "status": "pending",
                "cohort": str(info.get("assigned_cohort", "")),
                "feature_scope": "sc_features",
                "n_sc_features": None,
                "r2_reference": None,
                "r2_threshold": None,
                "retention_stat": retention_stat,
                "max_absent_frac": None,
                "min_present_frac": None,
                "stat_at_limit": None,
                "median_at_limit": None,
                "q25_at_limit": None,
                "median_r2_baseline": None,
                "meets_threshold": None,
            }

    n_done = sum(1 for v in targets_out.values() if v["status"] == "done")
    stat_label = retention_stat.lower()
    return {
        "version": 3,
        "method": "bisect_global_dropout",
        "rule": f"{stat_label}_r2_retention",
        "retention_stat": retention_stat,
        "retention_ratio": float(retention_ratio),
        "search_n_masks": int(search_n_masks),
        "confirm_n_masks": int(confirm_n_masks),
        "bisect_eps": float(bisect_eps),
        "p_max": float(C.P_MAX),
        "eval_split": "optimal_k_eval_half",
        "reference_metric": "sc_r2_median_or_test_sc_r2",
        "feature_scope": "sc_features",
        "n_eligible": len(eligible_mirs),
        "n_done": n_done,
        "n_pending": len(eligible_mirs) - n_done,
        "safety": f"{stat_label}(R2_mask) >= {retention_ratio:.0%} * r2_reference",
        "targets": targets_out,
    }
