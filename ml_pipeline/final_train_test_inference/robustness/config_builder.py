"""Assemble ``robustness_config.json`` for inference gating (v4 importance budget)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import robustness_paths as C


def load_bootstrap_references(path: Path | None = None) -> pd.DataFrame:
    path = path or C.BOOTSTRAP_SUMMARY_PATH
    df = pd.read_csv(path)
    ok = df["status"] == "ok"
    df = df.loc[ok].copy()
    df["target"] = df["target"].astype(str)
    return df.set_index("target", drop=False)


def _safe_name(mir: str) -> str:
    return mir.replace("/", "_")


def _inference_by_mir(inference_df: pd.DataFrame | None) -> dict[str, dict[str, dict]]:
    if inference_df is None or inference_df.empty:
        return {}
    out: dict[str, dict[str, dict]] = {}
    for _, row in inference_df.iterrows():
        mir = str(row["mirna"])
        ds = str(row["dataset"])
        out.setdefault(mir, {})[ds] = {
            "status": str(row.get("status", "")),
            "passes_importance_budget": bool(row.get("passes_importance_budget", False)),
            "absent_importance": float(row["absent_importance"])
            if pd.notna(row.get("absent_importance"))
            else None,
            "budget_used_fraction": float(row["budget_used_fraction"])
            if pd.notna(row.get("budget_used_fraction"))
            else None,
            "n_absent": int(row["n_absent"]) if pd.notna(row.get("n_absent")) else None,
            "pct_absent": float(row["pct_absent"]) if pd.notna(row.get("pct_absent")) else None,
        }
    return out


def build_robustness_config(
    eligible_mirs: list[str],
    target_info: dict,
    calibration_summary: pd.DataFrame | None,
    *,
    retention_ratio: float = C.RETENTION_RATIO,
    retention_stat: str = C.RETENTION_STAT,
    inference_datasets: pd.DataFrame | None = None,
    weights_dir_rel: str = C.WEIGHTS_DIR_REL,
) -> dict:
    """Build v4 config consumed by ``miRPredictor`` (importance-budget gate)."""
    done: dict[str, dict] = {}
    if calibration_summary is not None and len(calibration_summary):
        inf_map = _inference_by_mir(inference_datasets)
        for _, row in calibration_summary.iterrows():
            mir = str(row.get("mirna") or row.get("target"))
            status = str(row.get("status", "failed"))
            if status != "ok":
                done[mir] = {
                    "status": "failed",
                    "cohort": str(row.get("cohort", target_info.get(mir, {}).get("assigned_cohort", ""))),
                    "error": str(row.get("error", "")),
                }
                continue
            wfile = f"{_safe_name(mir)}.csv"
            wpath = row.get("weights_path")
            if wpath and Path(str(wpath)).is_file():
                wfile = Path(str(wpath)).name
            entry = {
                "status": "done",
                "cohort": str(row.get("cohort", "")),
                "assigned_k": str(row.get("cohort", "")),
                "feature_scope": "sc_features",
                "n_sc_features": int(row["n_sc_features"]),
                "r2_reference": float(row["r2_reference"]),
                "r2_threshold": float(row["r2_threshold"]),
                "r2_baseline_loo": float(row.get("r2_baseline_loo", row.get("r2_baseline_masks"))),
                "importance_budget": float(row["importance_budget"]),
                "weights_file": wfile,
                "meets_threshold": bool(row.get("meets_threshold", True)),
                "median_at_budget": float(row["median_at_budget"])
                if pd.notna(row.get("median_at_budget"))
                else None,
                "stat_at_budget": float(row["stat_at_budget"])
                if pd.notna(row.get("stat_at_budget"))
                else None,
            }
            if mir in inf_map:
                entry["inference_datasets"] = inf_map[mir]
            done[mir] = entry

    targets_out: dict[str, dict] = {}
    for mir in eligible_mirs:
        if mir in done:
            targets_out[mir] = done[mir]
        else:
            info = target_info.get(mir) or {}
            targets_out[mir] = {
                "status": "pending",
                "cohort": str(info.get("assigned_cohort", "")),
                "assigned_k": "",
                "feature_scope": "sc_features",
                "n_sc_features": None,
                "r2_reference": None,
                "r2_threshold": None,
                "importance_budget": None,
                "weights_file": None,
                "meets_threshold": None,
            }

    n_done = sum(1 for v in targets_out.values() if v.get("status") == "done")
    return {
        "version": 4,
        "rule": "importance_budget",
        "retention_ratio": float(retention_ratio),
        "retention_stat": str(retention_stat),
        "eval_split": "optimal_k_eval_half",
        "reference_metric": "sc_r2_median",
        "bootstrap_summary_path": str(C.BOOTSTRAP_SUMMARY_PATH),
        "feature_scope": "sc_features",
        "weights_dir": weights_dir_rel,
        "n_eligible": len(eligible_mirs),
        "n_done": n_done,
        "n_pending": len(eligible_mirs) - n_done,
        "safety": (
            f"{retention_stat} R2(mask) >= {retention_ratio:.0%} * sc_r2_median; "
            "inference gate: sum(weight[g] for absent g) <= importance_budget"
        ),
        "targets": targets_out,
    }
