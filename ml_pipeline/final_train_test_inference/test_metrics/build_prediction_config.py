"""Build final inference prediction_config.json from Optimal_K proto + TEST eval metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import COHORTS, PREDICTION_CONFIG_PATH, PROTO_PREDICTION_CONFIG


# Meta keys kept only in Optimal_K proto — stripped from production config.
_PROTO_ONLY_KEYS = (
    "version",
    "assignment_rule",
    "thresholds",
    "split_path",
    "features_source",
)


def build_prediction_config(proto: dict, per_target: pd.DataFrame) -> dict:
    """
    Production config for inference:

      eligible_mirs, n_eligible, n_ineligible, cohorts, cohort_counts,
      K1…K10 → {mirna: {features, test_bulk, test_optimal_k}}

    ``test_bulk`` / ``test_optimal_k`` are bootstrap median R² on the eval half
    (bulk and SC@assigned_K).
    """
    ok = per_target[per_target.get("status", "ok") == "ok"].copy()
    if "target" not in ok.columns:
        raise ValueError("per_target table must have a 'target' column")
    metrics = ok.set_index("target")

    missing = [t for t in proto["eligible_mirs"] if t not in metrics.index]
    if missing:
        raise KeyError(
            f"{len(missing)} eligible targets lack TEST metrics "
            f"(re-run test_metrics). e.g. {missing[:5]}"
        )

    by_k: dict[str, dict] = {k: {} for k in proto.get("cohorts", list(COHORTS))}
    for cohort in by_k:
        for mir, info in (proto.get(cohort) or {}).items():
            if mir not in metrics.index:
                continue
            row = metrics.loc[mir]
            by_k[cohort][mir] = {
                "features": list(info.get("features") or info.get("genes") or []),
                "test_bulk": float(row["bulk_r2_median"]),
                "test_optimal_k": float(row["sc_r2_median"]),
            }

    out = {
        "eligible_mirs": list(proto["eligible_mirs"]),
        "n_eligible": int(proto.get("n_eligible", len(proto["eligible_mirs"]))),
        "n_ineligible": int(proto.get("n_ineligible", 0)),
        "cohorts": list(proto.get("cohorts", list(COHORTS))),
        "cohort_counts": {
            k: len(by_k[k]) for k in proto.get("cohorts", list(COHORTS))
        },
        **by_k,
    }
    return out


def write_prediction_config(proto: dict, per_target: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path or PREDICTION_CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_prediction_config(proto, per_target)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def strip_proto_meta(proto: dict) -> dict:
    """Utility: drop proto-only keys (kept for tests / ad-hoc use)."""
    return {k: v for k, v in proto.items() if k not in _PROTO_ONLY_KEYS}


def load_proto_path() -> Path:
    if PROTO_PREDICTION_CONFIG.is_file():
        return PROTO_PREDICTION_CONFIG
    legacy = PROTO_PREDICTION_CONFIG.with_name("prediction_config.json")
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        f"missing {PROTO_PREDICTION_CONFIG} (or legacy prediction_config.json)"
    )
