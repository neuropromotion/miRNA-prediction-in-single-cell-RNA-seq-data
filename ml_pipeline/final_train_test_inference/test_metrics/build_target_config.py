#!/usr/bin/env python3
"""Build inference/target_config.json from test metrics + prediction_config.

Inputs (this directory):
  prediction_config.json   — eligible miRNAs + assigned K cohorts
  K1_K10_test_metrics.csv  — per-target R² on sc_TEST (K1–K10)
  bulk_test_metrics.csv    — per-target R² on bulk_TEST

Also needs:
  selected_features.json - selected features for each miRNA

Output:
  ../inference/target_config.json  (for downstream inference by StackPredictor / SingleCell)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FTTI = ROOT.parent
ML_PIPELINE = FTTI.parent

PREDICTION_CONFIG = ROOT / "prediction_config.json"
SC_METRICS = ROOT / "K1_K10_test_metrics.csv"
BULK_METRICS = ROOT / "bulk_test_metrics.csv"
OUT = FTTI / "inference" / "target_config.json"

COHORTS = ("K1", "K2", "K3", "K4", "K5", "K10")
ELIGIBLE_KEYS = ("eligible_mirs", "eligible mirs", "eligable_mirs")


def _features_path() -> Path:
    candidates = (
        ML_PIPELINE / "data" / "frozen" / "selected_features.json",
        FTTI / "selected_features.json",
        ML_PIPELINE
        / "pipeline_benchmarking"
        / "feature_selection"
        / "final_run"
        / "results"
        / "selected_features.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(
        "selected_features.json not found. Expected under data/frozen/ "
        "or final_train_test_inference/."
    )


def _load_eligible(cfg: dict) -> list[str]:
    for key in ELIGIBLE_KEYS:
        if key in cfg:
            return list(cfg[key])
    raise SystemExit(
        f"prediction_config.json missing eligible list "
        f"(tried keys: {ELIGIBLE_KEYS})"
    )


def _load_bulk_r2() -> dict[str, float]:
    if not BULK_METRICS.is_file():
        raise SystemExit(f"Missing bulk test metrics: {BULK_METRICS}")
    out: dict[str, float] = {}
    with BULK_METRICS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") == "ok":
                out[row["target"]] = float(row["R2_bulk"])
    return out


def _load_sc_metrics() -> dict[str, dict]:
    if not SC_METRICS.is_file():
        raise SystemExit(f"Missing SC test metrics: {SC_METRICS}")
    out: dict[str, dict] = {}
    with SC_METRICS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["target"]
            out[t] = {
                "test_r2": {k: float(row[f"R2_{k}"]) for k in COHORTS},
                "n_sc": int(row["n_sc"]),
                "n_bulk": int(row["n_bulk"]),
                "n_total": int(row["n_total"]),
            }
    return out


def _cohort_for(target: str, cohort_map: dict[str, list[str]]) -> str:
    for k in COHORTS:
        if target in cohort_map[k]:
            return k
    raise KeyError(f"{target} not assigned to any cohort in prediction_config.json")


def main() -> None:
    if not PREDICTION_CONFIG.is_file():
        raise SystemExit(f"Missing prediction config: {PREDICTION_CONFIG}")

    cfg = json.loads(PREDICTION_CONFIG.read_text(encoding="utf-8"))
    features = json.loads(_features_path().read_text(encoding="utf-8"))
    eligible = _load_eligible(cfg)
    eligible_set = set(eligible)

    raw_cohorts = {k: list(cfg[k]) for k in COHORTS}
    # Keep cohort lists consistent with eligible (drop any extras left in JSON).
    cohort_map = {k: [t for t in raw_cohorts[k] if t in eligible_set] for k in COHORTS}
    dropped = {
        k: sorted(set(raw_cohorts[k]) - eligible_set) for k in COHORTS if set(raw_cohorts[k]) - eligible_set
    }
    if dropped:
        print(f"Note: dropped non-eligible entries from cohorts: {dropped}")

    bulk_r2 = _load_bulk_r2()
    metrics_by_target = _load_sc_metrics()

    missing_sc = [t for t in eligible if t not in metrics_by_target]
    if missing_sc:
        raise SystemExit(
            f"Missing SC metrics for {len(missing_sc)} eligible targets, e.g. {missing_sc[:3]}"
        )
    missing_bulk = [t for t in eligible if t not in bulk_r2]
    if missing_bulk:
        raise SystemExit(
            f"Missing bulk R² for {len(missing_bulk)} eligible targets, e.g. {missing_bulk[:3]}"
        )
    missing_feat = [t for t in eligible if t not in features]
    if missing_feat:
        raise SystemExit(
            f"Missing features for {len(missing_feat)} eligible targets, e.g. {missing_feat[:3]}"
        )

    unassigned = [t for t in eligible if all(t not in cohort_map[k] for k in COHORTS)]
    if unassigned:
        raise SystemExit(
            f"{len(unassigned)} eligible targets missing from cohort lists, e.g. {unassigned[:3]}"
        )

    targets: dict[str, dict] = {}
    for target in eligible:
        test_r2 = dict(metrics_by_target[target]["test_r2"])
        test_r2["bulk"] = bulk_r2[target]
        targets[target] = {
            "assigned_cohort": _cohort_for(target, cohort_map),
            "genes": features[target],
            "test_r2": test_r2,
            "n_sc": metrics_by_target[target]["n_sc"],
            "n_bulk": metrics_by_target[target]["n_bulk"],
            "n_total": metrics_by_target[target]["n_total"],
        }

    manifest = {
        "version": "final_train_v3", # third attempt))
        "assignment_rule": (
            "Cohort assignment from prediction_config.json: among eligible miRNAs "
            "(R² > 0.4 on any K or bulk), assign the smallest K within -7.5% of the "
            "maximum sc_TEST R² across K1–K10."
        ),
        "cohorts": cohort_map,
        "eligible_mirs": eligible,
        "n_eligible": len(eligible),
        "cohort_counts": {k: len(cohort_map[k]) for k in COHORTS},
        "targets": targets,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"eligible={len(eligible)} cohort_counts={manifest['cohort_counts']}")
    print(f"sum_cohorts={sum(manifest['cohort_counts'].values())}")


if __name__ == "__main__":
    main()
