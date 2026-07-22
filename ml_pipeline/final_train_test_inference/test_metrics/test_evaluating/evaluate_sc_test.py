#!/usr/bin/env python3
"""Evaluate final CatBoost+TabM+ResNet stack on held-out sc_TEST (K1–K10).

Inputs (not in git):
  test_evaluating/sc_TEST/X_TEST_K1.parquet, Y_TEST_K1.parquet
  test_evaluating/sc_TEST/X_TEST_PB_K{2,3,4,5,10}.parquet (+ Y_*)
  data/splits/sc_k1/X_train.parquet  (KNN imputation reference)
  ../../models/ensemble/catboost_tabm_resnet_stack/weights/
  ../../train/results/{catboost_optuna,tabm,resnet}/models/

Output:
  ../K1_K10_test_metrics.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
TEST_METRICS = EVAL_DIR.parent
FTTI = TEST_METRICS.parent
ML_PIPELINE = FTTI.parent

if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))
_TRAIN = str(FTTI / "train")
if _TRAIN in sys.path:
    sys.path.remove(_TRAIN)
sys.path.insert(0, _TRAIN)

from stack import FitResult, apply_fit, fit_from_dict  # noqa: E402
from data import select_features  # noqa: E402
from impute import impute_k1_query  # noqa: E402
from io_splits import load_features, load_targets  # noqa: E402
from metrics import r2  # noqa: E402
from transforms import log2p1  # noqa: E402
from model_trainers import load_artifact, predict_one  # noqa: E402
from shared.paths import SPLITS  # noqa: E402

ENSEMBLE_ID = "catboost_tabm_resnet_stack"
STACK_MODELS = ("catboost_optuna", "tabm", "resnet")
PB_COHORTS = ("K2", "K3", "K4", "K5", "K10")
K_SPLITS = ("K1", *PB_COHORTS)

SC_TEST = EVAL_DIR / "sc_TEST"
WEIGHTS_DIR = FTTI / "models" / "ensemble" / ENSEMBLE_ID / "weights"
FEATURE_RESULTS = (
    ML_PIPELINE
    / "pipeline_benchmarking"
    / "feature_selection"
    / "final_run"
    / "results"
)

OUT_TABLE = TEST_METRICS / "K1_K10_test_metrics.csv"
OUT_DIR = EVAL_DIR / "sc_test_run"
OUT_PROGRESS = OUT_DIR / "test_metrics_table.csv"


def _r2_col(k_name: str) -> str:
    return f"R2_{k_name}"


@dataclass
class TestSplit:
    name: str
    x: pd.DataFrame
    y: pd.DataFrame


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "journal.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _align_xy(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


def _load_test_splits() -> dict[str, TestSplit]:
    if not SC_TEST.is_dir():
        raise FileNotFoundError(
            f"sc_TEST not found under {SC_TEST}. "
            "Place X_TEST_*.parquet / Y_TEST_*.parquet there (gitignored)."
        )
    k1_ref_path = SPLITS / "sc_k1" / "X_train.parquet"
    if not k1_ref_path.is_file():
        raise FileNotFoundError(
            f"KNN reference missing: {k1_ref_path}. "
            "Run data/prepare_splits/prepare_stage00_splits.py or copy prepared splits."
        )
    k1_ref = pd.read_parquet(k1_ref_path)
    out: dict[str, TestSplit] = {}

    x_k1 = log2p1(pd.read_parquet(SC_TEST / "X_TEST_K1.parquet"))
    y_k1 = log2p1(pd.read_parquet(SC_TEST / "Y_TEST_K1.parquet"))
    x_k1, y_k1 = _align_xy(x_k1, y_k1)
    x_k1 = impute_k1_query(x_k1, k1_ref)
    out["K1"] = TestSplit("K1", x_k1, y_k1)

    for cohort in PB_COHORTS:
        x = log2p1(pd.read_parquet(SC_TEST / f"X_TEST_PB_{cohort}.parquet"))
        y = log2p1(pd.read_parquet(SC_TEST / f"Y_TEST_PB_{cohort}.parquet"))
        x, y = _align_xy(x, y)
        out[cohort] = TestSplit(cohort, x, y)

    return out


def _load_feature_counts() -> pd.DataFrame:
    sc = json.loads((FEATURE_RESULTS / "sc_features.json").read_text(encoding="utf-8"))
    bulk = json.loads((FEATURE_RESULTS / "bulk_trimmed_features.json").read_text(encoding="utf-8"))
    final = json.loads((FEATURE_RESULTS / "selected_features.json").read_text(encoding="utf-8"))
    rows = []
    for target in final:
        rows.append(
            {
                "target": target,
                "n_sc": len(sc.get(target, [])),
                "n_bulk": len(bulk.get(target, [])),
                "n_total": len(final[target]),
            }
        )
    return pd.DataFrame(rows)


def _load_fit(target: str) -> FitResult:
    path = WEIGHTS_DIR / f"{target}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing stack weights: {path}")
    return fit_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _filter_targets(targets: list[str]) -> list[str]:
    if os.environ.get("FINAL_TARGETS"):
        wanted = {t.strip() for t in os.environ["FINAL_TARGETS"].split(",") if t.strip()}
        return [t for t in targets if t in wanted]
    shard = os.environ.get("FINAL_SHARD")
    if not shard:
        return targets
    idx_s, n_s = shard.split("/")
    idx, n = int(idx_s), int(n_s)
    return [t for j, t in enumerate(targets) if j % n == idx]


def _predict_stack(
    target: str,
    genes: list[str],
    fit: FitResult,
    splits: dict[str, TestSplit],
) -> dict[str, float]:
    arts = {m: load_artifact(m, target) for m in STACK_MODELS}
    scores: dict[str, float] = {}
    for k_name, split in splits.items():
        if target not in split.y.columns:
            scores[_r2_col(k_name)] = np.nan
            continue
        x = select_features(split.x, genes).to_numpy(dtype=np.float32)
        y_true = split.y[target].to_numpy(dtype=np.float64)
        preds = [predict_one(m, arts[m], x) for m in STACK_MODELS]
        mat = np.column_stack(preds)
        y_pred = apply_fit(fit, mat, STACK_MODELS)
        scores[_r2_col(k_name)] = r2(y_true, y_pred)
    return scores


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("=== sc_TEST stack evaluation (K1–K10) ===")
    _log(f"sc_TEST={SC_TEST}")
    _log(f"weights={WEIGHTS_DIR}")

    features = load_features()
    targets = _filter_targets(load_targets())
    feat_counts = _load_feature_counts().set_index("target")
    splits = _load_test_splits()
    for k, sp in splits.items():
        _log(f"split {k}: n={len(sp.x)}")

    rows: list[dict] = []
    if OUT_PROGRESS.exists():
        rows = pd.read_csv(OUT_PROGRESS).to_dict("records")
    done = {r["target"] for r in rows if r.get("status") == "ok"}

    ok = fail = skip = 0
    for i, target in enumerate(targets, 1):
        if target in done:
            skip += 1
            continue
        _log(f"({i}/{len(targets)}) {target}")
        try:
            fit = _load_fit(target)
            genes = features[target]
            scores = _predict_stack(target, genes, fit, splits)
            fc = feat_counts.loc[target]
            row = {
                "target": target,
                "status": "ok",
                **scores,
                "n_sc": int(fc["n_sc"]),
                "n_bulk": int(fc["n_bulk"]),
                "n_total": int(fc["n_total"]),
                "fallback": fit.fallback_best_solo,
            }
            ok += 1
            k1 = scores.get("R2_K1", np.nan)
            _log(f"  R2_K1={k1:.4f}")
        except Exception as exc:
            fail += 1
            row = {"target": target, "status": "fail", "error": str(exc)}
            _log(f"  FAIL: {exc}")

        rows = [r for r in rows if r.get("target") != target]
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_PROGRESS, index=False)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    cols = ["target", *(_r2_col(k) for k in K_SPLITS), "n_sc", "n_bulk", "n_total"]
    # Merge into existing published table so FINAL_TARGETS / shard runs do not wipe it.
    by_target: dict[str, dict] = {}
    if OUT_TABLE.exists():
        for r in pd.read_csv(OUT_TABLE).to_dict("records"):
            by_target[str(r["target"])] = r
    for r in ok_rows:
        by_target[str(r["target"])] = {c: r.get(c) for c in cols}
    table = pd.DataFrame(by_target.values()).sort_values("target")
    table[cols].to_csv(OUT_TABLE, index=False)

    summary = {
        "n_targets": len(targets),
        "n_ok": ok,
        "n_fail": fail,
        "n_skip": skip,
        "output": str(OUT_TABLE),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"done ok={ok} fail={fail} skip={skip} -> {OUT_TABLE}")


if __name__ == "__main__":
    main()
