"""TabPack Muon (paper) trainer for final_train.

Wraps ``shared.tabpack_trainer.train_tabpack_screen`` with Stage00 train/val
layout used by final_train. Artifacts: ``preds.npz`` + ``meta.json`` +
``inference_bundle.pt`` (live ``predict(x)`` via shared.predict_tabpack).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .constants import (
        ML_PIPELINE,
        PB_COHORTS,
        TABPACK_EXPERIMENT_NAMESPACE,
        TABPACK_N_MODELS,
        TABPACK_PROTOCOL,
    )
    from .data import TrainBundle, select_features
    from .metrics import clip_nonneg, r2
except ImportError:
    from final_train_test_inference.train.constants import (
        ML_PIPELINE,
        PB_COHORTS,
        TABPACK_EXPERIMENT_NAMESPACE,
        TABPACK_N_MODELS,
        TABPACK_PROTOCOL,
    )
    from final_train_test_inference.train.data import TrainBundle, select_features
    from final_train_test_inference.train.metrics import clip_nonneg, r2

if str(ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE))

from shared.tabpack_trainer import load_tabpack_screen, train_tabpack_screen  # noqa: E402


def _arrays(bundle: TrainBundle, target: str, genes: list[str]) -> dict[str, np.ndarray]:
    return {
        "x_train": select_features(bundle.x_train, genes).to_numpy(dtype=np.float32),
        "y_train": bundle.y_train[target].to_numpy(dtype=np.float64),
        "x_val": select_features(bundle.x_val, genes).to_numpy(dtype=np.float32),
        "y_val": bundle.y_val[target].to_numpy(dtype=np.float64),
    }


def outer_parts(
    bundle: TrainBundle, target: str, genes: list[str]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Modality-wise Stage00 val → TabPack test cache (stack / reports)."""
    parts: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "bulk",
            select_features(bundle.x_val_bulk, genes).to_numpy(dtype=np.float32),
            bundle.y_val_bulk[target].to_numpy(dtype=np.float64),
        ),
        (
            "k1",
            select_features(bundle.x_val_k1, genes).to_numpy(dtype=np.float32),
            bundle.y_val_k1[target].to_numpy(dtype=np.float64),
        ),
    ]
    for cohort in PB_COHORTS:
        x_c = bundle.x_val_pb_by_cohort[cohort]
        y_c = bundle.y_val_pb_by_cohort[cohort]
        if len(x_c) == 0:
            continue
        parts.append(
            (
                f"pb_{cohort}",
                select_features(x_c, genes).to_numpy(dtype=np.float32),
                y_c[target].to_numpy(dtype=np.float64),
            )
        )
    return parts


def train_tabpack_model(
    bundle: TrainBundle,
    target: str,
    genes: list[str],
    model_dir: Path,
) -> dict[str, Any]:
    """Train TabPack paper protocol; write ``model_dir/{preds.npz,meta.json}``."""
    arr = _arrays(bundle, target, genes)
    return train_tabpack_screen(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        outer_parts(bundle, target, genes),
        model_dir,
        target=target,
        n_models=int(os.environ.get("TABPACK_N_MODELS", TABPACK_N_MODELS)),
        protocol=os.environ.get("TABPACK_PROTOCOL", TABPACK_PROTOCOL),
        experiment_namespace=os.environ.get(
            "TABPACK_EXPERIMENT_NAMESPACE", TABPACK_EXPERIMENT_NAMESPACE
        ),
    )


def load_tabpack(model_dir: Path) -> dict[str, Any]:
    return load_tabpack_screen(model_dir)


def preds_by_split(
    artifact: dict,
    bundle: TrainBundle | None = None,
) -> dict[str, np.ndarray]:
    """Map cached preds onto final_train split names."""
    outer = artifact["outer_preds"]
    val_pred = clip_nonneg(np.asarray(artifact["val_pred"], dtype=np.float64))
    out: dict[str, np.ndarray] = {
        "val_mix": val_pred,
        "val_bulk": clip_nonneg(np.asarray(outer["bulk"], dtype=np.float64)),
        "val_k1": clip_nonneg(np.asarray(outer["k1"], dtype=np.float64)),
    }
    for cohort in PB_COHORTS:
        key = f"pb_{cohort}"
        if key in outer:
            out[f"val_pb_{cohort}"] = clip_nonneg(np.asarray(outer[key], dtype=np.float64))
    if bundle is not None:
        mod = np.asarray(bundle.mod_val)
        if len(val_pred) != len(mod):
            raise ValueError(
                f"tabpack val_pred length {len(val_pred)} != mod_val {len(mod)}"
            )
        out["val_pb"] = val_pred[mod == "pb"]
    return out


def eval_row(artifact: dict, bundle: TrainBundle, target: str) -> dict:
    """Val metrics from cached predictions."""
    preds = preds_by_split(artifact, bundle)
    row: dict = {
        "target": target,
        "model": "tabpack",
        "status": "ok",
        "train_sec": np.nan,
        "error": "",
        "val_mix_r2": r2(bundle.y_val[target].to_numpy(dtype=np.float64), preds["val_mix"]),
        "val_bulk_r2": r2(bundle.y_val_bulk[target].to_numpy(dtype=np.float64), preds["val_bulk"]),
        "val_k1_r2": r2(bundle.y_val_k1[target].to_numpy(dtype=np.float64), preds["val_k1"]),
        "val_pb_r2": r2(bundle.y_val_pb[target].to_numpy(dtype=np.float64), preds["val_pb"]),
    }
    for cohort in PB_COHORTS:
        key = f"val_pb_{cohort}"
        if key not in preds:
            continue
        y = bundle.y_val_pb_by_cohort[cohort][target].to_numpy(dtype=np.float64)
        row[f"{key}_r2"] = r2(y, preds[key])
    return row
