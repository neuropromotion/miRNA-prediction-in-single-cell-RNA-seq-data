"""Train/predict wrappers for model_tuning A/B (no Optuna).

Variants:
  tabm_muon, tabm_adamw_ema, dcnv2_muon, tabpack (paper MuonAdamWPack)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from constants import (
    BATCH_SIZE,
    SEED,
    TABM_BATCH_SIZE,
    TABPACK_N_MODELS,
    TABPACK_PROTOCOL,
)
from metrics import clip_nonneg
from shared.data import ModalityBundle, select_features
from shared.io_splits import PB_COHORTS
from shared.paths import PILOT_DIR

sys.path.insert(0, str(PILOT_DIR))

from dl_trainers import (  # noqa: E402
    predict_tabm,
    predict_torch_model,
    train_dcnv2,
    train_tabm,
)
from shared.tabpack_trainer import train_tabpack_screen  # noqa: E402

DEVICE = os.environ.get("STAGE03_DEVICE", os.environ.get("TUNING_DEVICE", "cuda"))

# Map tuning model id → (base_arch, optimizer_kind)
VARIANT_SPEC = {
    "tabm_muon": ("tabm", "muon"),
    "tabm_adamw_ema": ("tabm", "adamw_ema"),
    "dcnv2_muon": ("dcnv2", "muon"),
    "tabpack": ("tabpack", "paper"),
}


def _arrays(bundle: ModalityBundle, target: str, genes: list[str]) -> dict:
    return {
        "x_train": select_features(bundle.x_train, genes).to_numpy(dtype=np.float32),
        "y_train": bundle.y_train[target].to_numpy(dtype=np.float64),
        "x_val": select_features(bundle.x_val_inner, genes).to_numpy(dtype=np.float32),
        "y_val": bundle.y_val_inner[target].to_numpy(dtype=np.float64),
    }


def _outer_parts(
    bundle: ModalityBundle, target: str, genes: list[str]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    parts: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "bulk",
            select_features(bundle.x_outer_val_bulk, genes).to_numpy(dtype=np.float32),
            bundle.y_outer_val_bulk[target].to_numpy(dtype=np.float64),
        ),
        (
            "k1",
            select_features(bundle.x_outer_val_k1, genes).to_numpy(dtype=np.float32),
            bundle.y_outer_val_k1[target].to_numpy(dtype=np.float64),
        ),
    ]
    for cohort in PB_COHORTS:
        parts.append(
            (
                f"pb_{cohort}",
                select_features(bundle.x_outer_val_pb[cohort], genes).to_numpy(dtype=np.float32),
                bundle.y_outer_val_pb[cohort][target].to_numpy(dtype=np.float64),
            )
        )
    return parts


def _muon_lr() -> float:
    return float(os.environ.get("TUNING_MUON_LR", "0.02"))


def train_tabm_variant(arr: dict, model_dir: Path, optimizer_kind: str) -> Path:
    info = train_tabm(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        model_dir,
        DEVICE,
        TABM_BATCH_SIZE,
        optimizer_kind=optimizer_kind,
        muon_lr=_muon_lr() if optimizer_kind.startswith("muon") else None,
        save=True,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "train_info.json").write_text(
        json.dumps({"optimizer_kind": optimizer_kind, "seed": SEED, **info}, indent=2, default=str),
        encoding="utf-8",
    )
    return model_dir


def train_dcnv2_variant(arr: dict, model_dir: Path, optimizer_kind: str) -> Path:
    info = train_dcnv2(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        model_dir,
        DEVICE,
        BATCH_SIZE,
        optimizer_kind=optimizer_kind,
        muon_lr=_muon_lr() if optimizer_kind.startswith("muon") else None,
        save=True,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "train_info.json").write_text(
        json.dumps({"optimizer_kind": optimizer_kind, "seed": SEED, **info}, indent=2, default=str),
        encoding="utf-8",
    )
    return model_dir


def train_tabpack_model(
    bundle: ModalityBundle, target: str, genes: list[str], model_dir: Path
) -> dict:
    arr = _arrays(bundle, target, genes)
    n_models = int(os.environ.get("TABPACK_N_MODELS", TABPACK_N_MODELS))
    protocol = os.environ.get("TABPACK_PROTOCOL", TABPACK_PROTOCOL)
    return train_tabpack_screen(
        arr["x_train"],
        arr["y_train"],
        arr["x_val"],
        arr["y_val"],
        _outer_parts(bundle, target, genes),
        model_dir=model_dir,
        target=target,
        n_models=n_models,
        protocol=protocol,
        experiment_namespace="mirna_tuning",
    )


def train_one(
    model_name: str,
    bundle: ModalityBundle,
    target: str,
    genes: list[str],
    model_dir: Path,
):
    if model_name not in VARIANT_SPEC:
        raise ValueError(f"Unknown tuning model: {model_name}; allowed={list(VARIANT_SPEC)}")
    arch, kind = VARIANT_SPEC[model_name]
    arr = _arrays(bundle, target, genes)
    if arch == "tabm":
        return train_tabm_variant(arr, model_dir, kind)
    if arch == "dcnv2":
        return train_dcnv2_variant(arr, model_dir, kind)
    if arch == "tabpack":
        return train_tabpack_model(bundle, target, genes, model_dir)
    raise ValueError(model_name)


def load_artifact(model_name: str, model_dir: Path):
    arch = VARIANT_SPEC[model_name][0]
    if arch in ("tabm", "dcnv2"):
        return model_dir
    if arch == "tabpack":
        from shared.tabpack_trainer import load_tabpack_screen

        return load_tabpack_screen(model_dir)
    raise ValueError(model_name)


def predict_model(model_name: str, artifact, x: np.ndarray) -> np.ndarray:
    arch = VARIANT_SPEC[model_name][0]
    if arch == "tabm":
        return clip_nonneg(predict_tabm(artifact, x, DEVICE))
    if arch == "dcnv2":
        return clip_nonneg(predict_torch_model(artifact, x))
    if arch == "tabpack":
        raise TypeError("tabpack uses cached outer preds")
    raise ValueError(model_name)
