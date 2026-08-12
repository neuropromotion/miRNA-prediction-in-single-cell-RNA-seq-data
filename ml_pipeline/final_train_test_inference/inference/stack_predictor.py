"""Load final_train TabPack+DCNv2+TabM ridge stack for eligible miRNAs."""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# TabPack QT pickles were fit under sklearn 1.5.x; inference venv may be 1.7.x.
try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:  # pragma: no cover
    pass

try:
    from .constants import (
        ENSEMBLE_ID,
        FTTI_ROOT,
        ML_PIPELINE,
        CONFIG_PATH,
        MODELS_ROOT,
        STACK_MODELS,
        WEIGHTS_DIR,
        parse_prediction_config,
    )
except ImportError:
    from constants import (
        ENSEMBLE_ID,
        FTTI_ROOT,
        ML_PIPELINE,
        CONFIG_PATH,
        MODELS_ROOT,
        STACK_MODELS,
        WEIGHTS_DIR,
        parse_prediction_config,
    )

# Train helpers use bare `from constants import …`. Inference also has a
# `constants` module, so once it is cached in sys.modules the train imports
# resolve to the wrong file. Evict flat names, load train code with train/
# first on sys.path, then always restore inference constants.
_TRAIN = FTTI_ROOT / "train"
_INFERENCE_CONSTANTS = sys.modules.get("constants")
_FLAT_TRAIN_MODULES = (
    "constants",
    "metrics",
    "data",
    "stack",
    "model_trainers",
    "tabpack_trainer",
    "torch_trainers",
    "dl_trainers",
    "impute",
    "io_splits",
    "transforms",
    "journal",
)
# Point train model_trainers at packaged ../models or train/results.
os.environ.setdefault("FINAL_MODELS_ROOT", str(MODELS_ROOT))
_train_s = str(_TRAIN)
_ftti_s = str(FTTI_ROOT)
_ml_s = str(ML_PIPELINE)
for _p in (_train_s, _ftti_s, _ml_s):
    if _p in sys.path:
        sys.path.remove(_p)
# shared.tabpack_trainer lives under ml_pipeline/; train code under train/
sys.path.insert(0, _ml_s)
sys.path.insert(0, _ftti_s)
sys.path.insert(0, _train_s)

try:
    for _name in _FLAT_TRAIN_MODULES:
        sys.modules.pop(_name, None)
    from stack import FitResult, apply_fit, fit_from_dict  # noqa: E402
    from data import select_features  # noqa: E402
    import model_trainers as _model_trainers  # noqa: E402
    from model_trainers import load_artifact, predict_one  # noqa: E402
finally:
    if _INFERENCE_CONSTANTS is not None:
        sys.modules["constants"] = _INFERENCE_CONSTANTS


@dataclass(frozen=True)
class _TargetBundle:
    mirna: str
    cohort: str
    genes: list[str]
    fit: FitResult
    artifacts: dict[str, object]


class StackPredictor:
    """
    Ridge stack inference for eligible miRNAs (final_train production models).

    Base models: TabPack (Muon paper) + DCNv2 (AdamW) + TabM (AdamW).
    Expects expression matrix in log2(TPM+1) space, cells × ENSG genes
    (same transform as final_train training / sc_TEST evaluation).
    """

    def __init__(
        self,
        config_path: Path | str = CONFIG_PATH,
        weights_dir: Path | str = WEIGHTS_DIR,
        models_root: Path | str = MODELS_ROOT,
        device: str = "cuda",
        catboost_task: str = "CPU",  # unused; kept for API compatibility
        preload_all: bool = False,
    ) -> None:
        self._config_path = Path(config_path)
        self._weights_dir = Path(weights_dir)
        self._models_root = Path(models_root)
        device_key = str(device).strip().lower()
        if device_key in {"gpu", "cuda"}:
            device = "cuda"
        elif device_key == "cpu":
            device = "cpu"
        self._device = device
        self._catboost_task = catboost_task  # noqa: F841 — legacy kwarg

        os.environ["FINAL_DEVICE"] = device
        os.environ["FINAL_MODELS_ROOT"] = str(self._models_root)
        _model_trainers.DEVICE = device

        config = json.loads(self._config_path.read_text(encoding="utf-8"))
        self._config = config
        eligible, cohorts, target_info = parse_prediction_config(config)
        self._target_info: dict[str, dict] = target_info
        self._available_mirnas: list[str] = eligible
        self._cohorts: dict[str, list[str]] = cohorts

        self._cache: dict[str, _TargetBundle] = {}
        if preload_all:
            self.preload_all()

    @property
    def available_mirnas(self) -> list[str]:
        """miRNAs with trained stack models (eligible_mirs from config)."""
        return list(self._available_mirnas)

    @property
    def cohorts(self) -> dict[str, list[str]]:
        """Cohort assignment lists (K1, K2, …, K10)."""
        return {k: list(v) for k, v in self._cohorts.items()}

    @property
    def target_info(self) -> dict[str, dict]:
        """Per-miRNA metadata (assigned_cohort, genes/features, metrics)."""
        return {k: dict(v) for k, v in self._target_info.items()}

    def cohort_for(self, mirna: str) -> str:
        self._check_mirna(mirna)
        return str(self._target_info[mirna]["assigned_cohort"])

    def genes_for(self, mirna: str) -> list[str]:
        self._check_mirna(mirna)
        return list(self._target_info[mirna]["genes"])

    def preload_all(self) -> None:
        """Eager-load all eligible target bundles into memory."""
        for mirna in self._available_mirnas:
            self._get_bundle(mirna)

    def _check_mirna(self, mirna: str) -> None:
        if mirna not in self._target_info:
            raise KeyError(
                f"Unknown miRNA {mirna!r}. "
                f"Use one of {len(self._available_mirnas)} available_mirnas."
            )

    def _get_bundle(self, mirna: str) -> _TargetBundle:
        if mirna in self._cache:
            return self._cache[mirna]

        info = self._target_info[mirna]
        weight_path = self._weights_dir / f"{mirna}.json"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing stack weights: {weight_path}")

        fit = fit_from_dict(json.loads(weight_path.read_text(encoding="utf-8")))
        artifacts: dict[str, object] = {}
        for model in STACK_MODELS:
            artifacts[model] = load_artifact(model, mirna)

        bundle = _TargetBundle(
            mirna=mirna,
            cohort=str(info["assigned_cohort"]),
            genes=list(info["genes"]),
            fit=fit,
            artifacts=artifacts,
        )
        self._cache[mirna] = bundle
        return bundle

    @staticmethod
    def _as_gene_matrix(x: pd.DataFrame | np.ndarray, genes: list[str] | None) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x
        if genes is None:
            raise ValueError("Pass a DataFrame or provide `genes` for ndarray input.")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        if arr.shape[1] != len(genes):
            raise ValueError(f"Array has {arr.shape[1]} columns, expected {len(genes)} genes.")
        return pd.DataFrame(arr, columns=genes)

    def predict(self, mirna: str, x: pd.DataFrame | np.ndarray, *, genes: list[str] | None = None) -> np.ndarray:
        """
        Predict log2 miRNA expression for one target.

        Parameters
        ----------
        mirna
            miRNA name (must be in available_mirnas).
        x
            cells × genes matrix in log2(TPM+1) space.
        genes
            Column order when x is ndarray (ignored for DataFrame).

        Returns
        -------
        np.ndarray
            1D predictions per cell (non-negative).
        """
        bundle = self._get_bundle(mirna)
        x_df = self._as_gene_matrix(x, genes)
        x_sel = select_features(x_df, bundle.genes).to_numpy(dtype=np.float32)
        pred_matrix = np.column_stack(
            [predict_one(m, bundle.artifacts[m], x_sel) for m in STACK_MODELS]
        )
        return apply_fit(bundle.fit, pred_matrix, STACK_MODELS)

    def predict_many(
        self,
        x: pd.DataFrame | np.ndarray,
        mirnas: Iterable[str] | None = None,
        *,
        genes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Predict multiple miRNAs; returns cells × miRNAs DataFrame."""
        x_df = self._as_gene_matrix(x, genes)
        targets = list(mirnas) if mirnas is not None else self._available_mirnas
        out = pd.DataFrame(index=x_df.index)
        for mirna in targets:
            out[mirna] = self.predict(mirna, x_df)
        return out

    def predict_cohort(
        self,
        cohort: str,
        x: pd.DataFrame | np.ndarray,
        *,
        genes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Predict all miRNAs assigned to a cohort (K1, K2, …, K10)."""
        if cohort not in self._cohorts:
            raise KeyError(f"Unknown cohort {cohort!r}. Expected one of {list(self._cohorts)}.")
        return self.predict_many(x, self._cohorts[cohort], genes=genes)

    def __repr__(self) -> str:
        return (
            f"StackPredictor(n_mirnas={len(self._available_mirnas)}, "
            f"ensemble={ENSEMBLE_ID!r}, device={self._device!r})"
        )
