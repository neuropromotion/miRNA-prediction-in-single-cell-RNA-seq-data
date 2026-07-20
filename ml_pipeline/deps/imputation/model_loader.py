#!/usr/bin/env python3
"""
STACK inference loader: TabPFN-PCA + XGBoost + Ridge stacking cascade.

Provides CascadeStackLoader for lazy/eager loading of per-miRNA fold artifacts
under model_config/stage_*/<mirna>/ and batch predict_cascade() over stages.

Artifacts per target (n_splits folds):
    pca_fold{K}.joblib, tpfn_fold{K}.joblib, xgb_fold{K}.json,
    xgb_feat_list.json, stack_meta.joblib
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INFERENCE_ARTIFACTS_SHARED: Tuple[str, ...] = (
    "xgb_feat_list.json",
    "stack_meta.joblib",
)
INFERENCE_ARTIFACTS_PER_FOLD: Tuple[str, ...] = (
    "pca_fold{fold}.joblib",
    "tpfn_fold{fold}.joblib",
    "xgb_fold{fold}.json",
)
# All artifact filenames (shared + one fold); use list_required_artifact_paths for n_splits>1.
INFERENCE_ARTIFACTS: Tuple[str, ...] = INFERENCE_ARTIFACTS_SHARED + INFERENCE_ARTIFACTS_PER_FOLD

BULK_SIZE_SC = 1.0
DEFAULT_KNN_NEIGHBORS = 5

_SAFE_TABPFN_REGISTERED = False
_TORCH_LOAD_PATCHED = False
_SKLEARN_UNPICKLE_PATCHED = False
_XGB_MODULE = None


# ---------------------------------------------------------------------------
# SafeTabPFN — must match training pickles (TabPFNRegressor + ignore_pretraining_limits)
# ---------------------------------------------------------------------------
class SafeTabPFN:
    """Wrapper used only for unpickling tpfn_fold*.joblib saved during training."""

    def __init__(self, max_samples: int = 10000, device: str = "cuda", seed: int = 42):
        self.max_samples = int(max_samples)
        self.device = str(device)
        self.seed = int(seed)
        self.model = None

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if len(X) > self.max_samples:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X, y = X[idx], y[idx]
        self.model = TabPFNRegressor(
            device=self.device,
            ignore_pretraining_limits=True,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("SafeTabPFN.model is None — artifact not loaded correctly")
        arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(arr)


# ---------------------------------------------------------------------------
# Environment / version guards
# ---------------------------------------------------------------------------
def prefer_env_site_over_user_local() -> None:
    """Prefer conda/env site-packages over ~/.local (sklearn 1.7 shadowing 1.5.2)."""
    import site

    user_site = site.getusersitepackages()
    if not user_site:
        return

    def _is_user_local(path: str) -> bool:
        norm = path.replace("\\", "/")
        return "/.local/lib/python" in norm or path == user_site

    head = [p for p in sys.path if not _is_user_local(p)]
    tail = [p for p in sys.path if _is_user_local(p)]
    sys.path[:] = head + tail


def check_sklearn_version_for_tabpfn(max_version: Tuple[int, int] = (1, 5)) -> None:
    import sklearn

    parts = sklearn.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse scikit-learn version: {sklearn.__version__}") from exc

    if (major, minor) > max_version:
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} at {sklearn.__file__} is too new for "
            f"TabPFN stack weights (need <= {max_version[0]}.{max_version[1]}.x). "
            "Use prefer_env_site_over_user_local(), install scikit-learn==1.5.2, restart kernel."
        )


def check_tabpfn_version_for_stack_weights(max_major: int = 6) -> None:
    """Stack tpfn_fold*.joblib require tabpfn 6.4.x; tabpfn>=7 breaks predict (n_estimators_)."""
    import tabpfn

    ver_str = getattr(tabpfn, "__version__", "0")
    parts = ver_str.split(".")
    try:
        major = int(parts[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse tabpfn version: {ver_str}") from exc

    if major >= max_major + 1:
        raise RuntimeError(
            f"tabpfn {ver_str} is incompatible with stack weights (need 6.4.x, not >=7). "
            "Install tabpfn==6.4.1 from requirements-inference.txt and restart kernel."
        )


def configure_inference_device(mode: str = "auto") -> str:
    """
    Resolve TabPFN device from INFERENCE_DEVICE: 'cpu' | 'cuda' | 'auto'.
    Sets CUDA_VISIBLE_DEVICES='' for cpu mode.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if mode_norm == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if mode_norm == "cuda":
        return "cuda"
    raise ValueError(f"INFERENCE_DEVICE must be 'cpu', 'cuda', or 'auto', got {mode!r}")


# ---------------------------------------------------------------------------
# Unpickle / device patches (call before joblib.load tpfn_fold*.joblib)
# ---------------------------------------------------------------------------
def _patch_sklearn_unpickle_compat() -> None:
    """Shim missing sklearn private classes when unpickling across minor versions."""
    global _SKLEARN_UNPICKLE_PATCHED
    if _SKLEARN_UNPICKLE_PATCHED:
        return

    import sklearn.compose._column_transformer as ct_mod

    class _RemainderColsList(list):
        pass

    if not hasattr(ct_mod, "_RemainderColsList"):
        ct_mod._RemainderColsList = _RemainderColsList

    try:
        import sklearn.impute as impute_mod

        class _NoInverseImputer:  # pragma: no cover
            pass

        if not hasattr(impute_mod, "_NoInverseImputer"):
            impute_mod._NoInverseImputer = _NoInverseImputer
    except ImportError:
        pass

    _SKLEARN_UNPICKLE_PATCHED = True


def _register_safe_tabpfn_for_unpickle() -> None:
    global _SAFE_TABPFN_REGISTERED
    if _SAFE_TABPFN_REGISTERED:
        return

    _patch_sklearn_unpickle_compat()

    this_mod = sys.modules[__name__]
    setattr(this_mod, "SafeTabPFN", SafeTabPFN)

    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        setattr(main_mod, "SafeTabPFN", SafeTabPFN)

    _SAFE_TABPFN_REGISTERED = True


def _patch_torch_load_cpu() -> None:
    """Force torch.load map_location=cpu so CUDA-saved TabPFN weights load on CPU hosts."""
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return

    import torch

    original_load = torch.load

    def _load_with_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_load(*args, **kwargs)

    torch.load = _load_with_cpu
    _TORCH_LOAD_PATCHED = True


def move_tabpfn_artifact_to_device(artifact: Any, device: str) -> Any:
    """Move SafeTabPFN / TabPFNRegressor tensors to inference device after joblib.load."""
    dev = str(device).lower()

    inner = getattr(artifact, "model", artifact)
    if inner is None:
        return artifact

    if hasattr(inner, "to"):
        inner.to(dev)

    if hasattr(artifact, "device"):
        artifact.device = dev

    return artifact


def _get_xgb():
    """Lazy xgboost import — avoids GLIBCXX conflicts with tabpfn at import time."""
    global _XGB_MODULE
    if _XGB_MODULE is None:
        import xgboost as xgb

        _XGB_MODULE = xgb
    return _XGB_MODULE


# ---------------------------------------------------------------------------
# Matrix orientation / KNN impute (TabPFN branch uses imputed X_inf_sc)
# ---------------------------------------------------------------------------
def auto_orient_X(df: pd.DataFrame, gene_prefix: Sequence[str] = ("ENSG",)) -> pd.DataFrame:
    cols_ok = any(str(c).startswith(gene_prefix) for c in df.columns[:50])
    rows_ok = any(str(r).startswith(gene_prefix) for r in df.index[:50])
    if cols_ok and not rows_ok:
        return df
    if rows_ok and not cols_ok:
        return df.T
    return df


def knn_impute_cpu(
    X_df: pd.DataFrame,
    zero_mask: pd.DataFrame,
    indices: np.ndarray,
    donor_df: pd.DataFrame,
) -> pd.DataFrame:
    X = X_df.values.astype(np.float32).copy()
    mask = zero_mask.values
    donor = donor_df.values.astype(np.float32)
    _, n_features = X.shape
    for j in range(n_features):
        missing_idx = np.where(mask[:, j])[0]
        if len(missing_idx) == 0:
            continue
        neigh_vals = donor[indices[missing_idx], j]
        neigh_vals = np.where(neigh_vals == 0, np.nan, neigh_vals)
        with np.errstate(all="ignore"):
            imputed = np.nanmean(neigh_vals, axis=1)
        imputed = np.where(np.isnan(imputed), 0.0, imputed)
        X[missing_idx, j] = imputed
    return pd.DataFrame(X, index=X_df.index, columns=X_df.columns)


def run_imputer(
    X_ref: pd.DataFrame,
    X_query: pd.DataFrame,
    n_neighbors: int = DEFAULT_KNN_NEIGHBORS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(X_ref.columns) & set(X_query.columns))
    Xr = X_ref[common].copy()
    Xq = X_query[common].copy()
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(Xr.values.astype(np.float32))
    _, ind_ref = nn.kneighbors(Xr.values.astype(np.float32))
    _, ind_q = nn.kneighbors(Xq.values.astype(np.float32))
    Xr_filled = knn_impute_cpu(Xr, Xr == 0, ind_ref, Xr)
    Xq_filled = knn_impute_cpu(Xq, Xq == 0, ind_q, Xr_filled)
    return Xr_filled, Xq_filled


def ensure_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        miss_df = pd.DataFrame(fill_value, index=out.index, columns=missing, dtype=np.float32)
        out = pd.concat([out, miss_df], axis=1)
    return out[list(cols)]


def align_and_impute_for_inference(
    X_query: pd.DataFrame,
    required_cols: Sequence[str],
    X_ref_knn: pd.DataFrame,
    n_neighbors: int = DEFAULT_KNN_NEIGHBORS,
) -> pd.DataFrame:
    Xq = ensure_columns(X_query, required_cols, fill_value=0.0)
    Xref = ensure_columns(X_ref_knn, required_cols, fill_value=0.0)
    _, Xq_imp = run_imputer(Xref, Xq, n_neighbors=n_neighbors)
    return Xq_imp


# ---------------------------------------------------------------------------
# PCA reconstruction + feature builders
# ---------------------------------------------------------------------------
def pca_reconstruct(X_df: pd.DataFrame, bundle: Mapping[str, Any]) -> pd.DataFrame:
    common = bundle["common"]
    Xs = bundle["scaler"].transform(X_df[common].astype(np.float32))
    Z = bundle["pca"].transform(Xs)
    Xr = bundle["pca"].inverse_transform(Z)
    Xr = bundle["scaler"].inverse_transform(Xr)
    return pd.DataFrame(Xr, index=X_df.index, columns=common)


def prepare_tpfn_predict_X(
    X_gene_df: pd.DataFrame,
    feat_list: Sequence[str],
    cascade_test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    TabPFN input: explicit feature_map order (gene cols from PCA-reconstructed matrix;
    stack__/pred__ cols from cascade predictions). Unlike inference_genuine.py, do NOT
    concat the full cascade frame onto X_rec — only requested columns.
    """
    cascade_cols = [c for c in feat_list if str(c).startswith(("stack__", "pred__"))]
    gene_cols = [c for c in feat_list if c not in cascade_cols]
    Xq = ensure_columns(X_gene_df, gene_cols, fill_value=0.0).astype(np.float32)

    if cascade_cols:
        if cascade_test is not None and not cascade_test.empty:
            ct = cascade_test.reindex(Xq.index).fillna(0.0).astype(np.float32)
            ct = ct.reindex(columns=cascade_cols, fill_value=0.0)
        else:
            ct = pd.DataFrame(0.0, index=Xq.index, columns=cascade_cols, dtype=np.float32)
        Xq = pd.concat([Xq, ct[cascade_cols]], axis=1)

    return Xq[list(feat_list)]


def prepare_xgb_predict_X(
    X_query: pd.DataFrame,
    feat_list: Sequence[str],
    cascade_test: Optional[pd.DataFrame] = None,
    add_bulk_size: bool = True,
    bulk_size_value: float = BULK_SIZE_SC,
) -> pd.DataFrame:
    cascade_cols = [c for c in feat_list if str(c).startswith(("stack__", "pred__"))]
    base_cols = [c for c in feat_list if c not in cascade_cols and c != "bulk_size"]
    Xq = ensure_columns(X_query, base_cols, fill_value=0.0).astype(np.float32)

    if add_bulk_size and "bulk_size" in feat_list:
        Xq = Xq.copy()
        Xq["bulk_size"] = bulk_size_value

    if cascade_cols:
        if cascade_test is not None and not cascade_test.empty:
            ct = cascade_test.reindex(Xq.index).fillna(0.0).astype(np.float32)
            ct = ct.reindex(columns=cascade_cols, fill_value=0.0)
        else:
            ct = pd.DataFrame(0.0, index=Xq.index, columns=cascade_cols, dtype=np.float32)
        Xq = pd.concat([Xq, ct[cascade_cols]], axis=1)

    return Xq[list(feat_list)]


# ---------------------------------------------------------------------------
# Stage / artifact helpers
# ---------------------------------------------------------------------------
def stage_order_from_map(stage_map: Mapping[str, Sequence[str]]) -> List[str]:
    def _stage_num(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(stage_map.keys(), key=_stage_num)


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def list_required_artifact_paths(target_dir: Path, n_splits: int) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for name in INFERENCE_ARTIFACTS_SHARED:
        rows.append((name, target_dir / name))
    for fold in range(n_splits):
        for pattern in INFERENCE_ARTIFACTS_PER_FOLD:
            fname = pattern.format(fold=fold)
            rows.append((fname, target_dir / fname))
    return rows


def audit_artifacts_in_dir(target_dir: Path, n_splits: int) -> pd.DataFrame:
    records = []
    for fname, fpath in list_required_artifact_paths(target_dir, n_splits):
        records.append({"file": fname, "present": fpath.is_file(), "path": str(fpath)})
    return pd.DataFrame(records)


def target_ready_on_disk(
    models_dir: PathLike,
    stage_name: str,
    mir_name: str,
    n_splits: int,
) -> bool:
    target_dir = _as_path(models_dir) / stage_name / mir_name
    if not target_dir.is_dir():
        return False
    audit = audit_artifacts_in_dir(target_dir, n_splits)
    return bool(audit["present"].all())


def target_meta_only_on_disk(
    models_dir: PathLike,
    stage_name: str,
    mir_name: str,
    n_splits: int,
) -> bool:
    target_dir = _as_path(models_dir) / stage_name / mir_name
    meta_path = target_dir / "stack_meta.joblib"
    if not meta_path.is_file():
        return False
    return not target_ready_on_disk(models_dir, stage_name, mir_name, n_splits)


def find_stage_for_mir(stage_map: Mapping[str, Sequence[str]], mir_name: str) -> Optional[str]:
    for stage in stage_order_from_map(stage_map):
        if mir_name in stage_map.get(stage, []):
            return stage
    return None


def resolve_target_dir(
    models_dir: PathLike,
    stage_map: Mapping[str, Sequence[str]],
    mir_name: str,
    stage_hint: Optional[str] = None,
) -> Tuple[str, Path]:
    models_dir = _as_path(models_dir)
    if stage_hint is not None:
        stage = stage_hint
    else:
        stage = find_stage_for_mir(stage_map, mir_name)
        if stage is None:
            raise KeyError(f"miRNA {mir_name!r} not found in cascade stage_map")

    target_dir = models_dir / stage / mir_name
    if not target_dir.is_dir():
        raise FileNotFoundError(f"No model directory for {stage}/{mir_name}: {target_dir}")
    return stage, target_dir


# ---------------------------------------------------------------------------
# Per-target loaded model
# ---------------------------------------------------------------------------
class MicroRNAStackModel:
    """One miRNA stack: PCA+TabPFN branch, XGB branch, Ridge meta."""

    def __init__(
        self,
        mir_name: str,
        stage: str,
        target_dir: Path,
        n_splits: int,
        tpfn_feat_list: Sequence[str],
        xgb_feat_list: Sequence[str],
        pca_folds: Sequence[Any],
        tpfn_folds: Sequence[Any],
        xgb_folds: Sequence[Any],
        stack_meta: Any,
        add_bulk_size: bool = True,
        bulk_size_sc: float = BULK_SIZE_SC,
        ready: bool = True,
    ):
        self.mir_name = mir_name
        self.stage = stage
        self.target_dir = target_dir
        self.n_splits = int(n_splits)
        self.tpfn_feat_list = list(tpfn_feat_list)
        self.xgb_feat_list = list(xgb_feat_list)
        self.pca_folds = list(pca_folds)
        self.tpfn_folds = list(tpfn_folds)
        self.xgb_folds = list(xgb_folds)
        self.stack_meta = stack_meta
        self.add_bulk_size = bool(add_bulk_size)
        self.bulk_size_sc = float(bulk_size_sc)
        self.ready = bool(ready)

    def predict(
        self,
        X_inf_sc: pd.DataFrame,
        X_inf_raw: pd.DataFrame,
        cascade_preds_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        if not self.ready:
            raise RuntimeError(f"Model {self.mir_name} is not ready (missing artifacts on disk)")

        xgb = _get_xgb()
        tpfn_fold_preds: List[np.ndarray] = []

        for fold in range(self.n_splits):
            bundle = self.pca_folds[fold]
            X_rec = pca_reconstruct(X_inf_sc, bundle)
            X_tpfn = prepare_tpfn_predict_X(
                X_rec,
                self.tpfn_feat_list,
                cascade_test=cascade_preds_df,
            )
            tpfn = self.tpfn_folds[fold]
            tpfn_fold_preds.append(tpfn.predict(X_tpfn.values).astype(np.float32))

        tpfn_pred = np.mean(tpfn_fold_preds, axis=0)

        xgb_fold_preds: List[np.ndarray] = []
        for fold in range(self.n_splits):
            X_xgb = prepare_xgb_predict_X(
                X_inf_raw,
                self.xgb_feat_list,
                cascade_test=cascade_preds_df,
                add_bulk_size=self.add_bulk_size,
                bulk_size_value=self.bulk_size_sc,
            )
            booster = self.xgb_folds[fold]
            dmat = xgb.DMatrix(X_xgb.values, feature_names=list(X_xgb.columns))
            xgb_fold_preds.append(booster.predict(dmat).astype(np.float32))

        xgb_pred = np.mean(xgb_fold_preds, axis=0)
        Xm = np.column_stack([tpfn_pred, xgb_pred]).astype(np.float32)
        return self.stack_meta.predict(Xm).astype(np.float32)


# ---------------------------------------------------------------------------
# Cascade loader
# ---------------------------------------------------------------------------
class CascadeStackLoader:
    """
    Registry + lazy loader for STACK cascade weights under models_dir.

    Parameters
    ----------
    models_dir : path to model_config/
    features_path : features.json fallback when cascade_config lacks feature_map
    n_splits : number of CV folds (default 2)
    lazy : if True, weights load on first __getitem__/load
    strict : if True, raise when configured targets lack weights (when not filtering)
    only_available_on_disk : register only targets with full artifact set
    tabpfn_device : 'cpu' or 'cuda' for TabPFN after unpickle
    """

    def __init__(
        self,
        models_dir: PathLike,
        features_path: PathLike,
        n_splits: int = 2,
        lazy: bool = True,
        strict: bool = False,
        only_available_on_disk: bool = True,
        tabpfn_device: str = "cpu",
    ):
        self.models_dir = _as_path(models_dir)
        self.features_path = _as_path(features_path)
        self.n_splits = int(n_splits)
        self.lazy = bool(lazy)
        self.strict = bool(strict)
        self.only_available_on_disk = bool(only_available_on_disk)
        self.tabpfn_device = str(tabpfn_device)

        self._cache: Dict[str, MicroRNAStackModel] = {}
        self._load_config()
        self._build_registry()

        if not self.lazy:
            for mir in self._all_mirs:
                self.load(mir)

    def _load_config(self) -> None:
        cascade_path = self.models_dir / "cascade_config.json"
        if not cascade_path.is_file():
            raise FileNotFoundError(f"Missing cascade_config.json in {self.models_dir}")

        with open(cascade_path, encoding="utf-8") as f:
            self.cascade_config: Dict[str, Any] = json.load(f)

        self.stage_map: Dict[str, List[str]] = {
            k: list(v) for k, v in self.cascade_config["stage_map"].items()
        }
        self.feature_map: Dict[str, List[str]] = dict(self.cascade_config.get("feature_map") or {})
        self.n_splits = int(self.cascade_config.get("n_splits", self.n_splits))
        self.add_bulk_size = bool(self.cascade_config.get("add_bulk_size", True))
        self.bulk_size_sc = float(self.cascade_config.get("bulk_size_sc", BULK_SIZE_SC))

        if not self.feature_map and self.features_path.is_file():
            with open(self.features_path, encoding="utf-8") as f:
                features_payload = json.load(f)
            self.feature_map = dict(features_payload.get("features") or {})

        req_path = self.models_dir / "REQUIRED_GENES.json"
        if req_path.is_file():
            with open(req_path, encoding="utf-8") as f:
                self.required_genes: List[str] = json.load(f)
        else:
            self.required_genes = []

    def _build_registry(self) -> None:
        self._mir_to_stage: Dict[str, str] = {}
        self.stages: Dict[str, List[str]] = {}
        self._all_mirs: List[str] = []

        ordered_stages = stage_order_from_map(self.stage_map)
        missing_on_disk: List[str] = []

        for stage in ordered_stages:
            configured = list(self.stage_map.get(stage, []))
            available: List[str] = []
            for mir in configured:
                if self.only_available_on_disk:
                    if target_ready_on_disk(self.models_dir, stage, mir, self.n_splits):
                        available.append(mir)
                    else:
                        missing_on_disk.append(mir)
                else:
                    available.append(mir)
            if available:
                self.stages[stage] = available
                for mir in available:
                    self._mir_to_stage[mir] = stage
                    self._all_mirs.append(mir)

        self.stage_order = [s for s in ordered_stages if s in self.stages]

        if self.strict and missing_on_disk and not self.only_available_on_disk:
            sample = ", ".join(missing_on_disk[:5])
            raise RuntimeError(
                f"strict=True: {len(missing_on_disk)} configured targets lack full weights "
                f"(e.g. {sample})"
            )

    def __len__(self) -> int:
        return len(self._all_mirs)

    def __contains__(self, mir_name: str) -> bool:
        return mir_name in self._mir_to_stage

    def clear_cache(self) -> None:
        self._cache.clear()
        gc.collect()

    def audit_target_dir(self, mir_name: str) -> pd.DataFrame:
        stage = self._mir_to_stage.get(mir_name)
        if stage is None:
            stage = find_stage_for_mir(self.stage_map, mir_name)
        if stage is None:
            raise KeyError(f"Unknown miRNA: {mir_name!r}")

        target_dir = self.models_dir / stage / mir_name
        return audit_artifacts_in_dir(target_dir, self.n_splits)

    def readiness_report(self) -> pd.DataFrame:
        rows = []
        for stage in self.stage_order:
            for mir in self.stages.get(stage, []):
                audit = self.audit_target_dir(mir)
                n_missing = int((~audit["present"]).sum())
                missing_files = audit.loc[~audit["present"], "file"].tolist()
                ready = n_missing == 0
                meta_only = target_meta_only_on_disk(self.models_dir, stage, mir, self.n_splits)
                rows.append(
                    {
                        "mirna": mir,
                        "stage": stage,
                        "ready": ready,
                        "meta_only": meta_only,
                        "n_missing": n_missing,
                        "missing": ", ".join(missing_files),
                        "cached": mir in self._cache,
                    }
                )
        return pd.DataFrame(rows)

    def load(self, mir_name: str, force: bool = False) -> MicroRNAStackModel:
        if not force and mir_name in self._cache:
            return self._cache[mir_name]

        stage, target_dir = resolve_target_dir(
            self.models_dir,
            self.stage_map,
            mir_name,
            stage_hint=self._mir_to_stage.get(mir_name),
        )

        if not target_ready_on_disk(self.models_dir, stage, mir_name, self.n_splits):
            model = MicroRNAStackModel(
                mir_name=mir_name,
                stage=stage,
                target_dir=target_dir,
                n_splits=self.n_splits,
                tpfn_feat_list=self.feature_map.get(mir_name, []),
                xgb_feat_list=[],
                pca_folds=[],
                tpfn_folds=[],
                xgb_folds=[],
                stack_meta=None,
                add_bulk_size=self.add_bulk_size,
                bulk_size_sc=self.bulk_size_sc,
                ready=False,
            )
            self._cache[mir_name] = model
            return model

        tpfn_feat_list = self.feature_map.get(mir_name)
        if not tpfn_feat_list:
            raise KeyError(
                f"No TabPFN feature_map entry for {mir_name!r} "
                f"(cascade_config.json or features.json)"
            )

        xgb_feat_path = target_dir / "xgb_feat_list.json"
        with open(xgb_feat_path, encoding="utf-8") as f:
            xgb_feat_list = json.load(f)

        _register_safe_tabpfn_for_unpickle()
        _patch_torch_load_cpu()

        pca_folds = []
        tpfn_folds = []
        xgb_folds = []
        xgb = _get_xgb()

        for fold in range(self.n_splits):
            pca_path = target_dir / f"pca_fold{fold}.joblib"
            tpfn_path = target_dir / f"tpfn_fold{fold}.joblib"
            xgb_path = target_dir / f"xgb_fold{fold}.json"

            pca_folds.append(joblib.load(pca_path))
            tpfn = joblib.load(tpfn_path)
            move_tabpfn_artifact_to_device(tpfn, self.tabpfn_device)
            tpfn_folds.append(tpfn)

            booster = xgb.Booster()
            booster.load_model(str(xgb_path))
            xgb_folds.append(booster)

        stack_meta = joblib.load(target_dir / "stack_meta.joblib")

        model = MicroRNAStackModel(
            mir_name=mir_name,
            stage=stage,
            target_dir=target_dir,
            n_splits=self.n_splits,
            tpfn_feat_list=tpfn_feat_list,
            xgb_feat_list=xgb_feat_list,
            pca_folds=pca_folds,
            tpfn_folds=tpfn_folds,
            xgb_folds=xgb_folds,
            stack_meta=stack_meta,
            add_bulk_size=self.add_bulk_size,
            bulk_size_sc=self.bulk_size_sc,
            ready=True,
        )
        self._cache[mir_name] = model
        return model

    def __getitem__(self, mir_name: str) -> MicroRNAStackModel:
        return self.load(mir_name)

    def predict_cascade(
        self,
        X_inf_sc: pd.DataFrame,
        X_inf_raw: pd.DataFrame,
        skip_not_ready: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run stage-wise cascade inference.

        Returns DataFrame (cells × miRNA) with final stack predictions (no stack__ prefix).
        """
        if X_inf_sc.index.equals(X_inf_raw.index) is False:
            X_inf_raw = X_inf_raw.reindex(X_inf_sc.index, fill_value=0.0)

        cascade_preds = pd.DataFrame(index=X_inf_sc.index)
        all_stack: Dict[str, np.ndarray] = {}

        if verbose:
            print("Prediction of miRNAs expression (cascade by stages)..")

        for stage in self.stage_order:
            targets = self.stages.get(stage, [])
            if not targets:
                continue

            iterator: Iterable[str] = tqdm(targets, desc=stage) if verbose else targets
            stage_stack: Dict[str, np.ndarray] = {}

            for mir in iterator:
                try:
                    model = self[mir]
                    if not model.ready:
                        if skip_not_ready:
                            if verbose:
                                print(f"  [skip] {mir}: weights not ready on disk")
                            continue
                        raise RuntimeError(f"{mir}: weights not ready on disk")

                    stack_pred = model.predict(
                        X_inf_sc=X_inf_sc,
                        X_inf_raw=X_inf_raw,
                        cascade_preds_df=cascade_preds,
                    )
                    stage_stack[mir] = stack_pred
                    all_stack[mir] = stack_pred
                except Exception as exc:
                    if verbose:
                        print(f"  ✗ {mir} FAILED: {exc}")
                    if not skip_not_ready:
                        raise

            if stage_stack:
                new_cols = {f"stack__{mir}": pred for mir, pred in stage_stack.items()}
                cascade_preds = pd.concat(
                    [cascade_preds, pd.DataFrame(new_cols, index=X_inf_sc.index)],
                    axis=1,
                )

        if not all_stack:
            return pd.DataFrame(index=X_inf_sc.index)

        return pd.DataFrame(all_stack, index=X_inf_sc.index)


__all__ = [
    "BULK_SIZE_SC",
    "CascadeStackLoader",
    "INFERENCE_ARTIFACTS",
    "INFERENCE_ARTIFACTS_PER_FOLD",
    "INFERENCE_ARTIFACTS_SHARED",
    "MicroRNAStackModel",
    "SafeTabPFN",
    "align_and_impute_for_inference",
    "audit_artifacts_in_dir",
    "auto_orient_X",
    "check_sklearn_version_for_tabpfn",
    "check_tabpfn_version_for_stack_weights",
    "configure_inference_device",
    "ensure_columns",
    "find_stage_for_mir",
    "list_required_artifact_paths",
    "move_tabpfn_artifact_to_device",
    "pca_reconstruct",
    "prefer_env_site_over_user_local",
    "prepare_tpfn_predict_X",
    "prepare_xgb_predict_X",
    "stage_order_from_map",
    "target_meta_only_on_disk",
    "target_ready_on_disk",
]
