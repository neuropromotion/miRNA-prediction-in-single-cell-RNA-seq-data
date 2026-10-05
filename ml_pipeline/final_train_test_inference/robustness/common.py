"""Shared data-loading and prediction plumbing for the pipeline.

Mirrors the import pattern used by the repository's own evaluation scripts
(the same ``sys.path`` manipulation) rather than going through the
higher-level ``StackPredictor`` wrapper, so this module can call ``predict``
repeatedly under different feature masks without re-selecting features or
reloading artifacts on every call.

Imports of the ``train/*.py`` modules (which pull in catboost/torch and only
exist inside a full repository checkout) are lazy: performed on first use,
inside the functions that need them, rather than at module import time. This
lets ``masking.py`` / ``importance.py`` / ``conformal.py`` and the test suite
import ``common`` and exercise the pure-Python logic (with
``predict_with_bundle`` monkeypatched) in an environment that has neither the
``train/`` package nor its ML dependencies installed.

Model loading. Several of the repository's own ``predict_*`` functions
reload the model from disk on every call, which is appropriate for a
one-shot evaluation but not for this pipeline, which calls
``predict_with_bundle`` hundreds of times per target across Stages 1-3
(masking-curve repeats, SAGE's per-permutation reveal steps, PFI's
per-gene-per-repeat shuffles, top-K validation, conformal calibration). For
model types whose save format is well understood, ``predict_with_bundle``
loads the model once per ``(model_name, target, device)`` and reuses the
in-memory object:
  - "tabm"    -> ``shared/tabm_wrapper.py::TabMBundle`` (public load/predict API).
  - "dcnv2"   -> ``train/dl_trainers.py`` DCNv2 artifacts (public load function,
                 module-local rebuild).
  - "tabpack" -> ``shared/tabpack_trainer.py``'s ensemble bundle (rebuild uses
                 that module's internal, non-public helper functions; falls
                 back to the validated, reload-per-call ``predict_tabpack``
                 path if those are renamed -- see ``_get_cached_tabpack``).
Any other model name falls back to the repository's own
``load_artifact``/``predict_one`` (correct for any model type
``train/model_trainers.py`` itself supports, without the extra caching).

Model-stack agnosticism. Which models make up the stack is not hardcoded
here. Each target's stack-weight JSON records its own ``models`` tuple
(``train/stack.py::FitResult.models``), which is the source of truth the
ridge coefficients were fit against, so ``predict_with_bundle`` iterates
``bundle.fit.models`` rather than a fixed list. If the training side adds,
removes, or renames a model, this pipeline adapts without changes to the
dispatch logic; a new fast-cache case is opt-in only.

Devices can be pinned per target/process (see
``load_target_bundle(..., device=...)``) so ``run_sparsity_robustness.py``
can distribute targets across multiple GPUs.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import robustness_paths as C  # noqa: E402

# Model names with a dedicated, disk-reload-avoiding cache in this module.
_FAST_CACHED_MODELS = ("tabm", "tabpack", "xgb_optuna")
MAX_CACHED_MODELS = int(os.environ.get("SPARSITY_MODEL_CACHE_SIZE", "8"))  # bounded LRU, see _ModelCache

_repo = {}  # populated lazily by _ensure_repo_imports(); see module docstring
_tabpack_cache_broken = False  # set True after the first failed fast-path attempt; see _get_cached_tabpack


def _ensure_repo_imports() -> dict:
    """Import the ``train/*.py`` (and ``shared/*.py``) modules on first use,
    using the same ``sys.path`` convention as the repository's own
    evaluation scripts.

    Kept aligned with the current production stack
    (``tabpack`` / ``tabm`` / ``xgb_optuna``) — see
    ``train/model_trainers.py`` and ``Optimal_K/predict_stack.py``. DCNv2 is
    no longer part of the stack; the optional fast-cache path for it is
    retained only as a no-op branch for older weight JSONs.
    """
    if _repo:
        return _repo

    ml_s = str(C.ML_PIPELINE)
    train_s = str(C.FTTI / "train")
    for p in (ml_s, train_s):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, ml_s)
    sys.path.insert(0, train_s)

    from stack import FitResult, apply_fit, fit_from_dict
    from data import select_features
    from impute import impute_k1_query, zero_fraction
    from io_splits import load_features
    from transforms import log2p1
    from model_trainers import load_artifact, model_dir, predict_one
    from constants import ENSEMBLE_ID, RESULTS, TABM_DIR

    tabm_dir_s = str(TABM_DIR)
    if tabm_dir_s not in sys.path:
        sys.path.insert(0, tabm_dir_s)
    from tabm_wrapper import TabMBundle

    _repo.update(
        FitResult=FitResult,
        apply_fit=apply_fit,
        fit_from_dict=fit_from_dict,
        select_features=select_features,
        impute_k1_query=impute_k1_query,
        zero_fraction=zero_fraction,
        load_features=load_features,
        log2p1=log2p1,
        load_artifact=load_artifact,
        model_dir=model_dir,
        predict_one=predict_one,
        ENSEMBLE_ID=ENSEMBLE_ID,
        RESULTS=RESULTS,
        TabMBundle=TabMBundle,
    )
    return _repo


def _weights_dir() -> Path:
    """Directory holding the stack ridge-weight JSONs. Mirrors
    ``inference/constants.py``'s own fallback (prefer a packaged
    ``final_train_test_inference/models/`` directory if populated, else the
    local ``train/results/`` training-run cache) without importing that
    module directly, since it is also named ``constants.py`` and would
    collide with ``train/constants.py`` under this repository's flat import
    convention."""
    repo = _ensure_repo_imports()
    ensemble_id = repo["ENSEMBLE_ID"]
    packaged = C.FTTI / "models" / "ensemble" / ensemble_id / "weights"
    if packaged.is_dir() and any(packaged.glob("*.json")):
        return packaged
    return repo["RESULTS"] / "ensemble" / ensemble_id / "weights"


def r2(y_true, y_pred) -> float:
    """Same metric as ``train/metrics.py::r2`` (scikit-learn R^2), reimplemented
    here so callers do not need the repository's ``train/`` package just to
    score predictions (e.g. inside the test suite)."""
    return float(r2_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(explicit: str | None = None) -> str:
    """Pick a device string, in priority order: explicit argument >
    ``$FINAL_DEVICE`` > first visible CUDA device > CPU."""
    if explicit:
        return explicit
    env = os.environ.get("FINAL_DEVICE")
    if env:
        return env
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def available_devices() -> list[str]:
    """All CUDA devices visible to this process, or ``["cpu"]`` if none."""
    try:
        import torch

        n = torch.cuda.device_count()
        if n > 0:
            return [f"cuda:{i}" for i in range(n)]
    except ImportError:
        pass
    return ["cpu"]


# ---------------------------------------------------------------------------
# Bounded LRU cache for loaded model objects (tabm / dcnv2 / tabpack)
# ---------------------------------------------------------------------------


class _ModelCache:
    """Bounded LRU cache, avoiding unbounded memory growth when one process
    works through many targets sequentially (e.g. a single joblib worker
    handling several targets from a FINAL_SHARD chunk)."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[tuple, object] = OrderedDict()

    def get_or_build(self, key: tuple, builder) -> object:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        value = builder()
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        self._data.clear()


_MODEL_CACHE = _ModelCache(MAX_CACHED_MODELS)


def clear_model_cache() -> None:
    """Drop in-memory model objects (call between targets in long runs)."""
    _MODEL_CACHE.clear()


def _get_cached_tabm(target: str, device: str):
    repo = _ensure_repo_imports()

    def _build():
        d = repo["model_dir"]("tabm", target)
        return repo["TabMBundle"].load(d, device=device)

    return _MODEL_CACHE.get_or_build(("tabm", target, device), _build)


def _get_cached_xgb(target: str):
    repo = _ensure_repo_imports()

    def _build():
        return repo["load_artifact"]("xgb_optuna", target)

    return _MODEL_CACHE.get_or_build(("xgb_optuna", target, "cpu"), _build)


def _ensure_tabpack_project_path() -> Path:
    """Put ``deps/tabpack/src`` on ``sys.path`` (provides ``project.*``).

    Mirrors ``shared.tabpack_trainer._ensure_tabpack_on_path`` without importing
    that module first (which itself may need ``project`` already resolvable for
    some call paths). Honours ``$TABPACK_ROOT``.
    """
    env = os.environ.get("TABPACK_ROOT", "").strip()
    root = Path(env) if env else (C.ML_PIPELINE / "deps" / "tabpack")
    src = root / "src"
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)
    return root


def _get_cached_tabpack(target: str, device: str):
    """Load and rebuild TabPack's greedy-ensemble ``inference_bundle.pt``
    once, then cache it. TabPack is an ensemble of up to
    ``TABPACK_N_MODELS`` (32 by default) sub-models, so calling the public
    ``shared.tabpack_trainer.predict_tabpack`` (which performs the full
    ``torch.load`` and ensemble rebuild on every call) hundreds of times per
    target would dominate this pipeline's runtime.

    Splitting "load and rebuild" from "forward pass" relies on that module's
    internal ``_build_ensemble_model``/``_transform_raw_x`` functions, which
    carry no API stability guarantee. If they are renamed or removed, a
    warning is logged once and the pipeline falls back to the slower,
    validated ``predict_tabpack`` path for the remainder of the process.
    """
    global _tabpack_cache_broken
    if _tabpack_cache_broken:
        return None

    repo = _ensure_repo_imports()
    _ensure_tabpack_project_path()

    def _build():
        import torch

        # Package-qualified import (not bare): shared/tabpack_trainer.py
        # itself does ``from shared.paths import ...``, so it must be loaded
        # as ``shared.tabpack_trainer`` (relying on ML_PIPELINE already being
        # on sys.path, added in _ensure_repo_imports).
        from shared import tabpack_trainer as tpk

        d = repo["model_dir"]("tabpack", target)
        bundle_path = Path(d) / tpk.INFERENCE_BUNDLE
        if not bundle_path.exists():
            meta_path = Path(d) / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                alt = Path(meta.get("exp_dir", "")) / tpk.INFERENCE_BUNDLE
                if alt.exists():
                    bundle_path = alt
        if not bundle_path.exists():
            raise FileNotFoundError(f"TabPack inference bundle missing under {d}")

        dev = device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        with tpk._suppress_sklearn_pickle_version_warnings():
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        model, weights, _ = tpk._build_ensemble_model(bundle, dev)
        return dict(bundle=bundle, model=model, weights=weights, device=dev, module=tpk)

    try:
        return _MODEL_CACHE.get_or_build(("tabpack", target, device), _build)
    except Exception as exc:  # noqa: BLE001 -- internal-API dependency, degrade gracefully
        print(
            f"[sparsity_robustness] WARNING: TabPack fast-cache path failed ({exc!r}); "
            "falling back to the slower (but correct) predict_tabpack() for the rest of this run."
        )
        _tabpack_cache_broken = True
        return None


def _predict_tabpack_cached(cached: dict, x: np.ndarray) -> np.ndarray:
    import torch

    tpk = cached["module"]
    x_num_np, x_cat_np = tpk._transform_raw_x(cached["bundle"], x)
    with torch.inference_mode():
        x_num_t = (
            None
            if x_num_np is None
            else torch.as_tensor(x_num_np, device=cached["device"], dtype=torch.get_default_dtype())
        )
        x_cat_t = (
            None
            if x_cat_np is None
            else torch.as_tensor(x_cat_np, device=cached["device"], dtype=torch.long)
        )
        y = cached["model"](x_num_t, x_cat_t).squeeze(-1).float()  # (P, B)
        w = torch.as_tensor(cached["weights"], device=y.device, dtype=y.dtype)
        w = w / w.sum()
        y = (y * w[:, None]).sum(0)
        stats = cached["bundle"].get("regression_label_stats")
        if stats is not None:
            y = y * float(stats["std"]) + float(stats["mean"])
        out = y.detach().cpu().numpy().astype(np.float64)
    return np.maximum(out, 0.0)


@dataclass
class TestSplit:
    name: str
    x: pd.DataFrame  # log2(TPM+1); raw (not imputed) for K1, as-is for pseudobulk cohorts
    y: pd.DataFrame


@dataclass
class TargetBundle:
    target: str
    cohort: str
    genes: list[str]
    fit: object  # train.stack.FitResult once _ensure_repo_imports() has run
    artifacts: dict[str, object] = field(default_factory=dict)  # one entry per non-fast-cached model in fit.models
    device: str = "cpu"  # where tabm/dcnv2/tabpack are (or will be) cached for this bundle


def load_target_config() -> dict:
    """Parse ``inference/prediction_config.json`` (schema: ``cohorts`` is a
    list of cohort names, and ``config[cohort]`` is ``{mirna: {"features":
    [...], "test_bulk": ..., "test_optimal_k": ...}}``) into the shape the
    rest of this module expects: ``{"eligible_mirs": [...], "cohorts":
    [...], "targets": {mirna: {"assigned_cohort": ..., "genes": [...]}}}``.

    Mirrors the corresponding branch of
    ``inference/constants.py::parse_prediction_config`` without importing
    that module directly, since it is also named ``constants.py`` and would
    collide with ``train/constants.py`` under this repository's flat import
    convention; the parsing itself is simple enough not to warrant the
    import-eviction workaround that would require.
    """
    raw = json.loads(C.CONFIG_PATH.read_text(encoding="utf-8"))
    eligible = list(raw["eligible_mirs"])
    cohorts = list(raw["cohorts"])
    targets: dict[str, dict] = {}
    for cohort in cohorts:
        for mir, info in (raw.get(cohort) or {}).items():
            genes = list(info.get("features") or info.get("genes") or [])
            entry = {"assigned_cohort": cohort, "genes": genes}
            if info.get("test_sc_r2") is not None:
                entry["test_sc_r2"] = float(info["test_sc_r2"])
            targets[mir] = entry
    missing = [t for t in eligible if t not in targets]
    if missing:
        raise KeyError(
            f"{len(missing)} eligible_mirs not found under any cohort block in "
            f"{C.CONFIG_PATH} (e.g. {missing[:5]}) -- config may be stale or corrupt."
        )
    return {"eligible_mirs": eligible, "cohorts": cohorts, "targets": targets}


def load_sc_features() -> dict[str, list[str]]:
    """Per-target SC-only gene panels from ``sc_features.json``.

    Masking, importance, and minimal-set search operate only on these genes;
    non-SC genes in the trained ``selected_features`` panel stay untouched.
    """
    if not C.SC_FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"SC feature panels missing: {C.SC_FEATURES_PATH}. "
            "Expected final_train_test_inference/sc_features.json."
        )
    raw = json.loads(C.SC_FEATURES_PATH.read_text(encoding="utf-8"))
    return {str(k): list(v) for k, v in raw.items()}


def sc_genes_for_target(target: str, panel_genes: list[str], sc_features: dict[str, list[str]]) -> list[str]:
    """SC genes that also appear in this target's trained panel (preserve panel order)."""
    sc_set = set(sc_features.get(target) or [])
    if not sc_set:
        raise KeyError(f"{target!r} missing from {C.SC_FEATURES_PATH}")
    return [g for g in panel_genes if g in sc_set]


def _align_xy(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = x.index.intersection(y.index)
    return x.loc[common_idx], y.loc[common_idx]


_K1_REF_CACHE = Path(
    os.environ.get("ROBUSTNESS_K1_CACHE", "/tmp/mir_robustness_k1_ref.pkl")
)


def load_k1_reference() -> pd.DataFrame:
    """KNN donor pool for K1 re-imputation (the same reference used at train/eval time)."""
    if not C.K1_REF_PATH.is_file():
        raise FileNotFoundError(
            f"KNN reference missing: {C.K1_REF_PATH}. "
            "Run data/prepare_splits/prepare_stage00_splits.py or copy prepared splits."
        )
    mtime = C.K1_REF_PATH.stat().st_mtime
    if _K1_REF_CACHE.is_file():
        try:
            with _K1_REF_CACHE.open("rb") as f:
                stored_mtime, df = pickle.load(f)
            if stored_mtime == mtime:
                return df
        except Exception:
            pass
    df = pd.read_parquet(C.K1_REF_PATH)
    try:
        with _K1_REF_CACHE.open("wb") as f:
            pickle.dump((mtime, df), f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass
    return df


def load_eval_positions() -> list[int] | None:
    """Row positions (shared across K1 and all pseudobulk cohorts, since
    ``Optimal_K`` splits by shared row position across single-cell cohorts)
    belonging to the eval half of the TEST tune/eval split
    (``Optimal_K/splits.py``) -- the half not used for eligibility/optimal-K
    selection, and therefore the leakage-safe half for this pipeline's own
    analysis.

    Returns ``None`` (callers fall back to the full test set, with a
    warning) if ``Optimal_K`` has not been run and ``test_split.json`` is
    not present.
    """
    if not C.SPLIT_PATH.is_file():
        return None
    split = json.loads(C.SPLIT_PATH.read_text(encoding="utf-8"))
    return list(split["sc"]["eval_idx"])


_EVAL_SPLITS_CACHE = Path(
    os.environ.get("ROBUSTNESS_EVAL_CACHE", "/tmp/mir_robustness_eval_splits.pkl")
)


def _eval_splits_fingerprint(use_full: bool) -> tuple:
    paths = [
        C.SPLIT_PATH,
        C.SC_TEST / "X_TEST_K1.parquet",
        C.SC_TEST / "Y_TEST_K1.parquet",
    ]
    for cohort in C.PB_COHORTS:
        paths.append(C.SC_TEST / f"X_TEST_PB_{cohort}.parquet")
        paths.append(C.SC_TEST / f"Y_TEST_PB_{cohort}.parquet")
    return (use_full, tuple((str(p), p.stat().st_mtime if p.is_file() else 0) for p in paths))


def load_test_splits() -> dict[str, TestSplit]:
    """Held-out test matrices per cohort (K1..K10).

    By default (``paths.USE_FULL_TEST``) returns the **full** TEST matrices so
    Stage 1/2 / minimal-set have enough samples. If ``USE_FULL_TEST`` is False
    and ``test_split.json`` exists, restricts to the Optimal_K eval half.

    K1 is returned raw (not yet KNN-imputed) so callers can mask specific
    entries before choosing whether and how to re-impute; pseudobulk cohorts
    are never imputed in this pipeline, matching production behaviour, and
    are returned as-is.

    Cached on local disk (``/tmp``) so one-target-per-batch loops do not
    re-read large parquets from slow external volumes every iteration.
    """
    use_full = bool(getattr(C, "USE_FULL_TEST", True))
    fp = _eval_splits_fingerprint(use_full)
    if _EVAL_SPLITS_CACHE.is_file():
        try:
            with _EVAL_SPLITS_CACHE.open("rb") as f:
                stored_fp, out = pickle.load(f)
            if stored_fp == fp:
                print(
                    f"[robustness] eval splits from cache {_EVAL_SPLITS_CACHE}",
                    flush=True,
                )
                return out
        except Exception:
            pass

    out = _load_test_splits_from_disk(use_full)
    try:
        with _EVAL_SPLITS_CACHE.open("wb") as f:
            pickle.dump((fp, out), f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass
    return out


def _load_test_splits_from_disk(use_full: bool) -> dict[str, TestSplit]:
    if not C.SC_TEST.is_dir():
        raise FileNotFoundError(
            f"sc_TEST not found under {C.SC_TEST}. "
            "Place X_TEST_*.parquet / Y_TEST_*.parquet there."
        )
    log2p1 = _ensure_repo_imports()["log2p1"]
    eval_pos = None if use_full else load_eval_positions()
    if use_full:
        print(
            "[robustness] Using FULL sc_TEST "
            "(tune+eval). Optimal_K still owns the tune half for eligibility/K."
        )
    elif eval_pos is None:
        print(
            f"[robustness] WARNING: {C.SPLIT_PATH} not found -- using the FULL test "
            "set (including rows already used for Optimal_K's eligibility/K selection). "
            "Run Optimal_K first (or copy its results/test_split.json here) to avoid double-dipping."
        )
    else:
        print(
            f"[robustness] Using Optimal_K eval half only ({len(eval_pos)} row positions)."
        )

    def _restrict(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if eval_pos is None:
            return x, y
        pos = [p for p in eval_pos if p < len(x)]
        return x.iloc[pos], y.iloc[pos]

    out: dict[str, TestSplit] = {}

    x_k1 = log2p1(pd.read_parquet(C.SC_TEST / "X_TEST_K1.parquet"))
    y_k1 = log2p1(pd.read_parquet(C.SC_TEST / "Y_TEST_K1.parquet"))
    x_k1, y_k1 = _align_xy(x_k1, y_k1)
    x_k1, y_k1 = _restrict(x_k1, y_k1)
    out["K1"] = TestSplit("K1", x_k1, y_k1)

    for cohort in C.PB_COHORTS:
        x = log2p1(pd.read_parquet(C.SC_TEST / f"X_TEST_PB_{cohort}.parquet"))
        y = log2p1(pd.read_parquet(C.SC_TEST / f"Y_TEST_PB_{cohort}.parquet"))
        x, y = _align_xy(x, y)
        x, y = _restrict(x, y)
        out[cohort] = TestSplit(cohort, x, y)
    return out


def load_target_bundle(target: str, target_info: dict, device: str | None = None) -> TargetBundle:
    repo = _ensure_repo_imports()
    info = target_info[target]
    weight_path = _weights_dir() / f"{target}.json"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing stack weights: {weight_path}")
    fit = repo["fit_from_dict"](json.loads(weight_path.read_text(encoding="utf-8")))
    resolved_device = resolve_device(device)
    # Eagerly load an artifact for every model this target's stack actually
    # uses (bundle.fit.models -- see module docstring), except the ones with
    # their own dedicated cache (tabm/dcnv2/tabpack, loaded lazily on first
    # predict). load_artifact() is cheap and correct to call eagerly for
    # anything else: for a model whose artifact is a ready in-memory object
    # it is reused as-is; for anything that returns a path, predict_one()
    # knows how to consume it (without the extra caching).
    artifacts = {
        m: repo["load_artifact"](m, target) for m in fit.models if m not in _FAST_CACHED_MODELS
    }
    return TargetBundle(
        target=target,
        cohort=str(info["assigned_cohort"]),
        genes=list(info["genes"]),
        fit=fit,
        artifacts=artifacts,
        device=resolved_device,
    )


def predict_with_bundle(bundle: TargetBundle, x: pd.DataFrame, k1_ref: pd.DataFrame | None = None) -> np.ndarray:
    """Stack prediction for an arbitrary (already gene-selected or not) matrix.

    Iterates ``bundle.fit.models`` -- this target's own recorded model list
    -- rather than a hardcoded stack, so a change to which models are in the
    ensemble works without modification here. "tabm"/"tabpack"/"xgb_optuna"
    go through in-memory caches; every other name falls back to the
    repository's own ``predict_one``.

    ``k1_ref``: K1 re-imputation is always applied here, via
    ``maybe_reimpute`` (a no-op for non-K1 cohorts), rather than left to
    each caller.
    """
    # Pin model_trainers.DEVICE for any fallback predict_one() calls.
    os.environ.setdefault("FINAL_DEVICE", bundle.device)

    # Restrict K1 KNN re-imputation to this target's panel. Full-transcriptome
    # impute (~17k genes) on every predict is prohibitively slow for Stage 1/2
    # sweeps; panel space matches what the stack actually consumes.
    x = maybe_reimpute(bundle.cohort, x, k1_ref, genes=bundle.genes)
    repo = _ensure_repo_imports()
    x_sel = repo["select_features"](x, bundle.genes).to_numpy(dtype=np.float32)
    models = bundle.fit.models

    preds = []
    for m in models:
        if m == "tabm":
            tabm_bundle = _get_cached_tabm(bundle.target, bundle.device)
            preds.append(tabm_bundle.predict(x_sel))
        elif m == "tabpack":
            cached = _get_cached_tabpack(bundle.target, bundle.device)
            if cached is not None:
                preds.append(_predict_tabpack_cached(cached, x_sel))
            else:
                artifact = bundle.artifacts.get(m) or repo["load_artifact"](m, bundle.target)
                preds.append(repo["predict_one"](m, artifact, x_sel))
        elif m == "xgb_optuna":
            xgb_model = _get_cached_xgb(bundle.target)
            preds.append(repo["predict_one"](m, xgb_model, x_sel))
        else:
            artifact = bundle.artifacts.get(m) or repo["load_artifact"](m, bundle.target)
            preds.append(repo["predict_one"](m, artifact, x_sel))

    mat = np.column_stack(preds)
    return repo["apply_fit"](bundle.fit, mat, models)


def maybe_reimpute(
    cohort: str,
    x: pd.DataFrame,
    k1_ref: pd.DataFrame | None,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Apply KNN-based re-imputation for K1 only (pseudobulk cohorts are never imputed, matching production).

    If ``genes`` is given, both query and reference are restricted to that
    column set before imputation (robustness default: the target's selected
    feature panel). Neighbours are then found in panel space rather than the
    full transcriptome — a deliberate speed/accuracy trade-off for repeated
    masking/importance predicts.
    """
    if cohort == "K1" and k1_ref is not None:
        repo = _ensure_repo_imports()
        if genes:
            cols = list(dict.fromkeys(genes))
            x_use = x.reindex(columns=cols, fill_value=0.0)
            ref_use = k1_ref.reindex(columns=cols, fill_value=0.0)
            return repo["impute_k1_query"](x_use, ref_use)
        return repo["impute_k1_query"](x, k1_ref)
    return x


__all__ = [
    "TestSplit",
    "TargetBundle",
    "load_target_config",
    "load_sc_features",
    "sc_genes_for_target",
    "load_k1_reference",
    "load_eval_positions",
    "load_test_splits",
    "load_target_bundle",
    "predict_with_bundle",
    "maybe_reimpute",
    "resolve_device",
    "available_devices",
    "r2",
]
