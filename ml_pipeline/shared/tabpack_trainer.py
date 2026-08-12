"""TabPack trainer for model_selection / final_train (official yandex-research/tabpack).

Requires Python >=3.12 and deps/tabpack uv venv. Creates a temporary TabPack
dataset (train/val/test=outer concat), trains with online greedy ensemble, and
returns an artifact with predictions for inner_val + all outer_val cohorts.

Also persists ``inference_bundle.pt`` (ensemble member weights + preprocessor)
so live ``predict(x)`` works for production / TEST.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from shared.paths import ML_PIPELINE, SEED

DEFAULT_N_MODELS = int(os.environ.get("TABPACK_N_MODELS", "32"))
# paper: https://arxiv.org/pdf/2607.05380 — MuonAdamWPack + muon_lr in search space
DEFAULT_PROTOCOL = os.environ.get("TABPACK_PROTOCOL", "screen").strip().lower()

INFERENCE_BUNDLE = "inference_bundle.pt"


@contextlib.contextmanager
def _suppress_sklearn_pickle_version_warnings() -> Iterator[None]:
    """Bundles were fit under sklearn 1.5.x; host venv may be 1.7.x.

    Predictions are fine; InconsistentVersionWarning just floods notebook output.
    """
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # pragma: no cover
        yield
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        yield


def _tabpack_root() -> Path:
    env = os.environ.get("TABPACK_ROOT", "").strip()
    if env:
        return Path(env)
    return ML_PIPELINE / "deps" / "tabpack"


def _write_dataset(
    ds_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    outer_parts: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, tuple[int, int]]:
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "splits" / "default").mkdir(parents=True, exist_ok=True)

    xs = [np.asarray(x_train, dtype=np.float32), np.asarray(x_val, dtype=np.float32)]
    ys = [
        np.asarray(y_train, dtype=np.float32).reshape(-1),
        np.asarray(y_val, dtype=np.float32).reshape(-1),
    ]
    for _, x, y in outer_parts:
        xs.append(np.asarray(x, dtype=np.float32))
        ys.append(np.asarray(y, dtype=np.float32).reshape(-1))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    n_tr, n_va = len(xs[0]), len(xs[1])
    n_te = sum(len(a) for a in xs[2:])

    np.save(ds_dir / "x_num.npy", X)
    np.save(ds_dir / "y.npy", y)
    np.save(ds_dir / "splits" / "default" / "train.npy", np.arange(0, n_tr, dtype=np.int32))
    np.save(ds_dir / "splits" / "default" / "val.npy", np.arange(n_tr, n_tr + n_va, dtype=np.int32))
    np.save(
        ds_dir / "splits" / "default" / "test.npy",
        np.arange(n_tr + n_va, n_tr + n_va + n_te, dtype=np.int32),
    )
    (ds_dir / "info.json").write_text(
        json.dumps({"task": {"type": "regression", "score": "r2"}, "name": ds_dir.name}, indent=2),
        encoding="utf-8",
    )
    (ds_dir / "READY").write_text("")

    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, x, _ in outer_parts:
        n = len(x)
        slices[name] = (cursor, cursor + n)
        cursor += n
    (ds_dir / "outer_slices.json").write_text(json.dumps(slices), encoding="utf-8")
    return slices


def _config(rel_data_path: str, n_models: int, *, protocol: str = "screen") -> dict:
    """Build TabPack config.

    protocol='paper' matches deps/tabpack/experiments/tabpack/make.py
    (MuonAdamWPack + muon_lr search). protocol='screen' keeps the lighter
    AdamWPack setup used in model_selection.
    """
    use_paper = protocol in ("paper", "muon", "tabpack_paper")
    opt_space = {
        "lr": ["_tune_", "loguniform", 0.0001, 0.005],
        "weight_decay": ["_tune_", "loguniform", 0.001, 1.0],
    }
    if use_paper:
        opt_space["muon_lr"] = ["_tune_", "loguniform", 0.001, 0.1]
    return {
        "seed": SEED,
        "data": {
            "path": rel_data_path,
            "num_policy": "noisy-quantile",
            "extract_bin_from_num": True,
            "bin_policy": "convert-to-cat",
            "cache": False,
        },
        "n_models": n_models,
        "model": {"activation": "ReLU", "d_block": 384},
        "optimizer": {
            "type": "MuonAdamWPack" if use_paper else "AdamWPack",
            "shared_step": True,
        },
        "batch_size": 256,
        "n_epochs": -1,
        "patience": 16,
        "online_ensembles": {
            "greedy": {
                "type": "greedy",
                "update_type": "latest",
                "include_current_ensemble_in_pool": True,
                "patience": 32,
                "options": {"max_ensemble_size": min(32, n_models)},
            }
        },
        "sampler": {
            "type": "RandomSampler",
            "space": {
                "model": {
                    "n_blocks": ["_tune_", "int", 1, 4],
                    "dropout": ["_tune_", "?uniform", 0.0, 0.0, 0.5],
                },
                "optimizer": opt_space,
            },
        },
        "amp_dtype": "bfloat16",
        "save_all_predictions": False,
        "track_online_ensemble_history": False,
        "track_experiments": True,
        "save_final_predictions": True,
    }


def _ensemble_preds(exp_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Average greedy-ensemble member preds.

    TabPack's online greedy report can keep model ids that never landed in the
    final ``experiments`` / ``predictions.npz`` (pruned / not persisted). Use
    the intersection; if empty, fall back to the mean of all finished models.
    """
    rep = json.loads((exp_dir / "report.json").read_text(encoding="utf-8"))
    ids = list(rep["online_ensembles"]["greedy"]["report"]["ids"])
    order = [e["report"]["id"] for e in rep.get("experiments", [])]
    row = {mid: i for i, mid in enumerate(order)}
    missing = [i for i in ids if i not in row]
    available = [i for i in ids if i in row]
    if missing and available:
        print(
            f"[tabpack] warning: greedy ids missing from experiments, "
            f"skipping {missing}; using {len(available)}/{len(ids)} members",
            flush=True,
        )
        ids = available
    elif missing and not available:
        if not order:
            raise RuntimeError(
                f"ensemble ids not finished: {missing}; no finished experiments either"
            )
        print(
            f"[tabpack] warning: greedy ids {missing} all missing; "
            f"falling back to mean of {len(order)} finished models",
            flush=True,
        )
        ids = list(order)
    preds = np.load(exp_dir / "predictions.npz")
    idx = [row[i] for i in ids]
    val = np.maximum(preds["val"][idx].mean(0), 0.0)
    test = np.maximum(preds["test"][idx].mean(0), 0.0)
    return val.astype(np.float64), test.astype(np.float64), ids


def _ensure_tabpack_on_path(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _transform_raw_x(bundle: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply saved bin-extract + noisy-quantile pipeline to raw numeric X."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected X shape (N, F), got {x.shape}")

    bin_ext = bundle.get("bin_extractor")
    x_cat = None
    x_num = x
    if bin_ext is not None and bin_ext.get("encoder") is not None:
        bin_idx = np.asarray(bin_ext["bin_idx"])
        bin_mask = np.asarray(bin_ext["bin_mask"])
        enc = bin_ext["encoder"]
        x_bin = enc.transform(x[:, bin_idx]).astype(np.int64)
        x_cat = x_bin
        if bin_mask.all():
            x_num = None
        else:
            x_num = x[:, ~bin_mask]

    num_pp = bundle.get("num_preprocessor")
    if x_num is not None and num_pp is not None:
        normalizer = num_pp.get("normalizer")
        keep_mask = np.asarray(num_pp["keep_mask"])
        if normalizer is not None:
            with _suppress_sklearn_pickle_version_warnings():
                x_num = normalizer.transform(x_num)
        x_num = np.nan_to_num(x_num)
        x_num = x_num[:, keep_mask].astype(np.float32)

    if x_num is not None and x_num.shape[1] != int(bundle["n_num_features"]):
        raise ValueError(
            f"n_num_features mismatch after transform: "
            f"{x_num.shape[1]} vs bundle {bundle['n_num_features']}"
        )
    cats = bundle.get("cat_cardinalities") or []
    if x_cat is not None and len(cats) and x_cat.shape[1] != len(cats):
        raise ValueError(
            f"n_cat_features mismatch: {x_cat.shape[1]} vs {len(cats)}"
        )
    return x_num, x_cat


def _build_ensemble_model(bundle: dict[str, Any], device: str):
    import torch
    import project.nn as project_nn
    import project.tabpack as tabpack

    ens = bundle.get("ensemble") or {}
    ens_ids = [int(i) for i in ens.get("ids") or []]
    if not ens_ids:
        ens_ids = sorted(int(k) for k in bundle["members"].keys())
    unique, counts = np.unique(np.asarray(ens_ids, dtype=np.int64), return_counts=True)
    members = bundle["members"]
    missing = [int(i) for i in unique if int(i) not in members]
    if missing:
        raise KeyError(f"ensemble members missing from bundle: {missing}")

    model_meta = dict(bundle["model"])
    n_blocks = [int(members[int(i)]["config"]["model"]["n_blocks"]) for i in unique]
    dropout = []
    for i in unique:
        d = members[int(i)]["config"]["model"].get("dropout")
        if d is None:
            d = model_meta.get("dropout", 0.0)
            if isinstance(d, list):
                d = 0.0
        dropout.append(float(d))

    kwargs = {
        "n_num_features": int(bundle["n_num_features"]),
        "cat_cardinalities": list(bundle.get("cat_cardinalities") or []),
        "n_classes": bundle.get("n_classes"),
        "pack_size": len(unique),
        "n_blocks": n_blocks,
        "max_n_blocks": int(model_meta["max_n_blocks"]),
        "d_block": int(model_meta["d_block"]),
        "dropout": dropout,
        "activation": model_meta.get("activation", "ReLU"),
    }
    if model_meta.get("max_d_block") is not None:
        kwargs["max_d_block"] = int(model_meta["max_d_block"])
    if model_meta.get("num_embeddings") is not None:
        kwargs["num_embeddings"] = model_meta["num_embeddings"]

    model = tabpack.ModelPack(**kwargs).to(device)
    stacked: dict[str, Any] = {}
    keys = members[int(unique[0])]["state_dict"].keys()
    for k in keys:
        stacked[k] = torch.stack(
            [members[int(i)]["state_dict"][k] for i in unique], dim=0
        ).to(device)
    pack_idx = torch.arange(len(unique), device=device)
    project_nn.module_pack_load_state_dict(model, stacked, pack_idx=pack_idx)
    model.eval()
    weights = counts.astype(np.float64)
    return model, weights, unique


def predict_tabpack(
    model_dir: Path,
    x: np.ndarray,
    *,
    device: str | None = None,
    clip_nonneg: bool = True,
) -> np.ndarray:
    """Live predict from ``model_dir/inference_bundle.pt``."""
    import torch

    root = _tabpack_root()
    _ensure_tabpack_on_path(root)
    bundle_path = Path(model_dir) / INFERENCE_BUNDLE
    if not bundle_path.exists():
        # Fall back to experiment symlink / meta.exp_dir
        meta_path = Path(model_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            alt = Path(meta.get("exp_dir", "")) / INFERENCE_BUNDLE
            if alt.exists():
                bundle_path = alt
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"TabPack inference bundle missing under {model_dir} "
            f"(expected {INFERENCE_BUNDLE})"
        )

    device = device or os.environ.get("FINAL_DEVICE", os.environ.get("STAGE04_DEVICE", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with _suppress_sklearn_pickle_version_warnings():
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    x_num_np, x_cat_np = _transform_raw_x(bundle, x)
    model, weights, _ = _build_ensemble_model(bundle, device)

    with torch.inference_mode():
        x_num_t = (
            None
            if x_num_np is None
            else torch.as_tensor(x_num_np, device=device, dtype=torch.get_default_dtype())
        )
        x_cat_t = (
            None
            if x_cat_np is None
            else torch.as_tensor(x_cat_np, device=device, dtype=torch.long)
        )
        y = model(x_num_t, x_cat_t).squeeze(-1).float()  # (P, B)
        w = torch.as_tensor(weights, device=y.device, dtype=y.dtype)
        w = w / w.sum()
        y = (y * w[:, None]).sum(0)
        stats = bundle.get("regression_label_stats")
        if stats is not None:
            y = y * float(stats["std"]) + float(stats["mean"])
        out = y.detach().cpu().numpy().astype(np.float64)
    if clip_nonneg:
        out = np.maximum(out, 0.0)
    return out


def train_tabpack_screen(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    outer_parts: list[tuple[str, np.ndarray, np.ndarray]],
    model_dir: Path,
    *,
    target: str,
    n_models: int | None = None,
    protocol: str | None = None,
    experiment_namespace: str = "mirna_screen",
) -> dict[str, Any]:
    """Train TabPack; return artifact with val/outer preds + inference bundle."""
    root = _tabpack_root()
    if not (root / "src" / "project" / "tabpack.py").exists():
        raise FileNotFoundError(f"TabPack repo missing: {root}")

    protocol = (protocol or DEFAULT_PROTOCOL).strip().lower()
    n_models = int(n_models if n_models is not None else DEFAULT_N_MODELS)
    model_dir.mkdir(parents=True, exist_ok=True)
    safe = target.replace("/", "_")
    ds_dir = model_dir / "dataset"
    slices = _write_dataset(ds_dir, x_train, y_train, x_val, y_val, outer_parts)

    _ensure_tabpack_on_path(root)

    old_cwd = Path.cwd()
    os.chdir(root)
    try:
        try:
            from loguru import logger as _loguru

            _loguru.remove()
            _loguru.add(sys.stderr, level="WARNING")
        except Exception:
            pass
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        import lib.experiment as lib_experiment
        import lib.utils as lib_utils
        import project.tabpack as tabpack

        lib_utils.init()

        link_parent = root / "data" / experiment_namespace
        link_parent.mkdir(parents=True, exist_ok=True)
        link = link_parent / safe
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(ds_dir.resolve())
        rel = f"data/{experiment_namespace}/{safe}"

        cfg = _config(rel, n_models, protocol=protocol)
        exp_dir = root / "experiments" / experiment_namespace / safe / "main"
        lib_experiment.create(exp_dir, config=cfg, parents=True, force=True)
        (exp_dir / "_RUNNING").touch()
        try:
            report = tabpack.main(cfg, exp_dir)
        except RuntimeError as exc:
            if "already done" not in str(exc):
                raise
            report = lib_experiment.load_report(exp_dir)
        _ = report
        val_pred, test_pred, ids = _ensemble_preds(exp_dir)
    finally:
        os.chdir(old_cwd)

    # Copy inference bundle next to preds for production load.
    src_bundle = Path(exp_dir) / INFERENCE_BUNDLE
    dst_bundle = model_dir / INFERENCE_BUNDLE
    if src_bundle.exists():
        shutil.copy2(src_bundle, dst_bundle)
    else:
        raise FileNotFoundError(
            f"TabPack did not write {INFERENCE_BUNDLE} under {exp_dir}"
        )

    outer_preds = {name: test_pred[a:b].copy() for name, (a, b) in slices.items()}
    artifact = {
        "kind": "tabpack",
        "target": target,
        "val_pred": val_pred,
        "outer_preds": outer_preds,
        "ensemble_ids": ids,
        "n_models": n_models,
        "protocol": protocol,
        "exp_dir": str(exp_dir),
        "n_features": int(x_train.shape[1]),
        "model_dir": str(model_dir),
        "has_inference_bundle": True,
    }
    np.savez_compressed(
        model_dir / "preds.npz",
        val_pred=val_pred,
        **{f"outer_{k}": v for k, v in outer_preds.items()},
    )
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "kind": "tabpack",
                "target": target,
                "ensemble_ids": ids,
                "n_models": n_models,
                "protocol": protocol,
                "exp_dir": str(exp_dir),
                "outer_keys": list(outer_preds.keys()),
                "n_features": int(x_train.shape[1]),
                "has_inference_bundle": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    exp_link = model_dir / "experiment"
    if exp_link.exists() or exp_link.is_symlink():
        exp_link.unlink()
    exp_link.symlink_to(Path(artifact["exp_dir"]))
    return artifact


def load_tabpack_screen(model_dir: Path) -> dict[str, Any]:
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    z = np.load(model_dir / "preds.npz")
    outer = {k[len("outer_") :]: z[k] for k in z.files if k.startswith("outer_")}
    return {
        "kind": "tabpack",
        "target": meta["target"],
        "val_pred": np.asarray(z["val_pred"], dtype=np.float64),
        "outer_preds": {k: np.asarray(v, dtype=np.float64) for k, v in outer.items()},
        "ensemble_ids": meta.get("ensemble_ids", []),
        "n_models": meta.get("n_models"),
        "exp_dir": meta.get("exp_dir"),
        "n_features": meta.get("n_features"),
        "model_dir": str(model_dir),
        "has_inference_bundle": (Path(model_dir) / INFERENCE_BUNDLE).exists()
        or bool(meta.get("has_inference_bundle")),
    }
