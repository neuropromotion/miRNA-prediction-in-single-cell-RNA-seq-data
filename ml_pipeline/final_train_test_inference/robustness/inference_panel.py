"""Absent SC features in AnnData (no model inference)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import robustness_paths as C

INFERENCE_DIR = C.INFERENCE_DIR


def _ensure_inference_path() -> None:
    inf = str(INFERENCE_DIR)
    if inf not in sys.path:
        sys.path.insert(0, inf)


_ADATA_CACHE: dict[str, object] = {}


def load_adata(path: Path):
    _ensure_inference_path()
    import anndata as ad

    key = str(path.resolve())
    if key not in _ADATA_CACHE:
        _ADATA_CACHE[key] = ad.read_h5ad(path)
    return _ADATA_CACHE[key]


def absent_genes_and_stats(
    adata,
    panel: list[str],
    *,
    mapping_path: Path | None = None,
) -> tuple[list[str], dict]:
    """Return absent ENSG list (missing or all-zero) and count stats."""
    _ensure_inference_path()
    from mirpredictor import miRPredictor

    mp = miRPredictor(device="cpu", preload_models=False, log=False)
    measured = mp._measured_ensg_from_adata(adata, mapping_path=mapping_path)
    stats = mp._absent_feature_stats(measured, panel)

    norm = miRPredictor._normalize_ensg
    panel_n = [norm(g) for g in panel]
    present = set(map(norm, measured.index.astype(str)))
    nz = (measured != 0).any(axis=1)
    nonzero = {norm(g) for g, flag in nz.items() if bool(flag)}
    absent = [g for g in panel_n if g not in present or g not in nonzero]
    return absent, stats


def assess_dataset(
    adata_path: Path,
    dataset_name: str,
    mirna: str,
    panel: list[str],
    weights: dict[str, float],
    importance_budget: float,
    *,
    mapping_path: Path | None = None,
) -> dict:
    norm = lambda g: str(g).split(".", 1)[0] if str(g).startswith("ENSG") and "." in str(g) else str(g)
    wmap = {norm(k): float(v) for k, v in weights.items()}

    if not adata_path.is_file():
        return {
            "dataset": dataset_name,
            "mirna": mirna,
            "status": "missing_h5ad",
            "h5ad_path": str(adata_path),
        }

    adata = load_adata(adata_path)
    absent, stats = absent_genes_and_stats(adata, panel, mapping_path=mapping_path)
    absent_set = {norm(g) for g in absent}
    w_absent = sum(wmap.get(g, 0.0) for g in absent_set)
    # Genes absent but not in weight table: conservative mean weight among panel.
    unknown = [g for g in absent_set if g not in wmap and g in {norm(p) for p in panel}]
    if unknown and wmap:
        mean_w = sum(wmap.values()) / len(wmap)
        w_absent += len(unknown) * mean_w

    budget = float(importance_budget)
    frac = (w_absent / budget) if budget > 1e-9 else float("nan")
    return {
        "dataset": dataset_name,
        "mirna": mirna,
        "status": "ok",
        "h5ad_path": str(adata_path),
        "n_cells": int(adata.n_obs),
        "n_sc_features": int(stats["n_features"]),
        "n_absent": int(stats["n_absent"]),
        "pct_absent": float(stats["pct_absent"]),
        "absent_importance": float(w_absent),
        "importance_budget": budget,
        "budget_used_fraction": frac,
        "passes_importance_budget": bool(w_absent <= budget) if budget > 0 else False,
        "n_absent_genes_listed": len(absent),
    }
