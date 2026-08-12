"""Shared TEST tune/eval split (positional indices).

SC cohorts (K1, PB_K*) have equal length but different sample IDs, so we split
by row position with one RNG draw and apply the same iloc mask to every cohort.
Bulk has its own n → separate permutation with the same seed.
Persisted JSON is the contract for test_metrics (eval half).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import SEED, SPLIT_FRAC_TUNE, SPLIT_PATH


def make_half_split(n: int, *, seed: int, frac_tune: float) -> dict[str, list[int]]:
    if n < 2:
        raise ValueError(f"need n≥2 to split, got {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tune = int(round(n * frac_tune))
    n_tune = min(max(n_tune, 1), n - 1)
    tune = sorted(int(i) for i in perm[:n_tune])
    eval_idx = sorted(int(i) for i in perm[n_tune:])
    return {"tune_idx": tune, "eval_idx": eval_idx}


def build_test_split(n_sc: int, n_bulk: int, *, seed: int = SEED, frac_tune: float = SPLIT_FRAC_TUNE) -> dict[str, Any]:
    sc = make_half_split(n_sc, seed=seed, frac_tune=frac_tune)
    # Offset bulk seed stream so SC/bulk permutations are independent but reproducible.
    bulk = make_half_split(n_bulk, seed=seed + 1_000_003, frac_tune=frac_tune)
    return {
        "seed": seed,
        "frac_tune": frac_tune,
        "sc": {"n": n_sc, **sc},
        "bulk": {"n": n_bulk, **bulk},
        "note": (
            "SC tune/eval indices are positional (iloc) and shared across K1/PB cohorts. "
            "Bulk has its own positional split."
        ),
    }


def save_split(split: dict[str, Any], path: Path = SPLIT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return path


def load_split(path: Path = SPLIT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_split(n_sc: int, n_bulk: int, *, force: bool = False) -> dict[str, Any]:
    if SPLIT_PATH.is_file() and not force:
        split = load_split()
        if split["sc"]["n"] == n_sc and split["bulk"]["n"] == n_bulk:
            return split
    split = build_test_split(n_sc, n_bulk)
    save_split(split)
    return split
