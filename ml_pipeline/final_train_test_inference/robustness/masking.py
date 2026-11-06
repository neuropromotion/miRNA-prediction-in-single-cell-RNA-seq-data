"""Zero-out masks for SC genes (columns in TEST matrices)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zero_genes(x: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    cols = [g for g in genes if g in x.columns]
    if not cols:
        return x
    out = x.copy()
    out.loc[:, cols] = 0.0
    return out


def random_subset_at_budget(
    genes: list[str],
    weights: dict[str, float],
    budget: float,
    rng: np.random.Generator,
) -> list[str]:
    """Random permutation; take maximal prefix with cumulative weight <= budget."""
    if budget <= 0 or not genes:
        return []
    order = list(genes)
    rng.shuffle(order)
    chosen: list[str] = []
    total = 0.0
    for g in order:
        w = float(weights.get(g, 0.0))
        if total + w > budget + 1e-12:
            break
        chosen.append(g)
        total += w
    return chosen
