"""Merge bulk and SC feature lists (full union, no cap)."""

from __future__ import annotations


def union_full(sc_genes: list[str], bulk_genes: list[str]) -> tuple[list[str], dict]:
    """SC genes first, then bulk-only; no final cap."""
    sc_unique = list(dict.fromkeys(sc_genes))
    sc_set = set(sc_unique)
    bulk_only = [g for g in dict.fromkeys(bulk_genes) if g not in sc_set]
    merged = sc_unique + bulk_only
    overlap = len(sc_set & set(bulk_genes))
    meta = {
        "n_sc": len(sc_unique),
        "n_bulk": len(dict.fromkeys(bulk_genes)),
        "n_overlap": overlap,
        "n_bulk_only": len(bulk_only),
        "n_final": len(merged),
    }
    return merged, meta
