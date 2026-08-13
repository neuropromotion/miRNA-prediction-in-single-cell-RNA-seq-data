"""Merge bulk/SC feature lists with SC-first cap."""

from __future__ import annotations


def union_sc_priority(
    sc_genes: list[str],
    bulk_genes: list[str],
    max_features: int,
) -> tuple[list[str], dict]:
    """Keep all SC-derived genes first, then bulk-only, cap at max_features."""
    sc_unique = list(dict.fromkeys(sc_genes))
    sc_set = set(sc_unique)
    bulk_only = [g for g in dict.fromkeys(bulk_genes) if g not in sc_set]
    merged = sc_unique + bulk_only
    final = merged[:max_features]
    meta = {
        "n_sc_in_union": min(len(sc_unique), max_features),
        "n_bulk_only_in_union": max(0, len(final) - min(len(sc_unique), max_features)),
        "n_union_before_cap": len(merged),
        "n_union_final": len(final),
        "cap_applied": len(merged) > max_features,
    }
    return final, meta
