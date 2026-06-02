"""
Reciprocal Rank Fusion (RRF) — Phase 2.1 multi-leg score fusion.

RRF score for a memory m across L ranked lists:
    score(m) = Σ_l weight_l × 1 / (rank_l(m) + k)

where rank is 0-based and k=60 is the RRF constant (Cormack et al. 2009).

Phase 2.1: Unweighted (weight_l = 1.0 for all legs).
Phase 2.5: Weights become query-intent-dependent once labeled data is collected.

The three legs are: BM25 (lexical), dense (semantic), entity (structural).
Memories that appear in more legs and at higher ranks score highest.
"""
from __future__ import annotations

from collections import defaultdict


_DEFAULT_K = 60  # Cormack et al. RRF constant — balances impact of high vs low ranks


def rrf_fuse(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = _DEFAULT_K,
) -> list[tuple[str, float]]:
    """
    Fuse N ranked lists of memory IDs via Reciprocal Rank Fusion.

    Args:
        ranked_lists:  One list per retrieval leg, each containing memory IDs
                       sorted by that leg's score (best first).
        weights:       Per-leg weights (default: uniform 1.0).
                       Must match len(ranked_lists) if provided.
        k:             RRF constant (default 60).

    Returns:
        List of (memory_id, rrf_score) sorted descending.
        memory_ids that appeared in no ranked list are excluded.
    """
    if not ranked_lists:
        return []

    n_legs = len(ranked_lists)
    if weights is None:
        weights = [1.0] * n_legs
    elif len(weights) != n_legs:
        raise ValueError(f"weights length {len(weights)} != ranked_lists length {n_legs}")

    scores: dict[str, float] = defaultdict(float)

    for ranked_ids, weight in zip(ranked_lists, weights):
        for rank, memory_id in enumerate(ranked_ids):
            scores[memory_id] += weight * (1.0 / (rank + k))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_scores_to_ranked_ids(fused: list[tuple[str, float]]) -> list[str]:
    """Extract just the IDs from fuse result, in order."""
    return [mid for mid, _ in fused]


def rrf_explain(
    ranked_lists: list[list[str]],
    leg_names: list[str],
    memory_id: str,
    k: int = _DEFAULT_K,
) -> dict:
    """
    Return per-leg contribution for a single memory_id (for debugging / evaluation).

    Returns:
        {
          "memory_id": "...",
          "total_score": 0.042,
          "contributions": [
            {"leg": "bm25", "rank": 3, "contribution": 0.016},
            ...
          ]
        }
    """
    contributions = []
    total = 0.0

    for ranked_ids, name in zip(ranked_lists, leg_names):
        try:
            rank = ranked_ids.index(memory_id)
            contrib = 1.0 / (rank + k)
        except ValueError:
            rank = None
            contrib = 0.0
        total += contrib
        contributions.append({"leg": name, "rank": rank, "contribution": contrib})

    return {
        "memory_id": memory_id,
        "total_score": total,
        "contributions": contributions,
    }
