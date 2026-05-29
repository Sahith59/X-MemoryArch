"""
Sub-phase 2.5 — MMR (Maximal Marginal Relevance) diversification.

Runs AFTER the cross-encoder reranker (per spec: cross-encoder → MMR, never reversed).
Prevents the final result set from being dominated by semantically redundant memories.

Similarity between two memories is structural (no embedding required):
  - Same cluster_id     → 0.80
  - Same session_id     → 0.50
  - Entity token overlap → 0.30 × overlap_ratio (additive)
  Maximum combined similarity is capped at 1.0.

MMR selection formula:
  score(d, S) = λ × relevance(d) − (1−λ) × max_{s ∈ S} sim(d, s)

λ default: 0.70 — strong relevance bias with moderate diversification.
Range:     0.65–0.80 (lower λ = more diversity, higher = more relevance).

Usage:
  from app.services.retrieval.mmr import mmr_select
  selected_ids = mmr_select(memories, scores, top_k=10, lambda_=0.70)
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Structural similarity
# ---------------------------------------------------------------------------

def structural_similarity(
    m1,
    m2,
    entity_map: dict[str, set[str]] | None = None,
) -> float:
    """
    Structural similarity between two Memory objects.

    No embedding computation — uses metadata signals only.
    """
    sim = 0.0

    # Same cluster → very high similarity (same topic group)
    cid1 = getattr(m1, "cluster_id", None)
    cid2 = getattr(m2, "cluster_id", None)
    if cid1 is not None and cid2 is not None and cid1 == cid2:
        sim += 0.80

    # Same source session → likely corroborating from same conversation
    s1 = getattr(m1, "source_session_id", None)
    s2 = getattr(m2, "source_session_id", None)
    if s1 is not None and s2 is not None and s1 == s2:
        sim += 0.50

    # Entity token overlap
    if entity_map is not None:
        e1 = entity_map.get(m1.id, set())
        e2 = entity_map.get(m2.id, set())
        if e1 and e2:
            overlap = len(e1 & e2) / max(len(e1 | e2), 1)
            sim += 0.30 * overlap

    return min(sim, 1.0)


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------

def mmr_select(
    memories: list,
    scores: dict[str, float],
    top_k: int,
    lambda_: float = 0.70,
    entity_map: dict[str, set[str]] | None = None,
) -> list[str]:
    """
    Select top_k memory IDs using Maximal Marginal Relevance.

    Args:
        memories:   Candidate Memory objects (pre-sorted by relevance score desc).
        scores:     {memory_id: relevance_score} — from weighted ranker or cross-encoder.
        top_k:      Number of memories to select.
        lambda_:    Trade-off parameter (0.0 = pure diversity, 1.0 = pure relevance).
        entity_map: {memory_id: entity_token_set} for entity overlap signal.

    Returns:
        Ordered list of memory_id strings (most relevant/diverse first).
    """
    if not memories:
        return []

    # Index memories by id for fast lookup
    mem_by_id = {m.id: m for m in memories}
    candidates = [m.id for m in memories]  # ordered by input rank

    selected: list[str] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda mid: scores.get(mid, 0.0))
        else:
            # Subsequent picks: MMR score
            best = None
            best_mmr = float("-inf")
            for mid in remaining:
                relevance = scores.get(mid, 0.0)
                max_sim = max(
                    structural_similarity(
                        mem_by_id[mid],
                        mem_by_id[sel_id],
                        entity_map=entity_map,
                    )
                    for sel_id in selected
                )
                mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = mid

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)

    return selected
