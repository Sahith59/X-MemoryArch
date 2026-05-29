"""
Sub-phase 2.5 — Weighted ranking (5 signals, intent-dependent).

Signals:
  rrf         — reciprocal-rank-fusion score from the retrieval legs
  decay       — Ebbinghaus decay freshness signal (Phase 1 decay_score)
  importance  — user-assigned importance (1–5 scale, normalised to 0–1)
  quality     — extraction quality score (Phase 1 quality_score)
  entity      — entity overlap ratio between query tokens and memory entities

Weights are intent-dependent (see intent_classifier.get_intent_weights).

All signals are normalised independently to [0, 1] before weighting:
  rrf         → divided by max(rrf) in the candidate pool
  decay       → already in [0, 1]; None → 0.5 (neutral freshness)
  importance  → (value − 1) / 4
  quality     → already in [0, 1]; None → 0.5
  entity      → |query_tokens ∩ entity_tokens| / max(|query_tokens|, 1)

Usage:
  from app.services.retrieval.ranking import weighted_rank
  ranked = weighted_rank(memories, rrf_scores, intent, query_tokens, entity_map)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from app.services.retrieval.intent_classifier import IntentLabel, get_intent_weights


# ---------------------------------------------------------------------------
# Stop-words (mirrors graph_expansion to keep entity tokens consistent)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "what", "how",
    "when", "where", "why", "which", "that", "this", "these", "those",
})


# ---------------------------------------------------------------------------
# Score one memory
# ---------------------------------------------------------------------------

def _normalise_importance(importance: int | None) -> float:
    """Convert 1–5 importance to 0–1."""
    if importance is None:
        return 0.5
    return (max(1, min(5, importance)) - 1) / 4.0


def _entity_overlap(query_tokens: set[str], entity_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & entity_tokens)
    return overlap / max(len(query_tokens), 1)


def score_memory(
    memory,
    rrf_score_normalised: float,
    query_tokens: set[str],
    entity_tokens: set[str],
    weights: dict[str, float],
) -> float:
    """
    Compute a single weighted relevance score for one memory.

    Args:
        memory:                  Memory ORM object.
        rrf_score_normalised:    RRF score already normalised to [0, 1].
        query_tokens:            Set of lowercase non-stop query tokens.
        entity_tokens:           Set of normalised entity tokens for this memory.
        weights:                 Intent-dependent weight dict from get_intent_weights.

    Returns:
        Weighted score in [0, 1].
    """
    decay = memory.decay_score if memory.decay_score is not None else 0.5
    quality = memory.quality_score if memory.quality_score is not None else 0.5
    importance = _normalise_importance(getattr(memory, "importance", 3))
    entity = _entity_overlap(query_tokens, entity_tokens)

    return (
        weights["rrf"]       * rrf_score_normalised
        + weights["decay"]     * decay
        + weights["importance"] * importance
        + weights["quality"]   * quality
        + weights["entity"]    * entity
    )


# ---------------------------------------------------------------------------
# Bulk ranking
# ---------------------------------------------------------------------------

def weighted_rank(
    memories: list,
    rrf_scores: dict[str, float],
    intent: IntentLabel,
    query: str,
    entity_map: dict[str, set[str]] | None = None,
) -> list[tuple[str, float]]:
    """
    Re-rank a list of Memory objects using the 5-signal weighted function.

    Args:
        memories:    Candidate Memory objects (any order).
        rrf_scores:  {memory_id: rrf_score} from fusion / graph expansion.
        intent:      Detected or forced query intent.
        query:       Raw query string (used to compute query_tokens).
        entity_map:  {memory_id: set of normalised entity tokens}.
                     If None, entity signal is zero for all memories.

    Returns:
        List of (memory_id, weighted_score) sorted by score descending.
    """
    if not memories:
        return []

    weights = get_intent_weights(intent)

    # Query tokenisation
    query_tokens = {
        t.lower() for t in re.split(r"\W+", query)
        if t and t.lower() not in _STOP_WORDS
    }

    # Normalise RRF scores
    all_rrf = [rrf_scores.get(m.id, 0.0) for m in memories]
    max_rrf = max(all_rrf) if any(s > 0 for s in all_rrf) else 1.0

    entity_map = entity_map or {}

    scored: list[tuple[str, float]] = []
    for memory in memories:
        raw_rrf = rrf_scores.get(memory.id, 0.0)
        rrf_norm = raw_rrf / max_rrf if max_rrf > 0 else 0.0
        entity_tokens = entity_map.get(memory.id, set())

        ws = score_memory(
            memory=memory,
            rrf_score_normalised=rrf_norm,
            query_tokens=query_tokens,
            entity_tokens=entity_tokens,
            weights=weights,
        )
        scored.append((memory.id, ws))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Load entity map from DB (called by retrieval_service)
# ---------------------------------------------------------------------------

def load_entity_map(db, memory_ids: list[str]) -> dict[str, set[str]]:
    """
    Bulk-load entity tokens for the given memory IDs.

    Returns {memory_id: set_of_normalised_entity_tokens}.
    """
    if not memory_ids:
        return {}

    from app import models as phase1_models

    rows = (
        db.query(
            phase1_models.MemoryEntity.memory_id,
            phase1_models.MemoryEntity.normalized_text,
        )
        .filter(phase1_models.MemoryEntity.memory_id.in_(memory_ids))
        .all()
    )

    entity_map: dict[str, set[str]] = {}
    for memory_id, norm_text in rows:
        tokens = {t.lower() for t in re.split(r"\W+", norm_text) if t}
        entity_map.setdefault(memory_id, set()).update(tokens)

    return entity_map
