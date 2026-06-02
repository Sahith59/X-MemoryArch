"""
Sub-phase 2.4 — Entity/Graph Expansion.

Adds four retrieval expansion mechanisms on top of RRF-fused top-K:

  1. Entity soft-boost   — bump RRF scores when query shares entities with memory
  2. Code anchor retrieval — inject memories that anchor to file_path / symbol_name
  3. 1-hop link expansion — follow MemoryLink edges (both directions), damped score
  4. 2-hop link expansion — gated to multi-concept queries, bounded to prevent drift

All expansions:
  - Only inject memories already present in allowed_ids (hard filter respected)
  - Never drop memories already in top_k
  - Return an augmented {memory_id: score} dict; caller re-sorts by score
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Query classification helpers
# ---------------------------------------------------------------------------

_CODE_PATTERNS = re.compile(
    r"(?:"
    r"[a-zA-Z_][\w]*\.[a-z]{1,5}\b"  # filename.ext
    r"|def\s+\w+"                      # Python def
    r"|class\s+\w+"                    # class declaration
    r"|function\s+\w+"                 # JS function
    r"|import\s+[\w.]+"               # import statement
    r"|from\s+[\w.]+\s+import"        # from … import
    r"|/[\w./]+"                       # path segment
    r"|#\w+"                           # symbol anchor
    r")",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "what", "how",
    "when", "where", "why", "which", "that", "this", "these", "those",
})


def is_code_query(query: str) -> bool:
    """Return True when the query appears to reference code symbols or file paths."""
    return bool(_CODE_PATTERNS.search(query))


def is_multi_concept_query(query: str) -> bool:
    """
    Return True when the query spans multiple distinct concepts.

    Heuristic: explicit AND/OR connectors, or ≥5 meaningful (non-stop) tokens.
    Gating 2-hop expansion to multi-concept queries keeps retrieval focused.
    """
    lower = query.lower()
    if " and " in lower or " or " in lower:
        return True
    tokens = [t for t in re.split(r"\W+", lower) if t and t not in _STOP_WORDS]
    return len(tokens) >= 5


# ---------------------------------------------------------------------------
# 1. Entity soft-boost
# ---------------------------------------------------------------------------

def apply_entity_soft_boost(
    db: "Session",
    fused_scores: dict[str, float],
    query: str,
    entity_boost_weight: float = 0.15,
) -> dict[str, float]:
    """
    Boost RRF scores for memories whose stored entities overlap with query tokens.

    boost = entity_boost_weight × (overlap / max(len(query_tokens), 1))
    New score = original_score × (1 + boost)

    Operates in-place on a copy of fused_scores.
    """
    from app import models as phase1_models

    if not fused_scores:
        return fused_scores

    query_tokens = {
        t.lower() for t in re.split(r"\W+", query) if t and t.lower() not in _STOP_WORDS
    }
    if not query_tokens:
        return fused_scores

    memory_ids = list(fused_scores.keys())

    # Bulk-load entities for all top-K memories in one query
    rows = (
        db.query(
            phase1_models.MemoryEntity.memory_id,
            phase1_models.MemoryEntity.normalized_text,
        )
        .filter(phase1_models.MemoryEntity.memory_id.in_(memory_ids))
        .all()
    )

    # Build memory → entity token set
    entity_map: dict[str, set[str]] = {}
    for memory_id, norm_text in rows:
        entity_tokens = {t.lower() for t in re.split(r"\W+", norm_text) if t}
        entity_map.setdefault(memory_id, set()).update(entity_tokens)

    boosted = dict(fused_scores)
    for mid, base_score in fused_scores.items():
        entity_tokens = entity_map.get(mid, set())
        if not entity_tokens:
            continue
        overlap = len(query_tokens & entity_tokens)
        if overlap == 0:
            continue
        ratio = overlap / max(len(query_tokens), 1)
        boost = entity_boost_weight * ratio
        boosted[mid] = base_score * (1.0 + boost)

    return boosted


# ---------------------------------------------------------------------------
# 2. Code anchor retrieval
# ---------------------------------------------------------------------------

def code_anchor_retrieval(
    db: "Session",
    project_id: str,
    query: str,
    allowed_ids: set[str],
    max_results: int = 10,
) -> list[str]:
    """
    Return memory IDs whose file_path or symbol_name match tokens from the query.

    Matches are substring-based (LIKE %token%) on each non-stop meaningful token.
    Only returns IDs already in allowed_ids.
    """
    from app import models as phase1_models

    if not allowed_ids:
        return []

    tokens = [
        t.lower() for t in re.split(r"\W+", query)
        if t and t.lower() not in _STOP_WORDS and len(t) >= 2
    ]
    if not tokens:
        return []

    matched_ids: set[str] = set()
    for token in tokens[:5]:  # cap token scan to avoid N+1 explosion
        rows = (
            db.query(phase1_models.Memory.id)
            .filter(
                phase1_models.Memory.project_id == project_id,
                phase1_models.Memory.id.in_(allowed_ids),
                (
                    phase1_models.Memory.file_path.ilike(f"%{token}%")
                    | phase1_models.Memory.symbol_name.ilike(f"%{token}%")
                ),
            )
            .limit(max_results)
            .all()
        )
        for (mid,) in rows:
            matched_ids.add(mid)
            if len(matched_ids) >= max_results:
                break
        if len(matched_ids) >= max_results:
            break

    return list(matched_ids)


# ---------------------------------------------------------------------------
# 3. 1-hop MemoryLink expansion
# ---------------------------------------------------------------------------

def expand_1hop(
    db: "Session",
    top_k_ids: list[str],
    allowed_ids: set[str],
    top_k_scores: dict[str, float],
    max_per_hop: int = 10,
) -> dict[str, float]:
    """
    Follow MemoryLink edges one step in both directions from top_k_ids.

    Expanded score = 0.5 × originator_score  (damping factor).
    Only memories in allowed_ids are eligible.
    Returns NEW memories found (not already in top_k_ids), capped at max_per_hop.
    """
    from app import models as phase1_models

    if not top_k_ids or not allowed_ids:
        return {}

    existing = set(top_k_ids)

    # Edges where a top-K memory is the source (outbound)
    outbound = (
        db.query(phase1_models.MemoryLink.source_memory_id, phase1_models.MemoryLink.target_memory_id)
        .filter(
            phase1_models.MemoryLink.source_memory_id.in_(top_k_ids),
            phase1_models.MemoryLink.superseded_at.is_(None),
        )
        .all()
    )

    # Edges where a top-K memory is the target (inbound)
    inbound = (
        db.query(phase1_models.MemoryLink.source_memory_id, phase1_models.MemoryLink.target_memory_id)
        .filter(
            phase1_models.MemoryLink.target_memory_id.in_(top_k_ids),
            phase1_models.MemoryLink.superseded_at.is_(None),
        )
        .all()
    )

    expanded: dict[str, float] = {}

    for src_id, tgt_id in outbound:
        neighbor = tgt_id
        if neighbor in existing or neighbor not in allowed_ids:
            continue
        originator_score = top_k_scores.get(src_id, 0.0)
        candidate_score = 0.5 * originator_score
        # Keep the highest score if reached via multiple paths
        if neighbor not in expanded or expanded[neighbor] < candidate_score:
            expanded[neighbor] = candidate_score

    for src_id, tgt_id in inbound:
        neighbor = src_id
        if neighbor in existing or neighbor not in allowed_ids:
            continue
        originator_score = top_k_scores.get(tgt_id, 0.0)
        candidate_score = 0.5 * originator_score
        if neighbor not in expanded or expanded[neighbor] < candidate_score:
            expanded[neighbor] = candidate_score

    # Cap results
    if len(expanded) > max_per_hop:
        top_expanded = sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:max_per_hop]
        expanded = dict(top_expanded)

    return expanded


# ---------------------------------------------------------------------------
# 4. 2-hop expansion (gated)
# ---------------------------------------------------------------------------

def expand_2hop(
    db: "Session",
    hop1_expansion: dict[str, float],
    allowed_ids: set[str],
    already_included: set[str],
    max_total: int = 15,
) -> dict[str, float]:
    """
    Expand one further hop from the hop-1 results.

    Only called when is_multi_concept_query() returns True.
    Damping: 0.5 × hop1_score (so 0.25× original).
    Bounded by max_total to prevent drift.
    """
    from app import models as phase1_models

    if not hop1_expansion or not allowed_ids:
        return {}

    hop1_ids = list(hop1_expansion.keys())
    existing = already_included | set(hop1_ids)

    outbound = (
        db.query(phase1_models.MemoryLink.source_memory_id, phase1_models.MemoryLink.target_memory_id)
        .filter(
            phase1_models.MemoryLink.source_memory_id.in_(hop1_ids),
            phase1_models.MemoryLink.superseded_at.is_(None),
        )
        .all()
    )

    inbound = (
        db.query(phase1_models.MemoryLink.source_memory_id, phase1_models.MemoryLink.target_memory_id)
        .filter(
            phase1_models.MemoryLink.target_memory_id.in_(hop1_ids),
            phase1_models.MemoryLink.superseded_at.is_(None),
        )
        .all()
    )

    expanded: dict[str, float] = {}

    for src_id, tgt_id in outbound:
        neighbor = tgt_id
        if neighbor in existing or neighbor not in allowed_ids:
            continue
        candidate_score = 0.5 * hop1_expansion.get(src_id, 0.0)
        if neighbor not in expanded or expanded[neighbor] < candidate_score:
            expanded[neighbor] = candidate_score

    for src_id, tgt_id in inbound:
        neighbor = src_id
        if neighbor in existing or neighbor not in allowed_ids:
            continue
        candidate_score = 0.5 * hop1_expansion.get(tgt_id, 0.0)
        if neighbor not in expanded or expanded[neighbor] < candidate_score:
            expanded[neighbor] = candidate_score

    if len(expanded) > max_total:
        top = sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:max_total]
        expanded = dict(top)

    return expanded


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def graph_expand(
    db: "Session",
    project_id: str,
    top_k_ids: list[str],
    rrf_scores: dict[str, float],
    allowed_ids: set[str],
    query: str,
    enable_entity_boost: bool = True,
    enable_graph_expansion: bool = True,
    enable_2hop: bool = True,
    max_expansion_total: int = 15,
    entity_boost_weight: float = 0.15,
) -> tuple[dict[str, float], int]:
    """
    Run all Sub-phase 2.4 expansion passes and return an augmented score dict.

    Returns:
      (augmented_scores, expanded_count) where augmented_scores maps every
      memory ID (top-K + newly expanded) to its final score, and
      expanded_count is how many NEW memories were added via graph expansion.
    """
    scores = dict(rrf_scores)

    # 1. Entity soft-boost (modifies scores in-place for existing top-K)
    if enable_entity_boost and top_k_ids:
        scores = apply_entity_soft_boost(
            db=db,
            fused_scores=scores,
            query=query,
            entity_boost_weight=entity_boost_weight,
        )

    # 2. Code anchor retrieval — inject code-anchored memories
    if enable_graph_expansion and is_code_query(query):
        anchor_ids = code_anchor_retrieval(
            db=db,
            project_id=project_id,
            query=query,
            allowed_ids=allowed_ids,
            max_results=max_expansion_total,
        )
        existing_ids = set(scores.keys())
        for aid in anchor_ids:
            if aid not in existing_ids:
                # Score = half of the weakest top-K score (or small constant)
                min_score = min(scores.values()) if scores else 0.01
                scores[aid] = 0.5 * min_score

    # 3. 1-hop MemoryLink expansion
    expanded_1hop: dict[str, float] = {}
    if enable_graph_expansion and top_k_ids:
        expanded_1hop = expand_1hop(
            db=db,
            top_k_ids=top_k_ids,
            allowed_ids=allowed_ids,
            top_k_scores=rrf_scores,
            max_per_hop=max_expansion_total,
        )
        for mid, score in expanded_1hop.items():
            if mid not in scores:
                scores[mid] = score
            else:
                scores[mid] = max(scores[mid], score)

    # 4. 2-hop expansion — only for multi-concept queries
    if enable_graph_expansion and enable_2hop and expanded_1hop and is_multi_concept_query(query):
        already_included = set(top_k_ids) | set(scores.keys())
        expanded_2hop = expand_2hop(
            db=db,
            hop1_expansion=expanded_1hop,
            allowed_ids=allowed_ids,
            already_included=already_included,
            max_total=max_expansion_total,
        )
        for mid, score in expanded_2hop.items():
            if mid not in scores:
                scores[mid] = score
            else:
                scores[mid] = max(scores[mid], score)

    expanded_count = len(scores) - len(rrf_scores)
    return scores, max(0, expanded_count)
