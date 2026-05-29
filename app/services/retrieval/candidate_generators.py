"""
Phase 2.1 — Three parallel candidate generators.

Leg A (BM25):    FTS5 full-text search over search_text (title + content + tags).
Leg B (Dense):   Cosine similarity via the pluggable VectorBackend.
Leg C (Entity):  MemoryEntity table lookup for entities mentioned in the query.

All three legs operate on `allowed_ids` — a pre-filtered whitelist computed by
`apply_hard_filters()`. The privacy gate is always a SQLite authority; no leg
can surface a memory that failed the hard-filter pass.

Privacy clearance ordering: public < internal < sensitive < secret
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.services.vector_backends.base import VectorBackend


# ---------------------------------------------------------------------------
# Privacy clearance ordering
# ---------------------------------------------------------------------------

_PRIVACY_ORDER = ["public", "internal", "sensitive", "secret"]


def allowed_privacy_levels(max_clearance: str) -> list[str]:
    """
    Return all privacy levels visible to a caller with max_clearance.

    public → ['public']
    internal → ['public', 'internal']
    sensitive → ['public', 'internal', 'sensitive']
    secret → all four
    """
    try:
        idx = _PRIVACY_ORDER.index(max_clearance)
    except ValueError:
        idx = 1  # default to internal-level access
    return _PRIVACY_ORDER[: idx + 1]


# ---------------------------------------------------------------------------
# Hard filter gate — applied BEFORE any scoring leg
# ---------------------------------------------------------------------------

def apply_hard_filters(
    db: "Session",
    project_id: str,
    max_clearance: str = "internal",
    include_superseded: bool = False,
    now: datetime | None = None,
) -> tuple[list[str], int]:
    """
    Return (allowed_ids, forbidden_count).

    allowed_ids:      Memory IDs that pass all mandatory gates.
    forbidden_count:  Count of memories that exist in this project but were
                      blocked by the privacy gate — used for leakage auditing.
                      Must always be 0 for correctly configured callers.

    Hard gates (all mandatory — never bypassed by query parameters):
      1. project_id match
      2. status not 'superseded' (unless include_superseded=True)
      3. superseded_by IS NULL (not superseded by another memory)
      4. review_status != 'rejected'
      5. valid_until IS NULL OR valid_until > now
      6. privacy_level in allowed levels for max_clearance
    """
    from app import models

    if now is None:
        now = datetime.now(timezone.utc)

    # All active memories in this project (before privacy gate)
    base_q = (
        db.query(models.Memory.id)
        .filter(models.Memory.project_id == project_id)
        .filter(models.Memory.review_status != "rejected")
        .filter(
            (models.Memory.valid_until.is_(None))
            | (models.Memory.valid_until > now)
        )
    )

    if not include_superseded:
        base_q = base_q.filter(
            models.Memory.status != "superseded",
            models.Memory.superseded_by.is_(None),
        )

    pre_privacy_ids = {row[0] for row in base_q.all()}

    # Apply privacy gate
    visible_levels = allowed_privacy_levels(max_clearance)
    allowed_q = base_q.filter(models.Memory.privacy_level.in_(visible_levels))
    allowed_ids = [row[0] for row in allowed_q.all()]

    # forbidden_count = memories that pass base filters but blocked by privacy
    forbidden_count = len(pre_privacy_ids) - len(allowed_ids)

    return allowed_ids, max(0, forbidden_count)


# ---------------------------------------------------------------------------
# Leg A — BM25 via Phase 1's FTS5 table
# ---------------------------------------------------------------------------

_FTS_SPECIAL = re.compile(r'[^\w\s]')


def _fts_query(query: str) -> str:
    """Clean query text for FTS5 MATCH expression."""
    cleaned = _FTS_SPECIAL.sub(' ', query).strip()
    terms = [t for t in cleaned.split() if t]
    if not terms:
        return '""'
    return " ".join(terms)


def generate_bm25_candidates(
    db: "Session",
    project_id: str,
    query: str,
    allowed_ids: list[str],
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Run FTS5 BM25 search and return (memory_id, raw_bm25_score) pairs.

    Returns raw BM25 scores (positive — Phase 1's FTS5 returns negative values,
    we flip the sign). Filtered to allowed_ids only.
    """
    if not allowed_ids:
        return []

    fts_q = _fts_query(query)
    if not fts_q or fts_q == '""':
        return []

    try:
        rows = db.execute(
            text("""
                SELECT memories_fts.memory_id, bm25(memories_fts) AS bm25_score
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.memory_id
                WHERE memories_fts MATCH :q
                  AND m.project_id = :pid
                LIMIT 200
            """),
            {"q": fts_q, "pid": project_id},
        ).fetchall()
    except Exception:
        # FTS5 table may not exist (e.g., test DB not set up) — safe fallback
        return []

    if not rows:
        return []

    allowed_set = set(allowed_ids)
    scored = [
        (row[0], -row[1])  # flip negative BM25 to positive
        for row in rows
        if row[0] in allowed_set
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Leg B — Dense vector similarity via VectorBackend
# ---------------------------------------------------------------------------

def generate_dense_candidates(
    vector_backend: "VectorBackend",
    query_vector: list[float] | None,
    allowed_ids: list[str],
    top_k: int = 50,
    project_id: str = "",
) -> list[tuple[str, float]]:
    """
    Return (memory_id, cosine_similarity) from the vector backend.

    Returns empty list if query_vector is None (embedding unavailable) or
    if allowed_ids is empty.
    """
    if query_vector is None or not allowed_ids:
        return []

    try:
        return vector_backend.search(
            query_vector=query_vector,
            top_k=top_k,
            project_id=project_id,
            allowed_ids=allowed_ids,
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Leg C — Entity/link candidate generation
# ---------------------------------------------------------------------------

_MIN_ENTITY_WORD_LEN = 3


def _extract_query_tokens(query: str) -> list[str]:
    """
    Extract candidate entity tokens from a query string.

    Keeps words ≥ 3 chars. Lowercased for case-insensitive match against
    MemoryEntity.normalized_text.
    """
    words = re.findall(r'\b\w+\b', query)
    return [w.lower() for w in words if len(w) >= _MIN_ENTITY_WORD_LEN]


def generate_entity_candidates(
    db: "Session",
    project_id: str,
    query: str,
    allowed_ids: list[str],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Find memories that have entities matching tokens in the query.

    Score = fraction of query tokens that matched (0.0–1.0).
    Ties broken by number of matching entity rows.

    For Sub-phase 2.1 this uses simple token matching against
    MemoryEntity.normalized_text. Phase 2.5 will add spaCy entity extraction
    for query NER when the model is available.
    """
    if not allowed_ids:
        return []

    tokens = _extract_query_tokens(query)
    if not tokens:
        return []

    from app import models

    allowed_set = set(allowed_ids)

    # Fetch all entity rows for this project in allowed_ids
    rows = (
        db.query(models.MemoryEntity.memory_id, models.MemoryEntity.normalized_text)
        .filter(
            models.MemoryEntity.project_id == project_id,
            models.MemoryEntity.memory_id.in_(allowed_ids),
        )
        .all()
    )

    if not rows:
        return []

    # Group entity texts by memory_id
    mem_entities: dict[str, set[str]] = {}
    for mid, norm_text in rows:
        if mid not in mem_entities:
            mem_entities[mid] = set()
        mem_entities[mid].add(norm_text)

    # Score each memory by how many query tokens appear in its entity set
    token_set = set(tokens)
    scored: list[tuple[str, float]] = []
    for mid, entity_texts in mem_entities.items():
        matches = sum(
            1 for tok in token_set
            if any(tok in ent for ent in entity_texts)
        )
        if matches > 0:
            score = matches / len(token_set)
            scored.append((mid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
