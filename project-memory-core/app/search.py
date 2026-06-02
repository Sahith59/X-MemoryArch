"""
Sub-phase 1.12 — Hybrid Full-Text Search via SQLite FTS5.

BM25 (full-text relevance) × type_weight × importance_norm × recency_factor.
Half-life decay of 90 days from MemoryBank / FadeMem research.
"""
from __future__ import annotations
import math
import re
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app import models

# ---------------------------------------------------------------------------
# Type weights — from plan.md research notes
# ---------------------------------------------------------------------------
_TYPE_WEIGHTS: dict[str, float] = {
    "decision": 1.4,
    "problem": 1.3,       # was: bug
    "structure": 1.3,     # was: architecture
    "constraint": 1.2,
    "failed_approach": 1.2,
}
_DEFAULT_WEIGHT = 1.0
_HALF_LIFE_DAYS = 90.0


def _composite_score(
    bm25_raw: float,
    importance: int,
    mem_type: str,
    updated_at: datetime,
) -> float:
    bm25 = -bm25_raw  # FTS5 returns negative — flip to positive (higher = better)
    type_w = _TYPE_WEIGHTS.get(mem_type, _DEFAULT_WEIGHT)
    importance_norm = importance / 5.0
    # Ensure updated_at is timezone-aware
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    days_old = (datetime.now(timezone.utc) - updated_at).total_seconds() / 86400.0
    recency = math.pow(0.5, days_old / _HALF_LIFE_DAYS)
    return bm25 * type_w * importance_norm * recency


def _fts_query(q: str) -> str:
    """Convert a user search string to an FTS5 MATCH expression.

    Each token becomes a prefix match. Multiple tokens are ANDed together
    so all terms must appear. Special FTS5 chars are stripped.
    """
    clean = re.sub(r'["\(\)\*\:\^]', " ", q)
    tokens = [t for t in clean.split() if t]
    if not tokens:
        return '""'
    return " AND ".join(f"{tok}*" for tok in tokens)


# ---------------------------------------------------------------------------
# FTS5 setup — called once at app startup and once per test db fixture
# ---------------------------------------------------------------------------

# Bump this when the FTS column list changes so setup_fts auto-rebuilds the
# virtual table on existing databases instead of silently keeping stale schema.
_FTS_SCHEMA_VERSION = 2  # v1: title/content/tags/file_path  v2: +search_text/retrieval_hint


def _fts_needs_rebuild(conn) -> bool:
    """Return True when the live FTS table is missing columns added in v2+."""
    try:
        conn.execute(text("SELECT retrieval_hint FROM memories_fts LIMIT 0"))
        return False
    except Exception:
        return True


def _rebuild_fts(conn) -> None:
    """Drop old FTS artefacts and recreate with the current schema."""
    for obj in ("memories_fts_ai", "memories_fts_au", "memories_fts_ad"):
        conn.execute(text(f"DROP TRIGGER IF EXISTS {obj}"))
    conn.execute(text("DROP TABLE IF EXISTS memories_fts"))


def setup_fts(engine: Engine) -> None:
    """Create (or upgrade) the FTS5 virtual table and sync triggers.

    Schema v2 adds search_text (pre-computed composite including type_metadata
    values and source_quote) and retrieval_hint (curated TL;DR). When an older
    table is detected the function drops and recreates it, then backfills from
    the memories table so existing data is immediately searchable.
    """
    with engine.begin() as conn:
        if _fts_needs_rebuild(conn):
            _rebuild_fts(conn)

        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id    UNINDEXED,
                title,
                content,
                search_text,
                retrieval_hint,
                tags,
                file_path,
                tokenize='porter unicode61'
            )
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS memories_fts_ai
            AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(
                    memory_id, title, content,
                    search_text, retrieval_hint, tags, file_path
                )
                VALUES (
                    new.id,
                    new.title,
                    new.content,
                    COALESCE(new.search_text, ''),
                    COALESCE(new.retrieval_hint, ''),
                    COALESCE(new.tags, ''),
                    COALESCE(new.file_path, '')
                );
            END
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS memories_fts_au
            AFTER UPDATE ON memories BEGIN
                DELETE FROM memories_fts WHERE memory_id = old.id;
                INSERT INTO memories_fts(
                    memory_id, title, content,
                    search_text, retrieval_hint, tags, file_path
                )
                VALUES (
                    new.id,
                    new.title,
                    new.content,
                    COALESCE(new.search_text, ''),
                    COALESCE(new.retrieval_hint, ''),
                    COALESCE(new.tags, ''),
                    COALESCE(new.file_path, '')
                );
            END
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS memories_fts_ad
            AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE memory_id = old.id;
            END
        """))

        # Backfill: populate FTS for any memories not yet indexed
        # (runs on upgrade from v1, and on first setup with existing data)
        conn.execute(text("""
            INSERT INTO memories_fts(
                memory_id, title, content,
                search_text, retrieval_hint, tags, file_path
            )
            SELECT
                m.id,
                m.title,
                m.content,
                COALESCE(m.search_text, ''),
                COALESCE(m.retrieval_hint, ''),
                COALESCE(m.tags, ''),
                COALESCE(m.file_path, '')
            FROM memories m
            WHERE m.id NOT IN (SELECT memory_id FROM memories_fts)
        """))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_memories(
    db: Session,
    project_id: str,
    query: str,
    mem_type: str | None = None,
    status: str | None = None,
    importance_min: int | None = None,
    limit: int = 20,
) -> list[tuple[models.Memory, float]]:
    """Return memories ranked by BM25 × type_weight × importance × recency."""
    fts_q = _fts_query(query)

    # Step 1: FTS5 match — get IDs + raw BM25 scores for this project
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

    if not rows:
        return []

    bm25_by_id = {row[0]: row[1] for row in rows}

    # Step 2: Load full ORM objects with any extra filters applied
    orm_q = db.query(models.Memory).filter(
        models.Memory.id.in_(bm25_by_id.keys()),
        models.Memory.project_id == project_id,
    )
    if mem_type:
        orm_q = orm_q.filter(models.Memory.type == mem_type)
    if status:
        orm_q = orm_q.filter(models.Memory.status == status)
    if importance_min is not None:
        orm_q = orm_q.filter(models.Memory.importance >= importance_min)

    memories = orm_q.all()

    # Step 3: Score each using ORM datetime (no parsing issues)
    scored: list[tuple[models.Memory, float]] = []
    for m in memories:
        score = _composite_score(bm25_by_id[m.id], m.importance, m.type, m.updated_at)
        scored.append((m, score))

    # Step 4: Sort descending, apply limit
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]
