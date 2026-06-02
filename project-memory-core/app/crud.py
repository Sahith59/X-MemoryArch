from __future__ import annotations
import difflib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app import models, schemas

# ---------------------------------------------------------------------------
# Tier computation (Sub-phase 1.23)
# ---------------------------------------------------------------------------

_TIER_RECENCY_DAYS = 30   # memories updated within this window stay in "working"
_TIER_HIGH_IMPORTANCE = 4  # importance >= this always stays in "working"


def compute_tier(
    importance: int,
    updated_at: datetime,
    last_accessed_at: datetime | None = None,
    now: datetime | None = None,
) -> str:
    """
    Pure function — testable without a DB session.

    working  if:
      importance >= 4                          (high-value content)
      OR updated within the last 30 days       (recently modified)
      OR last_accessed_at within 30 days       (recently read — sub-phase 1.25)
    archival otherwise.
    """
    if importance >= _TIER_HIGH_IMPORTANCE:
        return "working"
    if now is None:
        now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if (now - updated_at).days <= _TIER_RECENCY_DAYS:
        return "working"
    if last_accessed_at is not None:
        if last_accessed_at.tzinfo is None:
            last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
        if (now - last_accessed_at).days <= _TIER_RECENCY_DAYS:
            return "working"
    return "archival"


# ---------------------------------------------------------------------------
# Decay score (Sub-phase 1.28) — Ebbinghaus exponential decay
# ---------------------------------------------------------------------------

_DECAY_HALF_LIFE_DAYS = 90.0   # memory freshness halves every 90 days


def compute_decay_score(
    importance: int,
    updated_at: datetime,
    last_accessed_at: datetime | None = None,
    now: datetime | None = None,
) -> float:
    """
    Pure function — testable without a DB session.

    Returns a freshness score in [0.0, 1.0] based on the Ebbinghaus forgetting
    curve (MemoryBank / FadeMem research).

    Formula:
      importance_weight  = importance / 5          (maps 1→0.20, 5→1.00)
      recency_update     = e^(-days_since_update  / 90)
      recency_access     = e^(-days_since_access  / 90)   (0 if never accessed)
      decay_score        = importance_weight × max(recency_update, recency_access)

    High-importance memories start high and decay slowly.
    A recent access refreshes the score the same way a recent write does.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    importance_weight = importance / 5.0

    days_update = max(0.0, (now - updated_at).total_seconds() / 86400)
    recency_update = math.exp(-days_update / _DECAY_HALF_LIFE_DAYS)

    recency_access = 0.0
    if last_accessed_at is not None:
        if last_accessed_at.tzinfo is None:
            last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
        days_access = max(0.0, (now - last_accessed_at).total_seconds() / 86400)
        recency_access = math.exp(-days_access / _DECAY_HALF_LIFE_DAYS)

    raw = importance_weight * max(recency_update, recency_access)
    return round(min(1.0, max(0.0, raw)), 4)


# ---------------------------------------------------------------------------

# Fields on Memory that are worth tracking in the changelog
_CHANGELOG_FIELDS = frozenset({
    "type", "title", "content", "importance", "confidence",
    "tags", "related_files", "related_tools", "status",
    "type_metadata", "file_path", "commit_sha", "superseded_by", "valid_until",
    "review_status", "source_type", "source_quote", "symbol_name",
    "line_start", "line_end", "branch_name",
    "privacy_level",
})


# ---------------------------------------------------------------------------
# Quality score + search text helpers (Sub-phase 1.15)
# ---------------------------------------------------------------------------

def _compute_quality_score(m: models.Memory) -> float:
    """
    Rule-based quality score (0.0–1.0) measuring how complete and trustworthy
    a memory is. Used later to prefer high-quality memories in retrieval.

    Components:
      Completeness (max 0.50): title length, content depth, type_metadata, tags
      Source traceability (max 0.25): session ref, source_quote, source_type
      Specificity (max 0.20): file_path, related_files, symbol_name
      Confidence contribution (max 0.05)
    """
    score = 0.0

    # — Completeness —
    title_len = len(m.title or "")
    if title_len >= 50:
        score += 0.20
    elif title_len >= 20:
        score += 0.12
    elif title_len >= 5:
        score += 0.06

    content_len = len(m.content or "")
    if content_len >= 200:
        score += 0.15
    elif content_len >= 100:
        score += 0.10
    elif content_len >= 20:
        score += 0.06

    if m.type_metadata:
        score += 0.10

    tags = json.loads(m.tags) if m.tags else []
    # Exclude the generic "auto-extracted" tag from quality signal
    meaningful_tags = [t for t in tags if t != "auto-extracted"]
    if meaningful_tags:
        score += 0.05

    # — Source traceability —
    if m.source_session_id:
        score += 0.15
    if m.source_quote:
        score += 0.05
    if m.source_type:
        score += 0.05

    # — Specificity —
    if m.file_path:
        score += 0.10
    related_files = json.loads(m.related_files) if m.related_files else []
    if related_files:
        score += 0.05
    if m.symbol_name:
        score += 0.05

    # — Confidence contribution —
    score += (m.confidence or 0.0) * 0.05

    return round(min(1.0, score), 3)


def _compute_search_text(m: models.Memory) -> str:
    """
    Flattened composite text field — the canonical input for Phase-2 vector
    embedding. Aggregates every searchable dimension of a memory into one string.
    """
    parts: list[str] = []

    if m.title:
        parts.append(m.title)
    if m.content:
        parts.append(m.content)

    tags = json.loads(m.tags) if m.tags else []
    if tags:
        parts.append("Tags: " + ", ".join(tags))

    if m.file_path:
        parts.append(f"File: {m.file_path}")
    if m.symbol_name:
        parts.append(f"Symbol: {m.symbol_name}")
    if m.branch_name:
        parts.append(f"Branch: {m.branch_name}")

    if m.type_metadata:
        md = json.loads(m.type_metadata)
        for key, val in md.items():
            if isinstance(val, str) and val:
                parts.append(f"{key}: {val}")
            elif isinstance(val, list):
                non_empty = [str(v) for v in val if v]
                if non_empty:
                    parts.append(f"{key}: {', '.join(non_empty)}")

    if m.source_quote:
        parts.append(f'Source: "{m.source_quote}"')

    return " ".join(filter(None, parts))


def _refresh_derived_fields(db: Session, m: models.Memory) -> None:
    """Recompute quality_score, search_text, retrieval_hint, and decay_score."""
    from app.services.retrieval_hint import compute_retrieval_hint
    now = datetime.now(timezone.utc)
    m.quality_score = _compute_quality_score(m)
    m.search_text = _compute_search_text(m)
    m.retrieval_hint = compute_retrieval_hint(
        mem_type=m.type,
        title=m.title,
        content=m.content,
        type_metadata=json.loads(m.type_metadata) if m.type_metadata else None,
    )
    # Use now as the reference updated_at (memory is being created/updated right now)
    m.decay_score = compute_decay_score(m.importance, now, m.last_accessed_at, now=now)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def create_project(db: Session, data: schemas.ProjectCreate) -> models.Project:
    project = models.Project(
        name=data.name,
        description=data.description,
        tech_stack=json.dumps(data.tech_stack),
        goals=json.dumps(data.goals),
        repo_path=data.repo_path,
        domain=data.domain,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_projects(db: Session, skip: int = 0, limit: int = 100) -> list[models.Project]:
    return db.query(models.Project).offset(skip).limit(limit).all()


def get_project(db: Session, project_id: str) -> models.Project:
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


def update_project(db: Session, project_id: str, data: schemas.ProjectUpdate) -> models.Project:
    project = get_project(db, project_id)
    updates = data.model_dump(exclude_unset=True)
    if "tech_stack" in updates:
        updates["tech_stack"] = json.dumps(updates["tech_stack"])
    if "goals" in updates:
        updates["goals"] = json.dumps(updates["goals"])
    for field, value in updates.items():
        setattr(project, field, value)
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id: str) -> None:
    project = get_project(db, project_id)
    # Collect memory IDs before cascade so we can clean up records SQLAlchemy
    # cascade doesn't reach (memory_links, memory_changelog have no ORM cascade).
    mem_ids = [m.id for m in db.query(models.Memory).filter(
        models.Memory.project_id == project_id
    ).all()]
    if mem_ids:
        db.query(models.MemoryLink).filter(
            or_(
                models.MemoryLink.source_memory_id.in_(mem_ids),
                models.MemoryLink.target_memory_id.in_(mem_ids),
            )
        ).delete(synchronize_session=False)
        db.query(models.MemoryChangelog).filter(
            models.MemoryChangelog.memory_id.in_(mem_ids)
        ).delete(synchronize_session=False)
    # Sub-phase 1.32: suggestions belong to the project — delete them explicitly
    db.query(models.MemorySuggestion).filter(
        models.MemorySuggestion.project_id == project_id
    ).delete(synchronize_session=False)
    # Sub-phase 1.33: context packets belong to the project — delete them explicitly
    db.query(models.ContextPacket).filter(
        models.ContextPacket.project_id == project_id
    ).delete(synchronize_session=False)
    # Sub-phase 1.34: handoff events belong to the project — delete them explicitly
    db.query(models.HandoffEvent).filter(
        models.HandoffEvent.project_id == project_id
    ).delete(synchronize_session=False)
    db.delete(project)
    db.commit()


# ---------------------------------------------------------------------------
# Messages (Sub-phase 1.31)
# ---------------------------------------------------------------------------

import re as _re

_CODE_BLOCK_RE = _re.compile(r"```[\w]*\n", _re.MULTILINE)
_ERROR_RE = _re.compile(
    r"(Traceback|Error:|Exception:|FAILED|assert.*Error)", _re.IGNORECASE
)
_LANG_RE = _re.compile(r"```(\w+)")

_CONTENT_TYPE_MAP = {
    schemas.MessageContentType.code_block: "code_block",
    schemas.MessageContentType.error_trace: "error_trace",
    schemas.MessageContentType.command: "command",
    schemas.MessageContentType.file_diff: "file_diff",
    schemas.MessageContentType.mixed: "mixed",
    schemas.MessageContentType.text: "text",
}


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def _detect_content_signals(content: str) -> dict:
    has_code = bool(_CODE_BLOCK_RE.search(content))
    has_error = bool(_ERROR_RE.search(content))
    lang_match = _LANG_RE.search(content)
    language = lang_match.group(1).lower() if lang_match else None
    if language in ("", "text", "plain"):
        language = None
    return {"has_code": has_code, "has_error": has_error, "language": language}


def _create_message_row(
    db: Session,
    project_id: str,
    session_id: str,
    msg: schemas.MessageCreate,
    index: int,
) -> models.Message:
    signals = _detect_content_signals(msg.content)
    message = models.Message(
        project_id=project_id,
        session_id=session_id,
        role=msg.role.value if hasattr(msg.role, "value") else msg.role,
        content=msg.content,
        content_type=msg.content_type.value if hasattr(msg.content_type, "value") else msg.content_type,
        message_index=index,
        token_estimate=_estimate_tokens(msg.content),
        has_code=signals["has_code"],
        has_error=signals["has_error"],
        language=signals["language"] or msg.language,
        extra_metadata=json.dumps(msg.metadata) if msg.metadata else None,
    )
    db.add(message)
    return message


def get_messages_for_session(
    db: Session, session_id: str
) -> list[models.Message]:
    get_session(db, session_id)
    return (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.message_index)
        .all()
    )


def add_message_to_session(
    db: Session, session_id: str, data: schemas.MessageCreate
) -> models.Message:
    session = get_session(db, session_id)
    existing_count = (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .count()
    )
    msg = _create_message_row(db, session.project_id, session_id, data, existing_count)
    db.commit()
    db.refresh(msg)
    return msg


def get_message(db: Session, message_id: str) -> models.Message:
    msg = db.query(models.Message).filter(models.Message.id == message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
    return msg


def delete_message(db: Session, message_id: str) -> None:
    msg = get_message(db, message_id)
    db.delete(msg)
    db.commit()


def get_message_count_for_session(db: Session, session_id: str) -> int:
    return (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .count()
    )


# ---------------------------------------------------------------------------
# AI Sessions
# ---------------------------------------------------------------------------

def create_session(
    db: Session, project_id: str, data: schemas.SessionCreate
) -> models.AISession:
    get_project(db, project_id)
    session = models.AISession(
        project_id=project_id,
        tool_name=data.tool_name,
        title=data.title,
        raw_content=data.raw_content,
        session_date=data.session_date,
    )
    db.add(session)
    db.flush()  # get session.id before creating messages

    if data.messages:
        # Structured messages provided — store each individually
        for idx, msg in enumerate(data.messages):
            _create_message_row(db, project_id, session.id, msg, idx)
    else:
        # Backward compat: raw_content → single mixed message
        _create_message_row(
            db, project_id, session.id,
            schemas.MessageCreate(
                role=schemas.MessageRole.unknown,
                content=data.raw_content,
                content_type=schemas.MessageContentType.mixed,
                message_index=0,
            ),
            index=0,
        )

    db.commit()
    db.refresh(session)
    return session


def get_sessions(
    db: Session, project_id: str, skip: int = 0, limit: int = 100
) -> list[models.AISession]:
    get_project(db, project_id)
    return (
        db.query(models.AISession)
        .filter(models.AISession.project_id == project_id)
        .order_by(models.AISession.session_date.desc(), models.AISession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_session(db: Session, session_id: str) -> models.AISession:
    session = db.query(models.AISession).filter(models.AISession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


def update_session(
    db: Session, session_id: str, data: schemas.SessionUpdate
) -> models.AISession:
    session = get_session(db, session_id)
    updates = data.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(session, field, value)
    db.commit()
    db.refresh(session)
    return session


def delete_session(db: Session, session_id: str) -> None:
    session = get_session(db, session_id)
    # Sub-phase 1.32: null out session reference on suggestions (preserve the suggestion, lose the link)
    db.query(models.MemorySuggestion).filter(
        models.MemorySuggestion.source_session_id == session_id
    ).update({"source_session_id": None}, synchronize_session=False)
    db.delete(session)
    db.commit()


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------

def create_memory(
    db: Session,
    project_id: str,
    data: schemas.MemoryCreate,
    embedding: Optional[bytes] = None,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
) -> models.Memory:
    get_project(db, project_id)
    memory = models.Memory(
        project_id=project_id,
        source_session_id=data.source_session_id,
        type=data.type.value,
        title=data.title,
        content=data.content,
        importance=data.importance,
        confidence=data.confidence,
        tags=json.dumps(data.tags),
        related_files=json.dumps(data.related_files),
        related_tools=json.dumps(data.related_tools),
        status=data.status.value,
        type_metadata=json.dumps(data.type_metadata) if data.type_metadata is not None else None,
        file_path=data.file_path,
        commit_sha=data.commit_sha,
        # Sub-phase 1.15 quality + traceability fields
        review_status=data.review_status.value if data.review_status else None,
        source_type=data.source_type.value if data.source_type else None,
        source_quote=data.source_quote,
        symbol_name=data.symbol_name,
        line_start=data.line_start,
        line_end=data.line_end,
        branch_name=data.branch_name,
        # Sub-phase 1.18 privacy field
        privacy_level=data.privacy_level.value if data.privacy_level else "internal",
        # Sub-phase 1.20 semantic embedding (None for manually created memories)
        embedding=embedding,
    )
    db.add(memory)
    db.flush()  # assign ID so quality helpers can reference it
    _refresh_derived_fields(db, memory)
    # Sub-phase 1.23: set tier (respect explicit override from caller, else auto-compute)
    if data.tier is not None:
        memory.tier = data.tier.value
    else:
        memory.tier = compute_tier(
            memory.importance,
            memory.updated_at or datetime.now(timezone.utc),
            last_accessed_at=None,  # new memory — never accessed yet
        )
    # Sub-phase 1.24: set embedding metadata when embedding is stored
    if embedding is not None:
        if embedding_model is None or embedding_dim is None:
            # Auto-populate from semantic_classifier constants (avoids coupling callers)
            try:
                from app.services.semantic_classifier import EMBEDDING_MODEL_NAME, EMBEDDING_DIM
                embedding_model = embedding_model or EMBEDDING_MODEL_NAME
                embedding_dim = embedding_dim or EMBEDDING_DIM
            except ImportError:
                pass
        memory.embedding_model = embedding_model
        memory.embedding_dim = embedding_dim
    db.commit()
    db.refresh(memory)
    return memory


_PRIVACY_ORDER = ["public", "internal", "sensitive", "secret"]


def get_memories(
    db: Session,
    project_id: str,
    type: Optional[str] = None,
    status: Optional[str] = None,
    importance_min: Optional[int] = None,
    tag: Optional[str] = None,
    max_privacy_level: Optional[str] = None,
    tier: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> list[models.Memory]:
    get_project(db, project_id)
    q = db.query(models.Memory).filter(models.Memory.project_id == project_id)
    if type:
        q = q.filter(models.Memory.type == type)
    if status:
        q = q.filter(models.Memory.status == status)
    if importance_min is not None:
        q = q.filter(models.Memory.importance >= importance_min)
    if tag:
        q = q.filter(models.Memory.tags.contains(f'"{tag}"'))
    if max_privacy_level and max_privacy_level in _PRIVACY_ORDER:
        cutoff = _PRIVACY_ORDER.index(max_privacy_level)
        allowed = _PRIVACY_ORDER[:cutoff + 1]
        q = q.filter(models.Memory.privacy_level.in_(allowed))
    if tier:
        q = q.filter(models.Memory.tier == tier)
    return q.order_by(models.Memory.importance.desc(), models.Memory.created_at.desc()).offset(skip).limit(limit).all()


def get_memory(db: Session, memory_id: str) -> models.Memory:
    memory = db.query(models.Memory).filter(models.Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return memory


# ---------------------------------------------------------------------------
# Deduplication (Sub-phase 1.44)
# ---------------------------------------------------------------------------

_DEDUP_THRESHOLD = 0.97  # cosine similarity above which two memories are considered duplicates


def find_near_duplicates(
    db: Session,
    project_id: str,
    embedding: bytes,
    threshold: float = _DEDUP_THRESHOLD,
    limit: int = 5,
) -> list[tuple[models.Memory, float]]:
    """Return existing memories whose embedding cosine similarity exceeds threshold.

    Loads all embedded memories in the project and computes similarity in Python.
    Suitable for Phase 1 project sizes (100s–1000s of memories). Phase 2 will
    replace this with an ANN index when projects grow larger.

    Returns list of (Memory, similarity) tuples sorted by similarity desc.
    """
    import numpy as np

    query_vec = np.frombuffer(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(query_vec))
    if norm == 0:
        return []
    query_vec = query_vec / norm

    candidates = (
        db.query(models.Memory)
        .filter(
            models.Memory.project_id == project_id,
            models.Memory.embedding.isnot(None),
        )
        .all()
    )

    results: list[tuple[models.Memory, float]] = []
    for mem in candidates:
        try:
            vec = np.frombuffer(mem.embedding, dtype=np.float32)
            n = float(np.linalg.norm(vec))
            if n == 0:
                continue
            sim = float(np.dot(query_vec, vec / n))
            if sim >= threshold:
                results.append((mem, round(sim, 4)))
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def backfill_embeddings(db: Session, project_id: str) -> dict[str, int]:
    """Embed every memory in the project that currently has no embedding.

    Uses title + content as the text so the vector represents the full
    memory, consistent with how manually created memories are embedded.
    Returns counts of backfilled and already-embedded memories.
    """
    from app.services import semantic_classifier as _sc
    from app.services.semantic_classifier import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

    get_project(db, project_id)
    rows = (
        db.query(models.Memory)
        .filter(models.Memory.project_id == project_id)
        .filter(models.Memory.embedding.is_(None))
        .all()
    )
    for row in rows:
        text = f"{row.title}. {row.content}"
        row.embedding = _sc.embed_text(text)
        row.embedding_model = EMBEDDING_MODEL_NAME
        row.embedding_dim = EMBEDDING_DIM
    if rows:
        db.commit()
    total = db.query(models.Memory).filter(models.Memory.project_id == project_id).count()
    return {"backfilled": len(rows), "already_embedded": total - len(rows)}


def update_memory(
    db: Session, memory_id: str, data: schemas.MemoryUpdate
) -> models.Memory:
    memory = get_memory(db, memory_id)
    updates = data.model_dump(exclude_unset=True)
    changelog_entries: list[models.MemoryChangelog] = []

    for field, value in updates.items():
        old_raw = getattr(memory, field, None)

        # Serialize to storage format
        if field == "type" and value is not None:
            value = value.value
        if field == "status" and value is not None:
            value = value.value
        if field == "review_status" and value is not None:
            value = value.value if hasattr(value, "value") else value
        if field == "source_type" and value is not None:
            value = value.value if hasattr(value, "value") else value
        if field == "privacy_level" and value is not None:
            value = value.value if hasattr(value, "value") else value
        if field in ("tags", "related_files", "related_tools") and value is not None:
            value = json.dumps(value)
        if field == "type_metadata":
            value = json.dumps(value) if value is not None else None

        setattr(memory, field, value)

        # Track changes for tracked fields
        if field in _CHANGELOG_FIELDS:
            old_str = str(old_raw) if old_raw is not None else None
            new_str = str(value) if value is not None else None
            if old_str != new_str:
                changelog_entries.append(models.MemoryChangelog(
                    memory_id=memory.id,
                    field=field,
                    old_value=old_str,
                    new_value=new_str,
                ))

    for entry in changelog_entries:
        db.add(entry)
    _refresh_derived_fields(db, memory)
    # Sub-phase 1.23/1.25: recalculate tier unless caller explicitly set it
    if "tier" not in updates or updates.get("tier") is None:
        memory.tier = compute_tier(
            memory.importance,
            datetime.now(timezone.utc),
            last_accessed_at=memory.last_accessed_at,
        )
    db.commit()
    db.refresh(memory)
    return memory


def delete_memory(db: Session, memory_id: str) -> None:
    memory = get_memory(db, memory_id)
    # Clean up orphan records before deleting — SQLite won't enforce FK cascades
    # unless PRAGMA foreign_keys = ON is set, which we avoid for Phase 1 safety.
    db.query(models.MemoryLink).filter(
        or_(
            models.MemoryLink.source_memory_id == memory_id,
            models.MemoryLink.target_memory_id == memory_id,
        )
    ).delete(synchronize_session=False)
    db.query(models.MemoryChangelog).filter(
        models.MemoryChangelog.memory_id == memory_id
    ).delete(synchronize_session=False)
    db.delete(memory)
    db.commit()


# ---------------------------------------------------------------------------
# Memory Links (Sub-phase 1.11)
# ---------------------------------------------------------------------------

def create_memory_link(
    db: Session, source_memory_id: str, data: schemas.MemoryLinkCreate
) -> models.MemoryLink:
    if source_memory_id == data.target_memory_id:
        raise HTTPException(status_code=400, detail="A memory cannot link to itself")
    # Validate both memories exist
    get_memory(db, source_memory_id)
    get_memory(db, data.target_memory_id)
    # Prevent exact duplicate links
    existing = (
        db.query(models.MemoryLink)
        .filter(
            models.MemoryLink.source_memory_id == source_memory_id,
            models.MemoryLink.target_memory_id == data.target_memory_id,
            models.MemoryLink.relationship_type == data.relationship_type.value,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail="A link with this relationship type already exists between these memories",
        )
    link = models.MemoryLink(
        source_memory_id=source_memory_id,
        target_memory_id=data.target_memory_id,
        relationship_type=data.relationship_type.value,
        note=data.note,
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


def get_memory_links(db: Session, memory_id: str) -> list[models.MemoryLink]:
    get_memory(db, memory_id)
    return (
        db.query(models.MemoryLink)
        .filter(
            or_(
                models.MemoryLink.source_memory_id == memory_id,
                models.MemoryLink.target_memory_id == memory_id,
            )
        )
        .order_by(models.MemoryLink.created_at.asc())
        .all()
    )


def get_memory_link(db: Session, link_id: str) -> models.MemoryLink:
    link = db.query(models.MemoryLink).filter(models.MemoryLink.id == link_id).first()
    if not link:
        raise HTTPException(status_code=404, detail=f"Memory link {link_id} not found")
    return link


def delete_memory_link(db: Session, link_id: str) -> None:
    link = get_memory_link(db, link_id)
    db.delete(link)
    db.commit()


# ---------------------------------------------------------------------------
# Temporal invalidation (Sub-phase 1.14)
# ---------------------------------------------------------------------------

def supersede_memory(
    db: Session, memory_id: str, new_data: schemas.MemoryCreate
) -> tuple[models.Memory, models.Memory]:
    """Atomically replace old_memory with new_memory in a single transaction."""
    old_memory = get_memory(db, memory_id)

    # Build new memory ORM object without committing yet
    new_mem = models.Memory(
        project_id=old_memory.project_id,
        source_session_id=new_data.source_session_id,
        type=new_data.type.value,
        title=new_data.title,
        content=new_data.content,
        importance=new_data.importance,
        confidence=new_data.confidence,
        tags=json.dumps(new_data.tags),
        related_files=json.dumps(new_data.related_files),
        related_tools=json.dumps(new_data.related_tools),
        status=new_data.status.value,
        type_metadata=json.dumps(new_data.type_metadata) if new_data.type_metadata else None,
        file_path=new_data.file_path,
        commit_sha=new_data.commit_sha,
        review_status=new_data.review_status.value if new_data.review_status else None,
        source_type=new_data.source_type.value if new_data.source_type else None,
        source_quote=new_data.source_quote,
        symbol_name=new_data.symbol_name,
        line_start=new_data.line_start,
        line_end=new_data.line_end,
        branch_name=new_data.branch_name,
        privacy_level=new_data.privacy_level.value if new_data.privacy_level else "internal",
    )
    db.add(new_mem)
    db.flush()  # assigns new_mem.id without committing
    _refresh_derived_fields(db, new_mem)

    # Mark old memory as superseded
    now = datetime.now(timezone.utc)
    old_status = old_memory.status
    old_memory.status = "superseded"
    old_memory.valid_until = now
    old_memory.superseded_by = new_mem.id

    # Write changelog entries for the three changed fields
    for field, old_val, new_val in [
        ("status",       old_status,    "superseded"),
        ("valid_until",  None,          str(now)),
        ("superseded_by", None,         new_mem.id),
    ]:
        db.add(models.MemoryChangelog(
            memory_id=old_memory.id,
            field=field,
            old_value=str(old_val) if old_val is not None else None,
            new_value=new_val,
        ))

    # Auto-create a supersedes link: old memory → new memory
    # superseded_at records exactly when the old memory was replaced
    db.add(models.MemoryLink(
        source_memory_id=old_memory.id,
        target_memory_id=new_mem.id,
        relationship_type="supersedes",
        superseded_at=now,
        note="Auto-created by supersede_memory",
    ))

    db.commit()
    db.refresh(old_memory)
    db.refresh(new_mem)
    return old_memory, new_mem


def auto_supersede_on_conflict(
    db: Session,
    new_memory: models.Memory,
    title_threshold: float = 0.85,
) -> list[models.Memory]:
    """Check whether the new memory contradicts existing active memories.

    Scans all active memories of the same type in the same project.
    If title similarity >= title_threshold, the old memory is marked superseded
    and a supersedes link (new → old) is created with superseded_at set.

    Uses the same SequenceMatcher difflib approach as detect_conflicts() for
    consistency.  Only fires for types where supersession makes semantic sense:
    decision, constraint, structure, how_to.

    Returns list of memories that were superseded.
    Called automatically after create_memory() in the router when
    auto_supersede=true is passed.
    """
    _SUPERSEDABLE_TYPES = frozenset({"decision", "constraint", "structure", "how_to"})
    if new_memory.type not in _SUPERSEDABLE_TYPES:
        return []

    candidates = (
        db.query(models.Memory)
        .filter(
            models.Memory.project_id == new_memory.project_id,
            models.Memory.type == new_memory.type,
            models.Memory.status == "active",
            models.Memory.id != new_memory.id,
        )
        .all()
    )

    now = datetime.now(timezone.utc)
    superseded: list[models.Memory] = []

    for old in candidates:
        sim = difflib.SequenceMatcher(
            None,
            new_memory.title.lower(),
            old.title.lower(),
        ).ratio()
        if sim < title_threshold:
            continue

        # Check no existing supersedes link already exists in this direction
        existing = (
            db.query(models.MemoryLink)
            .filter(
                models.MemoryLink.source_memory_id == old.id,
                models.MemoryLink.target_memory_id == new_memory.id,
                models.MemoryLink.relationship_type == "supersedes",
            )
            .first()
        )
        if existing:
            continue

        # Mark old memory as superseded
        old_status = old.status
        old.status = "superseded"
        old.valid_until = now
        old.superseded_by = new_memory.id

        for field, old_val, new_val in [
            ("status",        old_status,     "superseded"),
            ("valid_until",   None,           str(now)),
            ("superseded_by", None,           new_memory.id),
        ]:
            db.add(models.MemoryChangelog(
                memory_id=old.id,
                field=field,
                old_value=str(old_val) if old_val is not None else None,
                new_value=new_val,
            ))

        # Create the supersedes link with superseded_at
        db.add(models.MemoryLink(
            source_memory_id=old.id,
            target_memory_id=new_memory.id,
            relationship_type="supersedes",
            superseded_at=now,
            note=f"Auto-superseded by: {new_memory.title[:80]}",
        ))

        superseded.append(old)

    if superseded:
        db.commit()
        for m in superseded:
            db.refresh(m)

    return superseded


def get_memory_history(db: Session, memory_id: str) -> list[models.MemoryChangelog]:
    get_memory(db, memory_id)  # 404 if not found
    return (
        db.query(models.MemoryChangelog)
        .filter(models.MemoryChangelog.memory_id == memory_id)
        .order_by(models.MemoryChangelog.changed_at.asc())
        .all()
    )


def get_stale_memories(
    db: Session, project_id: str, days: int = 30
) -> list[models.Memory]:
    """Return active memories not updated in the last `days` days."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    get_project(db, project_id)
    return (
        db.query(models.Memory)
        .filter(
            models.Memory.project_id == project_id,
            models.Memory.status == "active",
            models.Memory.updated_at < cutoff,
        )
        .order_by(models.Memory.updated_at.asc())
        .all()
    )


# ---------------------------------------------------------------------------
# Current Truth Resolver (Sub-phase 1.17)
# ---------------------------------------------------------------------------

_CONFLICT_TITLE_THRESHOLD = 0.70   # memories with title similarity ≥ 0.70 are "conflicting"
_CONFLICT_CONTENT_THRESHOLD = 0.85  # memories with content similarity ≥ 0.85 are "duplicate content"


def detect_conflicts(db: Session, project_id: str) -> schemas.ConflictReport:
    """
    Scan all active memories in a project and identify conflicting pairs.

    Two memories conflict if:
      - They share the same type AND their titles are ≥70% similar  (title_overlap)
      - OR their content is ≥85% similar regardless of type            (content_overlap)
    """
    get_project(db, project_id)
    memories = (
        db.query(models.Memory)
        .filter(
            models.Memory.project_id == project_id,
            models.Memory.status == "active",
        )
        .order_by(models.Memory.created_at.asc())
        .all()
    )

    responses = [schemas.MemoryResponse.model_validate(m) for m in memories]
    pairs: list[schemas.ConflictPair] = []
    seen: set[frozenset] = set()  # prevent duplicate pair reporting

    for i, a in enumerate(responses):
        for j, b in enumerate(responses):
            if j <= i:
                continue
            pair_key = frozenset({a.id, b.id})
            if pair_key in seen:
                continue

            # Title overlap check (same type required)
            if a.type == b.type:
                t_sim = difflib.SequenceMatcher(
                    None, a.title.lower(), b.title.lower()
                ).ratio()
                if t_sim >= _CONFLICT_TITLE_THRESHOLD:
                    seen.add(pair_key)
                    pairs.append(schemas.ConflictPair(
                        memory_a=a,
                        memory_b=b,
                        similarity_score=round(t_sim, 3),
                        conflict_type="title_overlap",
                    ))
                    continue

            # Content overlap check (cross-type)
            c_norm_a = " ".join(a.content.lower().split())
            c_norm_b = " ".join(b.content.lower().split())
            c_sim = difflib.SequenceMatcher(None, c_norm_a, c_norm_b).ratio()
            if c_sim >= _CONFLICT_CONTENT_THRESHOLD:
                seen.add(pair_key)
                pairs.append(schemas.ConflictPair(
                    memory_a=a,
                    memory_b=b,
                    similarity_score=round(c_sim, 3),
                    conflict_type="content_overlap",
                ))

    return schemas.ConflictReport(
        project_id=project_id,
        total_conflicts=len(pairs),
        pairs=pairs,
    )


def resolve_conflicts(
    db: Session,
    project_id: str,
    dry_run: bool = False,
) -> schemas.ResolveConflictsResult:
    """
    Auto-resolve all detected conflicts in a project.

    Resolution strategy: keep the higher-quality memory (quality_score DESC,
    then importance DESC, then most recently updated). Mark the loser as
    superseded, pointing at the winner. Idempotent — already-superseded
    memories are never picked as a candidate.

    If dry_run=True, report what would happen without writing anything.
    """
    report = detect_conflicts(db, project_id)
    resolved_pairs: list[schemas.ConflictPair] = []

    for pair in report.pairs:
        # Reload from DB to get current status (a previous resolution may have changed it)
        a_orm = db.query(models.Memory).filter(models.Memory.id == pair.memory_a.id).first()
        b_orm = db.query(models.Memory).filter(models.Memory.id == pair.memory_b.id).first()

        if not a_orm or not b_orm:
            continue
        if a_orm.status != "active" or b_orm.status != "active":
            continue  # already resolved

        def _score(m: models.Memory) -> tuple:
            return (m.quality_score or 0.0, m.importance, m.updated_at or datetime.min)

        winner, loser = (a_orm, b_orm) if _score(a_orm) >= _score(b_orm) else (b_orm, a_orm)

        if not dry_run:
            now = datetime.now(timezone.utc)
            old_status = loser.status
            loser.status = "superseded"
            loser.valid_until = now
            loser.superseded_by = winner.id
            for field, old_val, new_val in [
                ("status",       old_status, "superseded"),
                ("valid_until",  None,       str(now)),
                ("superseded_by", None,      winner.id),
            ]:
                db.add(models.MemoryChangelog(
                    memory_id=loser.id,
                    field=field,
                    old_value=str(old_val) if old_val is not None else None,
                    new_value=new_val,
                ))
            # Auto-create supersedes link: loser → winner
            existing_link = (
                db.query(models.MemoryLink)
                .filter(
                    models.MemoryLink.source_memory_id == loser.id,
                    models.MemoryLink.target_memory_id == winner.id,
                    models.MemoryLink.relationship_type == "supersedes",
                )
                .first()
            )
            if not existing_link:
                db.add(models.MemoryLink(
                    source_memory_id=loser.id,
                    target_memory_id=winner.id,
                    relationship_type="supersedes",
                    superseded_at=now,
                    note="Auto-created by resolve_conflicts",
                ))
            db.flush()

        resolved_pairs.append(pair)

    if not dry_run and resolved_pairs:
        db.commit()

    return schemas.ResolveConflictsResult(
        resolved_count=len(resolved_pairs),
        pairs_resolved=resolved_pairs,
    )


# ---------------------------------------------------------------------------
# Memory Health Dashboard (Sub-phase 1.16)
# ---------------------------------------------------------------------------

def get_project_health(db: Session, project_id: str) -> schemas.ProjectHealthReport:
    """Compute a full health snapshot for a project's memory store."""
    get_project(db, project_id)  # 404 guard

    now = datetime.now(timezone.utc)
    memories = (
        db.query(models.Memory)
        .filter(models.Memory.project_id == project_id)
        .all()
    )
    total = len(memories)

    # Status counts
    status_counts: dict[str, int] = {
        "active": 0, "resolved": 0, "stale": 0, "archived": 0, "superseded": 0,
    }
    for m in memories:
        key = m.status if m.status in status_counts else "active"
        status_counts[key] = status_counts.get(key, 0) + 1

    # Quality buckets (only active memories contribute to quality metrics)
    active_mems = [m for m in memories if m.status == "active"]
    quality_scores = [m.quality_score for m in active_mems if m.quality_score is not None]
    buckets = schemas.QualityBucket(
        excellent=sum(1 for s in quality_scores if s >= 0.75),
        good=sum(1 for s in quality_scores if 0.50 <= s < 0.75),
        fair=sum(1 for s in quality_scores if 0.25 <= s < 0.50),
        poor=sum(1 for s in quality_scores if s < 0.25) + (len(active_mems) - len(quality_scores)),
    )
    avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None

    # Review status breakdown (active memories only)
    rs_counts: dict[str, int] = {
        "auto_extracted": 0, "verified": 0, "needs_review": 0,
        "rejected": 0, "draft": 0, "unset": 0,
    }
    for m in active_mems:
        rs = m.review_status or "unset"
        rs_counts[rs] = rs_counts.get(rs, 0) + 1

    review_breakdown = schemas.ReviewStatusBreakdown(**rs_counts)

    # Traceability (all memories)
    n = total or 1  # avoid division by zero
    pct_session = sum(1 for m in memories if m.source_session_id) / n
    pct_quote = sum(1 for m in memories if m.source_quote) / n
    pct_src_type = sum(1 for m in memories if m.source_type) / n

    # Type distribution (active only)
    type_dist: dict[str, int] = {}
    for m in active_mems:
        type_dist[m.type] = type_dist.get(m.type, 0) + 1

    # Confidence
    confidences = [m.confidence for m in active_mems if m.confidence is not None]
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else None
    low_conf = sum(1 for c in confidences if c < 0.60)

    # Actionable issues
    issues: list[schemas.HealthIssue] = []
    active_n = len(active_mems) or 1

    unreviewed = rs_counts.get("auto_extracted", 0)
    if unreviewed > 0 and (unreviewed / active_n) > 0.5:
        issues.append(schemas.HealthIssue(
            severity="warning",
            code="high_unreviewed_count",
            message=f"{unreviewed} memories ({unreviewed * 100 // active_n}%) are auto-extracted and unreviewed.",
        ))

    from datetime import timedelta

    def _aware(dt: datetime) -> datetime:
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    stale_cutoff = now - timedelta(days=30)
    stale_active = [
        m for m in active_mems if m.updated_at and _aware(m.updated_at) < stale_cutoff
    ]
    if stale_active:
        issues.append(schemas.HealthIssue(
            severity="warning",
            code="stale_active_memories",
            message=f"{len(stale_active)} active memories have not been updated in 30+ days.",
        ))

    if avg_quality is not None and avg_quality < 0.40:
        issues.append(schemas.HealthIssue(
            severity="warning",
            code="low_avg_quality",
            message=f"Average quality score is {avg_quality:.2f} — below the 0.40 threshold. Add file paths, tags, or type metadata.",
        ))

    no_source = sum(1 for m in memories if not m.source_session_id and not m.source_type)
    if total > 0 and (no_source / total) > 0.6:
        issues.append(schemas.HealthIssue(
            severity="info",
            code="low_traceability",
            message=f"{no_source} memories ({no_source * 100 // total}%) lack any source reference.",
        ))

    rejected = rs_counts.get("rejected", 0)
    if rejected > 0:
        issues.append(schemas.HealthIssue(
            severity="info",
            code="rejected_memories_present",
            message=f"{rejected} memories are marked rejected. Consider deleting or archiving them.",
        ))

    return schemas.ProjectHealthReport(
        project_id=project_id,
        generated_at=now,
        total_memories=total,
        active_count=status_counts["active"],
        resolved_count=status_counts["resolved"],
        stale_count=status_counts["stale"],
        archived_count=status_counts["archived"],
        superseded_count=status_counts["superseded"],
        avg_quality_score=avg_quality,
        quality_buckets=buckets,
        review_breakdown=review_breakdown,
        pct_with_session_ref=round(pct_session, 3),
        pct_with_source_quote=round(pct_quote, 3),
        pct_with_source_type=round(pct_src_type, 3),
        type_distribution=type_dist,
        avg_confidence=avg_conf,
        low_confidence_count=low_conf,
        issues=issues,
    )


def get_all_links_for_project(db: Session, project_id: str) -> list[models.MemoryLink]:
    """Fetch all links where source or target memory belongs to this project."""
    memory_ids = (
        db.query(models.Memory.id)
        .filter(models.Memory.project_id == project_id)
        .scalar_subquery()
    )
    return (
        db.query(models.MemoryLink)
        .filter(
            or_(
                models.MemoryLink.source_memory_id.in_(memory_ids),
                models.MemoryLink.target_memory_id.in_(memory_ids),
            )
        )
        .all()
    )


# ---------------------------------------------------------------------------
# Access tracking (Sub-phase 1.25)
# ---------------------------------------------------------------------------

def track_memory_access(db: Session, memory: models.Memory) -> None:
    """
    Increment access_count and update last_accessed_at.
    Called as a side effect of GET /memories/{id}.
    Also recalculates tier — recently accessed memories stay in "working".
    """
    now = datetime.now(timezone.utc)
    memory.access_count = (memory.access_count or 0) + 1
    memory.last_accessed_at = now
    memory.tier = compute_tier(memory.importance, memory.updated_at or now, last_accessed_at=now)
    memory.decay_score = compute_decay_score(
        memory.importance, memory.updated_at or now, last_accessed_at=now, now=now
    )
    db.commit()
    db.refresh(memory)


def get_most_accessed_memories(
    db: Session,
    project_id: str,
    limit: int = 10,
) -> list[models.Memory]:
    """Return memories sorted by access_count descending, then last_accessed_at descending."""
    get_project(db, project_id)
    from sqlalchemy import nulls_last
    return (
        db.query(models.Memory)
        .filter(models.Memory.project_id == project_id)
        .order_by(
            models.Memory.access_count.desc(),
            nulls_last(models.Memory.last_accessed_at.desc()),
        )
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Tier bulk operations (Sub-phase 1.23)
# ---------------------------------------------------------------------------

def retier_project_memories(
    db: Session,
    project_id: str,
    now: datetime | None = None,
) -> schemas.RetierResult:
    """Recompute tier for every memory in the project based on current rules."""
    get_project(db, project_id)
    if now is None:
        now = datetime.now(timezone.utc)
    memories = db.query(models.Memory).filter(
        models.Memory.project_id == project_id
    ).all()
    updated = 0
    working = 0
    archival = 0
    for mem in memories:
        new_tier = compute_tier(
            mem.importance,
            mem.updated_at or now,
            last_accessed_at=mem.last_accessed_at,
            now=now,
        )
        if mem.tier != new_tier:
            mem.tier = new_tier
            updated += 1
        if new_tier == "working":
            working += 1
        else:
            archival += 1
        mem.decay_score = compute_decay_score(
            mem.importance, mem.updated_at or now, mem.last_accessed_at, now=now
        )
    db.commit()
    return schemas.RetierResult(
        updated_count=updated,
        working_count=working,
        archival_count=archival,
    )


# ---------------------------------------------------------------------------
# Entity CRUD (Sub-phase 1.22)
# ---------------------------------------------------------------------------

def create_memory_entities(
    db: Session,
    memory_id: str,
    project_id: str,
    entities: list,  # list[entity_extractor.Entity]
) -> list[models.MemoryEntity]:
    """Persist extracted entities for a memory. Skips duplicates (same normalized_text)."""
    if not entities:
        return []
    seen: set[str] = set()
    rows = []
    for ent in entities:
        if ent.normalized in seen:
            continue
        seen.add(ent.normalized)
        row = models.MemoryEntity(
            memory_id=memory_id,
            project_id=project_id,
            entity_text=ent.text,
            entity_label=ent.label,
            normalized_text=ent.normalized,
        )
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def get_entities_for_memory(db: Session, memory_id: str) -> list[models.MemoryEntity]:
    return (
        db.query(models.MemoryEntity)
        .filter(models.MemoryEntity.memory_id == memory_id)
        .order_by(models.MemoryEntity.entity_label, models.MemoryEntity.normalized_text)
        .all()
    )


# ---------------------------------------------------------------------------
# Auto-linking via entity overlap (Sub-phase 1.29)
# ---------------------------------------------------------------------------

_AUTOLINK_LABELS = frozenset({"TECH", "ORG", "PRODUCT"})


def autolink_by_entities(
    db: Session,
    project_id: str,
    new_memory_id: str,
    min_shared: int = 1,
) -> list[models.MemoryLink]:
    """
    Create related_to links between new_memory_id and existing memories that
    share >= min_shared high-value entities (TECH / ORG / PRODUCT labels).

    Skips pairs that already have a related_to link in either direction.
    Returns the list of newly created MemoryLink records.
    """
    new_entities = (
        db.query(models.MemoryEntity)
        .filter(
            models.MemoryEntity.memory_id == new_memory_id,
            models.MemoryEntity.entity_label.in_(list(_AUTOLINK_LABELS)),
        )
        .all()
    )
    if not new_entities:
        return []

    high_value_normalized = {e.normalized_text for e in new_entities}

    candidates = (
        db.query(models.MemoryEntity)
        .filter(
            models.MemoryEntity.project_id == project_id,
            models.MemoryEntity.memory_id != new_memory_id,
            models.MemoryEntity.normalized_text.in_(list(high_value_normalized)),
            models.MemoryEntity.entity_label.in_(list(_AUTOLINK_LABELS)),
        )
        .all()
    )

    # Count shared entities per candidate memory
    overlap: dict[str, set[str]] = {}
    for ent in candidates:
        overlap.setdefault(ent.memory_id, set()).add(ent.normalized_text)

    created: list[models.MemoryLink] = []
    for other_id, shared in overlap.items():
        if len(shared) < min_shared:
            continue
        # Prevent duplicate links in either direction
        existing = (
            db.query(models.MemoryLink)
            .filter(
                or_(
                    and_(
                        models.MemoryLink.source_memory_id == new_memory_id,
                        models.MemoryLink.target_memory_id == other_id,
                    ),
                    and_(
                        models.MemoryLink.source_memory_id == other_id,
                        models.MemoryLink.target_memory_id == new_memory_id,
                    ),
                ),
                models.MemoryLink.relationship_type == "related_to",
            )
            .first()
        )
        if existing:
            continue
        link = models.MemoryLink(
            source_memory_id=new_memory_id,
            target_memory_id=other_id,
            relationship_type="related_to",
            note=f"Auto-linked: shared {', '.join(sorted(shared))}",
        )
        db.add(link)
        created.append(link)

    if created:
        db.commit()
        for lnk in created:
            db.refresh(lnk)
    return created


def autolink_project_memories(
    db: Session,
    project_id: str,
    min_shared: int = 1,
) -> schemas.AutolinkResult:
    """Run autolink for every memory in the project. Idempotent — skips existing links."""
    get_project(db, project_id)
    memories = (
        db.query(models.Memory)
        .filter(models.Memory.project_id == project_id)
        .all()
    )
    total = 0
    for mem in memories:
        links = autolink_by_entities(db, project_id, mem.id, min_shared=min_shared)
        total += len(links)
    return schemas.AutolinkResult(project_id=project_id, links_created=total)


# ---------------------------------------------------------------------------
# Cluster CRUD (Sub-phase 1.27)
# ---------------------------------------------------------------------------

def recluster_project_memories(
    db: Session,
    project_id: str,
) -> schemas.ClusterResult:
    """
    Run DBSCAN on all embedded memories in the project.
    Writes cluster_id and cluster_label back to each memory record.
    Memories without embeddings are set to cluster_id=None, cluster_label=None.
    """
    get_project(db, project_id)
    from app.services.clustering import compute_clusters

    memories = db.query(models.Memory).filter(
        models.Memory.project_id == project_id
    ).all()

    assignments = compute_clusters(memories)
    assignment_map = {a.memory_id: a for a in assignments}

    total_clustered = 0
    total_noise = 0
    total_unclustered = 0
    cluster_groups: dict[int, list[str]] = {}
    cluster_labels: dict[int, str] = {}

    for mem in memories:
        if mem.id in assignment_map:
            a = assignment_map[mem.id]
            mem.cluster_id = a.cluster_id
            mem.cluster_label = a.cluster_label
            if a.cluster_id >= 0:
                total_clustered += 1
                cluster_groups.setdefault(a.cluster_id, []).append(mem.id)
                cluster_labels[a.cluster_id] = a.cluster_label or ""
            else:
                total_noise += 1
        else:
            mem.cluster_id = None
            mem.cluster_label = None
            total_unclustered += 1

    db.commit()

    clusters = [
        schemas.ClusterInfo(
            cluster_id=cid,
            cluster_label=cluster_labels[cid],
            memory_count=len(mids),
            memory_ids=mids,
        )
        for cid, mids in sorted(cluster_groups.items())
    ]

    return schemas.ClusterResult(
        project_id=project_id,
        total_clustered=total_clustered,
        total_noise=total_noise,
        total_unclustered=total_unclustered,
        clusters=clusters,
    )


def get_project_clusters(
    db: Session,
    project_id: str,
) -> list[schemas.ClusterInfo]:
    """Return all non-noise clusters for the project (cluster_id >= 0)."""
    get_project(db, project_id)
    memories = (
        db.query(models.Memory)
        .filter(
            models.Memory.project_id == project_id,
            models.Memory.cluster_id >= 0,
        )
        .all()
    )
    groups: dict[int, list[str]] = {}
    labels: dict[int, str] = {}
    for m in memories:
        if m.cluster_id is not None and m.cluster_id >= 0:
            groups.setdefault(m.cluster_id, []).append(m.id)
            labels[m.cluster_id] = m.cluster_label or ""
    return [
        schemas.ClusterInfo(
            cluster_id=cid,
            cluster_label=labels[cid],
            memory_count=len(mids),
            memory_ids=mids,
        )
        for cid, mids in sorted(groups.items())
    ]


def get_entities_for_project(
    db: Session,
    project_id: str,
    label: str | None = None,
) -> list[schemas.ProjectEntitySummary]:
    """Return aggregated entity counts across all memories in a project.

    Groups by normalized_text across labels so "redis" appearing in two memories
    with different labels (TECH, ORG) counts as 2 — not two separate groups.
    The most common label per normalized_text is used as the representative label.
    """
    from collections import Counter

    q = db.query(models.MemoryEntity).filter(
        models.MemoryEntity.project_id == project_id
    )
    if label:
        q = q.filter(models.MemoryEntity.entity_label == label.upper())

    all_entities = q.all()

    # Group by normalized_text; track labels and distinct memory_ids
    groups: dict[str, dict] = {}
    for ent in all_entities:
        if ent.normalized_text not in groups:
            groups[ent.normalized_text] = {"labels": [], "memory_ids": []}
        groups[ent.normalized_text]["labels"].append(ent.entity_label)
        groups[ent.normalized_text]["memory_ids"].append(ent.memory_id)

    result = []
    for norm_text, data in sorted(
        groups.items(),
        key=lambda x: len(set(x[1]["memory_ids"])),
        reverse=True,
    ):
        most_common_label = Counter(data["labels"]).most_common(1)[0][0]
        memory_ids = list(set(data["memory_ids"]))
        result.append(
            schemas.ProjectEntitySummary(
                normalized_text=norm_text,
                entity_label=most_common_label,
                count=len(memory_ids),
                memory_ids=memory_ids,
            )
        )
    return result


# ---------------------------------------------------------------------------
# MemorySuggestions (Sub-phase 1.32)
# ---------------------------------------------------------------------------

def create_suggestion(
    db: Session,
    project_id: str,
    data: schemas.SuggestionCreate,
    *,
    status: str = "pending",
    created_by: str = "manual",
    memory_id: str | None = None,
) -> models.MemorySuggestion:
    row = models.MemorySuggestion(
        project_id=project_id,
        source_session_id=data.source_session_id,
        source_message_ids=json.dumps(data.source_message_ids) if data.source_message_ids else None,
        suggested_type=data.suggested_type.value if hasattr(data.suggested_type, "value") else data.suggested_type,
        title=data.title,
        content=data.content,
        confidence=data.confidence,
        status=status,
        created_by=created_by,
        memory_id=memory_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_suggestions(
    db: Session,
    project_id: str,
    status: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[models.MemorySuggestion]:
    if not db.query(models.Project).filter(models.Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
    q = db.query(models.MemorySuggestion).filter(
        models.MemorySuggestion.project_id == project_id
    )
    if status:
        q = q.filter(models.MemorySuggestion.status == status)
    return q.order_by(models.MemorySuggestion.created_at.desc()).offset(offset).limit(limit).all()


def get_suggestion(db: Session, suggestion_id: str) -> models.MemorySuggestion:
    row = db.query(models.MemorySuggestion).filter(
        models.MemorySuggestion.id == suggestion_id
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return row


def update_suggestion(
    db: Session,
    suggestion_id: str,
    data: schemas.SuggestionUpdate,
) -> models.MemorySuggestion:
    row = get_suggestion(db, suggestion_id)
    if data.title is not None:
        row.title = data.title
    if data.content is not None:
        row.content = data.content
    if data.suggested_type is not None:
        row.suggested_type = data.suggested_type.value
        # Mark as edited if status was pending
        if row.status == "pending":
            row.status = "edited"
    if data.status is not None:
        row.status = data.status.value
    if data.confidence is not None:
        row.confidence = data.confidence
    db.commit()
    db.refresh(row)
    return row


def approve_suggestion(
    db: Session,
    suggestion_id: str,
) -> tuple[models.MemorySuggestion, models.Memory]:
    """Promote a pending/edited suggestion to a real Memory row.

    Idempotent: if the suggestion already has a memory_id, returns the
    existing memory rather than creating a duplicate.
    """
    from datetime import datetime, timezone

    row = get_suggestion(db, suggestion_id)

    # Idempotent: already approved with an existing memory
    if row.memory_id:
        memory = db.query(models.Memory).filter(models.Memory.id == row.memory_id).first()
        if memory:
            return row, memory

    try:
        mem_type = schemas.MemoryType(row.suggested_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown memory type: {row.suggested_type}")

    mem_data = schemas.MemoryCreate(
        type=mem_type,
        title=row.title,
        content=row.content,
        confidence=row.confidence,
        source_session_id=row.source_session_id,
        review_status=schemas.ReviewStatus.auto_extracted,
        source_type=schemas.SourceType.ai_session if row.source_session_id else schemas.SourceType.manual,
    )
    from app.services import semantic_classifier as _sc
    embedding_bytes = _sc.embed_text(f"{row.title}. {row.content}")
    memory = create_memory(db, row.project_id, mem_data, embedding=embedding_bytes)

    now = datetime.now(timezone.utc)
    row.memory_id = memory.id
    row.status = "approved"
    row.reviewed_at = now
    db.commit()
    db.refresh(row)
    db.refresh(memory)
    return row, memory


def delete_suggestion(db: Session, suggestion_id: str) -> None:
    row = get_suggestion(db, suggestion_id)
    db.delete(row)
    db.commit()


# ---------------------------------------------------------------------------
# ContextPackets (Sub-phase 1.33)
# ---------------------------------------------------------------------------

def _assemble_packet_content(
    db: Session,
    project_id: str,
    memory_ids: list[str],
    session_ids: list[str],
    intent: str | None,
    target_tool: str,
) -> str:
    """Build a structured markdown handoff document from selected memories/sessions.

    The format is designed to be pasted directly into any AI tool's context
    window. Sections are ordered: intent → sessions (raw) → memories by type.
    """
    lines: list[str] = []

    # Header
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    project_name = project.name if project else project_id
    lines.append(f"# Context Packet — {project_name}")
    lines.append(f"**Target tool:** {target_tool}")
    if intent:
        lines.append(f"**Intent:** {intent}")
    lines.append("")

    # Sessions section
    if session_ids:
        fetched_sessions = (
            db.query(models.AISession)
            .filter(
                models.AISession.id.in_(session_ids),
                models.AISession.project_id == project_id,
            )
            .all()
        )
        if fetched_sessions:
            lines.append("## Sessions")
            for s in fetched_sessions:
                lines.append(f"### {s.title} ({s.tool_name})")
                if s.session_date:
                    lines.append(f"_Date: {s.session_date}_")
                if s.summary:
                    lines.append(f"**Summary:** {s.summary}")
                lines.append("")
                # Include up to 2000 chars of raw content to keep packets manageable
                raw = s.raw_content[:2000]
                if len(s.raw_content) > 2000:
                    raw += "\n\n_[content truncated]_"
                lines.append(raw)
                lines.append("")

    # Memories section — grouped by type
    if memory_ids:
        fetched_memories = (
            db.query(models.Memory)
            .filter(
                models.Memory.id.in_(memory_ids),
                models.Memory.project_id == project_id,
            )
            .all()
        )
        if fetched_memories:
            # Group by type for readability
            by_type: dict[str, list[models.Memory]] = {}
            for m in fetched_memories:
                by_type.setdefault(m.type, []).append(m)

            lines.append("## Memories")
            for mem_type, mems in sorted(by_type.items()):
                lines.append(f"### {mem_type.replace('_', ' ').title()}")
                for m in sorted(mems, key=lambda x: x.importance, reverse=True):
                    imp_stars = "★" * m.importance + "☆" * (5 - m.importance)
                    lines.append(f"**{m.title}** {imp_stars}")
                    lines.append(m.content)
                    if m.retrieval_hint:
                        lines.append(f"_TL;DR: {m.retrieval_hint}_")
                    tags = json.loads(m.tags) if m.tags else []
                    meaningful = [t for t in tags if t != "auto-extracted"]
                    if meaningful:
                        lines.append(f"Tags: {', '.join(meaningful)}")
                    lines.append("")

    return "\n".join(lines).strip()


def _estimate_packet_tokens(content: str) -> int:
    return max(0, int(len(content.split()) * 1.3))


def create_context_packet(
    db: Session,
    project_id: str,
    data: schemas.ContextPacketCreate,
) -> models.ContextPacket:
    get_project(db, project_id)  # 404 guard
    content = _assemble_packet_content(
        db,
        project_id,
        memory_ids=data.included_memory_ids,
        session_ids=data.included_session_ids,
        intent=data.intent,
        target_tool=data.target_tool,
    )
    row = models.ContextPacket(
        project_id=project_id,
        target_tool=data.target_tool,
        intent=data.intent,
        included_memory_ids=json.dumps(data.included_memory_ids) if data.included_memory_ids else None,
        included_session_ids=json.dumps(data.included_session_ids) if data.included_session_ids else None,
        content=content,
        token_estimate=_estimate_packet_tokens(content),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_context_packets(
    db: Session,
    project_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[models.ContextPacket]:
    if not db.query(models.Project).filter(models.Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
    return (
        db.query(models.ContextPacket)
        .filter(models.ContextPacket.project_id == project_id)
        .order_by(models.ContextPacket.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_context_packet(db: Session, packet_id: str) -> models.ContextPacket:
    row = db.query(models.ContextPacket).filter(models.ContextPacket.id == packet_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Context packet not found")
    return row


def delete_context_packet(db: Session, packet_id: str) -> None:
    row = get_context_packet(db, packet_id)
    # Null out context_packet_id on any handoff events referencing this packet
    db.query(models.HandoffEvent).filter(
        models.HandoffEvent.context_packet_id == packet_id
    ).update({"context_packet_id": None}, synchronize_session=False)
    db.delete(row)
    db.commit()


# ---------------------------------------------------------------------------
# HandoffEvents (Sub-phase 1.34)
# ---------------------------------------------------------------------------

def create_handoff_event(
    db: Session,
    project_id: str,
    data: schemas.HandoffEventCreate,
) -> models.HandoffEvent:
    from datetime import datetime, timezone

    get_project(db, project_id)  # 404 guard

    # Validate context_packet_id belongs to this project if provided
    if data.context_packet_id:
        packet = db.query(models.ContextPacket).filter(
            models.ContextPacket.id == data.context_packet_id,
            models.ContextPacket.project_id == project_id,
        ).first()
        if not packet:
            raise HTTPException(status_code=404, detail="Context packet not found in this project")

    handoff_at = data.handoff_at or datetime.now(timezone.utc)
    row = models.HandoffEvent(
        project_id=project_id,
        context_packet_id=data.context_packet_id,
        source_tool=data.source_tool,
        target_tool=data.target_tool,
        status=data.status.value,
        note=data.note,
        handoff_at=handoff_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_handoff_events(
    db: Session,
    project_id: str,
    status: str | None = None,
    source_tool: str | None = None,
    target_tool: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[models.HandoffEvent]:
    if not db.query(models.Project).filter(models.Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
    q = db.query(models.HandoffEvent).filter(
        models.HandoffEvent.project_id == project_id
    )
    if status:
        q = q.filter(models.HandoffEvent.status == status)
    if source_tool:
        q = q.filter(models.HandoffEvent.source_tool == source_tool)
    if target_tool:
        q = q.filter(models.HandoffEvent.target_tool == target_tool)
    return q.order_by(models.HandoffEvent.handoff_at.desc()).offset(offset).limit(limit).all()


def get_handoff_event(db: Session, event_id: str) -> models.HandoffEvent:
    row = db.query(models.HandoffEvent).filter(models.HandoffEvent.id == event_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Handoff event not found")
    return row


def update_handoff_event(
    db: Session,
    event_id: str,
    data: schemas.HandoffEventUpdate,
) -> models.HandoffEvent:
    row = get_handoff_event(db, event_id)
    if data.status is not None:
        row.status = data.status.value
    if data.note is not None:
        row.note = data.note
    db.commit()
    db.refresh(row)
    return row


def delete_handoff_event(db: Session, event_id: str) -> None:
    row = get_handoff_event(db, event_id)
    db.delete(row)
    db.commit()
