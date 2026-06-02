"""
Sub-phase 1.35 — Canonical YAML export service.

Generates a machine-readable .ai-memory/project-memory.yaml document that
any AI tool, script, or pipeline can parse without knowing the internal DB
schema. The output is stable across versions (schema_version field) and
complete enough that Phase 2 can bootstrap from it.

Schema version: 1.0

Top-level sections:
  schema_version    — "1.0", bump on breaking changes
  exported_at       — ISO-8601 UTC timestamp
  project           — project metadata
  stats             — aggregate counts and score averages
  memories          — full memory records, sorted by (importance desc, updated_at desc)
  sessions          — session metadata (no raw_content — too large; use summary)
  entity_index      — deduplicated named entities across all memories, sorted by count desc
  clusters          — DBSCAN cluster assignments (cluster_id >= 0 only; noise excluded)
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app import models

# ---------------------------------------------------------------------------
# Privacy level ordering — higher index = more restrictive
# ---------------------------------------------------------------------------
_PRIVACY_ORDER = ["public", "internal", "sensitive", "secret"]


def _privacy_allowed(memory_level: str, max_level: str) -> bool:
    """Return True if the memory's privacy level is <= max_level."""
    try:
        return _PRIVACY_ORDER.index(memory_level) <= _PRIVACY_ORDER.index(max_level)
    except ValueError:
        return True   # unknown level — allow to be safe


# ---------------------------------------------------------------------------
# Datetime serialiser — ISO-8601 with Z suffix
# ---------------------------------------------------------------------------
def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_project_section(project: models.Project) -> dict:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "domain": project.domain or "general",
        "tech_stack": project.get_tech_stack(),
        "goals": project.get_goals(),
        "repo_path": project.repo_path,
        "created_at": _iso(project.created_at),
        "updated_at": _iso(project.updated_at),
    }


def _build_stats_section(memories: list[models.Memory]) -> dict:
    if not memories:
        return {
            "total_memories": 0,
            "active_memories": 0,
            "resolved_memories": 0,
            "stale_memories": 0,
            "archived_memories": 0,
            "superseded_memories": 0,
            "avg_importance": None,
            "avg_confidence": None,
            "avg_quality_score": None,
            "avg_decay_score": None,
            "by_type": {},
            "by_tier": {"working": 0, "archival": 0},
            "by_review_status": {},
        }

    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {"working": 0, "archival": 0}
    review_counts: dict[str, int] = {}
    importance_sum = confidence_sum = quality_sum = decay_sum = 0.0
    quality_n = decay_n = 0

    for m in memories:
        status_counts[m.status] = status_counts.get(m.status, 0) + 1
        type_counts[m.type] = type_counts.get(m.type, 0) + 1
        tier = m.tier or "working"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        rs = m.review_status or "unset"
        review_counts[rs] = review_counts.get(rs, 0) + 1
        importance_sum += m.importance or 3
        confidence_sum += m.confidence or 1.0
        if m.quality_score is not None:
            quality_sum += m.quality_score
            quality_n += 1
        if m.decay_score is not None:
            decay_sum += m.decay_score
            decay_n += 1

    n = len(memories)
    return {
        "total_memories": n,
        "active_memories": status_counts.get("active", 0),
        "resolved_memories": status_counts.get("resolved", 0),
        "stale_memories": status_counts.get("stale", 0),
        "archived_memories": status_counts.get("archived", 0),
        "superseded_memories": status_counts.get("superseded", 0),
        "avg_importance": round(importance_sum / n, 2),
        "avg_confidence": round(confidence_sum / n, 4),
        "avg_quality_score": round(quality_sum / quality_n, 4) if quality_n else None,
        "avg_decay_score": round(decay_sum / decay_n, 4) if decay_n else None,
        "by_type": dict(sorted(type_counts.items())),
        "by_tier": tier_counts,
        "by_review_status": dict(sorted(review_counts.items())),
    }


def _build_memory_record(
    m: models.Memory,
    db: Session,
) -> dict:
    """Build a single complete memory record."""
    # Entities
    entities = db.query(models.MemoryEntity).filter(
        models.MemoryEntity.memory_id == m.id
    ).all()
    entity_records = [
        {"text": e.entity_text, "label": e.entity_label, "normalized": e.normalized_text}
        for e in entities
    ]

    # Links (both directions)
    outgoing = db.query(models.MemoryLink).filter(
        models.MemoryLink.source_memory_id == m.id
    ).all()
    link_records = [
        {
            "target_id": lk.target_memory_id,
            "relationship": lk.relationship_type,
            "note": lk.note,
        }
        for lk in outgoing
    ]

    return {
        "id": m.id,
        "type": m.type,
        "title": m.title,
        "content": m.content,
        "importance": m.importance,
        "confidence": m.confidence,
        "status": m.status,
        "tier": m.tier or "working",
        "decay_score": m.decay_score,
        "quality_score": m.quality_score,
        "review_status": m.review_status,
        "source_type": m.source_type,
        "privacy_level": m.privacy_level or "internal",
        "retrieval_hint": m.retrieval_hint,
        "cluster_id": m.cluster_id,
        "cluster_label": m.cluster_label,
        "tags": m.get_tags(),
        "related_files": m.get_related_files(),
        "related_tools": m.get_related_tools(),
        "source_quote": m.source_quote,
        "file_path": m.file_path,
        "commit_sha": m.commit_sha,
        "branch_name": m.branch_name,
        "symbol_name": m.symbol_name,
        "line_start": m.line_start,
        "line_end": m.line_end,
        "superseded_by": m.superseded_by,
        "valid_until": _iso(m.valid_until),
        "type_metadata": m.get_type_metadata(),
        "source_message_ids": m.get_source_message_ids(),
        "entities": entity_records,
        "links": link_records,
        "embedding_model": m.embedding_model,
        "embedding_dim": m.embedding_dim,
        "access_count": m.access_count or 0,
        "last_accessed_at": _iso(m.last_accessed_at),
        "source_session_id": m.source_session_id,
        "created_at": _iso(m.created_at),
        "updated_at": _iso(m.updated_at),
    }


def _build_sessions_section(
    project_id: str,
    db: Session,
) -> list[dict]:
    sessions = (
        db.query(models.AISession)
        .filter(models.AISession.project_id == project_id)
        .order_by(models.AISession.created_at.desc())
        .all()
    )
    records = []
    for s in sessions:
        memory_count = (
            db.query(models.Memory)
            .filter(models.Memory.source_session_id == s.id)
            .count()
        )
        message_count = (
            db.query(models.Message)
            .filter(models.Message.session_id == s.id)
            .count()
        )
        records.append({
            "id": s.id,
            "tool_name": s.tool_name,
            "title": s.title,
            "session_date": s.session_date,
            "summary": s.summary,
            "memory_count": memory_count,
            "message_count": message_count,
            "created_at": _iso(s.created_at),
            "updated_at": _iso(s.updated_at),
        })
    return records


def _build_entity_index(
    project_id: str,
    db: Session,
) -> list[dict]:
    """Return deduplicated entities sorted by memory_count descending."""
    all_entities = (
        db.query(models.MemoryEntity)
        .filter(models.MemoryEntity.project_id == project_id)
        .all()
    )
    groups: dict[str, dict] = {}
    for e in all_entities:
        key = e.normalized_text
        if key not in groups:
            groups[key] = {"label": e.entity_label, "memory_ids": set()}
        groups[key]["memory_ids"].add(e.memory_id)

    result = []
    for norm_text, data in sorted(
        groups.items(), key=lambda x: len(x[1]["memory_ids"]), reverse=True
    ):
        result.append({
            "text": norm_text,
            "label": data["label"],
            "memory_count": len(data["memory_ids"]),
            "memory_ids": sorted(data["memory_ids"]),
        })
    return result


def _build_clusters_section(memories: list[models.Memory]) -> list[dict]:
    """Return cluster summary for cluster_id >= 0 (noise excluded)."""
    groups: dict[int, dict] = {}
    for m in memories:
        if m.cluster_id is None or m.cluster_id < 0:
            continue
        cid = m.cluster_id
        if cid not in groups:
            groups[cid] = {"label": m.cluster_label or "", "memory_ids": []}
        groups[cid]["memory_ids"].append(m.id)

    return [
        {
            "cluster_id": cid,
            "label": data["label"],
            "memory_count": len(data["memory_ids"]),
            "memory_ids": data["memory_ids"],
        }
        for cid, data in sorted(groups.items())
    ]


# ---------------------------------------------------------------------------
# Custom YAML dumper — preserves insertion order, handles None as null
# ---------------------------------------------------------------------------

class _OrderedDumper(yaml.SafeDumper):
    pass


def _dict_representer(dumper: yaml.SafeDumper, data: dict) -> yaml.MappingNode:
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


_OrderedDumper.add_representer(dict, _dict_representer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_yaml_export(
    project_id: str,
    db: Session,
    *,
    status_filter: str | None = None,
    min_importance: int = 1,
    max_privacy_level: str = "internal",
) -> str:
    """Build and return a canonical YAML string for the given project.

    Args:
        project_id:        Target project.
        db:                SQLAlchemy session.
        status_filter:     If set, only include memories with this status.
        min_importance:    Only include memories with importance >= this (1–5).
        max_privacy_level: Most restrictive privacy level to include
                           (public < internal < sensitive < secret).

    Returns:
        A UTF-8 YAML string ready to write to .ai-memory/project-memory.yaml.
    """
    from fastapi import HTTPException

    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Fetch all memories for this project, applying DB-level filters where possible
    q = db.query(models.Memory).filter(models.Memory.project_id == project_id)
    if status_filter:
        q = q.filter(models.Memory.status == status_filter)
    if min_importance > 1:
        q = q.filter(models.Memory.importance >= min_importance)

    all_memories = q.order_by(
        models.Memory.importance.desc(),
        models.Memory.updated_at.desc(),
    ).all()

    # Apply privacy filter in Python (enum comparison)
    filtered_memories = [
        m for m in all_memories
        if _privacy_allowed(m.privacy_level or "internal", max_privacy_level)
    ]

    # Build the document sections
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "exported_at": _iso(datetime.now(timezone.utc)),
        "project": _build_project_section(project),
        "stats": _build_stats_section(filtered_memories),
        "memories": [_build_memory_record(m, db) for m in filtered_memories],
        "sessions": _build_sessions_section(project_id, db),
        "entity_index": _build_entity_index(project_id, db),
        "clusters": _build_clusters_section(filtered_memories),
    }

    return yaml.dump(
        doc,
        Dumper=_OrderedDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        indent=2,
        width=120,
    )
