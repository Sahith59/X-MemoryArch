"""
Sub-phase 2.4 — RAPTOR-style cluster summaries.

For each cluster with ≥ min_cluster_size memories, generate a single
"cluster_summary" Memory that distils the cluster's key themes.

When an LLM is available (llm_fn or ANTHROPIC_API_KEY), it synthesises a
rich summary.  Without one, a deterministic template is used so the feature
works in test and offline environments.

Summaries are stored as regular Memory rows (type="cluster_summary",
privacy_level="internal") so they participate in dense retrieval and BM25.
One summary per cluster_id — re-running updates the existing row.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ClusterSummaryResult:
    cluster_id: int
    cluster_label: str
    memory_count: int
    summary_memory_id: str
    summary_text: str
    used_llm: bool


# ---------------------------------------------------------------------------
# Template fallback (no LLM required)
# ---------------------------------------------------------------------------

def _template_summary(cluster_label: str, memories: list) -> str:
    titles = [m.title for m in memories[:8]]
    bullet_lines = "\n".join(f"- {t}" for t in titles)
    extras = len(memories) - len(titles)
    extra_note = f" (and {extras} more)" if extras > 0 else ""
    return (
        f"Cluster: {cluster_label}\n"
        f"Contains {len(memories)} memories covering the following topics{extra_note}:\n"
        f"{bullet_lines}"
    )


# ---------------------------------------------------------------------------
# Claude Haiku LLM summary (optional)
# ---------------------------------------------------------------------------

def _claude_summary(cluster_label: str, memories: list) -> str | None:
    """Call Claude Haiku to synthesise a cluster summary. Returns None on failure."""
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Build a compact corpus snapshot (truncated to avoid huge prompts)
    snippets = []
    for m in memories[:20]:
        content_preview = (m.content or "")[:300]
        snippets.append(f"[{m.type}] {m.title}: {content_preview}")
    corpus_text = "\n\n".join(snippets)

    prompt = (
        f"You are summarising a memory cluster labelled '{cluster_label}'.\n"
        f"The cluster contains {len(memories)} memories. "
        f"Here is a representative sample:\n\n{corpus_text}\n\n"
        f"Write a concise 2-3 sentence summary that captures the main themes, "
        f"key entities, and core topics of this cluster. "
        f"Be specific — mention concrete names, tools, and concepts."
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_cluster_summaries(
    db: "Session",
    project_id: str,
    min_cluster_size: int = 10,
    llm_fn: Callable[[str, list], str | None] | None = None,
) -> list[ClusterSummaryResult]:
    """
    Generate RAPTOR-style summaries for every cluster with ≥ min_cluster_size memories.

    Args:
        db:               SQLAlchemy session.
        project_id:       Scope to this project.
        min_cluster_size: Skip clusters smaller than this (default 10).
        llm_fn:           Optional custom LLM callable (cluster_label, memories) → str.
                          When None, tries Claude Haiku then falls back to template.

    Returns:
        List of ClusterSummaryResult, one per cluster processed.
    """
    from app import models as phase1_models

    _now = lambda: datetime.now(timezone.utc)  # noqa: E731

    # Load all active, non-cluster-summary memories with cluster assignments
    rows = (
        db.query(phase1_models.Memory)
        .filter(
            phase1_models.Memory.project_id == project_id,
            phase1_models.Memory.cluster_id.isnot(None),
            phase1_models.Memory.type != "cluster_summary",
            phase1_models.Memory.status == "active",
        )
        .all()
    )

    # Group by cluster_id
    clusters: dict[int, list] = {}
    for m in rows:
        clusters.setdefault(m.cluster_id, []).append(m)

    results: list[ClusterSummaryResult] = []

    for cluster_id, memories in sorted(clusters.items()):
        if len(memories) < min_cluster_size:
            continue

        # Derive cluster label from most common non-None value in the group
        label_counts: dict[str, int] = {}
        for m in memories:
            if m.cluster_label:
                label_counts[m.cluster_label] = label_counts.get(m.cluster_label, 0) + 1
        cluster_label = (
            max(label_counts, key=lambda l: label_counts[l])
            if label_counts
            else f"Cluster {cluster_id}"
        )

        # Generate summary text
        used_llm = False
        summary_text: str | None = None

        if llm_fn is not None:
            summary_text = llm_fn(cluster_label, memories)
            used_llm = summary_text is not None
        else:
            summary_text = _claude_summary(cluster_label, memories)
            used_llm = summary_text is not None

        if not summary_text:
            summary_text = _template_summary(cluster_label, memories)
            used_llm = False

        # Upsert: find existing cluster_summary for this cluster_id
        existing = (
            db.query(phase1_models.Memory)
            .filter(
                phase1_models.Memory.project_id == project_id,
                phase1_models.Memory.type == "cluster_summary",
                phase1_models.Memory.cluster_id == cluster_id,
            )
            .first()
        )

        if existing:
            existing.content = summary_text
            existing.title = f"[Summary] {cluster_label}"
            existing.updated_at = _now()
            summary_id = existing.id
        else:
            summary_id = str(uuid.uuid4())
            new_mem = phase1_models.Memory(
                id=summary_id,
                project_id=project_id,
                type="cluster_summary",
                title=f"[Summary] {cluster_label}",
                content=summary_text,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                privacy_level="internal",
                status="active",
                importance=3,
                confidence=0.9,
                source_type="auto_generated",
                created_at=_now(),
                updated_at=_now(),
            )
            db.add(new_mem)

        try:
            db.commit()
        except Exception:
            db.rollback()
            continue

        results.append(
            ClusterSummaryResult(
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                memory_count=len(memories),
                summary_memory_id=summary_id,
                summary_text=summary_text,
                used_llm=used_llm,
            )
        )

    return results
