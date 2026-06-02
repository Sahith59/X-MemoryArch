"""
Sub-phase 1.36 — Per-type Markdown export service.

Generates a focused Markdown document for a single memory type.
Each type has a tailored renderer that surfaces its type_metadata fields
in the most readable format for that type's purpose.

Endpoint: GET /projects/{id}/export/memories/{type_name}.md
Filters:  ?status=active  ?min_importance=1  ?max_privacy_level=internal

Supported type slugs (match the MemoryType enum values):
  decision, bug, task, insight, architecture, code_context,
  setup_instruction, open_question, constraint,
  conversation_note, workflow_pattern, failed_approach
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app import models

# ---------------------------------------------------------------------------
# Valid type set and display names
# ---------------------------------------------------------------------------

_TYPE_DISPLAY: dict[str, str] = {
    "decision":          "Decisions",
    "problem":           "Problems",
    "task":              "Tasks",
    "insight":           "Insights",
    "structure":         "Structure Notes",
    "reference_context": "Reference Context",
    "how_to":            "How-To Guides",
    "open_question":     "Open Questions",
    "constraint":        "Constraints",
    "conversation_note": "Conversation Notes",
    "workflow_pattern":  "Workflow Patterns",
    "failed_approach":   "Failed Approaches — Do Not Repeat",
}

# Domain-specific display overrides — software projects get familiar names
_DOMAIN_TYPE_DISPLAY: dict[str, dict[str, str]] = {
    "software": {
        "problem":           "Bugs",
        "structure":         "Architecture",
        "reference_context": "Code Context",
        "how_to":            "Setup Instructions",
    },
    "data": {
        "structure": "Architecture",
        "how_to":    "Setup Instructions",
    },
    "security": {
        "problem":   "Vulnerabilities & Issues",
        "structure": "Architecture",
        "how_to":    "Setup Instructions",
    },
    "design": {
        "problem":           "Design Problems",
        "structure":         "Design Structure",
        "reference_context": "Design References",
        "how_to":            "Design Processes",
    },
    "research": {
        "problem":           "Research Problems",
        "structure":         "Research Framework",
        "reference_context": "References & Sources",
        "how_to":            "Research Procedures",
    },
}

VALID_TYPES = set(_TYPE_DISPLAY.keys())

# Privacy level ordering
_PRIVACY_ORDER = ["public", "internal", "sensitive", "secret"]


def _privacy_allowed(level: str, max_level: str) -> bool:
    try:
        return _PRIVACY_ORDER.index(level) <= _PRIVACY_ORDER.index(max_level)
    except ValueError:
        return True


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _importance_stars(n: int) -> str:
    n = max(1, min(5, n))
    return "★" * n + "☆" * (5 - n)


def _confidence_pct(c: float) -> str:
    return f"{int(round(c * 100))}%"


def _freshness_bar(decay: float | None) -> str:
    if decay is None:
        return ""
    filled = round(decay * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"Freshness `[{bar}]` {int(decay * 100)}%"


# ---------------------------------------------------------------------------
# Type-specific metadata renderers
# ---------------------------------------------------------------------------

def _render_decision_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("rationale"):
        lines.append(f"**Rationale:** {tm['rationale']}")
    if tm.get("alternatives_considered"):
        alts = ", ".join(tm["alternatives_considered"])
        lines.append(f"**Alternatives considered:** {alts}")
    if tm.get("consequences"):
        lines.append(f"**Consequences:** {tm['consequences']}")
    if tm.get("decision_status"):
        lines.append(f"**Decision status:** `{tm['decision_status']}`")
    return lines


def _render_bug_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("error_message"):
        lines.append(f"**Error:** `{tm['error_message']}`")
    if tm.get("root_cause"):
        lines.append(f"**Root cause:** {tm['root_cause']}")
    if tm.get("fix_applied"):
        lines.append(f"**Fix applied:** {tm['fix_applied']}")
    if tm.get("environment"):
        lines.append(f"**Environment:** {tm['environment']}")
    if tm.get("affected_files"):
        files = ", ".join(f"`{f}`" for f in tm["affected_files"])
        lines.append(f"**Affected files:** {files}")
    return lines


def _render_architecture_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("pattern"):
        lines.append(f"**Pattern:** {tm['pattern']}")
    if tm.get("rationale"):
        lines.append(f"**Rationale:** {tm['rationale']}")
    if tm.get("components_affected"):
        comps = ", ".join(f"`{c}`" for c in tm["components_affected"])
        lines.append(f"**Components:** {comps}")
    return lines


def _render_setup_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("command"):
        lines.append(f"**Command:** `{tm['command']}`")
    if tm.get("prerequisites"):
        prereqs = ", ".join(tm["prerequisites"])
        lines.append(f"**Prerequisites:** {prereqs}")
    if tm.get("platform"):
        lines.append(f"**Platform:** {tm['platform']}")
    return lines


def _render_open_question_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("context"):
        lines.append(f"**Context:** {tm['context']}")
    if tm.get("options_considered"):
        opts = "; ".join(tm["options_considered"])
        lines.append(f"**Options:** {opts}")
    if tm.get("deadline"):
        lines.append(f"**Deadline:** {tm['deadline']}")
    return lines


def _render_constraint_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("source"):
        lines.append(f"**Source:** {tm['source']}")
    if tm.get("impact"):
        lines.append(f"**Impact:** {tm['impact']}")
    return lines


def _render_workflow_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("trigger"):
        lines.append(f"**Trigger:** {tm['trigger']}")
    if tm.get("steps"):
        lines.append("**Steps:**")
        for i, step in enumerate(tm["steps"], 1):
            lines.append(f"  {i}. {step}")
    if tm.get("tools_used"):
        tools = ", ".join(f"`{t}`" for t in tm["tools_used"])
        lines.append(f"**Tools:** {tools}")
    return lines


def _render_failed_approach_meta(tm: dict) -> list[str]:
    lines = []
    if tm.get("approach_tried"):
        lines.append(f"**Approach tried:** {tm['approach_tried']}")
    if tm.get("reason_failed"):
        lines.append(f"**Why it failed:** {tm['reason_failed']}")
    if tm.get("symptoms"):
        syms = ", ".join(tm["symptoms"])
        lines.append(f"**Symptoms:** {syms}")
    if tm.get("alternative_chosen"):
        lines.append(f"**Alternative chosen:** {tm['alternative_chosen']}")
    if tm.get("avoid_because"):
        lines.append(f"> ⚠ **Avoid because:** {tm['avoid_because']}")
    return lines


_META_RENDERERS = {
    "decision":          _render_decision_meta,
    "problem":           _render_bug_meta,
    "structure":         _render_architecture_meta,
    "how_to":            _render_setup_meta,
    "open_question":     _render_open_question_meta,
    "constraint":        _render_constraint_meta,
    "workflow_pattern":  _render_workflow_meta,
    "failed_approach":   _render_failed_approach_meta,
}


# ---------------------------------------------------------------------------
# Per-memory renderer
# ---------------------------------------------------------------------------

def _render_memory(m: models.Memory, db: Session) -> list[str]:
    lines: list[str] = []

    # Title and importance
    stars = _importance_stars(m.importance)
    lines.append(f"### {m.title}")
    lines.append(
        f"**Importance:** {stars} &nbsp;|&nbsp; "
        f"**Confidence:** {_confidence_pct(m.confidence)} &nbsp;|&nbsp; "
        f"**Status:** `{m.status}`"
    )

    # Dates
    meta_parts = []
    if m.created_at:
        meta_parts.append(f"Created {_iso(m.created_at)}")
    if m.updated_at and m.updated_at != m.created_at:
        meta_parts.append(f"Updated {_iso(m.updated_at)}")
    if meta_parts:
        lines.append("_" + " · ".join(meta_parts) + "_")

    # Source session attribution
    if m.source_session_id:
        session = db.query(models.AISession).filter(
            models.AISession.id == m.source_session_id
        ).first()
        if session:
            date_part = f" · {session.session_date}" if session.session_date else ""
            lines.append(f"_Source: {session.tool_name} — {session.title}{date_part}_")

    # Retrieval hint (TL;DR)
    if m.retrieval_hint:
        lines.append(f"> _TL;DR: {m.retrieval_hint}_")

    lines.append("")

    # Main content
    lines.append(m.content)
    lines.append("")

    # Type-specific metadata
    tm = json.loads(m.type_metadata) if m.type_metadata else {}
    renderer = _META_RENDERERS.get(m.type)
    if renderer and tm:
        meta_lines = renderer(tm)
        if meta_lines:
            lines.extend(meta_lines)
            lines.append("")

    # Source quote
    if m.source_quote:
        lines.append(f'> "{m.source_quote}"')
        lines.append("")

    # Code anchors
    anchor_parts = []
    if m.file_path:
        anchor_parts.append(f"`{m.file_path}`")
    if m.line_start:
        span = f"L{m.line_start}"
        if m.line_end:
            span += f"–{m.line_end}"
        anchor_parts.append(span)
    if m.commit_sha:
        anchor_parts.append(f"commit `{m.commit_sha[:8]}`")
    if m.branch_name:
        anchor_parts.append(f"branch `{m.branch_name}`")
    if anchor_parts:
        lines.append(f"**Code anchor:** {' · '.join(anchor_parts)}")
        lines.append("")

    # Tags and tools
    tags = json.loads(m.tags) if m.tags else []
    meaningful_tags = [t for t in tags if t != "auto-extracted"]
    if meaningful_tags:
        lines.append(f"**Tags:** {', '.join(f'`{t}`' for t in meaningful_tags)}")

    related_files = json.loads(m.related_files) if m.related_files else []
    if related_files:
        lines.append(f"**Related files:** {', '.join(f'`{f}`' for f in related_files)}")

    related_tools = json.loads(m.related_tools) if m.related_tools else []
    if related_tools:
        lines.append(f"**Tools:** {', '.join(related_tools)}")

    # Freshness
    freshness = _freshness_bar(m.decay_score)
    if freshness:
        lines.append(freshness)

    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_type_markdown(
    project_id: str,
    memory_type: str,
    db: Session,
    *,
    status_filter: str | None = None,
    min_importance: int = 1,
    max_privacy_level: str = "internal",
) -> str:
    """Generate a focused Markdown document for one memory type.

    Args:
        project_id:        Target project.
        memory_type:       One of the MemoryType enum values (e.g. "decision").
        db:                SQLAlchemy session.
        status_filter:     Restrict to this status (None = all statuses).
        min_importance:    Only include memories with importance >= this.
        max_privacy_level: Privacy gate.

    Returns:
        Markdown string ready to write to {type}.md.

    Raises:
        HTTPException 404 if project not found.
        HTTPException 422 if memory_type is not a valid type.
    """
    from fastapi import HTTPException

    if memory_type not in VALID_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown memory type '{memory_type}'. Valid types: {sorted(VALID_TYPES)}",
        )

    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Fetch memories of this type with DB-level filters
    q = (
        db.query(models.Memory)
        .filter(
            models.Memory.project_id == project_id,
            models.Memory.type == memory_type,
        )
    )
    if status_filter:
        q = q.filter(models.Memory.status == status_filter)
    if min_importance > 1:
        q = q.filter(models.Memory.importance >= min_importance)

    memories = q.order_by(
        models.Memory.importance.desc(),
        models.Memory.updated_at.desc(),
    ).all()

    # Apply privacy filter in Python
    memories = [
        m for m in memories
        if _privacy_allowed(m.privacy_level or "internal", max_privacy_level)
    ]

    # Resolve domain-aware display name
    domain = project.domain or "general"
    domain_overrides = _DOMAIN_TYPE_DISPLAY.get(domain, {})
    display_name = domain_overrides.get(memory_type, _TYPE_DISPLAY[memory_type])

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []

    # ── Document header ──────────────────────────────────────────────────────
    lines.append(f"# {display_name} — {project.name}")
    lines.append(f"_Project: {project.name} · Domain: `{domain}` · Exported: {now_str} · Type: `{memory_type}`_")
    lines.append("")

    # ── Summary bar ─────────────────────────────────────────────────────────
    n = len(memories)
    if n == 0:
        lines.append(f"_No `{memory_type}` memories found matching the current filters._")
        lines.append("")
        return "\n".join(lines)

    # Stats for this type
    avg_imp = sum(m.importance for m in memories) / n
    avg_conf = sum(m.confidence for m in memories) / n
    status_counts: dict[str, int] = {}
    for m in memories:
        status_counts[m.status] = status_counts.get(m.status, 0) + 1

    status_str = " · ".join(f"{v} {k}" for k, v in sorted(status_counts.items()))
    lines.append(
        f"**{n} {'memory' if n == 1 else 'memories'}** · "
        f"Avg importance {avg_imp:.1f}/5 · "
        f"Avg confidence {int(avg_conf * 100)}% · "
        f"{status_str}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per-memory records ───────────────────────────────────────────────────
    for m in memories:
        lines.extend(_render_memory(m, db))

    # ── Footer ───────────────────────────────────────────────────────────────
    lines.append(f"_End of {display_name} export · {n} records_")

    return "\n".join(lines)
