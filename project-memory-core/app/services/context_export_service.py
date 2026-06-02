"""
Context export service — generates a portable Markdown context packet.

The exported Markdown is designed to be pasted directly into ChatGPT, Claude,
Gemini, Cursor, or any other AI tool to instantly restore project context.
"""
from __future__ import annotations
import difflib
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app import crud, schemas

_TOP_ENTITIES_LIMIT = 15
_TOP_ACCESSED_LIMIT = 5
_ENTITY_LABELS_SHOWN = {"TECH", "ORG", "PRODUCT"}


_SECTION_ORDER = [
    ("decision",          "## Active Decisions"),
    ("problem",           "## Problems"),
    ("task",              "## Current Tasks"),
    ("structure",         "## Structure Notes"),
    ("workflow_pattern",  "## Workflow Patterns"),
    ("how_to",            "## How-To Guides"),
    ("open_question",     "## Open Questions"),
    ("constraint",        "## Constraints"),
    ("insight",           "## Insights"),
    ("reference_context", "## Reference Context"),
    ("conversation_note", "## Conversation Notes"),
    ("failed_approach",   "## Failed Approaches"),
    ("unclassified",      "## Review Queue — Needs Classification"),
]

# Domain-specific section display names — software domain gets familiar names
_DOMAIN_SECTION_NAMES: dict[str, dict[str, str]] = {
    "software": {
        "problem":           "## Active Bugs",
        "structure":         "## Architecture Notes",
        "how_to":            "## Setup Instructions",
        "reference_context": "## Code Context",
    },
    "data": {
        "problem":    "## Issues & Blockers",
        "structure":  "## Architecture Notes",
        "how_to":     "## Setup Instructions",
    },
    "security": {
        "problem":   "## Vulnerabilities & Issues",
        "structure": "## Architecture Notes",
        "how_to":    "## Setup Instructions",
    },
    "design": {
        "problem":           "## Design Problems",
        "structure":         "## Design Structure",
        "how_to":            "## Design Processes",
        "reference_context": "## Design References",
    },
    "research": {
        "problem":           "## Research Problems",
        "structure":         "## Research Framework",
        "how_to":            "## Research Procedures",
        "reference_context": "## References & Sources",
    },
}

_STALE_DAYS = 30
_CONTENT_DEDUP_THRESHOLD = 0.80


def _days_since(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days


def _content_norm(text: str) -> str:
    return " ".join(text.lower().split())


def _is_content_dupe(content: str, seen_norms: list[str]) -> bool:
    norm = _content_norm(content)
    return any(
        difflib.SequenceMatcher(None, norm, s).ratio() >= _CONTENT_DEDUP_THRESHOLD
        for s in seen_norms
    )


def _freshness_bar(decay_score: float | None) -> str:
    if decay_score is None:
        return ""
    pct = int(decay_score * 100)
    filled = round(decay_score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar} {pct}%"


def _memory_row(
    m: schemas.MemoryResponse,
    link_lines: list[str] | None = None,
    stale_days: int | None = None,
) -> str:
    tags = ", ".join(m.tags) if m.tags else "—"
    tools = ", ".join(m.related_tools) if m.related_tools else "—"
    files = ", ".join(m.related_files) if m.related_files else "—"
    lines = [f"### {m.title}"]
    if m.retrieval_hint:
        lines.append(f"_{m.retrieval_hint}_")
    lines.append(f"{m.content}")
    decay_str = f"  **Freshness:** {_freshness_bar(m.decay_score)}" if m.decay_score is not None else ""
    lines.append(f"- **Importance:** {m.importance}/5  **Confidence:** {m.confidence:.0%}{decay_str}")
    if stale_days is not None and stale_days >= _STALE_DAYS:
        lines.append(f"- **[POSSIBLY STALE — {stale_days} days since last update]**")
    if m.tags:
        lines.append(f"- **Tags:** {tags}")
    if m.related_tools and m.related_tools != ["—"]:
        lines.append(f"- **Tools:** {tools}")
    if m.related_files and m.related_files != ["—"]:
        lines.append(f"- **Files:** {files}")
    if link_lines:
        lines.append(f"- **Relationships:** {' | '.join(link_lines)}")
    return "\n".join(lines)


_PRIVACY_EXPORT_DEFAULT = "internal"  # exclude sensitive + secret by default


def generate_context_markdown(
    project_id: str,
    db: Session,
    max_privacy_level: str = _PRIVACY_EXPORT_DEFAULT,
) -> str:
    project = crud.get_project(db, project_id)
    tech_stack = project.get_tech_stack()
    goals = project.get_goals()

    # Fetch active memories respecting privacy ceiling
    memories = crud.get_memories(
        db, project_id, status="active", limit=2000, max_privacy_level=max_privacy_level
    )
    memory_responses = [schemas.MemoryResponse.model_validate(m) for m in memories]

    # Build title lookup for relationship display
    title_by_id = {m.id: m.title for m in memory_responses}

    # Fetch all links and build per-memory relationship lines
    all_links = crud.get_all_links_for_project(db, project_id)
    links_by_memory: dict[str, list[str]] = {}
    for lnk in all_links:
        rel = lnk.relationship_type.replace("_", " ")
        src_id, tgt_id = lnk.source_memory_id, lnk.target_memory_id
        tgt_title = title_by_id.get(tgt_id, tgt_id[:8])
        src_title = title_by_id.get(src_id, src_id[:8])
        links_by_memory.setdefault(src_id, []).append(f"→ {rel}: _{tgt_title}_")
        links_by_memory.setdefault(tgt_id, []).append(f"← {rel}: _{src_title}_")

    # Group active memories by type
    by_type: dict[str, list[schemas.MemoryResponse]] = {}
    for m in memory_responses:
        by_type.setdefault(m.type, []).append(m)

    # Fetch superseded memories for the history section
    superseded_mems = crud.get_memories(db, project_id, status="superseded", limit=1000)
    superseded_responses = [schemas.MemoryResponse.model_validate(m) for m in superseded_mems]

    # Fetch last 5 sessions
    sessions = crud.get_sessions(db, project_id, limit=5)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []

    domain = project.domain or "general"
    lines.append(f"# Project Context: {project.name}")
    lines.append(f"_Domain: `{domain}`_")
    lines.append("")

    lines.append("## Overview")
    lines.append(project.description or "_No description provided._")
    lines.append("")

    lines.append("## Goals")
    if goals:
        for g in goals:
            lines.append(f"- {g}")
    else:
        lines.append("_No goals defined._")
    lines.append("")

    lines.append("## Tech Stack")
    if tech_stack:
        for t in tech_stack:
            lines.append(f"- {t}")
    else:
        lines.append("_No tech stack defined._")
    if project.repo_path:
        lines.append(f"\n**Repo:** `{project.repo_path}`")
    lines.append("")

    # Current State — quick status snapshot for any AI reading this context
    active_count = len(memory_responses)
    type_counts = {t: len(v) for t, v in by_type.items() if v}
    dominant_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    dominant_str = ", ".join(f"{t} ({n})" for t, n in dominant_types) if dominant_types else "none yet"
    last_session = sessions[0] if sessions else None
    last_session_str = (
        f"Last AI session: [{last_session.tool_name}] {last_session.title}"
        if last_session else "No AI sessions recorded yet"
    )
    stack_str = ", ".join(tech_stack) if tech_stack else "not specified"
    lines.append("## Current State")
    lines.append(
        f"{project.description or 'Local-first AI memory system.'} "
        f"Tech stack: {stack_str}. "
        f"{active_count} active memories ({dominant_str}). "
        f"{last_session_str}."
    )
    lines.append("")

    # Key Entities section — top TECH/ORG/PRODUCT entities by mention count
    all_entities = crud.get_entities_for_project(db, project_id)
    tech_entities = [
        e for e in all_entities if e.entity_label in _ENTITY_LABELS_SHOWN
    ]
    tech_entities.sort(key=lambda e: e.count, reverse=True)
    top_entities = tech_entities[:_TOP_ENTITIES_LIMIT]
    if top_entities:
        lines.append("## Key Entities")
        lines.append("_Technologies, organizations, and products mentioned across memories._")
        lines.append("")
        lines.append("| Entity | Label | Memories |")
        lines.append("|--------|-------|----------|")
        for e in top_entities:
            lines.append(f"| {e.normalized_text} | {e.entity_label} | {e.count} |")
        lines.append("")

    # Memory Clusters section — DBSCAN cluster overview
    clusters = crud.get_project_clusters(db, project_id)
    if clusters:
        lines.append("## Memory Clusters")
        lines.append("_Semantically related memory groups discovered by automatic clustering._")
        lines.append("")
        lines.append("| Cluster | Topic | Memories |")
        lines.append("|---------|-------|----------|")
        for c in clusters:
            lines.append(f"| {c.cluster_id} | {c.cluster_label} | {c.memory_count} |")
        lines.append("")

    # Most Referenced section — top memories by access_count
    top_accessed = crud.get_most_accessed_memories(db, project_id, limit=_TOP_ACCESSED_LIMIT)
    top_accessed_responses = [schemas.MemoryResponse.model_validate(m) for m in top_accessed]
    top_accessed_nonzero = [m for m in top_accessed_responses if m.access_count > 0]
    if top_accessed_nonzero:
        lines.append("## Most Referenced")
        lines.append("_Memories retrieved most often — likely the most load-bearing context._")
        lines.append("")
        for m in top_accessed_nonzero:
            hint = f" — {m.retrieval_hint}" if m.retrieval_hint else ""
            lines.append(f"- **{m.title}** ({m.access_count}x){hint}")
        lines.append("")

    # Active memory sections with stale signals + content-level dedup
    rendered_content_norms: list[str] = []

    domain = project.domain or "general"
    domain_overrides = _DOMAIN_SECTION_NAMES.get(domain, {})
    for mem_type, section_header in _SECTION_ORDER:
        section_header = domain_overrides.get(mem_type, section_header)
        items = by_type.get(mem_type, [])
        if not items:
            continue

        # Within each section, filter memories whose content is near-identical to
        # something already rendered (importance-sorted, so highest wins).
        section_items: list[schemas.MemoryResponse] = []
        for m in items:
            if _is_content_dupe(m.content, rendered_content_norms):
                continue
            rendered_content_norms.append(_content_norm(m.content))
            section_items.append(m)

        if not section_items:
            continue

        lines.append(section_header)
        for m in section_items:
            age = _days_since(m.updated_at)
            lines.append("")
            lines.append(_memory_row(
                m,
                link_lines=links_by_memory.get(m.id),
                stale_days=age if age >= _STALE_DAYS else None,
            ))
        lines.append("")

    # Superseded decisions (history preserved, not lost)
    if superseded_responses:
        lines.append("## Superseded Decisions")
        lines.append("_These decisions were replaced. Kept for historical context._")
        for m in superseded_responses:
            replacement = f" → replaced by memory `{m.superseded_by[:8]}...`" if m.superseded_by else ""
            expired = f" (expired: {m.valid_until.strftime('%Y-%m-%d') if m.valid_until else 'unknown'})"
            lines.append("")
            lines.append(f"### ~~{m.title}~~{expired}{replacement}")
            lines.append(m.content)
        lines.append("")

    # Recent AI sessions
    lines.append("## Recent AI Sessions")
    if sessions:
        for s in sessions:
            date_str = s.session_date or s.created_at.strftime("%Y-%m-%d")
            lines.append(f"### [{s.tool_name}] {s.title} ({date_str})")
            if s.summary:
                lines.append(s.summary)
            else:
                preview = s.raw_content.strip()[:200].replace("\n", " ")
                if len(s.raw_content) > 200:
                    preview += "..."
                lines.append(preview)
            lines.append("")
    else:
        lines.append("_No AI sessions recorded yet._")
        lines.append("")

    lines.append(f"---\n_Last Updated: {now_str}_")

    return "\n".join(lines)
