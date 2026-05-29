"""
Sub-phase 2.7 — Contextual Embeddings.

Architectural basis (Anthropic research):
  Prepending a 50-100 token LLM-generated context prefix to each memory
  before embedding reduces retrieval failure by 67%.

Why this works:
  Raw memory text is often phrasing-fragile: "we chose Redis" only matches
  queries that mention Redis. A context prefix — "This memory records an
  architectural decision made during the auth session about caching strategy.
  It captures: we chose Redis for session caching due to sub-millisecond
  reads." — gives the embedding model much richer signal for semantic matching.

Prefix format:
  "This memory captures [type] made in [project context]. It records: [title]."
  For LLM path: a richer 60-100 token elaboration.

Workflow:
  1. generate_contextual_prefix(memory, project, llm_fn=None) → str
  2. build_contextual_text(memory, prefix) → str  (prefix + "\n\n" + content)
  3. generate_contextual_embeddings(db, project_id, ...) → ContextualEmbeddingResult
     — backfill: iterates all active memories, generates prefix, re-embeds, writes back

Dual-index safety rule:
  Never replace embeddings in place. generate_contextual_embeddings() writes
  contextual_prefix + new embedding to the same row (it's an in-place upgrade),
  but only after all prefixes are generated so the index is consistent.
  For full ANN index rebuild, call BackfillService.backfill_project() after.

Usage:
  from app.services.retrieval.contextual_embeddings import generate_contextual_embeddings
  result = generate_contextual_embeddings(db, project_id)
  # Then backfill ANN backend:
  from app.services.vector_backends.migration import BackfillService
  svc = BackfillService()
  state = svc.backfill_project(db, project_id, ann_backend)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_BATCH = 50
_MAX_PREFIX_TOKENS = 100    # guideline — the LLM may vary slightly
_MIN_PREFIX_TOKENS = 40     # minimum useful prefix length (chars ÷ 4 ≈ tokens)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContextualEmbeddingResult:
    project_id: str
    total_processed: int
    already_had_prefix: int     # skipped (prefix unchanged, no re-embed)
    newly_prefixed: int         # prefix generated + embedding updated
    failed: int                 # errors during prefix gen or embedding
    used_llm: bool
    memory_ids_updated: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template prefix generator (no LLM required)
# ---------------------------------------------------------------------------

_TYPE_VERBS: dict[str, str] = {
    "decision":         "records an architectural decision",
    "constraint":       "records a technical constraint",
    "problem":          "describes a problem or bug",
    "preference":       "records a team preference",
    "plan":             "describes a plan or roadmap item",
    "procedure":        "documents a procedure or process",
    "fact":             "states a technical fact",
    "open_question":    "records an open question or uncertainty",
    "workflow_pattern": "documents a workflow pattern",
    "reference":        "provides a reference or link",
    "failed_approach":  "records a failed approach or dead end",
    "insight":          "captures a technical insight",
    "how_to":           "documents a how-to or recipe",
    "cluster_summary":  "is a cluster-level summary",
    "task":             "records a task or TODO item",
    "update":           "records an update or status change",
}


def generate_contextual_prefix(
    memory,
    project,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """
    Generate a 50-100 token context prefix for a memory.

    If llm_fn is provided (callable that takes a prompt string and returns text),
    it will be used to generate a richer prefix. Falls back to template.

    Args:
        memory:   Phase 1 Memory ORM row.
        project:  Phase 1 Project ORM row.
        llm_fn:   Optional LLM callable (prompt: str) -> str.

    Returns:
        Prefix string (never empty — template always succeeds).
    """
    if llm_fn is not None:
        try:
            return _llm_prefix(memory, project, llm_fn)
        except Exception:
            pass  # fall through to template

    return _template_prefix(memory, project)


def _template_prefix(memory, project) -> str:
    """Fast, deterministic prefix — no LLM needed."""
    mem_type = getattr(memory, "type", "fact") or "fact"
    verb = _TYPE_VERBS.get(mem_type, f"records a {mem_type}")
    project_name = getattr(project, "name", "this project") or "this project"
    title = getattr(memory, "title", "") or ""
    return (
        f"This memory {verb} in the project '{project_name}'. "
        f"It captures: {title}."
    )


_LLM_PROMPT_TEMPLATE = """\
You are generating a short context prefix for a memory stored in an AI memory system.
The prefix will be prepended to the memory text before embedding. It should be 50-80 words,
written in third person, describing what this memory is about and its significance.

Project: {project_name}
Memory type: {memory_type}
Memory title: {title}
Memory content (first 300 chars): {content_preview}

Write ONLY the context prefix. No preamble, no quotes. One paragraph."""


def _llm_prefix(memory, project, llm_fn: Callable[[str], str]) -> str:
    """Generate prefix using an LLM callable."""
    content = getattr(memory, "content", "") or ""
    prompt = _LLM_PROMPT_TEMPLATE.format(
        project_name=getattr(project, "name", "Unknown") or "Unknown",
        memory_type=getattr(memory, "type", "fact") or "fact",
        title=getattr(memory, "title", "") or "",
        content_preview=content[:300],
    )
    result = llm_fn(prompt)
    if not result or not result.strip():
        raise ValueError("LLM returned empty prefix")
    # Trim to reasonable length (rough token cap: 4 chars/token × 100 tokens = 400 chars)
    return result.strip()[:400]


def build_contextual_text(memory, prefix: str) -> str:
    """
    Combine prefix + memory content into a single string for embedding.

    Format: "{prefix}\n\n{content}"
    """
    content = getattr(memory, "content", "") or ""
    return f"{prefix}\n\n{content}"


# ---------------------------------------------------------------------------
# Backfill service
# ---------------------------------------------------------------------------

def generate_contextual_embeddings(
    db: "Session",
    project_id: str,
    llm_fn: Callable[[str], str] | None = None,
    batch_size: int = _DEFAULT_BATCH,
    force_regenerate: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> ContextualEmbeddingResult:
    """
    Generate context prefixes and re-embed all active memories for a project.

    Args:
        db:               SQLAlchemy session.
        project_id:       Project to process.
        llm_fn:           Optional LLM callable for richer prefixes.
        batch_size:       Memories per batch.
        force_regenerate: If True, re-generate even if prefix already exists.
        embedding_model:  Model to use for re-embedding. Must be in registry.

    Returns:
        ContextualEmbeddingResult with counts and updated memory IDs.
    """
    from app import models as phase1_models
    from app.services.vector_backends.embedding_models import embed_with_model

    # Load project
    project = db.query(phase1_models.Project).filter_by(id=project_id).first()
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    # Load active memories with content (skip cluster_summary type)
    rows = (
        db.query(phase1_models.Memory)
        .filter(
            phase1_models.Memory.project_id == project_id,
            phase1_models.Memory.status == "active",
            phase1_models.Memory.type != "cluster_summary",
        )
        .all()
    )

    result = ContextualEmbeddingResult(
        project_id=project_id,
        total_processed=len(rows),
        already_had_prefix=0,
        newly_prefixed=0,
        failed=0,
        used_llm=llm_fn is not None,
    )

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        for mem in batch:
            try:
                # Skip if prefix already exists and force_regenerate=False
                if mem.contextual_prefix and not force_regenerate:
                    result.already_had_prefix += 1
                    continue

                prefix = generate_contextual_prefix(mem, project, llm_fn=llm_fn)
                contextual_text = build_contextual_text(mem, prefix)

                # Re-embed using contextual text
                if embedding_model == "all-MiniLM-L6-v2":
                    from app.services.semantic_classifier import embed_text
                    new_embedding = embed_text(contextual_text)
                else:
                    new_embedding = embed_with_model(contextual_text, embedding_model)

                # Update in-place (prefix + new embedding)
                mem.contextual_prefix = prefix
                mem.embedding = new_embedding
                mem.embedding_model = embedding_model

                result.newly_prefixed += 1
                result.memory_ids_updated.append(mem.id)

            except Exception:
                result.failed += 1
                continue

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    return result


def get_embed_text_for_memory(memory) -> str:
    """
    Return the text to embed for a given memory.

    If contextual_prefix is set, returns prefix + content (contextual embedding).
    Otherwise returns raw content (Phase 1 behavior).

    Used by retrieval pipeline to embed query using the same strategy as stored memories.
    """
    prefix = getattr(memory, "contextual_prefix", None)
    content = getattr(memory, "content", "") or ""
    if prefix:
        return f"{prefix}\n\n{content}"
    return content
