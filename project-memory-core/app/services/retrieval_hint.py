"""
Sub-phase 1.26 — Retrieval hint synthesis.

Computes a concise, query-facing description for each memory at write time.
The hint is designed to maximise semantic overlap with likely search queries —
it answers "what would a developer type to find this memory?" rather than
describing what the memory says.

No external model required — all logic is rule-based using the memory's
own type, title, content, and type_metadata fields.

Phase-2 uses retrieval_hint as a lightweight alternative to full-content
embedding: small batch re-ranks can compare query embeddings against the
hint rather than the full content string.
"""
from __future__ import annotations
import re


_MAX_HINT_LEN = 300

# Query-facing labels per type — chosen to match how developers phrase searches
_TYPE_LABELS: dict[str, str] = {
    "decision":           "Decision",
    "problem":            "Problem",      # was: bug
    "failed_approach":    "Avoid",
    "structure":          "Structure",    # was: architecture
    "how_to":             "How to",       # was: setup_instruction
    "open_question":      "Question",
    "task":               "Task",
    "constraint":         "Constraint",
    "insight":            "Insight",
    "reference_context":  "Reference",    # was: code_context
    "workflow_pattern":   "Workflow",
    "conversation_note":  "Note",
}


def _first_sentence(text: str, max_len: int = 120) -> str:
    """Extract the first complete sentence, capped at max_len chars."""
    if not text:
        return ""
    m = re.search(r"\.(?:\s|$)", text)
    sentence = text[: m.start()].strip() if m else text.strip()
    return sentence[:max_len]


def _truncate(text: str) -> str:
    return text[:_MAX_HINT_LEN].rstrip()


def compute_retrieval_hint(
    mem_type: str,
    title: str,
    content: str,
    type_metadata: dict | None,
) -> str:
    """
    Synthesize a retrieval hint for the given memory fields.

    Pure function — no DB access, no model inference.
    Returns a non-empty string of at most 300 characters.
    """
    meta = type_metadata or {}
    label = _TYPE_LABELS.get(mem_type, "Memory")

    # ------------------------------------------------------------------ #
    # Type-specific enrichment: pull the most discriminative metadata     #
    # field into the hint so queries for that field land here.            #
    # ------------------------------------------------------------------ #

    if mem_type == "decision":
        rationale = meta.get("rationale", "")
        alts = meta.get("alternatives_considered", [])
        if rationale:
            return _truncate(f"{label}: {title}. Rationale: {rationale}")
        if alts:
            return _truncate(f"{label}: {title}. Alternatives: {', '.join(alts[:3])}")
        return _truncate(f"{label}: {title}")

    if mem_type == "problem":
        error = meta.get("error_message", "")
        cause = meta.get("root_cause", "")
        env = meta.get("environment", "")
        if error:
            return _truncate(f"{label}: {title}. Error: {error}")
        if cause:
            return _truncate(f"{label}: {title}. Cause: {cause}")
        if env:
            return _truncate(f"{label}: {title}. Context: {env}")
        first = _first_sentence(content)
        return _truncate(f"{label}: {title}. {first}" if first else f"{label}: {title}")

    if mem_type == "failed_approach":
        approach = meta.get("approach_tried", "")
        avoid = meta.get("avoid_because", "") or meta.get("reason_failed", "")
        alt = meta.get("alternative_chosen", "")
        parts = [f"{label}: {approach or title}"]
        if avoid:
            parts.append(f"Avoid because: {avoid}")
        if alt:
            parts.append(f"Use {alt} instead")
        return _truncate(". ".join(parts))

    if mem_type == "structure":
        pattern = meta.get("pattern", "")
        rationale = meta.get("rationale", "")
        if pattern:
            base = f"{label}: {title}. Pattern: {pattern}"
            return _truncate(f"{base}. {rationale}" if rationale else base)
        return _truncate(f"{label}: {title}")

    if mem_type == "how_to":
        cmd = meta.get("command", "")
        platform = meta.get("platform", "")
        if cmd:
            suffix = f" ({platform})" if platform else ""
            return _truncate(f"{label}: {title}. Step: {cmd}{suffix}")
        return _truncate(f"{label}: {title}")

    if mem_type == "open_question":
        ctx = meta.get("context", "")
        if ctx:
            return _truncate(f"{label}: {title}. Context: {ctx}")
        return _truncate(f"{label}: {title}")

    if mem_type == "constraint":
        impact = meta.get("impact", "")
        source = meta.get("source", "")
        if impact:
            return _truncate(f"{label}: {title}. Impact: {impact}")
        if source:
            return _truncate(f"{label}: {title}. Source: {source}")
        return _truncate(f"{label}: {title}")

    if mem_type == "workflow_pattern":
        trigger = meta.get("trigger", "")
        steps = meta.get("steps", [])
        if trigger:
            return _truncate(f"{label}: when {trigger} — {title}")
        if steps:
            return _truncate(f"{label}: {title}. Steps: {', '.join(steps[:3])}")
        return _truncate(f"{label}: {title}")

    if mem_type == "insight":
        first = _first_sentence(content)
        if first and first.lower() != title.lower():
            return _truncate(f"{label}: {title}. {first}")
        return _truncate(f"{label}: {title}")

    # Default: label + title + first sentence of content
    first = _first_sentence(content)
    if first and len(first) > 10 and first.lower() not in title.lower():
        return _truncate(f"{label}: {title}. {first}")
    return _truncate(f"{label}: {title}")
