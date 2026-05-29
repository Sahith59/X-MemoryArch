"""
Pre-Phase 2 — Extraction and HyDE prompt templates.

Prompt design principles:
  - Temperature 0.0 for deterministic extraction (no creative variation).
  - Strict JSON mode enforced — LLM must return valid JSON, no prose.
  - canonical_type is constrained to 13 families — prevents open proliferation.
  - type_label is open — lets the LLM add project-specific nuance.
  - source_quote + source_message_ids are REQUIRED — no provenance = no memory.
  - review_recommended=true for confidence < 0.6 → routes to review queue.
  - Contradictions → supersedes_memory_id field (temporal invalidation).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical type families shown to the LLM in the extraction prompt
# ---------------------------------------------------------------------------
CANONICAL_FAMILIES = (
    "decision",         # architectural or product choice with rationale
    "problem",          # bug, issue, error, failure, pain point
    "constraint",       # hard requirement, limitation, preference, non-negotiable
    "open_question",    # unresolved question, pending decision
    "workflow_pattern", # repeatable process or procedure
    "failed_approach",  # tried-and-rejected approach, prevents repeating mistakes
    "insight",          # realization, lesson learned, fact, observation
    "how_to",           # step-by-step instructions, commands, setup
    "task",             # TODO, action item, plan
    "structure",        # architecture, design, system/org structure
    "reference_context",# code context, reference material, notes about a specific piece of code
    "conversation_note",# general note from a conversation, doesn't fit other types
)

CANONICAL_FAMILIES_STR = ", ".join(CANONICAL_FAMILIES)


# ---------------------------------------------------------------------------
# Extraction system prompt (shared across Claude and OpenAI)
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """\
You extract durable, reusable memories from AI/developer conversations.

Return ONLY a JSON array. No prose. No markdown. No explanation outside the JSON.
Temperature: 0. Be precise, not creative.

WHAT TO EXTRACT:
Prefer: decisions, constraints, problems, failed approaches, procedures, architectural choices,
        open questions, tasks, insights, code references, workflow patterns.
Skip:   greetings, filler, acknowledgements, chat niceties ("great!", "sure!", "of course"),
        meta-commentary about the conversation itself.
If the entire chunk contains only filler, return: []

OUTPUT FORMAT — return a JSON array of objects, each matching this schema exactly:
[
  {
    "canonical_type": "<one of: """ + CANONICAL_FAMILIES_STR + """>",
    "type_label": "<open label — can be same as canonical_type or more specific, e.g. 'security_requirement'>",
    "title": "<concise title, max 100 chars>",
    "content": "<full memory content, preserve technical details>",
    "importance": <integer 1-5>,
    "confidence": <float 0.0-1.0>,
    "source_quote": "<verbatim quote from the source text — REQUIRED>",
    "source_message_ids": ["<message_id or empty list if unavailable>"],
    "entities": [{"label": "<entity text>", "entity_type": "<TECH|ORG|PRODUCT|PERSON|OTHER>"}],
    "links": [{"target_memory_id": "<existing memory ID or null>", "relation_type": "<supersedes|conflicts_with|resolves|related_to>", "confidence": <float>}],
    "tags": ["<tag1>", "<tag2>"],
    "related_files": ["<file path if mentioned>"],
    "privacy_level": "<public|internal|sensitive|secret>",
    "review_recommended": <true|false>,
    "llm_reasoning": "<one sentence: why extracted and typed this way>",
    "temporal": {
      "valid_from": "<ISO datetime or null>",
      "valid_until": "<ISO datetime or null>",
      "supersedes_memory_id": "<ID of older memory this replaces, or null>"
    }
  }
]

RULES:
1. source_quote is REQUIRED. If you cannot find a verbatim quote, do not create the memory.
2. canonical_type MUST be one of the 13 families listed above. Use type_label for nuance.
3. importance: 1=trivial, 2=minor, 3=normal, 4=significant, 5=critical.
4. confidence: how certain you are this is a durable, reusable memory (not a one-off comment).
5. Set review_recommended=true for confidence < 0.6 or ambiguous extractions.
6. If a statement clearly contradicts or replaces an existing memory, set temporal.supersedes_memory_id.
7. Extract entities only for genuinely named entities (tools, frameworks, companies, people).
8. privacy_level: public=safe to share, internal=team only, sensitive=need-to-know, secret=never export.
9. Do NOT invent facts. Extract only what is explicitly stated.
"""

EXTRACTION_SYSTEM_PROMPT_OLLAMA = """\
Extract memories from this conversation. Return ONLY valid JSON — no prose.

Output a JSON array. Each item:
{
  "canonical_type": "<""" + CANONICAL_FAMILIES_STR + """>",
  "type_label": "<specific label>",
  "title": "<max 100 chars>",
  "content": "<full content>",
  "importance": <1-5>,
  "confidence": <0.0-1.0>,
  "source_quote": "<verbatim quote — REQUIRED>",
  "source_message_ids": [],
  "entities": [{"label": "<text>", "entity_type": "<TECH|ORG|PRODUCT|PERSON|OTHER>"}],
  "links": [],
  "tags": [],
  "related_files": [],
  "privacy_level": "internal",
  "review_recommended": false,
  "llm_reasoning": "<one sentence why>",
  "temporal": {"valid_from": null, "valid_until": null, "supersedes_memory_id": null}
}

Skip filler/greetings. Return [] if nothing meaningful. Include source_quote always.
"""


def build_extraction_user_prompt(
    chunk_text: str,
    project_summary: str,
    existing_memory_titles: list[str] | None = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    """
    Build the per-chunk user turn for extraction.

    Args:
        chunk_text:             The substantive session text for this chunk.
        project_summary:        Brief description of the project context.
        existing_memory_titles: Titles of existing memories (for supersession detection).
        chunk_index:            0-based index of this chunk.
        total_chunks:           Total number of chunks in this session.
    """
    parts: list[str] = []

    if project_summary:
        parts.append(f"PROJECT CONTEXT:\n{project_summary}\n")

    if existing_memory_titles:
        titles_str = "\n".join(f"  - {t}" for t in existing_memory_titles[:20])
        parts.append(
            f"EXISTING MEMORIES (for supersession detection — do NOT re-extract these):\n"
            f"{titles_str}\n"
        )

    if total_chunks > 1:
        parts.append(
            f"CHUNK {chunk_index + 1} of {total_chunks} "
            f"(extract only from this chunk — duplicates across chunks will be merged)\n"
        )

    parts.append(f"CONVERSATION TEXT TO EXTRACT FROM:\n{chunk_text}")

    return "\n".join(parts)


def build_project_summary(project) -> str:
    """Build a compact project summary string for the LLM context."""
    lines = [f"Project: {project.name}"]
    if project.description:
        lines.append(f"Description: {project.description}")
    if project.domain and project.domain != "general":
        lines.append(f"Domain: {project.domain}")
    try:
        tech = project.get_tech_stack()
        if tech:
            lines.append(f"Tech stack: {', '.join(tech)}")
    except Exception:
        pass
    try:
        goals = project.get_goals()
        if goals:
            lines.append(f"Goals: {'; '.join(goals[:3])}")
    except Exception:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings prompt
# ---------------------------------------------------------------------------

HYDE_SYSTEM_PROMPT = """\
Write a short hypothetical memory note that would likely answer the user's query.
Do NOT say it is hypothetical. Write as if it is a stored fact in a developer's memory system.
Return ONLY the hypothetical memory text — no preamble, no explanation.
"""


def build_hyde_user_prompt(query: str) -> str:
    """Build the HyDE user turn for a given retrieval query."""
    return f"""\
Query: {query}

Write a concise memory note (60-120 tokens) that would directly answer this query.
Include: likely entities (tools, frameworks, decisions), timeframe if relevant, core relation.
Use factual, terse, professional language — as if written by a developer taking notes.
Do NOT start with "This memory..." or "Here is...".
"""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = frozenset({
    "canonical_type", "type_label", "title", "content",
    "importance", "confidence", "source_quote",
})

OPTIONAL_FIELDS_WITH_DEFAULTS: dict = {
    "source_message_ids": [],
    "entities": [],
    "links": [],
    "tags": [],
    "related_files": [],
    "privacy_level": "internal",
    "review_recommended": False,
    "llm_reasoning": None,
    "temporal": {"valid_from": None, "valid_until": None, "supersedes_memory_id": None},
}


def validate_draft(raw: dict) -> dict | None:
    """
    Validate and normalise a raw LLM-output memory dict.

    Returns the normalised dict on success, None if invalid.
    Fills missing optional fields with defaults rather than raising.
    """
    if not isinstance(raw, dict):
        return None

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in raw or raw[field] is None:
            return None

    # source_quote must be non-empty string
    if not isinstance(raw.get("source_quote"), str) or not raw["source_quote"].strip():
        return None

    # title and content must be non-empty strings
    for f in ("title", "content"):
        if not isinstance(raw.get(f), str) or not raw[f].strip():
            return None

    # importance must be int 1–5
    try:
        imp = int(raw["importance"])
        if not 1 <= imp <= 5:
            imp = max(1, min(5, imp))
        raw["importance"] = imp
    except (TypeError, ValueError):
        raw["importance"] = 3

    # confidence must be float 0–1
    try:
        conf = float(raw["confidence"])
        raw["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        raw["confidence"] = 0.5

    # Fill optional fields with defaults
    for key, default in OPTIONAL_FIELDS_WITH_DEFAULTS.items():
        if key not in raw:
            raw[key] = default

    # Ensure list fields are actually lists
    for list_field in ("source_message_ids", "entities", "links", "tags", "related_files"):
        if not isinstance(raw.get(list_field), list):
            raw[list_field] = []

    # Normalise temporal block
    if not isinstance(raw.get("temporal"), dict):
        raw["temporal"] = {"valid_from": None, "valid_until": None, "supersedes_memory_id": None}

    return raw
