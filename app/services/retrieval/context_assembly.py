"""
Sub-phase 2.2 — Token-budgeted context assembly.

Assembly strategy (priority order per memory):
  1. retrieval_hint   — most compact, pre-computed signal
  2. title + type + timestamp + provenance header
  3. source_quote     — verbatim evidence / provenance
  4. content          — full memory body

Budget enforcement:
  - Memories are added in RRF rank order (highest score first).
  - A memory is dropped if it would exceed the remaining budget.
  - At least one memory is always included (even if it blows the budget).
  - Never truncate mid-sentence.

Abstractive fallback (when memories are excluded):
  - Excluded memories that belong to a cluster emit a one-line summary:
    "Cluster: <label> — N memories excluded (budget)"
  - This is a lightweight stand-in until RAPTOR cluster summaries (Sub-phase 2.4).

Metrics:
  Compression Ratio = total_input_tokens / packet_tokens  (target ≥ 8x)
  RCD proxy         = Σ(tok[m] × importance[m]/5) / packet_tokens  (target ≥ 0.65)
    RCD proxy is an importance-weighted estimate; gold labels arrive in Sub-phase 2.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Approximate token count (word-count × 1.3 heuristic, same as Phase 1)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Extractive unit formatter
# ---------------------------------------------------------------------------

def format_extractive_unit(
    memory: Any,
    include_source_quote: bool = True,
    include_retrieval_hint: bool = True,
) -> str:
    """Format one Memory object as a retrievable extractive context unit.

    Fields used (per plan §2.2):
      retrieval_hint + source_quote + title + canonical_type + timestamp + provenance
    """
    lines: list[str] = []

    mem_type = getattr(memory, "canonical_type", None) or getattr(memory, "type", "memory")
    ts = ""
    ca = getattr(memory, "created_at", None)
    if ca is not None:
        if hasattr(ca, "strftime"):
            ts = ca.strftime("%Y-%m-%d")
        else:
            ts = str(ca)[:10]

    importance = getattr(memory, "importance", 3)
    lines.append(f"## [{mem_type}] {memory.title}")

    meta: list[str] = [f"Recorded: {ts}", f"Importance: {importance}/5"]
    sid = getattr(memory, "source_session_id", None)
    if sid:
        meta.append(f"Session: {str(sid)[:8]}…")
    lines.append(f"*{' | '.join(meta)}*")

    if include_retrieval_hint:
        hint = getattr(memory, "retrieval_hint", None)
        if hint:
            lines.append(f"**Hint:** {hint}")

    if include_source_quote:
        quote = getattr(memory, "source_quote", None)
        if quote:
            lines.append(f'**Source:** "{quote}"')

    lines.append(getattr(memory, "content", ""))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly result
# ---------------------------------------------------------------------------

@dataclass
class AssemblyResult:
    """Output of assemble_context_packet."""
    content: str
    token_count: int
    included_memory_ids: list[str] = field(default_factory=list)
    excluded_memory_ids: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    compression_ratio: float = 0.0       # target ≥ 8x
    rcd_proxy: float = 0.0               # target ≥ 0.65
    used_abstractive_fallback: bool = False


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def assemble_context_packet(
    memories: list[Any],
    scores: dict[str, float] | None = None,
    token_budget: int = 4000,
    include_source_quote: bool = True,
    include_retrieval_hint: bool = True,
    query: str = "",
    project_name: str = "",
) -> AssemblyResult:
    """
    Build a token-budgeted context packet from RRF-ranked memories.

    Args:
        memories:               Memory objects in RRF rank order (best first).
        scores:                 RRF score per memory_id (not used for ordering here —
                                caller already sorted by score; kept for future weighting).
        token_budget:           Maximum tokens for the assembled packet (default 4000).
        include_source_quote:   Include source_quote in each extractive unit.
        include_retrieval_hint: Include retrieval_hint in each extractive unit.
        query:                  Original query string (prepended as packet header).
        project_name:           Project name for the packet header.

    Returns:
        AssemblyResult with content, metrics, and included/excluded ID lists.
    """
    if scores is None:
        scores = {}

    # ------------------------------------------------------------------
    # Build packet header (always included; counted in total token budget)
    # ------------------------------------------------------------------
    header_parts: list[str] = []
    if project_name:
        header_parts.append(f"# Context Packet — {project_name}")
    if query:
        header_parts.append(f"**Query:** {query}")
    header_parts.append("---")
    header = "\n".join(header_parts)
    header_tokens = count_tokens(header)

    # ------------------------------------------------------------------
    # Compute extractive unit + token count for every memory
    # ------------------------------------------------------------------
    # (memory, unit_text, unit_token_count, content_token_count)
    memory_units: list[tuple[Any, str, int, int]] = []
    total_content_tokens = 0
    for m in memories:
        unit = format_extractive_unit(m, include_source_quote, include_retrieval_hint)
        unit_tok = count_tokens(unit)
        content_tok = count_tokens(getattr(m, "content", ""))
        total_content_tokens += content_tok
        memory_units.append((m, unit, unit_tok, content_tok))

    # ------------------------------------------------------------------
    # Greedy inclusion: iterate rank order (best first)
    # ------------------------------------------------------------------
    remaining_budget = max(0, token_budget - header_tokens)
    included_ids: list[str] = []
    included_units: list[str] = []
    included_unit_tokens: list[int] = []
    excluded_ids: list[str] = []
    excluded_memories: list[Any] = []

    for m, unit_text, unit_tok, _ in memory_units:
        if unit_tok <= remaining_budget:
            included_ids.append(m.id)
            included_units.append(unit_text)
            included_unit_tokens.append(unit_tok)
            remaining_budget -= unit_tok
        else:
            excluded_ids.append(m.id)
            excluded_memories.append(m)

    # ------------------------------------------------------------------
    # Guarantee: always include at least one memory (even over budget)
    # ------------------------------------------------------------------
    if not included_ids and memory_units:
        m0, unit0, tok0, _ = memory_units[0]
        included_ids.append(m0.id)
        included_units.append(unit0)
        included_unit_tokens.append(tok0)
        if m0.id in excluded_ids:
            excluded_ids.remove(m0.id)
            excluded_memories = [m for m in excluded_memories if m.id != m0.id]

    # ------------------------------------------------------------------
    # Abstractive fallback: cluster_label summary for excluded memories
    # ------------------------------------------------------------------
    used_abstractive_fallback = False
    abstractive_lines: list[str] = []
    if excluded_memories:
        clusters: dict[int, list[Any]] = {}
        for m in excluded_memories:
            cid = getattr(m, "cluster_id", None)
            if cid is not None:
                clusters.setdefault(cid, []).append(m)

        if clusters:
            abstractive_lines.append("**Excluded clusters (budget):**")
            for cid in sorted(clusters):
                cluster_mems = clusters[cid]
                label = getattr(cluster_mems[0], "cluster_label", None) or f"Cluster {cid}"
                n = len(cluster_mems)
                abstractive_lines.append(
                    f"- {label} — {n} memor{'ies' if n > 1 else 'y'} excluded"
                )
            used_abstractive_fallback = True

    # ------------------------------------------------------------------
    # Assemble final content
    # ------------------------------------------------------------------
    sections: list[str] = [header]
    sections.extend(included_units)
    if abstractive_lines:
        sections.append("\n".join(abstractive_lines))

    content = "\n\n".join(sections)
    packet_tokens = count_tokens(content)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    total_input_tokens = max(total_content_tokens, 1)

    # Compression Ratio: how much was compressed (higher = leaner packet)
    compression_ratio = total_input_tokens / max(packet_tokens, 1)

    # RCD proxy: importance-weighted fraction of packet tokens
    included_set = set(included_ids)
    weighted_relevant = 0.0
    for m, _, unit_tok, _ in memory_units:
        if m.id in included_set:
            importance = getattr(m, "importance", 3)
            weighted_relevant += unit_tok * (importance / 5)
    rcd_proxy = weighted_relevant / max(packet_tokens, 1)

    return AssemblyResult(
        content=content,
        token_count=packet_tokens,
        included_memory_ids=included_ids,
        excluded_memory_ids=excluded_ids,
        total_input_tokens=total_input_tokens,
        compression_ratio=compression_ratio,
        rcd_proxy=rcd_proxy,
        used_abstractive_fallback=used_abstractive_fallback,
    )
