"""
Pre-Phase 2 — Cross-chunk near-duplicate detection and deduplication.

After LLM extracts memories from multiple chunks, the same fact can appear
in overlapping regions (due to chunk overlap) or be stated twice in different
sessions. This module removes near-duplicates before DB write.

Strategy:
  1. Title-based dedup: SequenceMatcher ratio >= title_threshold (0.85).
     Catches "Use Redis for caching" vs "Use Redis as cache layer".
  2. Content-based dedup: SequenceMatcher ratio >= content_threshold (0.90).
     Catches identical content with different titles.
  3. Source-quote dedup: identical source_quote strings (exact match).
     Catches the same sentence extracted twice from chunk overlap.

Order matters: deduplicate across chunks in extraction order.
When a duplicate is detected, keep the one with higher confidence.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from .base import ExtractedMemoryDraft


@dataclass
class DeduplicationResult:
    """Result of running deduplication over a list of draft memories."""
    kept: list[ExtractedMemoryDraft]
    removed: list[ExtractedMemoryDraft]
    removed_titles: list[str]

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def _similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings (case-insensitive)."""
    return difflib.SequenceMatcher(
        None, a.lower().strip(), b.lower().strip()
    ).ratio()


def deduplicate_drafts(
    drafts: list[ExtractedMemoryDraft],
    title_threshold: float = 0.85,
    content_threshold: float = 0.90,
) -> DeduplicationResult:
    """
    Remove near-duplicate drafts from LLM extraction output.

    For each pair (a, b) where a comes before b:
      - If title similarity >= title_threshold AND same canonical_type → b is duplicate.
      - If content similarity >= content_threshold (any type) → b is duplicate.
      - If source_quote is identical (non-empty) → b is duplicate.
      - Duplicate: keep whichever has higher confidence. If b wins, swap.

    Args:
        drafts:             List of ExtractedMemoryDraft (in extraction order).
        title_threshold:    Similarity floor for title-based dedup (default 0.85).
        content_threshold:  Similarity floor for content-based dedup (default 0.90).

    Returns:
        DeduplicationResult with kept/removed splits.
    """
    if not drafts:
        return DeduplicationResult(kept=[], removed=[], removed_titles=[])

    kept: list[ExtractedMemoryDraft] = []
    removed: list[ExtractedMemoryDraft] = []

    for candidate in drafts:
        is_dup = False
        for i, existing in enumerate(kept):
            # --- Source quote exact match (fastest check) ---
            sq_c = (candidate.source_quote or "").strip()
            sq_e = (existing.source_quote or "").strip()
            if sq_c and sq_e and sq_c == sq_e:
                is_dup = True
                # Keep higher confidence
                if candidate.confidence > existing.confidence:
                    kept[i] = candidate
                    removed.append(existing)
                else:
                    removed.append(candidate)
                break

            # --- Title similarity (same canonical_type) ---
            if candidate.canonical_type == existing.canonical_type:
                t_sim = _similarity(candidate.title, existing.title)
                if t_sim >= title_threshold:
                    is_dup = True
                    if candidate.confidence > existing.confidence:
                        kept[i] = candidate
                        removed.append(existing)
                    else:
                        removed.append(candidate)
                    break

            # --- Content similarity (cross-type) ---
            c_sim = _similarity(candidate.content, existing.content)
            if c_sim >= content_threshold:
                is_dup = True
                if candidate.confidence > existing.confidence:
                    kept[i] = candidate
                    removed.append(existing)
                else:
                    removed.append(candidate)
                break

        if not is_dup:
            kept.append(candidate)

    return DeduplicationResult(
        kept=kept,
        removed=removed,
        removed_titles=[d.title for d in removed],
    )


def deduplicate_against_existing(
    drafts: list[ExtractedMemoryDraft],
    existing_titles: list[str],
    existing_contents: list[str],
    title_threshold: float = 0.85,
    content_threshold: float = 0.90,
) -> DeduplicationResult:
    """
    Filter out drafts that are near-duplicates of memories already in the DB.

    Used to prevent re-extracting memories that a previous session already stored.
    Much cheaper than embedding-based dedup — no model needed.

    Args:
        drafts:             New draft memories from LLM extraction.
        existing_titles:    Titles of memories already in the project DB.
        existing_contents:  Contents of memories already in the project DB.
        title_threshold:    Similarity floor for title-based dedup.
        content_threshold:  Similarity floor for content-based dedup.

    Returns:
        DeduplicationResult — kept = not duplicates of existing DB memories.
    """
    if not drafts:
        return DeduplicationResult(kept=[], removed=[], removed_titles=[])

    if not existing_titles and not existing_contents:
        return DeduplicationResult(kept=list(drafts), removed=[], removed_titles=[])

    kept: list[ExtractedMemoryDraft] = []
    removed: list[ExtractedMemoryDraft] = []

    for draft in drafts:
        is_dup = False

        # Title check against existing DB memories
        for ex_title in existing_titles:
            if _similarity(draft.title, ex_title) >= title_threshold:
                is_dup = True
                break

        # Content check against existing DB memories
        if not is_dup:
            for ex_content in existing_contents:
                if _similarity(draft.content, ex_content) >= content_threshold:
                    is_dup = True
                    break

        if is_dup:
            removed.append(draft)
        else:
            kept.append(draft)

    return DeduplicationResult(
        kept=kept,
        removed=removed,
        removed_titles=[d.title for d in removed],
    )
