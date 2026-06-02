"""
Phase 4.6 — Haiku Temporal Reranker at Retrieval.

One Claude Haiku 4.5 call per query: given the top-20 sessions from A11 (each
with its best memory text and session position), ask Haiku to rerank them
applying temporal reasoning the ms-marco cross-encoder cannot perform.

What Haiku resolves that ms-marco cannot:
  - "Where does Alice work NOW?" → prefers LATER session position
  - "What was Alice's PREVIOUS job?" → prefers EARLIER session position
  - "When did Alice FIRST mention X?" → finds EARLIEST relevant session
  - Session-position-aware disambiguation of state-change queries

Why ms-marco fails at this:
  ms-marco is trained on (query, Wikipedia passage) relevance. It has no concept
  of session ordering, temporal state changes, or conversational memory. It scores
  ALL semantically similar memories equally, regardless of whether they represent
  the current or historical state.

Caching:
  Haiku outputs are deterministic (temperature=0). Cached to disk keyed by
  hash(query + candidate memory texts). After first run = $0 on reruns.

Cost per full run:
  690 LoCoMo + 500 LME = 1,190 queries × ~950 input tokens + ~80 output tokens
  ≈ $0.94 input + $0.38 output = ~$1.32 total per model_tag run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a precise ranking assistant. You return only valid JSON, "
    "no explanation or commentary."
)

_PROMPT = """\
Rank these memory snippets to best answer the user query.

Query: {query}

Memories (session_pos = chronological position, higher = more recent):
{memories_block}

Follow these two steps in order:

STEP 1 — RELEVANCE (always the primary criterion):
  Which memories directly answer or clearly relate to the query topic?
  An irrelevant memory must rank BELOW all relevant ones — regardless of session_pos.

STEP 2 — TEMPORAL TIE-BREAKING (secondary, only among relevant memories):
  Apply ONLY when the query explicitly mentions time or ordering:
  • "now / currently / still / today" → prefer HIGHER session_pos among relevant memories
  • "previously / used to / before / at the time" → prefer LOWER session_pos among relevant
  • "first time / first [specific thing]" → prefer the LOWEST session_pos where that SPECIFIC thing is actually described (not just any tangentially related session)
  • "most recent / latest" → prefer HIGHEST session_pos among relevant memories
  • No temporal keyword in query → rank purely by relevance, ignore session_pos entirely

Critical: The "first" rule means the earliest session where the EXACT THING being asked about appears, not the earliest session that mentions a related topic.

Return ONLY this JSON (no markdown, no explanation):
{{"ranked": [1, 2, 3, ...]}}

1-based indices ordered MOST to LEAST relevant. Include all {n} indices exactly once."""


def _build_prompt(query: str, candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        text = c["memory_text"][:300].replace('"', "'")
        pos  = c.get("session_position", 1)
        lines.append(f'{i}. [pos={pos}] "{text}"')
    memories_block = "\n".join(lines)
    return _PROMPT.format(
        query=query,
        memories_block=memories_block,
        n=len(candidates),
    )


def _parse_response(text: str, n: int) -> list[int]:
    """Parse Haiku JSON response into a list of 1-based indices.

    Tries in order:
      1. Parse JSON and extract 'ranked' array
      2. Find a JSON array anywhere in the text with regex
      3. Return identity (1..n) as fallback

    Invalid indices (out of range or duplicate) are dropped; missing indices
    are appended at the end in natural order.
    """
    def _extract_list(raw: Any) -> list[int]:
        if not isinstance(raw, list):
            return []
        result = []
        seen: set[int] = set()
        for v in raw:
            try:
                i = int(v)
                if 1 <= i <= n and i not in seen:
                    result.append(i)
                    seen.add(i)
            except (TypeError, ValueError):
                pass
        # Append any missing indices
        missing = [i for i in range(1, n + 1) if i not in seen]
        return result + missing

    # Attempt 1: clean JSON parse
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "ranked" in obj:
            return _extract_list(obj["ranked"])
        if isinstance(obj, list):
            return _extract_list(obj)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find JSON block with regex
    match = re.search(r'\{.*?"ranked"\s*:\s*(\[[\d\s,]+\]).*?\}', text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(1))
            return _extract_list(arr)
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: find any number array in the text
    match = re.search(r'\[[\d\s,]+\]', text)
    if match:
        try:
            arr = json.loads(match.group(0))
            return _extract_list(arr)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: identity
    return list(range(1, n + 1))


def _cache_key(query: str, candidates: list[dict]) -> str:
    """Deterministic cache key from query + candidate memory texts."""
    parts = [query] + [c["memory_text"][:150] for c in candidates]
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class HaikuTemporalReranker:
    """
    Phase 4.6 temporal reranker using Claude Haiku 4.5.

    Takes the top-20 sessions from A11 (each with its best memory text and
    session position) and reranks them using Haiku's temporal reasoning.

    Parameters
    ----------
    api_key       : Anthropic API key (None = read from ANTHROPIC_API_KEY env)
    model         : Model ID to use (default: claude-haiku-4-5-20251001)
    max_candidates: Max sessions to show Haiku (default: 20)
    cache_path    : Path to JSON cache file. Loaded on init, saved after each call.
    """

    def __init__(
        self,
        api_key:        str | None = None,
        model:          str = "claude-haiku-4-5-20251001",
        max_candidates: int = 20,
        cache_path:     Path | None = None,
    ):
        self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model    = model
        self._max      = max_candidates
        self._client   = None   # lazy-loaded
        self._cache_path = cache_path
        self._cache: dict[str, list[int]] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
    ) -> list[str]:
        """Rerank session IDs using Haiku temporal reasoning.

        Parameters
        ----------
        query      : The user query
        candidates : List of dicts, each with:
                       "session_id":       str
                       "memory_text":      str (best memory for this session)
                       "session_position": int (chronological ordinal)

        Returns
        -------
        list[str] : session_ids in reranked order (best first)
        """
        candidates = candidates[: self._max]
        n = len(candidates)
        if n == 0:
            return []
        if n == 1:
            return [candidates[0]["session_id"]]

        key = _cache_key(query, candidates)
        if key in self._cache:
            ranked_idx = self._cache[key]
        else:
            prompt = _build_prompt(query, candidates)
            ranked_idx = self._call_haiku(prompt, n)
            self._cache[key] = ranked_idx
            self._save_cache()

        return [candidates[i - 1]["session_id"] for i in ranked_idx]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for HaikuTemporalReranker. "
                    "Install: pip install anthropic"
                ) from e
        return self._client

    def _call_haiku(self, prompt: str, n: int) -> list[int]:
        """Call Haiku and return parsed ranked indices. Falls back to identity on error."""
        if not self._api_key:
            return list(range(1, n + 1))
        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self._model,
                max_tokens=200,
                temperature=0,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = msg.content[0].text.strip()
            return _parse_response(raw_text, n)
        except Exception:
            return list(range(1, n + 1))

    def _save_cache(self) -> None:
        if self._cache_path:
            try:
                self._cache_path.write_text(
                    json.dumps(self._cache, ensure_ascii=False)
                )
            except OSError:
                pass

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def cache_size(self) -> int:
        """Number of cached query results."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear in-memory cache (does not delete the cache file)."""
        self._cache.clear()
