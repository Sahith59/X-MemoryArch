"""
Phase 4.5 — Multi-Signal Retrieval Engine (Mem0 architecture, zero API cost).

Three parallel signals fused into a combined score:
  1. Dense semantic:  max cosine across query variants (GTE-large, pre-computed)
  2. BM25 keyword:    rank_bm25 over lemmatized 15-80 word memories, adaptive sigmoid
  3. Entity boost:    spread-attenuated (score × ENTITY_BOOST_WEIGHT × 1/memory_count)

Plus recency bias for temporal queries, then ms-marco reranker on top-40 pool.

Why spread attenuation matters:
  "John" (309 memories) → boost per memory = 0.3/309 ≈ 0.001 (noise-level)
  "Caroline" (40 memories) → boost per memory = 0.3/40 = 0.0075 (small but real)
  unique entity (1 memory) → boost per memory = 0.3/1 = 0.3 (dominant signal)
  This prevents high-frequency entities from flooding the ranking pool.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable

import numpy as np

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _BM25Okapi = None  # type: ignore[assignment,misc]
    _HAS_BM25 = False

try:
    from nltk.stem import WordNetLemmatizer as _WNL
    _lemmatizer_instance = _WNL()

    def _lemmatize(text: str) -> list[str]:
        return [_lemmatizer_instance.lemmatize(w) for w in text.lower().split()]
except Exception:
    def _lemmatize(text: str) -> list[str]:  # type: ignore[misc]
        return text.lower().split()

# Capitalized proper-name pattern (each word ≥ 3 chars to avoid "I", "He", etc.)
_ENTITY_RE = re.compile(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b')

# Words to skip even if capitalized (question words, determiners, common verbs)
_SKIP_WORDS = frozenset({
    "what", "where", "when", "who", "how", "why", "which", "this", "that",
    "these", "those", "they", "their", "there", "the", "and", "but", "for",
    "are", "was", "were", "does", "did", "can", "could", "would", "should",
    "has", "had", "have", "will", "been", "his", "her", "him", "she", "they",
    "during", "about", "from", "with", "into", "then", "than", "also",
    "have", "does", "did", "any", "all", "some", "one", "two", "now",
    "did", "not", "yet", "just", "let", "get",
})

# Recency indicator keywords — activate recency bias when present in query
_RECENCY_KEYWORDS = frozenset({
    "now", "currently", "still", "latest", "recent", "recently", "today",
    "current", "present", "anymore", "nowadays", "right now",
})

# ── Module-level constants (override per-instance for grid search) ─────────────
DEFAULT_ENTITY_BOOST_WEIGHT: float = 5.0   # Phase 5.1: tuned up from 0.3 — at 0.3/memory_count the boost was sub-0.001, unmeasurable
DEFAULT_RECENCY_WEIGHT:      float = 0.1
DEFAULT_SEMANTIC_THRESHOLD:  float = 0.3
DEFAULT_RERANKER_POOL_SIZE:  int   = 40
_TOP_PER_SEARCH:             int   = 25   # raised from 20 — more memories/session means we need wider initial pool per variant
_MAX_PER_SESSION:            int   = 2
_RRF_K:                      int   = 60


def _adaptive_sigmoid(raw: float, n_tokens: int) -> float:
    """Normalize raw BM25 score to [0,1]. Midpoint adapts to query token count.

    Longer queries produce higher raw BM25 totals, so we raise the midpoint.
    Mem0 formula: long query (≥15 tokens) → mid=12, slope=0.5
                  short query (<15 tokens) → mid=5,  slope=0.8
    """
    mid, slope = (12.0, 0.5) if n_tokens >= 15 else (5.0, 0.8)
    return 1.0 / (1.0 + math.exp(-slope * (raw - mid)))


def _adaptive_sigmoid_vec(raw: np.ndarray, n_tokens: int) -> np.ndarray:
    """Vectorized adaptive sigmoid over a numpy array."""
    mid, slope = (12.0, 0.5) if n_tokens >= 15 else (5.0, 0.8)
    return (1.0 / (1.0 + np.exp(-slope * (raw - mid)))).astype(np.float32)


def _extract_query_entities(query: str) -> list[str]:
    """Extract lowercased proper names from a query string (max 8).

    Uses a capitalized-word regex then filters out skip words. Multi-word
    matches (e.g. "Did Melanie") have leading skip-word tokens stripped so
    "Did Melanie" → "melanie" rather than being dropped entirely.
    """
    results: list[str] = []
    seen: set[str] = set()
    for m in _ENTITY_RE.finditer(query):
        tokens = m.group(1).split()
        # Strip leading tokens that are skip words (e.g. "Did" in "Did Melanie")
        while tokens and tokens[0].lower() in _SKIP_WORDS:
            tokens = tokens[1:]
        if not tokens:
            continue
        name = " ".join(tokens).lower()
        if name not in _SKIP_WORDS and name not in seen and len(name) >= 3:
            results.append(name)
            seen.add(name)
            if len(results) >= 8:
                break
    return results


def _is_recency_query(query: str) -> bool:
    """Return True if the query uses recency indicators."""
    tokens = set(query.lower().split())
    return bool(tokens & _RECENCY_KEYWORDS)


def _session_id_from_memory_id(mid: str) -> str | None:
    """Parse the session_id component from a memory_id.

    Memory ID format: "mem_{session_id}_{type}_{ordinal}"
    Examples:
      "mem_c0_session_1_state_000"    → "c0_session_1"
      "mem_c0_session_1_episodic_000" → "c0_session_1"
      "mem_answer_abc_2_state_001"    → "answer_abc_2"
    """
    if not mid.startswith("mem_"):
        return None
    rest = mid[4:]  # strip "mem_" prefix
    for suffix in ("_state_", "_episodic_"):
        idx = rest.rfind(suffix)
        if idx > 0:
            return rest[:idx]
    # Fallback: strip last two underscore-separated segments
    parts = rest.rsplit("_", 2)
    return "_".join(parts[:-2]) if len(parts) >= 3 else rest


class MultiSignalRetriever:
    """
    Phase 4.5 multi-signal retrieval engine.

    Dense semantic + BM25 + entity boost + recency → score fusion → ms-marco reranker.

    Parameters
    ----------
    mem_texts        : list[str]         One text per memory row (15-80 words)
    mem_embs         : np.ndarray        (N, D) float32 GTE-large embeddings
    mem_session_keys : list[str]         Session ID per row
    mem_positions    : list[int]         Session ordinal position per row (for recency)
    entity_store     : list[dict]        From entity_store_{model}_{ds}.json
    embed_fn         : Callable          GTE-large embed function, returns list[float]
    reranker         : optional          ms-marco reranker with .rerank(query, mems, top_n) API
    mem_ids          : list[str] | None  Memory IDs (for precise memory-level entity boost)
    entity_boost_weight : float          ENTITY_BOOST_WEIGHT hyperparameter
    recency_weight      : float          RECENCY_WEIGHT hyperparameter
    semantic_threshold  : float          Min cosine to include in combined scoring
    reranker_pool_size  : int            Number of candidates fed to the reranker
    """

    def __init__(
        self,
        mem_texts:        list[str],
        mem_embs:         np.ndarray,
        mem_session_keys: list[str],
        mem_positions:    list[int],
        entity_store:     list[dict],
        embed_fn:         Callable[[str], list[float]],
        reranker=None,
        mem_ids:          list[str] | None = None,
        entity_boost_weight: float = DEFAULT_ENTITY_BOOST_WEIGHT,
        recency_weight:      float = DEFAULT_RECENCY_WEIGHT,
        semantic_threshold:  float = DEFAULT_SEMANTIC_THRESHOLD,
        reranker_pool_size:  int   = DEFAULT_RERANKER_POOL_SIZE,
    ):
        n = len(mem_texts)
        if mem_embs.shape[0] != n:
            raise ValueError(
                f"mem_embs rows ({mem_embs.shape[0]}) != mem_texts length ({n})"
            )
        if len(mem_session_keys) != n:
            raise ValueError(
                f"mem_session_keys length ({len(mem_session_keys)}) != {n}"
            )
        if len(mem_positions) != n:
            raise ValueError(
                f"mem_positions length ({len(mem_positions)}) != {n}"
            )
        if mem_ids is not None and len(mem_ids) != n:
            raise ValueError(
                f"mem_ids length ({len(mem_ids)}) != {n}"
            )

        self._texts        = mem_texts
        self._embs         = mem_embs.astype(np.float32)
        self._session_keys = mem_session_keys
        self._embed_fn     = embed_fn
        self._reranker     = reranker

        self._entity_boost_weight = entity_boost_weight
        self._recency_weight      = recency_weight
        self._semantic_threshold  = semantic_threshold
        self._reranker_pool_size  = reranker_pool_size

        # Raw session positions (for Haiku reranker metadata)
        self._positions: list[int] = list(mem_positions)

        # Recency array: session_position normalized to [0.0, 1.0]
        max_pos = max(mem_positions) if mem_positions else 1
        self._recency = np.array(
            [p / max(max_pos, 1) for p in mem_positions], dtype=np.float32
        )

        # session_id → list of row indices (needed for entity boost fallback)
        self._session_to_idxs: dict[str, list[int]] = defaultdict(list)
        for i, sk in enumerate(mem_session_keys):
            self._session_to_idxs[sk].append(i)

        # BM25 index over lemmatized memory texts
        self._bm25 = None
        if _HAS_BM25:
            self._bm25 = _BM25Okapi([_lemmatize(t) for t in mem_texts])

        # Entity index: entity_text (lowercase) → (sorted_unique_row_idxs, memory_count)
        # If mem_ids provided → memory-level precision (only the specific memory rows)
        # If not provided → session-level fallback (all memories from linked sessions)
        mid_to_idx: dict[str, int] = {}
        if mem_ids is not None:
            mid_to_idx = {mid: i for i, mid in enumerate(mem_ids)}

        self._entity_index: dict[str, tuple[list[int], int]] = {}
        for ent in entity_store:
            et: str = ent["entity_text"]
            raw_mids: list[str] = ent.get("linked_memory_ids", [])
            mc: int = ent.get("memory_count", len(raw_mids))

            if mid_to_idx:
                idxs = [mid_to_idx[m] for m in raw_mids if m in mid_to_idx]
            else:
                # Session-level fallback
                idxs = []
                seen_sids: set[str] = set()
                for m in raw_mids:
                    sid = _session_id_from_memory_id(m)
                    if sid and sid not in seen_sids:
                        seen_sids.add(sid)
                        idxs.extend(self._session_to_idxs.get(sid, []))

            if idxs:
                unique_idxs = list(dict.fromkeys(idxs))  # deduplicate, preserve order
                self._entity_index[et] = (unique_idxs, mc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:    str,
        rephrases: list[str] | None = None,
        top_k:    int = 10,
    ) -> list[str]:
        """Return up to top_k ranked session IDs for the given query.

        Two-stage pipeline:

        Stage 1 — Multi-query RRF pool selection:
          Each query variant gets top-_TOP_PER_SEARCH by cosine. RRF fuses the
          ranked lists (memories in multiple top-K get boosted). Returns up to
          2× reranker_pool_size indices, sorted by descending RRF score.

        Stage 2 — Pool re-scoring (RRF base + entity boost + recency):
          The RRF score (normalized to [0,1]) is the base. Entity boost and
          recency bias are added as small adjustments. Sorting by this combined
          score changes the pool ordering slightly — entity-linked memories move
          up, recent memories move up for temporal queries.

          When entity_boost_weight=0 and recency_weight=0, this is identical to
          A11s (Phase 4.4): RRF top-40 → reranker. That is the guaranteed floor.

        BM25 is NOT used in the default pipeline (see bm25_scores() for diagnostics).
        """
        variants = [query] + (rephrases or [])[:2]

        # Stage 1: multi-query RRF pool (returns indices + RRF scores)
        pool, rrf_scores = self._rrf_select_with_scores(variants)
        if not pool:
            return []

        pool_arr = np.array(pool, dtype=np.int64)

        # Normalize RRF scores to [0, 1] so entity boost is on the same scale
        rrf_vals = np.array([rrf_scores[i] for i in pool], dtype=np.float32)
        max_rrf = rrf_vals.max()
        if max_rrf > 0:
            rrf_vals /= max_rrf

        # Stage 2: adjustments on top of normalized RRF
        entity_all = self._entity_boost(query)
        entity_pool = entity_all[pool_arr]

        combined = rrf_vals + entity_pool
        if _is_recency_query(query):
            combined = combined + self._recency_weight * self._recency[pool_arr]

        # Sort descending → take reranker pool
        top_n = min(self._reranker_pool_size, len(pool))
        top_local = np.argsort(combined)[::-1][:top_n]
        final_pool = [pool[i] for i in top_local.tolist()]

        # Rerank on memory text (ms-marco — 50-word memories match its training distribution)
        if self._reranker and final_pool:
            final_pool = self._rerank(query, final_pool)

        # Session diversity → top-k unique sessions
        return self._session_diversity(final_pool, top_k=top_k)

    def retrieve_with_details(
        self,
        query:    str,
        rephrases: list[str] | None = None,
        top_k:    int = 20,
    ) -> list[dict]:
        """Like retrieve(), but returns rich metadata per session for Haiku reranking.

        Returns list of dicts (best-first):
          {"session_id": str, "memory_text": str, "session_position": int}

        "memory_text" is the memory from that session with the highest cosine
        similarity to the primary query — gives Haiku the most relevant snippet
        for temporal reasoning.
        """
        variants = [query] + (rephrases or [])[:2]

        # Same pool selection and scoring as retrieve()
        pool, rrf_scores = self._rrf_select_with_scores(variants)
        if not pool:
            return []

        pool_arr = np.array(pool, dtype=np.int64)
        rrf_vals = np.array([rrf_scores[i] for i in pool], dtype=np.float32)
        max_rrf = rrf_vals.max()
        if max_rrf > 0:
            rrf_vals /= max_rrf

        entity_all = self._entity_boost(query)
        entity_pool = entity_all[pool_arr]

        combined = rrf_vals + entity_pool
        if _is_recency_query(query):
            combined = combined + self._recency_weight * self._recency[pool_arr]

        top_n = min(self._reranker_pool_size, len(pool))
        top_local = np.argsort(combined)[::-1][:top_n]
        final_pool = [pool[i] for i in top_local.tolist()]

        if self._reranker and final_pool:
            final_pool = self._rerank(query, final_pool)

        # For best-memory selection within each session
        qvec = np.array(self._embed_fn(variants[0]), dtype=np.float32)

        selected: list[dict] = []
        seen:     set[str]   = set()
        counts:   dict[str, int] = defaultdict(int)

        for idx in final_pool:
            if len(selected) >= top_k:
                break
            sk = self._session_keys[idx]
            if counts[sk] < _MAX_PER_SESSION:
                counts[sk] += 1
                if sk not in seen:
                    # Best memory for this session = highest cosine to primary query
                    sess_idxs = self._session_to_idxs.get(sk, [idx])
                    sess_arr = np.array(sess_idxs, dtype=np.int64)
                    sess_cos = self._embs[sess_arr] @ qvec
                    best_local = int(np.argmax(sess_cos))
                    best_idx = sess_idxs[best_local]

                    selected.append({
                        "session_id":       sk,
                        "memory_text":      self._texts[best_idx],
                        "session_position": self._positions[best_idx],
                    })
                    seen.add(sk)

        return selected

    def dense_scores(self, query: str) -> np.ndarray:
        """Return raw semantic cosine scores for all memories (for diagnostics)."""
        qvec = np.array(self._embed_fn(query), dtype=np.float32)
        return np.clip(self._embs @ qvec, 0.0, 1.0).astype(np.float32)

    def bm25_scores(self, query: str) -> np.ndarray:
        """Return BM25 signal scores for all memories (for diagnostics)."""
        return self._bm25_signal(query)

    def entity_scores(self, query: str) -> np.ndarray:
        """Return entity boost scores for all memories (for diagnostics)."""
        return self._entity_boost(query)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rrf_select_with_scores(
        self, variants: list[str]
    ) -> tuple[list[int], dict[int, float]]:
        """Multi-query RRF pool selection. Returns (sorted pool indices, {idx: rrf_score}).

        For each query variant, takes top-_TOP_PER_SEARCH memories by cosine.
        RRF fuses across variants — memories in multiple top-K get boosted.
        Returns up to 2×reranker_pool_size indices, sorted by descending RRF score.
        """
        n = len(self._texts)
        rrf: dict[int, float] = {}
        for q in variants[:3]:
            qvec = np.array(self._embed_fn(q), dtype=np.float32)
            sims = self._embs @ qvec
            top_n = min(_TOP_PER_SEARCH, n)
            top = np.argpartition(sims, -top_n)[-top_n:]
            top = top[np.argsort(sims[top])[::-1]]
            for rank, idx in enumerate(top.tolist()):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + _RRF_K)

        target = min(self._reranker_pool_size * 2, n)
        sorted_pool = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:target]
        pool_idxs = [idx for idx, _ in sorted_pool]
        return pool_idxs, dict(sorted_pool)

    def _rrf_select(self, variants: list[str]) -> list[int]:
        """Convenience wrapper — returns pool indices only."""
        idxs, _ = self._rrf_select_with_scores(variants)
        return idxs

    def _dense_signal(self, variants: list[str]) -> np.ndarray:
        """Diagnostic: multi-query max cosine over all N memories. Returns (N,) float32 in [0, 1].

        Not used in the default retrieve() pipeline (RRF select is used instead).
        Useful for ablation studies and debugging.
        """
        best = np.zeros(len(self._texts), dtype=np.float32)
        for q in variants[:3]:
            qvec = np.array(self._embed_fn(q), dtype=np.float32)
            sims = np.clip(self._embs @ qvec, 0.0, 1.0).astype(np.float32)
            np.maximum(best, sims, out=best)
        return best

    def _bm25_signal(self, query: str) -> np.ndarray:
        """BM25 scores, adaptive sigmoid normalized. Returns (N,) float32 in [0, 1]."""
        n = len(self._texts)
        if self._bm25 is None:
            return np.zeros(n, dtype=np.float32)
        tokens = _lemmatize(query)
        n_tokens = len(tokens)
        raw = np.array(self._bm25.get_scores(tokens), dtype=np.float32)
        return _adaptive_sigmoid_vec(raw, n_tokens)

    def _entity_boost(self, query: str) -> np.ndarray:
        """Spread-attenuated entity boost. Returns (N,) float32."""
        boost = np.zeros(len(self._texts), dtype=np.float32)
        extracted = _extract_query_entities(query)
        if not extracted:
            return boost

        for name in extracted:
            ent_data = self._entity_index.get(name)
            if ent_data is None:
                # Prefix match: entity_text startswith name OR name startswith entity_text
                for ek, data in self._entity_index.items():
                    if len(name) >= 3 and (ek.startswith(name) or name.startswith(ek)):
                        ent_data = data
                        break

            if ent_data is None:
                continue

            row_idxs, memory_count = ent_data
            b = self._entity_boost_weight / max(memory_count, 1)
            for idx in row_idxs:
                boost[idx] += b

        return boost

    def _rerank(self, query: str, pool: list[int]) -> list[int]:
        """Rerank row indices by ms-marco cross-encoder on memory text."""
        class _Mem:
            def __init__(self, id_: str, content_: str):
                self.id      = id_
                self.title   = f"mem[{id_}]"
                self.content = content_

        pseudo   = [_Mem(str(idx), self._texts[idx]) for idx in pool]
        reranked = self._reranker.rerank(query, pseudo, top_n=len(pool))
        return [int(mid) for mid, _ in reranked]

    def _session_diversity(
        self,
        pool:  list[int],
        top_k: int = 10,
    ) -> list[str]:
        """Pick top-k unique sessions, capping at _MAX_PER_SESSION memories per session."""
        selected: list[str]    = []
        seen:     set[str]     = set()
        counts:   dict[str, int] = defaultdict(int)
        for idx in pool:
            if len(selected) >= top_k:
                break
            sk = self._session_keys[idx]
            if counts[sk] < _MAX_PER_SESSION:
                counts[sk] += 1
                if sk not in seen:
                    selected.append(sk)
                    seen.add(sk)
        return selected
