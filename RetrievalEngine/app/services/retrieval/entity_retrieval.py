"""
Phase 4.3 — Entity Graph Walk Retrieval.

Pipeline (per query):
  1. Semantic leg: multi-query RRF over all triple embeddings (same as A10)
  2. Entity search leg: for matched entity sessions, take top-N triples by cosine
     similarity to the query (focused, not ALL entity triples)
  3. Weighted RRF blend: entity leg weight 2.0 vs semantic weight 1.0
  4. Rerank top-40 blended pool on triple text (same cross-encoder as A10)
  5. Session-diversity MMR → top-k unique sessions

Why "top-N entity triples" instead of "all entity triples":
  - Caroline appears in 19/272 LoCoMo sessions = 228 entity triples.
  - Using ALL entity triples floods the RRF pool and hurts precision.
  - Top-10 cosine-similar entity triples = exactly the relevant ones.
    E.g. for "What did Caroline research?", these are
    "Caroline studies at University", "Caroline has role researcher" — not
    "Caroline attended LGBTQ support group" (irrelevant, low cosine).
  - 10 entity triples at w=2.0 vs 60 semantic triples at w=1.0 → targeted boost
    without pool degradation.

Why triple-text reranking (not full-session):
  - ms-marco cross-encoder is trained on passage QA (Wikipedia-style).
  - Triple text ("Caroline studies at University of Colorado") looks like a
    factual passage that ANSWERS a question → ms-marco scores it correctly.
  - Full session text (conversational turns) does not look like a factual answer
    → ms-marco scores it lower, hurting retrieval.

Usage (from benchmark):
    retriever = EntityRetriever(
        triple_embs=triple_embs,
        triple_session_keys=triple_session_keys,
        triple_texts=triple_texts,
        registry=registry,
        extractor=extractor,
        embed_fn=st_embed_one,
        reranker=get_default_reranker(),
    )
    sessions = retriever.retrieve(query, rephrases=["alt1", "alt2"])
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable

import numpy as np

from app.services.retrieval.fusion import rrf_fuse

if TYPE_CHECKING:
    from app.services.knowledge_graph.entity_registry import EntityRegistry
    from app.services.knowledge_graph.query_entity_extractor import QueryEntityExtractor

# RRF weights for each leg. Entity leg gets 2× weight to boost entity-matched triples.
# Semantic leg is 1.0 (baseline).
_W_SEMANTIC: float = 1.0
_W_ENTITY:   float = 2.0

# Top-N entity triples per search. Keep small to avoid flooding the pool.
_ENTITY_TOP_N: int = 5   # hyperparameter sweep showed N=5 best (+0.003 over N=0)

# Expose for tests / diagnostics.
_GRAPH_WEIGHTS: dict[str, float] = {
    "ENTITY_STATE": _W_ENTITY,
    "EPISODE": _W_ENTITY,
    "SEMANTIC": _W_ENTITY,
}

_TOP_PER_SEARCH  = 20   # candidate triples per query variant in semantic search
_RRF_POOL        = 40   # triple indices fed to the reranker
_MAX_PER_SESSION = 2    # session-diversity cap


class EntityRetriever:
    """
    Phase 4.3 hybrid retriever: targeted entity search leg + triple semantic + reranker.

    Thread-unsafe (embed_fn / reranker may have internal state). Create one
    instance per evaluation thread or protect externally.
    """

    def __init__(
        self,
        triple_embs: np.ndarray,
        triple_session_keys: list[str],
        triple_texts: list[str],
        registry: "EntityRegistry",
        extractor: "QueryEntityExtractor",
        embed_fn: Callable[[str], list[float]],
        session_content: dict[str, str] | None = None,
        reranker=None,
    ):
        if triple_embs.shape[0] != len(triple_session_keys):
            raise ValueError(
                f"triple_embs rows ({triple_embs.shape[0]}) != "
                f"triple_session_keys length ({len(triple_session_keys)})"
            )
        if triple_embs.shape[0] != len(triple_texts):
            raise ValueError(
                f"triple_embs rows ({triple_embs.shape[0]}) != "
                f"triple_texts length ({len(triple_texts)})"
            )

        self._embs         = triple_embs          # (N, dim) float32
        self._session_keys = triple_session_keys  # length N
        self._texts        = triple_texts          # length N
        self._registry     = registry
        self._extractor    = extractor
        self._embed_fn     = embed_fn
        self._reranker     = reranker
        # session_content kept for API compatibility but not used in current pipeline
        self._session_content = session_content or {}

        # Reverse index: session_id → list of triple indices
        self._session_to_idxs: dict[str, list[int]] = {}
        for i, sk in enumerate(triple_session_keys):
            self._session_to_idxs.setdefault(sk, []).append(i)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        rephrases: list[str] | None = None,
        top_k: int = 10,
    ) -> list[str]:
        """
        Return up to top_k ranked session IDs for the given query.

        Args:
            query:     Raw query string.
            rephrases: 0-2 alternative phrasings (from MQR cache). Optional.
            top_k:     Number of sessions to return.

        Returns:
            Ranked list of session IDs (best first).
        """
        # 1. Entity extraction + intent
        qe = self._extractor.extract(query)

        # 2. Embed query once (shared by both legs)
        qvec = np.array(self._embed_fn(query), dtype=np.float32)

        # 3. Semantic leg: multi-query RRF over all triple embeddings
        variants = [query] + (rephrases or [])[:2]
        semantic_idxs = self._semantic_search(variants, _TOP_PER_SEARCH)

        # 4. Entity search leg (if entity found): top-N entity triples by cosine
        entity_idxs: list[int] = []
        if qe.entity_ids:
            entity_idxs = self._entity_search(qe.entity_ids, qvec, _ENTITY_TOP_N)

        # 5. Weighted RRF blend at triple-index level
        if entity_idxs:
            fused = rrf_fuse(
                [[str(i) for i in semantic_idxs], [str(i) for i in entity_idxs]],
                weights=[_W_SEMANTIC, _W_ENTITY],
            )
        else:
            fused = rrf_fuse([[str(i) for i in semantic_idxs]], weights=[_W_SEMANTIC])
        pool = [int(idx_str) for idx_str, _ in fused[:_RRF_POOL]]

        # 6. Cross-encoder rerank on triple texts (same as A10 baseline)
        if self._reranker and pool:
            pool = self._rerank(query, pool)

        # 7. Session-diversity → top-k unique sessions
        return self._session_diversity(pool, top_k=top_k)

    def intent_for(self, query: str) -> str:
        """Return intent classification for a query (for diagnostics)."""
        return self._extractor.extract(query).intent

    def entity_sessions_for(self, entity_ids: list[str]) -> set[str]:
        """Return all session IDs where any of the given entities appears (for diagnostics)."""
        result: set[str] = set()
        for eid in entity_ids:
            ent = self._registry.get_entity(eid)
            if ent:
                result.update(self._registry.sessions_for_entity(ent.canonical))
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _semantic_search(self, queries: list[str], top_per: int) -> list[int]:
        """Multi-query RRF over triple embeddings. Returns triple indices (best first)."""
        ranked_lists: list[list[str]] = []
        for q in queries[:3]:
            qvec = np.array(self._embed_fn(q), dtype=np.float32)
            sims = self._embs @ qvec
            n    = min(top_per, len(sims))
            top  = np.argpartition(sims, -n)[-n:]
            top  = top[np.argsort(sims[top])[::-1]]
            ranked_lists.append([str(i) for i in top.tolist()])

        fused = rrf_fuse(ranked_lists)
        return [int(idx_str) for idx_str, _ in fused]

    def _entity_search(
        self,
        entity_ids: list[str],
        qvec: np.ndarray,
        top_n: int = _ENTITY_TOP_N,
    ) -> list[int]:
        """
        For matched entity sessions, return top-N triple indices ranked by cosine
        similarity to the query vector.

        Uses only triples from sessions where the entity appears as a subject.
        Taking top-N (not all) prevents flooding the RRF pool with irrelevant
        entity triples that happen to have low cosine similarity.
        """
        entity_sessions: set[str] = set()
        for eid in entity_ids:
            ent = self._registry.get_entity(eid)
            if ent:
                entity_sessions.update(self._registry.sessions_for_entity(ent.canonical))

        if not entity_sessions:
            return []

        entity_triple_idxs: list[int] = []
        for sid in entity_sessions:
            entity_triple_idxs.extend(self._session_to_idxs.get(sid, []))

        if not entity_triple_idxs:
            return []

        idxs = np.array(entity_triple_idxs, dtype=np.int64)
        sims = self._embs[idxs] @ qvec
        # Take top-N by cosine similarity
        n = min(top_n, len(sims))
        top = np.argpartition(sims, -n)[-n:]
        top = top[np.argsort(sims[top])[::-1]]
        return idxs[top].tolist()

    def _rerank(self, query: str, triple_idxs: list[int]) -> list[int]:
        """Cross-encoder rerank on triple text. Returns sorted triple indices."""
        class _Mem:
            def __init__(self, id_: str, content_: str):
                self.id = id_
                self.title = f"triple[{id_}]"
                self.content = content_

        pseudo = [_Mem(str(idx), self._texts[idx]) for idx in triple_idxs]
        reranked = self._reranker.rerank(query, pseudo, top_n=len(triple_idxs))
        return [int(mid) for mid, _ in reranked]

    def _session_diversity(
        self,
        triple_idxs: list[int],
        top_k: int = 10,
        max_per_session: int = _MAX_PER_SESSION,
    ) -> list[str]:
        """Pick top-k unique sessions, capping at max_per_session triples per session."""
        selected: list[str] = []
        seen:     set[str]  = set()
        counts:   dict[str, int] = defaultdict(int)

        for idx in triple_idxs:
            if len(selected) >= top_k:
                break
            sk = self._session_keys[idx]
            if counts[sk] < max_per_session:
                counts[sk] += 1
                if sk not in seen:
                    selected.append(sk)
                    seen.add(sk)

        return selected

    def _entity_post_sort(
        self,
        sessions: list[str],
        entity_ids: list[str],
    ) -> list[str]:
        """
        Move sessions where the query entity appears to the front.
        Preserves internal rank order within each partition.
        Kept for API compatibility and experimental use.
        """
        entity_sess_set: set[str] = set()
        for eid in entity_ids:
            ent = self._registry.get_entity(eid)
            if ent:
                entity_sess_set.update(self._registry.sessions_for_entity(ent.canonical))

        if not entity_sess_set:
            return sessions

        entity_group = [s for s in sessions if s in entity_sess_set]
        other_group  = [s for s in sessions if s not in entity_sess_set]
        return entity_group + other_group
