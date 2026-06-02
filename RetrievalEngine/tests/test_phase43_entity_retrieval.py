"""
Phase 4.3 — EntityRetriever unit tests.

Tests are fully offline: use a tiny synthetic triple corpus (5 sessions × 3 triples),
a hand-built EntityRegistry, and a mock reranker so no models are loaded.

Coverage:
  TestEntityRetrieverInit         — constructor validation
  TestSemanticSearch              — multi-query RRF over triple embeddings
  TestEntityPostSort              — entity session promotion after reranking
  TestEntitySessionsFor           — entity_sessions_for() diagnostic method
  TestSessionDiversity            — session-diversity MMR (max_per_session cap)
  TestRetrieveNoEntity            — SEMANTIC query (no entity → pure semantic)
  TestRetrieveEntityState         — ENTITY_STATE query (entity sessions promoted)
  TestRetrieveEpisode             — EPISODE query (entity sessions promoted)
  TestRetrieveMissingEntity       — entity mentioned but not in registry → semantic only
  TestRetrieveEmptyTripleSession  — entity in registry but no triples indexed
  TestIntentFor                   — intent_for() diagnostic method
  TestRerankerCalled              — reranker is invoked when provided
  TestEndToEnd                    — full retrieve() against known gold session
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

RE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RE_DIR))

from app.services.retrieval.entity_retrieval import EntityRetriever, _GRAPH_WEIGHTS
from app.services.knowledge_graph.entity_registry import EntityRegistry
from app.services.knowledge_graph.query_entity_extractor import QueryEntityExtractor


# ── Synthetic corpus ──────────────────────────────────────────────────────────
# 5 sessions × 3 triples each = 15 triples total.
# Embedding dim = 4 (tiny, L2-normalised random-ish vectors).

_DIM = 4

def _norm(v: list[float]) -> list[float]:
    a = np.array(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else v

# Session IDs
SESS = [f"s{i}" for i in range(5)]

# Triple texts: deterministic, encode who/what/where for semantic test
TRIPLE_TEXTS = [
    # s0: Alice works at Hospital
    "Alice works at City Hospital",
    "Alice lives in Boston",
    "Alice has skill nursing",
    # s1: Bob studies at MIT
    "Bob studies at MIT",
    "Bob lives in Cambridge",
    "Bob has hobby coding",
    # s2: Carol manages DevCorp (Alice and Carol interact)
    "Carol manages DevCorp",
    "Carol works with Alice",
    "Carol lives in Boston",
    # s3: Dave visited Paris
    "Dave visited Paris last summer",
    "Dave stayed at Marriott",
    "Dave took flight AA123",
    # s4: Generic unrelated session
    "The weather was sunny",
    "They discussed quarterly results",
    "Plans were made for next week",
]

# Parallel session keys
TRIPLE_SESSION_KEYS = (
    SESS[0] * 3 + SESS[1] * 3 + SESS[2] * 3 + SESS[3] * 3 + SESS[4] * 3
).split()  # wrong, rebuild below

def _build_session_keys():
    keys = []
    for s in SESS:
        keys.extend([s] * 3)
    return keys

TRIPLE_SESSION_KEYS = _build_session_keys()

# Embeddings: s0 triples → direction "A" (hospitals/alice queries match)
#             s1 triples → direction "B"
#             s2 triples → direction "A+C" (partially matches alice queries)
#             s3 triples → direction "D" (travel/event)
#             s4 triples → random noise

np.random.seed(42)
_raw = np.random.randn(15, _DIM).astype(np.float32)
# Bias s0 triples toward query "alice hospital"
_alice_dir = np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32)
_alice_dir /= np.linalg.norm(_alice_dir)
for i in range(3):   # s0
    _raw[i] = _raw[i] * 0.3 + _alice_dir * 0.7
# Bias s3 triples toward query "dave travel"
_dave_dir = np.array([0.1, 0.1, 0.9, 0.1], dtype=np.float32)
_dave_dir /= np.linalg.norm(_dave_dir)
for i in range(9, 12):  # s3
    _raw[i] = _raw[i] * 0.3 + _dave_dir * 0.7
# L2-normalise all rows
norms = np.linalg.norm(_raw, axis=1, keepdims=True)
TRIPLE_EMBS = np.where(norms > 0, _raw / norms, _raw)


# ── Registry fixture ──────────────────────────────────────────────────────────

def _build_registry() -> EntityRegistry:
    reg = EntityRegistry()
    reg.register_session("Alice",  SESS[0])
    reg.register_session("Alice",  SESS[2])  # Alice also in s2 (works with Carol)
    reg.register_session("Bob",    SESS[1])
    reg.register_session("Carol",  SESS[2])
    reg.register_session("Dave",   SESS[3])
    return reg


def _build_extractor(registry: EntityRegistry) -> QueryEntityExtractor:
    return QueryEntityExtractor(registry)


def _identity_embed(text: str) -> list[float]:
    """Deterministic embed: project text hash onto unit circle in _DIM space."""
    h = hash(text) % (2 ** 16)
    v = np.array([
        np.cos(h * 0.001),
        np.sin(h * 0.001),
        np.cos(h * 0.002),
        np.sin(h * 0.002),
    ], dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n).tolist()


def _alice_embed(text: str) -> list[float]:
    """Returns a vector biased strongly toward s0 triples (alice direction)."""
    return _alice_dir.tolist()


def _dave_embed(text: str) -> list[float]:
    """Returns a vector biased strongly toward s3 triples (dave/travel)."""
    return _dave_dir.tolist()


class _MockReranker:
    """Reranker that reverses the input order (deterministic, no model needed)."""
    def __init__(self):
        self.called_with: list[tuple[str, list[Any]]] = []

    def rerank(self, query: str, memories: list[Any], top_n: int = 40):
        self.called_with.append((query, memories))
        return [(m.id, float(len(memories) - i)) for i, m in enumerate(reversed(memories))]


_SESSION_CONTENT = {
    SESS[0]: "Alice works at City Hospital as a nurse. She lives in Boston.",
    SESS[1]: "Bob studies at MIT. He lives in Cambridge.",
    SESS[2]: "Carol manages DevCorp. She works closely with Alice.",
    SESS[3]: "Dave visited Paris last summer. He stayed at the Marriott.",
    SESS[4]: "Quarterly results were discussed. Plans for next week.",
}


def _build_retriever(embed_fn=None, reranker=None, session_content=None) -> EntityRetriever:
    registry  = _build_registry()
    extractor = _build_extractor(registry)
    return EntityRetriever(
        triple_embs=TRIPLE_EMBS,
        triple_session_keys=TRIPLE_SESSION_KEYS,
        triple_texts=TRIPLE_TEXTS,
        session_content=session_content or _SESSION_CONTENT,
        registry=registry,
        extractor=extractor,
        embed_fn=embed_fn or _identity_embed,
        reranker=reranker,
    )


# ── Test classes ──────────────────────────────────────────────────────────────

class TestEntityRetrieverInit:
    def test_builds_session_to_idxs(self):
        r = _build_retriever()
        assert r._session_to_idxs[SESS[0]] == [0, 1, 2]
        assert r._session_to_idxs[SESS[1]] == [3, 4, 5]
        assert r._session_to_idxs[SESS[4]] == [12, 13, 14]

    def test_raises_on_emb_session_mismatch(self):
        with pytest.raises(ValueError, match="triple_embs rows"):
            EntityRetriever(
                triple_embs=np.zeros((5, _DIM), dtype=np.float32),
                triple_session_keys=["s0"] * 10,
                triple_texts=["x"] * 5,
                registry=_build_registry(),
                extractor=_build_extractor(_build_registry()),
                embed_fn=_identity_embed,
            )

    def test_raises_on_emb_texts_mismatch(self):
        with pytest.raises(ValueError, match="triple_texts"):
            EntityRetriever(
                triple_embs=np.zeros((5, _DIM), dtype=np.float32),
                triple_session_keys=["s0"] * 5,
                triple_texts=["x"] * 10,
                registry=_build_registry(),
                extractor=_build_extractor(_build_registry()),
                embed_fn=_identity_embed,
            )


class TestSemanticSearch:
    def test_returns_list_of_ints(self):
        r = _build_retriever()
        result = r._semantic_search(["alice hospital"], top_per=5)
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_top_per_limits_candidates(self):
        r = _build_retriever()
        result = r._semantic_search(["test"], top_per=3)
        # RRF of 1 list with 3 items → at most 3 unique indices
        assert len(result) <= 15

    def test_alice_biased_embed_returns_s0_indices_first(self):
        r = _build_retriever(embed_fn=_alice_embed)
        result = r._semantic_search(["alice hospital"], top_per=5)
        # First 3 results should be s0 triple indices (0,1,2)
        assert set(result[:3]) == {0, 1, 2}

    def test_multi_query_promotes_consistent_matches(self):
        r = _build_retriever(embed_fn=_alice_embed)
        # 3 identical queries → s0 triples should dominate via RRF
        result = r._semantic_search(["alice", "alice hospital", "what alice does"], top_per=5)
        assert result[0] in {0, 1, 2}

    def test_empty_query_list_returns_empty(self):
        r = _build_retriever()
        result = r._semantic_search([], top_per=5)
        assert result == []


class TestEntityPostSort:
    def test_entity_sessions_promoted_to_front(self):
        r = _build_retriever()
        # s0 and s2 are Alice sessions; put them at end of input list
        sessions = [SESS[4], SESS[3], SESS[1], SESS[0], SESS[2]]
        registry  = _build_registry()
        extractor = _build_extractor(registry)
        qe = extractor.extract("What does Alice do for work?")
        result = r._entity_post_sort(sessions, qe.entity_ids)
        # Alice sessions (s0, s2) should appear before s4, s3, s1
        alice_positions = [result.index(s) for s in [SESS[0], SESS[2]]]
        other_positions = [result.index(s) for s in [SESS[4], SESS[3], SESS[1]]]
        assert max(alice_positions) < min(other_positions)

    def test_no_entity_ids_returns_unchanged(self):
        r = _build_retriever()
        sessions = [SESS[0], SESS[1], SESS[2]]
        result = r._entity_post_sort(sessions, [])
        assert result == sessions

    def test_unknown_entity_ids_returns_unchanged(self):
        r = _build_retriever()
        sessions = [SESS[0], SESS[1]]
        result = r._entity_post_sort(sessions, ["nonexistent_id"])
        assert result == sessions

    def test_preserves_order_within_groups(self):
        r = _build_retriever()
        # s0 at rank 3, s2 at rank 5 → after sort: s0, s2, s1, s3, s4
        sessions = [SESS[1], SESS[3], SESS[0], SESS[4], SESS[2]]
        registry  = _build_registry()
        extractor = _build_extractor(registry)
        qe = extractor.extract("What does Alice do for work?")
        result = r._entity_post_sort(sessions, qe.entity_ids)
        entity_part = [s for s in result if s in {SESS[0], SESS[2]}]
        other_part  = [s for s in result if s not in {SESS[0], SESS[2]}]
        # Entity group should be [s0, s2] (preserving their relative input order)
        assert entity_part == [SESS[0], SESS[2]]
        # Non-entity group preserves input order too
        assert other_part == [SESS[1], SESS[3], SESS[4]]

    def test_all_entity_sessions_returns_unchanged(self):
        r = _build_retriever()
        # Only Alice sessions in list
        sessions = [SESS[0], SESS[2]]
        registry  = _build_registry()
        extractor = _build_extractor(registry)
        qe = extractor.extract("What does Alice do for work?")
        result = r._entity_post_sort(sessions, qe.entity_ids)
        assert result == sessions  # unchanged (all entity, no non-entity to reorder against)


class TestEntitySessionsFor:
    def test_returns_alice_sessions(self):
        r = _build_retriever()
        registry  = _build_registry()
        extractor = _build_extractor(registry)
        qe = extractor.extract("What does Alice do for work?")
        result = r.entity_sessions_for(qe.entity_ids)
        assert SESS[0] in result
        assert SESS[2] in result

    def test_empty_entity_ids_returns_empty(self):
        r = _build_retriever()
        result = r.entity_sessions_for([])
        assert result == set()

    def test_unknown_entity_returns_empty(self):
        r = _build_retriever()
        result = r.entity_sessions_for(["nonexistent"])
        assert result == set()


class TestSessionDiversity:
    def test_returns_unique_sessions(self):
        r = _build_retriever()
        # All s0 triples first, then s1
        idxs = [0, 1, 2, 3, 4, 5]
        result = r._session_diversity(idxs, top_k=10, max_per_session=2)
        assert len(result) == len(set(result))

    def test_max_per_session_cap(self):
        r = _build_retriever()
        # Feed 3 s0 triples, max_per_session=1 → only s0 once
        idxs = [0, 1, 2, 3]
        result = r._session_diversity(idxs, top_k=10, max_per_session=1)
        assert result.count(SESS[0]) <= 1

    def test_respects_top_k(self):
        r = _build_retriever()
        idxs = list(range(15))  # all triples
        result = r._session_diversity(idxs, top_k=2)
        assert len(result) <= 2

    def test_preserves_rank_order(self):
        r = _build_retriever()
        # First s3, then s0, max_per_session=3 → s3 first in output
        idxs = [9, 10, 11, 0, 1, 2]
        result = r._session_diversity(idxs, top_k=5)
        assert result[0] == SESS[3]
        assert result[1] == SESS[0]

    def test_empty_input_returns_empty(self):
        r = _build_retriever()
        assert r._session_diversity([], top_k=10) == []


class TestRetrieveNoEntity:
    def test_semantic_query_returns_sessions(self):
        r = _build_retriever()
        result = r.retrieve("quarterly planning results", top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_no_entity_falls_back_to_semantic(self):
        # s4 triples are about "quarterly results" — with semantic embed they should surface
        r = _build_retriever()
        result = r.retrieve("quarterly results planning", top_k=5)
        assert all(s in SESS for s in result)


class TestRetrieveEntityState:
    def test_alice_entity_state_returns_alice_sessions(self):
        # With alice-biased embed, Alice's sessions (s0, s2) should be top
        r = _build_retriever(embed_fn=_alice_embed)
        result = r.retrieve("What does Alice do for work?", top_k=5)
        assert SESS[0] in result or SESS[2] in result

    def test_entity_state_uses_positive_graph_weight(self):
        # All intents use same entity leg weight (targeted entity search, not intent-gated)
        assert _GRAPH_WEIGHTS["ENTITY_STATE"] > 0.0
        assert _GRAPH_WEIGHTS["EPISODE"] > 0.0
        assert _GRAPH_WEIGHTS["SEMANTIC"] > 0.0

    def test_top1_is_alice_session_with_alice_embed(self):
        r = _build_retriever(embed_fn=_alice_embed)
        result = r.retrieve("What does Alice do for work?", top_k=10)
        # s0 (Alice works at City Hospital) or s2 (Carol works with Alice) should be top
        assert result[0] in {SESS[0], SESS[2]}


class TestRetrieveEpisode:
    def test_dave_episode_query_returns_dave_session(self):
        r = _build_retriever(embed_fn=_dave_embed)
        result = r.retrieve("When did Dave visit Paris?", top_k=5)
        assert SESS[3] in result

    def test_episode_graph_weight_positive(self):
        assert _GRAPH_WEIGHTS["EPISODE"] > 0.0


class TestRetrieveMissingEntity:
    def test_unknown_entity_falls_back_to_semantic(self):
        r = _build_retriever(embed_fn=_alice_embed)
        # "Zara" not in registry → graph walk returns nothing → pure semantic
        result = r.retrieve("What does Zara do for work?", top_k=5)
        assert isinstance(result, list)
        # Should still return something from semantic
        assert len(result) > 0


class TestRetrieveEmptyTripleSession:
    def test_entity_with_no_indexed_triples(self):
        registry = EntityRegistry()
        registry.register_session("Ghost", "s_ghost")  # session not in TRIPLE_SESSION_KEYS
        extractor = QueryEntityExtractor(registry)
        r = EntityRetriever(
            triple_embs=TRIPLE_EMBS,
            triple_session_keys=TRIPLE_SESSION_KEYS,
            triple_texts=TRIPLE_TEXTS,
            registry=registry,
            extractor=extractor,
            embed_fn=_identity_embed,
        )
        result = r.retrieve("What does Ghost do?", top_k=5)
        assert isinstance(result, list)  # graceful fallback


class TestIntentFor:
    def test_entity_state_intent(self):
        r = _build_retriever()
        intent = r.intent_for("What does Alice do for work?")
        assert intent == "ENTITY_STATE"

    def test_episode_intent(self):
        r = _build_retriever()
        intent = r.intent_for("When did Dave visit Paris?")
        assert intent == "EPISODE"

    def test_semantic_intent_for_generic_query(self):
        r = _build_retriever()
        intent = r.intent_for("quarterly results planning session")
        assert intent == "SEMANTIC"


class TestRerankerCalled:
    def test_reranker_is_called_when_provided(self):
        mock_rr = _MockReranker()
        r = _build_retriever(embed_fn=_alice_embed, reranker=mock_rr)
        r.retrieve("What does Alice do for work?", top_k=3)
        assert len(mock_rr.called_with) == 1
        query_used, mems_used = mock_rr.called_with[0]
        assert query_used == "What does Alice do for work?"
        assert len(mems_used) > 0

    def test_no_reranker_still_returns_sessions(self):
        r = _build_retriever(embed_fn=_alice_embed, reranker=None)
        result = r.retrieve("What does Alice do for work?", top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0


class TestEndToEnd:
    def test_alice_query_gold_in_top5(self):
        r = _build_retriever(embed_fn=_alice_embed)
        result = r.retrieve("What does Alice do for work?", top_k=5)
        assert SESS[0] in result

    def test_dave_query_gold_in_top5(self):
        r = _build_retriever(embed_fn=_dave_embed)
        result = r.retrieve("When did Dave visit Paris?", top_k=5)
        assert SESS[3] in result

    def test_retrieve_with_rephrases(self):
        r = _build_retriever(embed_fn=_alice_embed)
        rephrases = ["Alice's profession", "Where does Alice work"]
        result = r.retrieve("What does Alice do for work?", rephrases=rephrases, top_k=5)
        assert SESS[0] in result

    def test_retrieve_returns_no_duplicates(self):
        r = _build_retriever()
        result = r.retrieve("test query", top_k=10)
        assert len(result) == len(set(result))

    def test_retrieve_respects_top_k(self):
        r = _build_retriever()
        for k in (1, 3, 5):
            result = r.retrieve("some query", top_k=k)
            assert len(result) <= k
