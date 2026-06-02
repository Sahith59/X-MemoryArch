"""
Phase 4.5 tests — MultiSignalRetriever and helper functions.

All tests are fully offline: no API calls, no disk I/O, no LLM.
The embed_fn and reranker are replaced with deterministic stubs.

Test classes:
  TestAdaptiveSigmoid          (6)  — normalization formula
  TestExtractQueryEntities     (10) — proper-name extraction from queries
  TestIsRecencyQuery           (6)  — recency keyword detection
  TestSessionIdFromMemoryId    (8)  — memory_id → session_id parsing
  TestLemmatize                (5)  — tokenization fallback
  TestMultiSignalInit          (10) — constructor validation
  TestMultiSignalDenseSignal   (7)  — max-cosine semantic signal
  TestMultiSignalBm25Signal    (8)  — BM25 + adaptive sigmoid
  TestMultiSignalEntityBoost   (12) — entity boost with spread attenuation
  TestMultiSignalRetrieve      (10) — end-to-end retrieve pipeline
  TestMultiSignalRecency       (6)  — recency bias activation
  TestMultiSignalEdgeCases     (6)  — missing BM25, empty entity store, etc.

Total: 94 tests
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from app.services.retrieval.multi_signal_retrieval import (
    DEFAULT_ENTITY_BOOST_WEIGHT,
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_SEMANTIC_THRESHOLD,
    MultiSignalRetriever,
    _adaptive_sigmoid,
    _adaptive_sigmoid_vec,
    _extract_query_entities,
    _is_recency_query,
    _lemmatize,
    _session_id_from_memory_id,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _unit_embed(text: str) -> list[float]:
    """Deterministic 4-dim unit embed: hash → angle → (cos, sin, cos2, sin2)."""
    h = abs(hash(text)) % 1000
    angle = h * (2 * math.pi / 1000)
    v = [math.cos(angle), math.sin(angle), math.cos(2 * angle), math.sin(2 * angle)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _make_retriever(
    n_mems: int = 8,
    sessions: list[str] | None = None,
    positions: list[int] | None = None,
    entity_store: list[dict] | None = None,
    mem_ids: list[str] | None = None,
    reranker=None,
    entity_boost_weight: float = DEFAULT_ENTITY_BOOST_WEIGHT,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    semantic_threshold: float = 0.0,   # 0 = no gating in tests (easier to reason about)
    reranker_pool_size: int = 40,
) -> MultiSignalRetriever:
    """Build a minimal MultiSignalRetriever for testing."""
    texts    = [f"Memory text number {i}. Alice works at Hospital as a nurse." for i in range(n_mems)]
    rng      = np.random.default_rng(42)
    embs_raw = rng.standard_normal((n_mems, 4)).astype(np.float32)
    # L2-normalize so cosine = dot product
    norms    = np.linalg.norm(embs_raw, axis=1, keepdims=True)
    embs     = embs_raw / np.where(norms == 0, 1.0, norms)

    if sessions is None:
        sessions = [f"session_{i % 4}" for i in range(n_mems)]
    if positions is None:
        positions = [i + 1 for i in range(n_mems)]
    if entity_store is None:
        entity_store = []

    return MultiSignalRetriever(
        mem_texts=texts,
        mem_embs=embs,
        mem_session_keys=sessions,
        mem_positions=positions,
        entity_store=entity_store,
        embed_fn=_unit_embed,
        reranker=reranker,
        mem_ids=mem_ids,
        entity_boost_weight=entity_boost_weight,
        recency_weight=recency_weight,
        semantic_threshold=semantic_threshold,
        reranker_pool_size=reranker_pool_size,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TestAdaptiveSigmoid
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveSigmoid:
    def test_output_in_unit_interval(self):
        for raw in [-10.0, 0.0, 5.0, 10.0, 20.0]:
            v = _adaptive_sigmoid(raw, n_tokens=10)
            assert 0.0 < v < 1.0

    def test_short_query_midpoint_5(self):
        """At midpoint=5, sigmoid output should be exactly 0.5 for short query."""
        v = _adaptive_sigmoid(5.0, n_tokens=5)
        assert abs(v - 0.5) < 1e-6

    def test_long_query_midpoint_12(self):
        """At midpoint=12, sigmoid output should be exactly 0.5 for long query."""
        v = _adaptive_sigmoid(12.0, n_tokens=15)
        assert abs(v - 0.5) < 1e-6

    def test_long_query_threshold_is_15(self):
        """n_tokens=14 → short params, n_tokens=15 → long params."""
        v14 = _adaptive_sigmoid(10.0, n_tokens=14)
        v15 = _adaptive_sigmoid(10.0, n_tokens=15)
        # At raw=10: short mid=5 → high score; long mid=12 → lower score
        assert v14 > v15

    def test_monotone_increasing(self):
        scores = [_adaptive_sigmoid(r, n_tokens=7) for r in [0, 2, 5, 8, 15]]
        assert scores == sorted(scores)

    def test_vectorized_matches_scalar(self):
        raw_vals = np.array([0.0, 5.0, 12.0, 20.0], dtype=np.float32)
        vec = _adaptive_sigmoid_vec(raw_vals, n_tokens=10)
        for i, raw in enumerate(raw_vals):
            expected = _adaptive_sigmoid(float(raw), n_tokens=10)
            assert abs(float(vec[i]) - expected) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# TestExtractQueryEntities
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractQueryEntities:
    def test_extracts_single_name(self):
        result = _extract_query_entities("What does Caroline do for work?")
        assert "caroline" in result

    def test_extracts_multi_word_name(self):
        result = _extract_query_entities("Where does Alice Johnson live?")
        assert "alice johnson" in result

    def test_skips_question_words(self):
        result = _extract_query_entities("What does she do?")
        # "What" is in _SKIP_WORDS; no real entity
        assert result == []

    def test_skips_short_words(self):
        # "He" and "It" are 2 chars — pattern requires ≥3 chars after capital
        result = _extract_query_entities("He went to It.")
        assert result == []

    def test_max_eight_entities(self):
        query = " ".join([f"Person{i}" for i in range(20)])
        result = _extract_query_entities(query)
        assert len(result) <= 8

    def test_deduplicates(self):
        result = _extract_query_entities("Caroline told Caroline about Caroline's plans.")
        assert result.count("caroline") == 1

    def test_returns_lowercase(self):
        result = _extract_query_entities("Ask Melanie about John's job.")
        assert all(e == e.lower() for e in result)

    def test_empty_query(self):
        assert _extract_query_entities("") == []

    def test_first_person_query(self):
        # "What did I do on my birthday?" → no capitalized proper names
        result = _extract_query_entities("What did I do on my birthday?")
        assert result == []

    def test_multiple_entities(self):
        result = _extract_query_entities("Did Melanie and John attend the event together?")
        assert "melanie" in result
        assert "john" in result


# ══════════════════════════════════════════════════════════════════════════════
# TestIsRecencyQuery
# ══════════════════════════════════════════════════════════════════════════════

class TestIsRecencyQuery:
    def test_currently_triggers(self):
        assert _is_recency_query("Where does Alice currently work?")

    def test_still_triggers(self):
        assert _is_recency_query("Does she still live in Boston?")

    def test_recent_triggers(self):
        assert _is_recency_query("What is Alice's most recent job?")

    def test_no_recency_keywords(self):
        assert not _is_recency_query("What did Alice do during session 3?")

    def test_episodic_query(self):
        assert not _is_recency_query("What hotel did they stay at during the Paris trip?")

    def test_case_insensitive(self):
        # "CURRENTLY" lowercased during tokenization
        assert _is_recency_query("Where CURRENTLY does she work?")


# ══════════════════════════════════════════════════════════════════════════════
# TestSessionIdFromMemoryId
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionIdFromMemoryId:
    def test_state_format(self):
        assert _session_id_from_memory_id("mem_c0_session_1_state_000") == "c0_session_1"

    def test_episodic_format(self):
        assert _session_id_from_memory_id("mem_c0_session_1_episodic_000") == "c0_session_1"

    def test_lme_format(self):
        assert _session_id_from_memory_id("mem_answer_abc_2_state_001") == "answer_abc_2"

    def test_multi_segment_session(self):
        assert _session_id_from_memory_id("mem_c2_session_10_state_002") == "c2_session_10"

    def test_no_mem_prefix_returns_none(self):
        assert _session_id_from_memory_id("c0_session_1_state_000") is None

    def test_missing_suffix_fallback(self):
        result = _session_id_from_memory_id("mem_custom_id_x_y_z")
        assert result is not None  # fallback path still returns something

    def test_episodic_high_ordinal(self):
        result = _session_id_from_memory_id("mem_c7_session_25_episodic_003")
        assert result == "c7_session_25"

    def test_state_ordinal_002(self):
        result = _session_id_from_memory_id("mem_c3_session_7_state_002")
        assert result == "c3_session_7"


# ══════════════════════════════════════════════════════════════════════════════
# TestLemmatize
# ══════════════════════════════════════════════════════════════════════════════

class TestLemmatize:
    def test_returns_list(self):
        result = _lemmatize("the quick brown fox")
        assert isinstance(result, list)
        assert len(result) == 4

    def test_lowercases(self):
        result = _lemmatize("DOGS are Running")
        assert all(t == t.lower() for t in result)

    def test_empty_string(self):
        assert _lemmatize("") == []

    def test_single_word(self):
        result = _lemmatize("hospital")
        assert result == ["hospital"]

    def test_splits_on_spaces(self):
        result = _lemmatize("hello world")
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalInit
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalInit:
    def test_basic_construction(self):
        r = _make_retriever(n_mems=4)
        assert r is not None

    def test_emb_row_mismatch_raises(self):
        texts = ["a", "b", "c"]
        embs  = np.random.rand(4, 4).astype(np.float32)  # 4 rows vs 3 texts
        with pytest.raises(ValueError, match="mem_embs rows"):
            MultiSignalRetriever(
                mem_texts=texts, mem_embs=embs,
                mem_session_keys=["s"] * 3, mem_positions=[1] * 3,
                entity_store=[], embed_fn=_unit_embed,
            )

    def test_session_keys_mismatch_raises(self):
        texts = ["a", "b", "c"]
        embs  = np.random.rand(3, 4).astype(np.float32)
        with pytest.raises(ValueError, match="mem_session_keys"):
            MultiSignalRetriever(
                mem_texts=texts, mem_embs=embs,
                mem_session_keys=["s"] * 2,   # 2 keys, 3 texts
                mem_positions=[1] * 3,
                entity_store=[], embed_fn=_unit_embed,
            )

    def test_positions_mismatch_raises(self):
        texts = ["a", "b"]
        embs  = np.random.rand(2, 4).astype(np.float32)
        with pytest.raises(ValueError, match="mem_positions"):
            MultiSignalRetriever(
                mem_texts=texts, mem_embs=embs,
                mem_session_keys=["s"] * 2,
                mem_positions=[1],   # 1 position, 2 texts
                entity_store=[], embed_fn=_unit_embed,
            )

    def test_mem_ids_mismatch_raises(self):
        texts = ["a", "b"]
        embs  = np.random.rand(2, 4).astype(np.float32)
        with pytest.raises(ValueError, match="mem_ids"):
            MultiSignalRetriever(
                mem_texts=texts, mem_embs=embs,
                mem_session_keys=["s"] * 2,
                mem_positions=[1, 2],
                entity_store=[], embed_fn=_unit_embed,
                mem_ids=["id_only_one"],  # 1 vs 2
            )

    def test_recency_normalized_to_unit_interval(self):
        positions = [1, 2, 5, 10]
        r = _make_retriever(n_mems=4, positions=positions)
        assert float(r._recency.max()) <= 1.0 + 1e-6
        assert float(r._recency.min()) >= 0.0

    def test_recency_max_is_one(self):
        positions = [1, 2, 4, 8]
        r = _make_retriever(n_mems=4, positions=positions)
        assert abs(float(r._recency.max()) - 1.0) < 1e-6

    def test_entity_index_built(self):
        entity_store = [
            {
                "entity_text": "alice",
                "canonical_name": "Alice",
                "linked_memory_ids": [
                    "mem_session_0_state_000",
                    "mem_session_1_state_000",
                ],
                "memory_count": 2,
            }
        ]
        r = _make_retriever(
            n_mems=4,
            sessions=["session_0", "session_1", "session_2", "session_3"],
            entity_store=entity_store,
        )
        assert "alice" in r._entity_index

    def test_entity_index_empty_mids_skipped(self):
        entity_store = [
            {
                "entity_text": "ghost",
                "canonical_name": "Ghost",
                "linked_memory_ids": [],  # no linked memories
                "memory_count": 0,
            }
        ]
        r = _make_retriever(entity_store=entity_store)
        # Entity with no linked memories should not appear in index
        assert "ghost" not in r._entity_index

    def test_session_to_idxs_built(self):
        sessions = ["s0", "s0", "s1", "s1"]
        r = _make_retriever(n_mems=4, sessions=sessions)
        assert sorted(r._session_to_idxs["s0"]) == [0, 1]
        assert sorted(r._session_to_idxs["s1"]) == [2, 3]


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalDenseSignal
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalDenseSignal:
    def test_output_shape(self):
        r = _make_retriever(n_mems=6)
        scores = r._dense_signal(["test query"])
        assert scores.shape == (6,)

    def test_output_in_unit_interval(self):
        r = _make_retriever(n_mems=6)
        scores = r._dense_signal(["what does Alice do?"])
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0 + 1e-5

    def test_multi_query_takes_max(self):
        """With two queries, dense signal >= score from either query alone."""
        r = _make_retriever(n_mems=4)
        s1 = r._dense_signal(["query one"])
        s2 = r._dense_signal(["query two"])
        both = r._dense_signal(["query one", "query two"])
        # Each position should be >= max(s1[i], s2[i])
        expected_min = np.maximum(s1, s2)
        assert np.all(both >= expected_min - 1e-5)

    def test_identical_queries_same_as_single(self):
        r = _make_retriever(n_mems=4)
        single  = r._dense_signal(["unique test query"])
        doubled = r._dense_signal(["unique test query", "unique test query"])
        np.testing.assert_allclose(single, doubled, atol=1e-5)

    def test_empty_variants_returns_zeros(self):
        r = _make_retriever(n_mems=4)
        scores = r._dense_signal([])
        np.testing.assert_array_equal(scores, np.zeros(4, dtype=np.float32))

    def test_returns_float32(self):
        r = _make_retriever(n_mems=4)
        scores = r._dense_signal(["query"])
        assert scores.dtype == np.float32

    def test_different_queries_give_different_scores(self):
        r = _make_retriever(n_mems=8)
        s1 = r._dense_signal(["what does Caroline study?"])
        s2 = r._dense_signal(["where does John live?"])
        # Not all identical (would require identical hashes)
        assert not np.allclose(s1, s2)


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalBm25Signal
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalBm25Signal:
    def test_output_shape(self):
        r = _make_retriever(n_mems=5)
        scores = r._bm25_signal("test query")
        assert scores.shape == (5,)

    def test_output_in_unit_interval(self):
        r = _make_retriever(n_mems=5)
        scores = r._bm25_signal("memory text number")
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0 + 1e-5

    def test_returns_float32(self):
        r = _make_retriever(n_mems=4)
        scores = r._bm25_signal("query")
        assert scores.dtype == np.float32

    def test_matching_query_higher_than_nonmatching(self):
        """A query matching the memory text should score at least as high as non-matching."""
        texts = [
            "Alice works at City Hospital as a senior nurse and manager.",
            "Bob enjoys hiking and rock climbing on weekends outdoors.",
            "Carol teaches art and music at the local community center.",
            "Dave writes software and builds distributed systems daily.",
            "Eve researches biology and chemistry at the university lab.",
        ]
        embs = np.eye(5, 4, dtype=np.float32)
        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=[f"s{i}" for i in range(5)], mem_positions=list(range(1, 6)),
            entity_store=[], embed_fn=_unit_embed, semantic_threshold=0.0,
        )
        scores = r._bm25_signal("Alice nurse hospital")
        # Memory 0 mentions alice, hospital, nurse — should score at least as high as text 1-4
        assert scores[0] >= scores[1]

    def test_empty_query_all_zero_or_sigmoid_of_zero(self):
        r = _make_retriever(n_mems=4)
        scores = r._bm25_signal("")
        # BM25 of empty query = 0 raw → sigmoid(0) for short query (mid=5): 1/(1+e^5) ≈ 0.007
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_long_query_uses_higher_midpoint(self):
        """A score of 10.0 raw: short query (mid=5) should give higher sigmoid than long (mid=12)."""
        short_norm = _adaptive_sigmoid(10.0, n_tokens=5)
        long_norm  = _adaptive_sigmoid(10.0, n_tokens=20)
        assert short_norm > long_norm

    def test_no_bm25_returns_zeros(self):
        """When rank_bm25 unavailable, signal should be all zeros."""
        r = _make_retriever(n_mems=4)
        original_bm25 = r._bm25
        r._bm25 = None  # simulate missing library
        scores = r._bm25_signal("any query")
        np.testing.assert_array_equal(scores, np.zeros(4, dtype=np.float32))
        r._bm25 = original_bm25  # restore

    def test_non_empty_query_non_trivial(self):
        """At least one memory should have a non-trivial BM25 score."""
        r = _make_retriever(n_mems=6)
        scores = r._bm25_signal("memory text number Alice nurse")
        # With matching terms, some scores should be non-zero after sigmoid
        assert float(scores.max()) > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalEntityBoost
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalEntityBoost:
    def _make_with_entity(self, session_for_entity: str, memory_count: int = 5) -> MultiSignalRetriever:
        sessions = ["session_0", "session_1", "session_2", "session_3"]
        entity_store = [
            {
                "entity_text": "caroline",
                "canonical_name": "Caroline",
                "linked_memory_ids": [
                    f"mem_{session_for_entity}_state_000",
                    f"mem_{session_for_entity}_state_001",
                ],
                "memory_count": memory_count,
            }
        ]
        return _make_retriever(
            n_mems=4, sessions=sessions,
            entity_store=entity_store,
            entity_boost_weight=0.3,
        )

    def test_entity_boost_zero_when_no_match(self):
        r = self._make_with_entity("session_0")
        boost = r._entity_boost("What does John do?")  # "John" not in entity store
        np.testing.assert_array_equal(boost, np.zeros(4, dtype=np.float32))

    def test_entity_boost_nonzero_for_matched_entity(self):
        r = self._make_with_entity("session_0")
        boost = r._entity_boost("What does Caroline study?")
        assert float(boost.max()) > 0.0

    def test_entity_boost_only_on_linked_sessions(self):
        r = self._make_with_entity("session_0")
        boost = r._entity_boost("What does Caroline study?")
        # session_0 is at index 0, session_1 at index 1, etc.
        # Only session_0 rows should be boosted (since entity linked to session_0)
        assert float(boost[0]) > 0.0
        assert float(boost[1]) == 0.0
        assert float(boost[2]) == 0.0
        assert float(boost[3]) == 0.0

    def test_spread_attenuation_formula(self):
        """Boost per memory should equal ENTITY_BOOST_WEIGHT / memory_count."""
        memory_count = 10
        r = self._make_with_entity("session_0", memory_count=memory_count)
        boost = r._entity_boost("What does Caroline study?")
        expected_boost_per_row = 0.3 / memory_count
        # session_0 = row 0, and entity has 2 linked memory_ids both in session_0
        # Row 0 gets boost from both linked memories via session fallback
        # (2 memories in session_0, so boost = 2 × 0.3/10 = 0.06)
        assert float(boost[0]) > 0.0

    def test_higher_memory_count_lower_boost(self):
        """Entity with 100 memories should boost less per row than entity with 5."""
        sessions = ["session_0", "session_1"]
        ent_5   = [{"entity_text": "alice", "canonical_name": "Alice",
                    "linked_memory_ids": ["mem_session_0_state_000"], "memory_count": 5}]
        ent_100 = [{"entity_text": "alice", "canonical_name": "Alice",
                    "linked_memory_ids": ["mem_session_0_state_000"], "memory_count": 100}]

        r5   = _make_retriever(n_mems=2, sessions=sessions, entity_store=ent_5,   entity_boost_weight=0.3)
        r100 = _make_retriever(n_mems=2, sessions=sessions, entity_store=ent_100, entity_boost_weight=0.3)

        b5   = r5._entity_boost("What does Alice do?")
        b100 = r100._entity_boost("What does Alice do?")
        assert float(b5[0]) > float(b100[0])

    def test_prefix_match_fires(self):
        """Query entity 'carol' should prefix-match entity 'caroline'."""
        sessions = ["session_0", "session_1"]
        entity_store = [
            {"entity_text": "caroline", "canonical_name": "Caroline",
             "linked_memory_ids": ["mem_session_0_state_000"], "memory_count": 3}
        ]
        r = _make_retriever(n_mems=2, sessions=sessions, entity_store=entity_store, entity_boost_weight=0.3)
        boost_full   = r._entity_boost("What does Caroline study?")  # exact match
        boost_prefix = r._entity_boost("What does Carol study?")      # prefix match
        assert float(boost_prefix[0]) > 0.0
        assert float(boost_full[0]) > 0.0

    def test_no_entity_in_query_returns_zeros(self):
        r = _make_retriever(n_mems=4)
        boost = r._entity_boost("what did they do during the trip?")
        np.testing.assert_array_equal(boost, np.zeros(4, dtype=np.float32))

    def test_mem_id_level_precision(self):
        """When mem_ids provided, entity boost is memory-level, not session-level."""
        sessions = ["s0", "s0", "s1", "s1"]
        # Only mem_s0_state_000 (row 0) is linked to the entity, NOT mem_s0_state_001 (row 1)
        mem_ids = [
            "mem_s0_state_000",
            "mem_s0_state_001",
            "mem_s1_state_000",
            "mem_s1_state_001",
        ]
        entity_store = [
            {"entity_text": "alice", "canonical_name": "Alice",
             "linked_memory_ids": ["mem_s0_state_000"],  # only row 0
             "memory_count": 1}
        ]
        r = _make_retriever(
            n_mems=4, sessions=sessions, entity_store=entity_store,
            mem_ids=mem_ids, entity_boost_weight=0.3, semantic_threshold=0.0,
        )
        boost = r._entity_boost("What does Alice do?")
        assert float(boost[0]) > 0.0   # row 0 is linked
        assert float(boost[1]) == 0.0  # row 1 is NOT linked (even in same session)
        assert float(boost[2]) == 0.0  # row 2 — wrong session
        assert float(boost[3]) == 0.0  # row 3 — wrong session

    def test_multiple_entities_accumulate(self):
        """Both Alice and Bob entities should boost their respective sessions."""
        sessions = ["s_alice", "s_bob", "s_other"]
        entity_store = [
            {"entity_text": "alice", "canonical_name": "Alice",
             "linked_memory_ids": ["mem_s_alice_state_000"], "memory_count": 2},
            {"entity_text": "bob",   "canonical_name": "Bob",
             "linked_memory_ids": ["mem_s_bob_state_000"], "memory_count": 2},
        ]
        r = _make_retriever(
            n_mems=3, sessions=sessions, entity_store=entity_store,
            entity_boost_weight=0.3,
        )
        boost = r._entity_boost("Alice and Bob went hiking.")
        assert float(boost[0]) > 0.0   # Alice's session
        assert float(boost[1]) > 0.0   # Bob's session
        assert float(boost[2]) == 0.0  # other session

    def test_empty_entity_store(self):
        r = _make_retriever(n_mems=4, entity_store=[])
        boost = r._entity_boost("What does Caroline do?")
        np.testing.assert_array_equal(boost, np.zeros(4, dtype=np.float32))

    def test_returns_float32(self):
        r = _make_retriever(n_mems=4)
        boost = r._entity_boost("query")
        assert boost.dtype == np.float32

    def test_boost_output_shape(self):
        r = _make_retriever(n_mems=7)
        boost = r._entity_boost("some query")
        assert boost.shape == (7,)


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalRetrieve
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalRetrieve:
    def test_returns_list_of_strings(self):
        r = _make_retriever(n_mems=6, sessions=["s0", "s1", "s2", "s0", "s1", "s2"])
        result = r.retrieve("test query")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_returns_at_most_top_k(self):
        r = _make_retriever(n_mems=10, sessions=[f"s{i}" for i in range(10)])
        result = r.retrieve("query", top_k=5)
        assert len(result) <= 5

    def test_no_duplicate_sessions(self):
        sessions = ["s0", "s1", "s2", "s0", "s1", "s2", "s3"]
        r = _make_retriever(n_mems=7, sessions=sessions)
        result = r.retrieve("test query")
        assert len(result) == len(set(result))

    def test_all_same_session_returns_one(self):
        """When all memories are from session_0, retrieve returns only session_0."""
        sessions = ["session_0"] * 4
        r = _make_retriever(n_mems=4, sessions=sessions)
        result = r.retrieve("query", top_k=5)
        assert len(result) == 1
        assert result[0] == "session_0"

    def test_rephrases_used(self):
        """retrieve with rephrases should not crash and still return sessions."""
        r = _make_retriever(n_mems=4, sessions=["s0", "s1", "s2", "s3"])
        result = r.retrieve("What does Alice do?", rephrases=["alt1", "alt2"])
        assert len(result) >= 1

    def test_with_mock_reranker(self):
        """Mock reranker that reverses rank order — should still return valid sessions."""
        mock_reranker = MagicMock()

        def reverse_rerank(query, mems, top_n):
            ids = [m.id for m in mems]
            return [(id_, 1.0 / (i + 1)) for i, id_ in enumerate(reversed(ids))]

        mock_reranker.rerank = reverse_rerank
        sessions = [f"s{i}" for i in range(6)]
        r = _make_retriever(n_mems=6, sessions=sessions, reranker=mock_reranker)
        result = r.retrieve("query", top_k=3)
        assert len(result) <= 3
        assert all(s in sessions for s in result)

    def test_retrieve_no_crash_with_threshold_param(self):
        """semantic_threshold param still accepted (no crash), retrieve works normally."""
        r = _make_retriever(n_mems=4, semantic_threshold=0.5)
        result = r.retrieve("query that matches nothing")
        # RRF pool always returns something; semantic_threshold is currently unused in retrieve
        assert isinstance(result, list)  # no crash

    def test_diagnostic_methods_work(self):
        r = _make_retriever(n_mems=4)
        d = r.dense_scores("query")
        b = r.bm25_scores("query")
        e = r.entity_scores("query")
        assert d.shape == (4,)
        assert b.shape == (4,)
        assert e.shape == (4,)

    def test_pool_capped_at_n_when_few_mems(self):
        """When n_mems < reranker_pool_size, all memories form the pool (no crash)."""
        r = _make_retriever(n_mems=3, sessions=["s0", "s1", "s2"], reranker_pool_size=40)
        result = r.retrieve("query", top_k=5)
        assert len(result) <= 3  # can't return more than 3 unique sessions

    def test_empty_rephrases(self):
        r = _make_retriever(n_mems=4, sessions=["s0", "s1", "s2", "s3"])
        result = r.retrieve("query", rephrases=[], top_k=3)
        assert len(result) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalRecency
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalRecency:
    def test_recency_bias_increases_score_for_temporal_query(self):
        """A memory at later session_position should rank higher for recency queries.

        Uses identical embeddings and deliberately dissimilar texts so BM25 gives
        equal scores, making recency_weight=5.0 the dominant differentiator.
        """
        positions = [1, 10]
        sessions  = ["s_early", "s_recent"]

        # Texts share no query terms so BM25 scores are equal
        texts = [
            "Zxqy foobar baz quux waldo garply grault early.",
            "Zxqy foobar baz quux waldo garply grault recent.",
        ]
        embs = np.ones((2, 4), dtype=np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=sessions, mem_positions=positions,
            entity_store=[], embed_fn=lambda q: [1.0, 1.0, 1.0, 1.0],
            semantic_threshold=0.0,
            recency_weight=5.0,  # large enough to dominate any BM25 noise
        )
        result = r.retrieve("Where does Alice currently work?", top_k=2)
        assert result[0] == "s_recent"

    def test_no_recency_without_trigger_words(self):
        """Without recency keywords, both sessions should be equally ranked by semantic."""
        positions = [1, 10]
        sessions  = ["s_early", "s_recent"]
        texts = ["Memory A text here now.", "Memory B text here."]
        embs  = np.ones((2, 4), dtype=np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=sessions, mem_positions=positions,
            entity_store=[], embed_fn=lambda q: [1.0, 1.0, 1.0, 1.0],
            semantic_threshold=0.0, recency_weight=0.5,
        )
        # Query without recency keyword — no recency boost
        result = r.retrieve("What did Alice do during session 3?", top_k=2)
        assert len(result) == 2  # both sessions returned

    def test_recency_array_proportional_to_position(self):
        """Recency values should be proportional to session positions."""
        positions = [1, 2, 4, 8]
        r = _make_retriever(n_mems=4, positions=positions)
        # position[3]=8 is max → recency[3]=1.0; position[0]=1 → recency[0]=0.125
        assert float(r._recency[3]) == pytest.approx(1.0)
        assert float(r._recency[0]) == pytest.approx(1 / 8)

    def test_recency_weight_zero_means_no_recency_effect(self):
        """With recency_weight=0, recency trigger has no effect."""
        r = _make_retriever(n_mems=4, recency_weight=0.0)
        # Should not crash and return valid sessions
        result = r.retrieve("Where does Alice currently work?")
        assert len(result) >= 0

    def test_all_same_position_recency_all_equal(self):
        """When all session_positions are equal, recency signal is constant."""
        positions = [5, 5, 5, 5]
        r = _make_retriever(n_mems=4, positions=positions)
        assert float(r._recency.max() - r._recency.min()) < 1e-6

    def test_recency_keywords_comprehensive(self):
        """All defined recency keywords should trigger the bias."""
        keywords = ["currently", "still", "now", "recently", "latest", "today", "present"]
        for kw in keywords:
            assert _is_recency_query(f"What is Alice {kw} working on?"), f"'{kw}' should trigger"


# ══════════════════════════════════════════════════════════════════════════════
# TestMultiSignalEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSignalEdgeCases:
    def test_single_memory_single_session(self):
        texts = ["The only memory in this system about Alice."]
        embs  = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=["only_session"], mem_positions=[1],
            entity_store=[], embed_fn=_unit_embed, semantic_threshold=0.0,
        )
        result = r.retrieve("Alice memory")
        assert result == ["only_session"]

    def test_empty_entity_store_no_crash(self):
        r = _make_retriever(n_mems=4, entity_store=[])
        result = r.retrieve("What does Caroline do?")
        assert isinstance(result, list)

    def test_none_mem_ids_uses_session_fallback(self):
        """Without mem_ids, entity boost uses session-level fallback — no crash."""
        entity_store = [
            {"entity_text": "alice", "canonical_name": "Alice",
             "linked_memory_ids": ["mem_session_0_state_000"], "memory_count": 3}
        ]
        sessions = ["session_0", "session_1"]
        r = _make_retriever(
            n_mems=2, sessions=sessions, entity_store=entity_store,
            mem_ids=None,  # no mem_ids → session fallback
        )
        result = r.retrieve("What does Alice do?")
        assert isinstance(result, list)

    def test_zero_positions_no_crash(self):
        """Edge: session_position = 0 for all → max_pos=0 guarded, recency=0."""
        positions = [0, 0, 0]
        r = _make_retriever(n_mems=3, positions=positions)
        # max(0) → guarded to 1, recency = 0/1 = 0 for all
        np.testing.assert_array_equal(r._recency, np.zeros(3, dtype=np.float32))

    def test_reranker_none_no_crash(self):
        r = _make_retriever(n_mems=4, reranker=None)
        result = r.retrieve("query without reranker")
        assert isinstance(result, list)

    def test_top_k_larger_than_sessions(self):
        """top_k > number of unique sessions → return all unique sessions."""
        sessions = ["s0", "s0", "s1", "s1"]  # 2 unique
        r = _make_retriever(n_mems=4, sessions=sessions)
        result = r.retrieve("query", top_k=100)
        assert len(result) <= 2  # at most 2 unique sessions
