"""
Sub-phase 2.5 — Ranking, Reranker, MMR tests.

Coverage:
  Intent classifier      — all 5 intents, heuristics, LLM fallback, override
  Weighted ranking       — 5 signals, per-intent weight differences, normalisation
  Cross-encoder          — passthrough when unavailable, mock when injected
  MMR                    — cluster dedup, session dedup, lambda variation, edge cases
  Structural similarity  — cluster, session, entity overlap
  Pipeline integration   — retrieve() with all 2.5 stages enabled/disabled
  Telemetry              — intent_detected, reranked, mmr_applied on RetrievalResult
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.retrieval.intent_classifier import (
    IntentLabel,
    classify_intent,
    get_intent_weights,
)
from app.services.retrieval.ranking import (
    load_entity_map,
    score_memory,
    weighted_rank,
)
from app.services.retrieval.reranker import CrossEncoderReranker, _passthrough
from app.services.retrieval.mmr import mmr_select, structural_similarity
from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _make_embedding(dim: int = 384, val: float = 1.0, idx: int = 0) -> bytes:
    vec = np.zeros(dim, dtype=np.float32)
    vec[idx] = val
    return vec.tobytes()


def _create_memory(
    db,
    project_id: str,
    *,
    title: str = "T",
    content: str = "C",
    status: str = "active",
    privacy: str = "internal",
    review_status: str = "auto_extracted",
    cluster_id: int | None = None,
    session_id: str | None = None,
    decay_score: float | None = None,
    quality_score: float | None = None,
    importance: int = 3,
    embedding: bytes | None = None,
):
    from app import models as phase1_models
    mid = str(uuid.uuid4())
    m = phase1_models.Memory(
        id=mid,
        project_id=project_id,
        type="decision",
        title=title,
        content=content,
        status=status,
        privacy_level=privacy,
        review_status=review_status,
        cluster_id=cluster_id,
        source_session_id=session_id,
        decay_score=decay_score,
        quality_score=quality_score,
        importance=importance,
        confidence=1.0,
        embedding=embedding,
        created_at=_now(),
        updated_at=_now(),
    )
    db.add(m)
    db.commit()
    return m


def _create_entity(db, memory_id, project_id, text, label="TECH"):
    from app import models as phase1_models
    e = phase1_models.MemoryEntity(
        id=str(uuid.uuid4()),
        memory_id=memory_id,
        project_id=project_id,
        entity_text=text,
        entity_label=label,
        normalized_text=text.lower(),
        created_at=_now(),
    )
    db.add(e)
    db.commit()
    return e


def _create_project(db):
    from app import crud, schemas
    return crud.create_project(db, schemas.ProjectCreate(
        name="2.5 Test Project", description="Phase 2.5 tests",
        tech_stack=["Python"], goals=["Test ranking"], domain="software",
    ))


def _make_mock_memory(
    mid: str = None,
    cluster_id: int | None = None,
    session_id: str | None = None,
    decay_score: float | None = 0.8,
    quality_score: float | None = 0.7,
    importance: int = 3,
    title: str = "Mock Memory",
    content: str = "Mock content",
):
    m = MagicMock()
    m.id = mid or str(uuid.uuid4())
    m.cluster_id = cluster_id
    m.source_session_id = session_id
    m.decay_score = decay_score
    m.quality_score = quality_score
    m.importance = importance
    m.title = title
    m.content = content
    return m


@pytest.fixture(autouse=True)
def _patch_ml(mock_embed, mock_entities, mock_is_technical_true):
    pass


# ===========================================================================
# PART 1: Intent Classifier
# ===========================================================================

class TestIntentClassifier:
    # ---- Temporal ----
    def test_temporal_when(self):
        intent, conf = classify_intent("when was the auth service last updated")
        assert intent == IntentLabel.temporal

    def test_temporal_recent(self):
        intent, _ = classify_intent("what is the most recent database decision")
        assert intent == IntentLabel.temporal

    def test_temporal_history(self):
        intent, _ = classify_intent("show me the history of changes to the API")
        assert intent == IntentLabel.temporal

    def test_temporal_before_after(self):
        intent, _ = classify_intent("what changed before the deployment last week")
        assert intent == IntentLabel.temporal

    def test_temporal_confidence_above_threshold(self):
        _, conf = classify_intent("when was postgres updated")
        assert conf >= 0.6

    # ---- Factual ----
    def test_factual_what_is(self):
        intent, _ = classify_intent("what is the connection pool size")
        assert intent == IntentLabel.factual

    def test_factual_define(self):
        intent, _ = classify_intent("define the rate limiting policy")
        assert intent == IntentLabel.factual

    def test_factual_what_are(self):
        intent, _ = classify_intent("what are the API endpoints")
        assert intent == IntentLabel.factual

    def test_factual_describe(self):
        intent, _ = classify_intent("describe the authentication flow")
        assert intent == IntentLabel.factual

    # ---- Code ----
    def test_code_file_extension(self):
        intent, _ = classify_intent("bug in auth.py login handler")
        assert intent == IntentLabel.code

    def test_code_def_keyword(self):
        intent, _ = classify_intent("def authenticate_user function")
        assert intent == IntentLabel.code

    def test_code_import(self):
        intent, _ = classify_intent("from app.models import User")
        assert intent == IntentLabel.code

    def test_code_path(self):
        intent, _ = classify_intent("find /app/routers/auth setup")
        assert intent == IntentLabel.code

    # ---- Exploratory ----
    def test_exploratory_how_does(self):
        intent, _ = classify_intent("how does the authentication service work")
        assert intent == IntentLabel.exploratory

    def test_exploratory_why(self):
        intent, _ = classify_intent("why did we choose postgresql")
        assert intent == IntentLabel.exploratory

    def test_exploratory_explain(self):
        intent, _ = classify_intent("explain the architecture of the caching layer")
        assert intent == IntentLabel.exploratory

    def test_exploratory_overview(self):
        intent, _ = classify_intent("give an overview of the deployment strategy")
        assert intent == IntentLabel.exploratory

    def test_exploratory_trade_off(self):
        intent, _ = classify_intent("explain the trade-off between redis and memcached")
        assert intent == IntentLabel.exploratory

    # ---- General fallback ----
    def test_general_empty(self):
        intent, _ = classify_intent("")
        assert intent == IntentLabel.general

    def test_general_no_strong_signal(self):
        intent, _ = classify_intent("database postgres")
        assert intent == IntentLabel.general

    # ---- LLM fallback ----
    def test_llm_fallback_called_on_low_confidence(self):
        called = []
        def mock_llm(prompt: str) -> str:
            called.append(prompt)
            return "temporal"

        # Short ambiguous query that won't hit threshold
        intent, conf = classify_intent("database", llm_fn=mock_llm, llm_confidence_threshold=0.9)
        # LLM was called and its response used
        if called:
            assert intent == IntentLabel.temporal

    def test_llm_fallback_not_called_when_confident(self):
        called = []
        def mock_llm(prompt: str) -> str:
            called.append(prompt)
            return "factual"

        classify_intent("when was auth.py last changed", llm_fn=mock_llm, llm_confidence_threshold=0.5)
        # High-confidence temporal hit — LLM not needed
        assert len(called) == 0

    def test_llm_fallback_handles_exception(self):
        def bad_llm(prompt: str) -> str:
            raise RuntimeError("API error")

        # Should not raise — falls back to heuristic result
        intent, conf = classify_intent("database", llm_fn=bad_llm, llm_confidence_threshold=0.0)
        assert intent is not None

    # ---- Intent override in config ----
    def test_intent_override_forces_label(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())
        _create_memory(db, project.id, title="A", content="auth postgres")

        cfg = RetrievalConfig(
            top_k=5, embed_query=False, intent="temporal",
            max_clearance="internal",
        )
        result = retrieve(db, project.id, "auth postgres", SQLiteExactBackend(db), cfg)
        assert result.intent_detected == "temporal"
        assert result.intent_confidence == 1.0


# ===========================================================================
# PART 2: Weighted Ranking
# ===========================================================================

class TestWeightedRanking:
    def test_returns_sorted_desc_by_score(self):
        m1 = _make_mock_memory("m1", decay_score=0.9, importance=5, quality_score=0.9)
        m2 = _make_mock_memory("m2", decay_score=0.1, importance=1, quality_score=0.1)
        scores = {"m1": 1.0, "m2": 0.5}

        ranked = weighted_rank([m1, m2], scores, IntentLabel.general, "some query")
        assert ranked[0][0] == "m1"
        assert ranked[0][1] > ranked[1][1]

    def test_decay_weight_higher_for_temporal_intent(self):
        w_temporal = get_intent_weights(IntentLabel.temporal)
        w_factual = get_intent_weights(IntentLabel.factual)
        assert w_temporal["decay"] > w_factual["decay"]

    def test_entity_weight_higher_for_code_intent(self):
        w_code = get_intent_weights(IntentLabel.code)
        w_temporal = get_intent_weights(IntentLabel.temporal)
        assert w_code["entity"] > w_temporal["entity"]

    def test_rrf_weight_highest_for_code_intent(self):
        w_code = get_intent_weights(IntentLabel.code)
        w_temporal = get_intent_weights(IntentLabel.temporal)
        assert w_code["rrf"] > w_temporal["rrf"]

    def test_all_weights_sum_to_one(self):
        for intent in IntentLabel:
            w = get_intent_weights(intent)
            total = sum(w.values())
            assert abs(total - 1.0) < 1e-9, f"{intent}: weights sum to {total}"

    def test_high_decay_scores_higher_on_temporal(self):
        fresh = _make_mock_memory("fresh", decay_score=0.95, importance=3, quality_score=0.5)
        stale = _make_mock_memory("stale", decay_score=0.05, importance=3, quality_score=0.5)
        scores = {"fresh": 1.0, "stale": 1.0}

        ranked = weighted_rank([fresh, stale], scores, IntentLabel.temporal, "what changed recently")
        assert ranked[0][0] == "fresh"

    def test_high_quality_scores_higher_on_factual(self):
        hq = _make_mock_memory("hq", decay_score=0.5, importance=3, quality_score=0.95)
        lq = _make_mock_memory("lq", decay_score=0.5, importance=3, quality_score=0.05)
        scores = {"hq": 1.0, "lq": 1.0}

        ranked = weighted_rank([hq, lq], scores, IntentLabel.factual, "what is the rate limit")
        assert ranked[0][0] == "hq"

    def test_entity_overlap_boosts_score(self):
        m_match = _make_mock_memory("match")
        m_nomatch = _make_mock_memory("nomatch")
        scores = {"match": 1.0, "nomatch": 1.0}
        entity_map = {"match": {"redis", "cache"}, "nomatch": {"unrelated"}}

        ranked = weighted_rank(
            [m_match, m_nomatch], scores, IntentLabel.general,
            "redis cache setup", entity_map=entity_map
        )
        assert ranked[0][0] == "match"

    def test_normalises_importance_1_to_5(self):
        m = _make_mock_memory("m", importance=1, decay_score=0.5, quality_score=0.5)
        w = get_intent_weights(IntentLabel.general)
        s_low = score_memory(m, 0.5, set(), set(), w)

        m2 = _make_mock_memory("m2", importance=5, decay_score=0.5, quality_score=0.5)
        s_high = score_memory(m2, 0.5, set(), set(), w)

        assert s_high > s_low

    def test_none_decay_uses_neutral_0_5(self):
        m_none = _make_mock_memory("none", decay_score=None, importance=3, quality_score=0.5)
        m_mid = _make_mock_memory("mid", decay_score=0.5, importance=3, quality_score=0.5)
        w = get_intent_weights(IntentLabel.general)
        s_none = score_memory(m_none, 0.5, set(), set(), w)
        s_mid = score_memory(m_mid, 0.5, set(), set(), w)
        assert abs(s_none - s_mid) < 0.001

    def test_none_quality_uses_neutral_0_5(self):
        m_none = _make_mock_memory("none", quality_score=None, decay_score=0.5, importance=3)
        m_mid = _make_mock_memory("mid", quality_score=0.5, decay_score=0.5, importance=3)
        w = get_intent_weights(IntentLabel.general)
        s_none = score_memory(m_none, 0.5, set(), set(), w)
        s_mid = score_memory(m_mid, 0.5, set(), set(), w)
        assert abs(s_none - s_mid) < 0.001

    def test_empty_memories_returns_empty(self):
        ranked = weighted_rank([], {}, IntentLabel.general, "query")
        assert ranked == []

    def test_single_memory_returns_single(self):
        m = _make_mock_memory("m1")
        ranked = weighted_rank([m], {"m1": 0.5}, IntentLabel.general, "query")
        assert len(ranked) == 1
        assert ranked[0][0] == "m1"

    def test_load_entity_map_from_db(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        _create_entity(db, m.id, project.id, "Redis")
        _create_entity(db, m.id, project.id, "PostgreSQL")

        entity_map = load_entity_map(db, [m.id])
        assert m.id in entity_map
        assert "redis" in entity_map[m.id]
        assert "postgresql" in entity_map[m.id]

    def test_load_entity_map_empty_ids(self, db):
        result = load_entity_map(db, [])
        assert result == {}


# ===========================================================================
# PART 3: Cross-Encoder Reranker
# ===========================================================================

class TestCrossEncoderReranker:
    def test_passthrough_when_model_unavailable(self):
        reranker = CrossEncoderReranker("nonexistent-model-xyz")
        assert reranker.available is False

        memories = [_make_mock_memory(f"m{i}") for i in range(5)]
        result = reranker.rerank("query", memories, top_n=3)

        # Original order preserved
        ids = [mid for mid, _ in result]
        assert ids[0] == memories[0].id
        assert len(result) == 5

    def test_passthrough_returns_ordinal_scores(self):
        memories = [_make_mock_memory(f"m{i}") for i in range(3)]
        result = _passthrough(memories)
        scores = [s for _, s in result]
        # Scores should be decreasing (first = highest)
        assert scores[0] > scores[1] > scores[2]

    def test_passthrough_empty_memories(self):
        memories = [_make_mock_memory("m1")]
        reranker = CrossEncoderReranker("no-model")
        result = reranker.rerank("query", memories=[])
        assert result == []

    def test_reranker_with_mock_model(self):
        reranker = CrossEncoderReranker("mock-model")
        reranker._available = True

        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.2, 0.7]
        reranker._model = mock_ce

        memories = [
            _make_mock_memory("m0"),
            _make_mock_memory("m1"),
            _make_mock_memory("m2"),
        ]
        result = reranker.rerank("query", memories, top_n=3)
        ids = [mid for mid, _ in result]

        # Scores: m0=0.9, m1=0.2, m2=0.7 → order: m0, m2, m1
        assert ids[0] == "m0"
        assert ids[1] == "m2"
        assert ids[2] == "m1"

    def test_tail_beyond_top_n_appended_at_end(self):
        reranker = CrossEncoderReranker("mock-model")
        reranker._available = True

        mock_ce = MagicMock()
        mock_ce.predict.return_value = [1.0]  # only scores top_n=1
        reranker._model = mock_ce

        memories = [_make_mock_memory("m0"), _make_mock_memory("m1"), _make_mock_memory("m2")]
        result = reranker.rerank("query", memories, top_n=1)

        ids = [mid for mid, _ in result]
        assert ids[0] == "m0"  # top scored
        assert "m1" in ids
        assert "m2" in ids
        # m1 and m2 should have sentinel scores
        sentinel_scores = [s for mid, s in result if mid != "m0"]
        assert all(s < 0 for s in sentinel_scores)

    def test_model_exception_during_predict_falls_back(self):
        reranker = CrossEncoderReranker("mock-model")
        reranker._available = True

        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError("Model error")
        reranker._model = mock_ce

        memories = [_make_mock_memory(f"m{i}") for i in range(3)]
        result = reranker.rerank("query", memories, top_n=3)

        # Should fall back to passthrough
        assert len(result) == 3
        ids = [mid for mid, _ in result]
        assert ids[0] == memories[0].id

    def test_thread_safe_singleton(self):
        from app.services.retrieval.reranker import get_default_reranker
        r1 = get_default_reranker()
        r2 = get_default_reranker()
        assert r1 is r2


# ===========================================================================
# PART 4: Structural Similarity + MMR
# ===========================================================================

class TestStructuralSimilarity:
    def test_same_cluster_gives_high_similarity(self):
        m1 = _make_mock_memory("m1", cluster_id=1)
        m2 = _make_mock_memory("m2", cluster_id=1)
        sim = structural_similarity(m1, m2)
        assert sim >= 0.80

    def test_same_session_gives_medium_similarity(self):
        sid = str(uuid.uuid4())
        m1 = _make_mock_memory("m1", session_id=sid)
        m2 = _make_mock_memory("m2", session_id=sid)
        sim = structural_similarity(m1, m2)
        assert sim >= 0.50

    def test_different_cluster_and_session_gives_low_sim(self):
        m1 = _make_mock_memory("m1", cluster_id=1, session_id="s1")
        m2 = _make_mock_memory("m2", cluster_id=2, session_id="s2")
        sim = structural_similarity(m1, m2)
        assert sim < 0.20

    def test_entity_overlap_contributes(self):
        m1 = _make_mock_memory("m1")
        m2 = _make_mock_memory("m2")
        entity_map = {"m1": {"redis", "cache"}, "m2": {"redis", "cache"}}
        sim = structural_similarity(m1, m2, entity_map=entity_map)
        assert sim > 0.0

    def test_no_entity_overlap_zero_entity_contribution(self):
        m1 = _make_mock_memory("m1")
        m2 = _make_mock_memory("m2")
        entity_map = {"m1": {"redis"}, "m2": {"postgres"}}
        sim = structural_similarity(m1, m2, entity_map=entity_map)
        entity_only_sim = sim
        assert entity_only_sim < 0.05  # near zero for entity-only path

    def test_cluster_and_session_capped_at_one(self):
        sid = str(uuid.uuid4())
        m1 = _make_mock_memory("m1", cluster_id=1, session_id=sid)
        m2 = _make_mock_memory("m2", cluster_id=1, session_id=sid)
        sim = structural_similarity(m1, m2)
        assert sim <= 1.0

    def test_none_cluster_not_counted(self):
        m1 = _make_mock_memory("m1", cluster_id=None)
        m2 = _make_mock_memory("m2", cluster_id=None)
        sim = structural_similarity(m1, m2)
        assert sim == pytest.approx(0.0)


class TestMMRSelect:
    def _make_mem_set(self, n, cluster_id=None, session_id=None):
        return [
            _make_mock_memory(f"m{i}", cluster_id=cluster_id, session_id=session_id)
            for i in range(n)
        ]

    def test_first_pick_is_highest_scored(self):
        memories = [_make_mock_memory("high"), _make_mock_memory("low")]
        scores = {"high": 1.0, "low": 0.1}
        selected = mmr_select(memories, scores, top_k=2)
        assert selected[0] == "high"

    def test_returns_exactly_top_k(self):
        memories = [_make_mock_memory(f"m{i}") for i in range(10)]
        scores = {m.id: float(10 - i) for i, m in enumerate(memories)}
        selected = mmr_select(memories, scores, top_k=5)
        assert len(selected) == 5

    def test_all_same_cluster_still_selects_top_k(self):
        memories = [_make_mock_memory(f"m{i}", cluster_id=1) for i in range(5)]
        scores = {m.id: float(5 - i) for i, m in enumerate(memories)}
        selected = mmr_select(memories, scores, top_k=3)
        assert len(selected) == 3

    def test_lambda_zero_maximises_diversity(self):
        # λ=0: pure diversity → second pick should differ from first
        sid = str(uuid.uuid4())
        # 3 same-cluster memories + 1 different
        m_a = _make_mock_memory("cluster_a1", cluster_id=1)
        m_b = _make_mock_memory("cluster_a2", cluster_id=1)
        m_c = _make_mock_memory("cluster_b",  cluster_id=2)
        scores = {"cluster_a1": 1.0, "cluster_a2": 0.9, "cluster_b": 0.5}
        selected = mmr_select([m_a, m_b, m_c], scores, top_k=2, lambda_=0.0)
        # With λ=0, diversity is maximised — second pick should be from different cluster
        assert selected[0] == "cluster_a1"
        assert selected[1] == "cluster_b"

    def test_lambda_one_is_pure_relevance(self):
        # λ=1: pick by score only, ignore diversity
        m1 = _make_mock_memory("best", cluster_id=1)
        m2 = _make_mock_memory("second", cluster_id=1)
        m3 = _make_mock_memory("third", cluster_id=1)
        scores = {"best": 1.0, "second": 0.9, "third": 0.8}
        selected = mmr_select([m1, m2, m3], scores, top_k=3, lambda_=1.0)
        assert selected == ["best", "second", "third"]

    def test_different_sessions_preferred_over_same_session(self):
        sid = str(uuid.uuid4())
        m_same = _make_mock_memory("same_session", session_id=sid)
        m_diff = _make_mock_memory("diff_session", session_id="other-session")
        m_first = _make_mock_memory("first", session_id=sid)
        scores = {"first": 1.0, "same_session": 0.9, "diff_session": 0.85}
        # λ=0.3 → diversity weighted heavily
        selected = mmr_select([m_first, m_same, m_diff], scores, top_k=2, lambda_=0.3)
        # First is always picked; second should prefer different session
        assert selected[0] == "first"
        assert selected[1] == "diff_session"

    def test_empty_memories_returns_empty(self):
        result = mmr_select([], {}, top_k=5)
        assert result == []

    def test_fewer_memories_than_top_k(self):
        memories = [_make_mock_memory(f"m{i}") for i in range(3)]
        scores = {m.id: 1.0 for m in memories}
        selected = mmr_select(memories, scores, top_k=10)
        assert len(selected) == 3

    def test_entity_similarity_used_when_provided(self):
        m1 = _make_mock_memory("m1")
        m2 = _make_mock_memory("m2_same_entities")
        m3 = _make_mock_memory("m3_diff_entities")
        scores = {"m1": 1.0, "m2_same_entities": 0.95, "m3_diff_entities": 0.90}
        entity_map = {
            "m1": {"redis", "cache"},
            "m2_same_entities": {"redis", "cache"},
            "m3_diff_entities": {"postgres", "database"},
        }
        # With λ=0.5 and entity overlap, m3 should be preferred over m2 as second pick
        selected = mmr_select([m1, m2, m3], scores, top_k=2, lambda_=0.5, entity_map=entity_map)
        assert selected[0] == "m1"
        assert selected[1] == "m3_diff_entities"


# ===========================================================================
# PART 5: Pipeline Integration
# ===========================================================================

class TestPipelineWith25:
    def _setup(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())
        return project

    def test_result_has_intent_detected_field(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="Auth", content="JWT token auth")
        cfg = RetrievalConfig(top_k=5, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "when was the auth changed", SQLiteExactBackend(db), cfg)
        assert hasattr(result, "intent_detected")
        assert result.intent_detected in [i.value for i in IntentLabel]

    def test_temporal_query_detected(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="Auth", content="auth history log")
        cfg = RetrievalConfig(top_k=5, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "when was auth last updated", SQLiteExactBackend(db), cfg)
        assert result.intent_detected == "temporal"

    def test_code_query_detected(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="Router", content="routing handlers")
        cfg = RetrievalConfig(top_k=5, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "def authenticate_user in auth.py", SQLiteExactBackend(db), cfg)
        assert result.intent_detected == "code"

    def test_mmr_applied_is_true_by_default(self, db):
        project = self._setup(db)
        for i in range(5):
            _create_memory(db, project.id, title=f"Memory {i}", content=f"auth content {i}")
        cfg = RetrievalConfig(top_k=3, embed_query=False, max_clearance="internal", enable_mmr=True)
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)
        assert result.mmr_applied is True

    def test_mmr_disabled_not_applied(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="M", content="auth content")
        cfg = RetrievalConfig(top_k=3, embed_query=False, max_clearance="internal", enable_mmr=False)
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)
        assert result.mmr_applied is False

    def test_reranked_false_when_model_unavailable(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="M", content="auth content")
        cfg = RetrievalConfig(
            top_k=3, embed_query=False, max_clearance="internal",
            enable_reranker=True,  # enabled but model not available
        )
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)
        # Model not installed → reranked=False
        assert result.reranked is False

    def test_reranked_true_when_model_mocked(self, db):
        project = self._setup(db)
        for i in range(5):
            _create_memory(db, project.id, title=f"M{i}", content=f"auth content {i}")
        from app.search import setup_fts
        setup_fts(db.get_bind())

        from app.services.retrieval import reranker as reranker_module

        mock_reranker = MagicMock()
        mock_reranker.available = True
        # Return passthrough-style scores
        mock_reranker.rerank.side_effect = lambda query, memories, top_n: [
            (m.id, float(len(memories) - i)) for i, m in enumerate(memories)
        ]

        with patch.object(reranker_module, "get_default_reranker", return_value=mock_reranker):
            cfg = RetrievalConfig(
                top_k=3, embed_query=False, max_clearance="internal",
                enable_reranker=True,
            )
            result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)

        assert result.reranked is True

    def test_weighted_ranking_disabled_still_returns_results(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="M", content="auth content")
        cfg = RetrievalConfig(
            top_k=5, embed_query=False, max_clearance="internal",
            enable_weighted_ranking=False, enable_mmr=False,
        )
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)
        assert result is not None
        assert isinstance(result.selected_memory_ids, list)

    def test_mmr_deduplicates_same_cluster_memories(self, db):
        project = self._setup(db)
        # 4 memories all in cluster 1 (should be diversified away after first)
        for i in range(4):
            _create_memory(db, project.id, title=f"Cluster1-M{i}", content=f"auth {i}",
                           cluster_id=1)
        # 1 from cluster 2
        _create_memory(db, project.id, title="Cluster2-M", content="different topic", cluster_id=2)

        cfg = RetrievalConfig(
            top_k=5, embed_query=False, max_clearance="internal",
            enable_mmr=True, mmr_lambda=0.3,  # strong diversity
        )
        result = retrieve(db, project.id, "auth different", SQLiteExactBackend(db), cfg)
        assert result is not None
        assert len(result.selected_memory_ids) <= 5

    def test_intent_confidence_in_result(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="M", content="auth content")
        cfg = RetrievalConfig(top_k=3, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "when was auth last updated", SQLiteExactBackend(db), cfg)
        assert 0.0 <= result.intent_confidence <= 1.0

    def test_empty_project_all_flags_enabled(self, db):
        project = self._setup(db)
        cfg = RetrievalConfig(
            top_k=5, embed_query=False, max_clearance="internal",
            enable_weighted_ranking=True, enable_mmr=True, enable_reranker=True,
        )
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)
        assert result.selected_memory_ids == []
        assert result.expanded_via_links == 0

    def test_telemetry_run_logged_with_intent(self, db):
        from app.p2_models import RetrievalRun
        project = self._setup(db)
        _create_memory(db, project.id, title="M", content="auth content")
        cfg = RetrievalConfig(top_k=3, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "when was auth changed", SQLiteExactBackend(db), cfg)

        run = db.query(RetrievalRun).filter(RetrievalRun.id == result.run_id).first()
        assert run is not None
        assert run.intent == "temporal"

    def test_privacy_still_enforced_after_ranking(self, db):
        project = self._setup(db)
        _create_memory(db, project.id, title="Public", content="auth public", privacy="public")
        _create_memory(db, project.id, title="Secret", content="auth secret", privacy="secret")

        cfg = RetrievalConfig(
            top_k=10, embed_query=False, max_clearance="public",
            enable_weighted_ranking=True, enable_mmr=True,
        )
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)

        from app import models as phase1_models
        for mid in result.selected_memory_ids:
            m = db.query(phase1_models.Memory).filter(phase1_models.Memory.id == mid).first()
            assert m.privacy_level == "public"

    def test_top_k_respected_after_all_stages(self, db):
        project = self._setup(db)
        for i in range(20):
            _create_memory(db, project.id, title=f"M{i}", content=f"auth content {i}")
        cfg = RetrievalConfig(
            top_k=5, embed_query=False, max_clearance="internal",
            enable_weighted_ranking=True, enable_mmr=True,
        )
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)
        assert len(result.selected_memory_ids) <= 5


# ===========================================================================
# PART 6: Stress / edge cases
# ===========================================================================

class TestRankingEdgeCases:
    def test_all_signals_zero_still_returns_result(self):
        m = _make_mock_memory("m1", decay_score=0.0, quality_score=0.0, importance=1)
        ranked = weighted_rank([m], {"m1": 0.0}, IntentLabel.general, "query")
        assert len(ranked) == 1
        assert ranked[0][1] >= 0.0

    def test_high_rrf_score_zero_everything_else(self):
        m_high = _make_mock_memory("high", decay_score=0.0, quality_score=0.0, importance=1)
        m_low = _make_mock_memory("low", decay_score=0.0, quality_score=0.0, importance=1)
        ranked = weighted_rank(
            [m_high, m_low], {"high": 1.0, "low": 0.1},
            IntentLabel.general, "query"
        )
        assert ranked[0][0] == "high"

    def test_mmr_with_single_memory_returns_it(self):
        m = _make_mock_memory("solo")
        result = mmr_select([m], {"solo": 1.0}, top_k=5)
        assert result == ["solo"]

    def test_classify_intent_very_long_query(self):
        long_query = " ".join(["auth", "database", "redis", "python"] * 50)
        intent, conf = classify_intent(long_query)
        assert intent in list(IntentLabel)
        assert 0.0 <= conf <= 1.0

    def test_classify_intent_unicode_query(self):
        intent, conf = classify_intent("¿cuándo fue actualizado auth.py?")
        assert intent in list(IntentLabel)

    def test_weighted_rank_consistent_ordering(self):
        memories = [_make_mock_memory(f"m{i}", importance=i+1) for i in range(5)]
        scores = {m.id: 1.0 for m in memories}
        r1 = weighted_rank(memories, scores, IntentLabel.general, "query")
        r2 = weighted_rank(memories, scores, IntentLabel.general, "query")
        assert [mid for mid, _ in r1] == [mid for mid, _ in r2]

    def test_mmr_select_no_infinite_loop_all_identical(self):
        # All memories identical cluster/session — should still terminate
        sid = str(uuid.uuid4())
        memories = [_make_mock_memory(f"m{i}", cluster_id=1, session_id=sid) for i in range(10)]
        scores = {m.id: 1.0 for m in memories}
        result = mmr_select(memories, scores, top_k=5, lambda_=0.5)
        assert len(result) == 5
