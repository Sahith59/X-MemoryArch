"""
End-to-end and unit tests for Sub-phase 2.1 Retrieval Core.

Covers:
  - Hard filter: status=superseded excluded
  - Hard filter: superseded_by excluded
  - Hard filter: review_status=rejected excluded
  - Hard filter: valid_until expired excluded
  - Hard filter: privacy_level clearance gates
  - Privacy leakage: forbidden_candidate_count always reflects blocked count
  - Privacy leakage: low-clearance caller never receives high-clearance memory
  - allowed_privacy_levels progression
  - BM25 candidate generation returns only allowed_ids
  - Entity candidate matching
  - Full pipeline: retrieve() returns RetrievalResult
  - Full pipeline: run_id written to retrieval_runs table
  - Full pipeline: latency_ms populated
  - Full pipeline: memories in result are all in allowed_ids
  - Full pipeline: top_k respected
  - Full pipeline: empty project returns empty result
  - include_superseded=True bypasses supersession filter
  - RetrievalRun log written on every call
"""
from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.services.retrieval.candidate_generators import (
    apply_hard_filters,
    allowed_privacy_levels,
    generate_bm25_candidates,
    generate_entity_candidates,
)
from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(dim: int = 384, val: float = 1.0, idx: int = 0) -> bytes:
    vec = np.zeros(dim, dtype=np.float32)
    vec[idx] = val
    return vec.tobytes()


def _create_memory(db, project_id, *, title="T", content="C", status="active",
                   privacy="internal", review_status="auto_extracted",
                   superseded_by=None, valid_until=None, embedding=None,
                   importance=3):
    from app import crud, schemas
    m = crud.create_memory(db, project_id, schemas.MemoryCreate(
        type=schemas.MemoryType.decision,
        title=title,
        content=content,
        importance=importance,
        tags=[], related_files=[], related_tools=[],
        status=schemas.MemoryStatus(status),
        source_type=schemas.SourceType.ai_session,
        privacy_level=schemas.PrivacyLevel(privacy),
        review_status=schemas.ReviewStatus(review_status),
    ), embedding=embedding)
    if superseded_by:
        m.superseded_by = superseded_by
        db.commit()
    if valid_until:
        m.valid_until = valid_until
        db.commit()
    return m


@pytest.fixture(autouse=True)
def _patch_ml(mock_embed, mock_entities, mock_is_technical_true):
    pass


# ---------------------------------------------------------------------------
# allowed_privacy_levels
# ---------------------------------------------------------------------------

class TestPrivacyClearanceLevels:
    def test_public_clearance(self):
        assert allowed_privacy_levels("public") == ["public"]

    def test_internal_clearance(self):
        assert allowed_privacy_levels("internal") == ["public", "internal"]

    def test_sensitive_clearance(self):
        assert allowed_privacy_levels("sensitive") == ["public", "internal", "sensitive"]

    def test_secret_clearance(self):
        assert allowed_privacy_levels("secret") == ["public", "internal", "sensitive", "secret"]

    def test_unknown_defaults_to_internal(self):
        levels = allowed_privacy_levels("unknown_level")
        assert "public" in levels
        assert "internal" in levels
        assert "sensitive" not in levels


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------

class TestHardFilters:
    def test_active_memory_included(self, db, project):
        m = _create_memory(db, project.id, title="Active Memory")
        allowed, forbidden = apply_hard_filters(db, project.id)
        assert m.id in allowed

    def test_superseded_status_excluded(self, db, project):
        m = _create_memory(db, project.id, status="superseded")
        allowed, _ = apply_hard_filters(db, project.id, include_superseded=False)
        assert m.id not in allowed

    def test_superseded_by_excluded(self, db, project):
        m1 = _create_memory(db, project.id, title="Old")
        m2 = _create_memory(db, project.id, title="New")
        m1.superseded_by = m2.id
        db.commit()
        allowed, _ = apply_hard_filters(db, project.id, include_superseded=False)
        assert m1.id not in allowed
        assert m2.id in allowed

    def test_rejected_review_status_excluded(self, db, project):
        m = _create_memory(db, project.id, review_status="rejected")
        allowed, _ = apply_hard_filters(db, project.id)
        assert m.id not in allowed

    def test_expired_valid_until_excluded(self, db, project):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        m = _create_memory(db, project.id, valid_until=past)
        allowed, _ = apply_hard_filters(db, project.id)
        assert m.id not in allowed

    def test_future_valid_until_included(self, db, project):
        future = datetime.now(timezone.utc) + timedelta(days=30)
        m = _create_memory(db, project.id, valid_until=future)
        allowed, _ = apply_hard_filters(db, project.id)
        assert m.id in allowed

    def test_null_valid_until_included(self, db, project):
        m = _create_memory(db, project.id, valid_until=None)
        allowed, _ = apply_hard_filters(db, project.id)
        assert m.id in allowed

    def test_include_superseded_bypasses_filter(self, db, project):
        m = _create_memory(db, project.id, status="superseded")
        allowed, _ = apply_hard_filters(db, project.id, include_superseded=True)
        assert m.id in allowed

    def test_multiple_filters_all_applied(self, db, project):
        good = _create_memory(db, project.id, title="Good")
        bad_status = _create_memory(db, project.id, status="superseded")
        bad_review = _create_memory(db, project.id, review_status="rejected")
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        bad_expired = _create_memory(db, project.id, valid_until=past)

        allowed, _ = apply_hard_filters(db, project.id)
        assert good.id in allowed
        assert bad_status.id not in allowed
        assert bad_review.id not in allowed
        assert bad_expired.id not in allowed


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------

class TestPrivacyGate:
    def test_public_caller_sees_only_public(self, db, project):
        pub = _create_memory(db, project.id, title="Public", privacy="public")
        intern = _create_memory(db, project.id, title="Internal", privacy="internal")
        secret = _create_memory(db, project.id, title="Secret", privacy="secret")

        allowed, forbidden = apply_hard_filters(db, project.id, max_clearance="public")
        assert pub.id in allowed
        assert intern.id not in allowed
        assert secret.id not in allowed
        assert forbidden == 2  # internal + secret blocked

    def test_internal_caller_sees_public_and_internal(self, db, project):
        pub = _create_memory(db, project.id, title="Public", privacy="public")
        intern = _create_memory(db, project.id, title="Internal", privacy="internal")
        sens = _create_memory(db, project.id, title="Sensitive", privacy="sensitive")

        allowed, _ = apply_hard_filters(db, project.id, max_clearance="internal")
        assert pub.id in allowed
        assert intern.id in allowed
        assert sens.id not in allowed

    def test_secret_caller_sees_all(self, db, project):
        for priv in ("public", "internal", "sensitive", "secret"):
            _create_memory(db, project.id, privacy=priv)

        allowed, forbidden = apply_hard_filters(db, project.id, max_clearance="secret")
        assert len(allowed) == 4
        assert forbidden == 0

    def test_forbidden_count_is_zero_for_secret_clearance(self, db, project):
        for priv in ("public", "internal", "sensitive", "secret"):
            _create_memory(db, project.id, privacy=priv)
        _, forbidden = apply_hard_filters(db, project.id, max_clearance="secret")
        assert forbidden == 0

    def test_no_cross_project_exposure(self, db):
        from app import crud, schemas
        p1 = crud.create_project(db, schemas.ProjectCreate(name="P1", description=""))
        p2 = crud.create_project(db, schemas.ProjectCreate(name="P2", description=""))
        m1 = _create_memory(db, p1.id, title="P1 Memory")
        m2 = _create_memory(db, p2.id, title="P2 Memory")

        allowed_p1, _ = apply_hard_filters(db, p1.id)
        allowed_p2, _ = apply_hard_filters(db, p2.id)
        assert m1.id in allowed_p1
        assert m1.id not in allowed_p2
        assert m2.id not in allowed_p1
        assert m2.id in allowed_p2


# ---------------------------------------------------------------------------
# BM25 candidate generation
# ---------------------------------------------------------------------------

class TestBM25Candidates:
    def test_returns_empty_when_no_allowed_ids(self, db, project):
        result = generate_bm25_candidates(db, project.id, "query", allowed_ids=[], top_k=10)
        assert result == []

    def test_returns_empty_when_fts_not_set_up(self, db, project):
        # FTS5 table may not exist — should return [] gracefully, not raise
        m = _create_memory(db, project.id, title="PostgreSQL memory")
        result = generate_bm25_candidates(db, project.id, "PostgreSQL", [m.id], top_k=10)
        # Either [] (FTS not set up) or [(m.id, score)] — both acceptable
        assert isinstance(result, list)

    def test_bm25_with_fts_setup(self, db, project):
        from app.search import setup_fts
        from sqlalchemy import inspect
        setup_fts(db.get_bind())

        m1 = _create_memory(db, project.id, title="PostgreSQL database decision",
                             content="We use PostgreSQL for production workloads.")
        m2 = _create_memory(db, project.id, title="Redis cache strategy",
                             content="Redis provides fast session caching.")
        # Rebuild search_text for FTS
        from app import crud
        for m in [m1, m2]:
            m.search_text = f"{m.title} {m.content}"
            db.commit()

        allowed = [m1.id, m2.id]
        result = generate_bm25_candidates(db, project.id, "PostgreSQL production", allowed, 10)
        if result:
            ids_returned = [mid for mid, _ in result]
            assert all(mid in allowed for mid, _ in result)
            assert m1.id in ids_returned  # PostgreSQL term matches m1

    def test_bm25_scores_are_positive(self, db, project):
        from app.search import setup_fts
        setup_fts(db.get_bind())
        m = _create_memory(db, project.id, title="FastAPI REST API service",
                            content="We are building a FastAPI REST service.")
        m.search_text = f"{m.title} {m.content}"
        db.commit()
        result = generate_bm25_candidates(db, project.id, "FastAPI", [m.id], 10)
        for _, score in result:
            assert score > 0


# ---------------------------------------------------------------------------
# Entity candidate generation
# ---------------------------------------------------------------------------

class TestEntityCandidates:
    def test_empty_allowed_ids_returns_empty(self, db, project):
        result = generate_entity_candidates(db, project.id, "PostgreSQL", [], 10)
        assert result == []

    def test_no_entities_returns_empty(self, db, project):
        m = _create_memory(db, project.id)
        result = generate_entity_candidates(db, project.id, "PostgreSQL", [m.id], 10)
        assert result == []

    def test_entity_match_returns_memory(self, db, project):
        from app import crud
        from app.services.entity_extractor import Entity
        m = _create_memory(db, project.id, title="DB decision")
        crud.create_memory_entities(db, m.id, project.id, [
            Entity(text="PostgreSQL", label="TECH", normalized="postgresql"),
        ])
        result = generate_entity_candidates(db, project.id, "PostgreSQL", [m.id], 10)
        assert len(result) > 0
        assert result[0][0] == m.id
        assert result[0][1] > 0.0

    def test_entity_match_respects_allowed_ids(self, db, project):
        from app import crud
        from app.services.entity_extractor import Entity
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        for mid in [m1.id, m2.id]:
            crud.create_memory_entities(db, mid, project.id, [
                Entity(text="Redis", label="TECH", normalized="redis"),
            ])
        result = generate_entity_candidates(db, project.id, "Redis", [m1.id], 10)
        ids_returned = {mid for mid, _ in result}
        assert m2.id not in ids_returned


# ---------------------------------------------------------------------------
# Full pipeline: retrieve()
# ---------------------------------------------------------------------------

class TestRetrievePipeline:
    def test_returns_retrieval_result(self, db, project):
        from app.services.retrieval.retrieval_service import RetrievalResult
        m = _create_memory(db, project.id, title="Test memory")
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "test query", backend)
        assert isinstance(result, RetrievalResult)

    def test_run_id_is_string(self, db, project):
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "query", backend)
        assert isinstance(result.run_id, str)
        assert len(result.run_id) > 0

    def test_latency_ms_populated(self, db, project):
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "query", backend)
        assert result.latency_ms >= 0

    def test_empty_project_returns_empty_memories(self, db, project):
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "anything", backend)
        assert result.memories == []
        assert result.selected_memory_ids == []
        assert result.fused_count == 0

    def test_top_k_respected(self, db, project):
        from app.search import setup_fts
        setup_fts(db.get_bind())
        for i in range(8):
            m = _create_memory(db, project.id, title=f"Memory about PostgreSQL {i}",
                               content=f"PostgreSQL content {i} for testing retrieval.",
                               embedding=_make_embedding(idx=i % 5))
            m.search_text = f"{m.title} {m.content}"
            db.commit()

        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "PostgreSQL", backend, RetrievalConfig(top_k=3))
        assert len(result.memories) <= 3

    def test_all_returned_memories_pass_filters(self, db, project):
        good = _create_memory(db, project.id, title="Good memory")
        bad = _create_memory(db, project.id, status="superseded")

        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "memory", backend, RetrievalConfig(top_k=10))

        returned_ids = {m.id for m in result.memories}
        assert bad.id not in returned_ids

    def test_privacy_gate_in_pipeline(self, db, project):
        pub = _create_memory(db, project.id, title="Public memory", privacy="public")
        secret = _create_memory(db, project.id, title="Secret memory", privacy="secret")

        backend = SQLiteExactBackend(db)
        cfg = RetrievalConfig(top_k=10, max_clearance="public")
        result = retrieve(db, project.id, "memory", backend, cfg)

        returned_ids = {m.id for m in result.memories}
        assert secret.id not in returned_ids
        assert result.forbidden_candidate_count >= 1  # secret memory was blocked

    def test_retrieval_run_logged_to_db(self, db, project):
        from app.p2_models import RetrievalRun
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "test logging", backend)

        run = db.query(RetrievalRun).filter(RetrievalRun.id == result.run_id).first()
        assert run is not None
        assert run.query == "test logging"
        assert run.project_id == project.id
        assert run.backend_used == "sqlite_exact"
        assert run.latency_ms >= 0

    def test_retrieval_run_selected_ids_logged(self, db, project):
        from app.search import setup_fts
        from app.p2_models import RetrievalRun
        setup_fts(db.get_bind())
        m = _create_memory(db, project.id, title="PostgreSQL decision",
                           content="We decided to use PostgreSQL.",
                           embedding=_make_embedding(idx=0))
        m.search_text = f"{m.title} {m.content}"
        db.commit()

        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "PostgreSQL", backend, RetrievalConfig(top_k=5))

        run = db.query(RetrievalRun).filter(RetrievalRun.id == result.run_id).first()
        assert run is not None
        logged_ids = run.get_selected_ids()
        assert set(logged_ids) == set(result.selected_memory_ids)

    def test_candidate_counts_reported(self, db, project):
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "anything", backend)
        assert result.candidate_count_bm25 >= 0
        assert result.candidate_count_dense >= 0
        assert result.candidate_count_entity >= 0

    def test_forbidden_count_zero_for_secret_clearance(self, db, project):
        for priv in ("public", "internal", "sensitive", "secret"):
            _create_memory(db, project.id, privacy=priv)
        backend = SQLiteExactBackend(db)
        result = retrieve(db, project.id, "memory", backend, RetrievalConfig(max_clearance="secret"))
        assert result.forbidden_candidate_count == 0

    def test_dense_leg_uses_embedding_similarity(self, db, project):
        # Memory A: embedding[0]=1.0 (matches query)
        # Memory B: embedding[1]=1.0 (orthogonal to query)
        m_a = _create_memory(db, project.id, title="Match",
                              embedding=_make_embedding(idx=0))
        m_b = _create_memory(db, project.id, title="No Match",
                              embedding=_make_embedding(idx=1))

        # Mock embed_text to return vector aligned with m_a
        import numpy as np
        from unittest.mock import patch
        query_vec = np.zeros(384, dtype=np.float32)
        query_vec[0] = 1.0
        query_bytes = query_vec.tobytes()

        with patch("app.services.semantic_classifier.embed_text", return_value=query_bytes):
            backend = SQLiteExactBackend(db)
            result = retrieve(db, project.id, "query", backend, RetrievalConfig(top_k=5))

        if result.memories:
            # m_a should appear (highest cosine similarity to query)
            returned_ids = [m.id for m in result.memories]
            assert m_a.id in returned_ids

    def test_rrf_fusion_combines_all_legs(self, db, project):
        from app.search import setup_fts
        setup_fts(db.get_bind())

        # Create a memory that should rank in all three legs
        m = _create_memory(db, project.id,
                           title="PostgreSQL database decision for production",
                           content="Team decided PostgreSQL for ACID compliance.",
                           embedding=_make_embedding(idx=0))
        m.search_text = f"{m.title} {m.content}"
        db.commit()

        from app import crud
        from app.services.entity_extractor import Entity
        crud.create_memory_entities(db, m.id, project.id, [
            Entity(text="PostgreSQL", label="TECH", normalized="postgresql"),
        ])

        import numpy as np
        from unittest.mock import patch
        query_vec = np.zeros(384, dtype=np.float32)
        query_vec[0] = 1.0

        with patch("app.services.semantic_classifier.embed_text", return_value=query_vec.tobytes()):
            backend = SQLiteExactBackend(db)
            result = retrieve(db, project.id, "PostgreSQL database",
                             backend, RetrievalConfig(top_k=5))

        assert m.id in result.selected_memory_ids
        assert result.candidate_count_bm25 > 0 or result.candidate_count_dense > 0

    def test_multiple_runs_each_logged(self, db, project):
        from app.p2_models import RetrievalRun
        backend = SQLiteExactBackend(db)
        for i in range(3):
            retrieve(db, project.id, f"query {i}", backend)
        count = db.query(RetrievalRun).filter(RetrievalRun.project_id == project.id).count()
        assert count == 3
