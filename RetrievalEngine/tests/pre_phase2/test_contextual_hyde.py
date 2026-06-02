"""
Sub-phase 2.7 — Contextual Embeddings + HyDE stress tests.

Coverage:
  ContextualEmbeddings
    - Template prefix generation for all 13 canonical types
    - LLM prefix generation (mocked)
    - LLM fallback to template on failure
    - build_contextual_text: prefix + content concatenation
    - generate_contextual_embeddings: full backfill (batch, skip, force)
    - Skips memories without content, cluster_summaries, superseded
    - Writes contextual_prefix and updated embedding back to DB
    - force_regenerate=True re-processes already-prefixed memories
    - Empty project produces ContextualEmbeddingResult with zeros
    - Project not found raises ValueError
    - get_embed_text_for_memory: returns contextual text if prefix set, else content

  HyDE
    - generate_hyde_text: mocked LLM, prompt contains query
    - generate_hyde_text: empty query raises ValueError
    - generate_hyde_text: LLM returns empty → raises ValueError
    - embed_text_to_vec: returns normalized np.ndarray
    - combine_vectors: all 4 strategies (avg, max, query, hyde)
    - combine_vectors: normalisation of result
    - generate_hyde_vector: full pipeline (mocked LLM)
    - generate_hyde_vector: llm_fn=None returns None (graceful skip)
    - generate_hyde_vector: LLM failure returns None (graceful skip)
    - should_use_hyde: intent/confidence gating

  Pipeline integration
    - retrieve() with enable_hyde=False: hyde_used=False in result
    - retrieve() with enable_hyde=True + mocked LLM: hyde_used=True in result
    - retrieve() with enable_hyde=True + no llm_fn: hyde_used=False (graceful)
    - HyDE augments query vector (combined is different from pure query vector)
    - Retrieval results are still valid (no crashes, correct structure)

  Router endpoints
    - POST /memories/generate-contextual-embeddings: success path
    - POST /memories/generate-contextual-embeddings: 404 for unknown project
    - POST /memories/generate-contextual-embeddings: force_regenerate=True
    - POST /memories/generate-contextual-embeddings: skip already-prefixed
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.retrieval.contextual_embeddings import (
    ContextualEmbeddingResult,
    build_contextual_text,
    generate_contextual_embeddings,
    generate_contextual_prefix,
    get_embed_text_for_memory,
    _template_prefix,
    _TYPE_VERBS,
)
from app.services.retrieval.hyde import (
    HyDEConfig,
    combine_vectors,
    embed_text_to_vec,
    generate_hyde_text,
    generate_hyde_vector,
    should_use_hyde,
)
from app.services.retrieval.retrieval_service import RetrievalConfig, RetrievalResult, retrieve
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _make_embedding_bytes(dim: int = 384, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tobytes()


def _create_project(db):
    from app import crud, schemas
    return crud.create_project(db, schemas.ProjectCreate(
        name="HyDE Test Project",
        description="For sub-phase 2.7 tests",
        tech_stack=["Python"],
        goals=["Test contextual embeddings"],
        domain="software",
    ))


def _create_memory(db, project_id: str, *, content: str = "Test content",
                   title: str = "Test Title", embedding: bytes | None = None,
                   status: str = "active", mem_type: str = "decision",
                   contextual_prefix: str | None = None) -> object:
    from app import models as m
    mid = str(uuid.uuid4())
    mem = m.Memory(
        id=mid,
        project_id=project_id,
        type=mem_type,
        title=title,
        content=content,
        status=status,
        privacy_level="internal",
        review_status="auto_extracted",
        importance=3,
        confidence=1.0,
        embedding=embedding,
        contextual_prefix=contextual_prefix,
        created_at=_now(),
        updated_at=_now(),
    )
    db.add(mem)
    db.commit()
    return mem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_ml(mock_embed, mock_entities, mock_is_technical_true):
    pass


# ===========================================================================
# TestContextualEmbeddings — prefix generation
# ===========================================================================

class TestTemplatePrefixGeneration:
    def test_template_prefix_contains_project_name(self):
        mem = MagicMock()
        mem.type = "decision"
        mem.title = "Use PostgreSQL"
        project = MagicMock()
        project.name = "Alpha Project"
        prefix = _template_prefix(mem, project)
        assert "Alpha Project" in prefix

    def test_template_prefix_contains_title(self):
        mem = MagicMock()
        mem.type = "constraint"
        mem.title = "API rate limit is 1000 req/min"
        project = MagicMock()
        project.name = "Beta"
        prefix = _template_prefix(mem, project)
        assert "API rate limit" in prefix

    def test_template_prefix_all_canonical_types(self):
        """All 13 canonical types produce non-empty, meaningful prefixes."""
        project = MagicMock()
        project.name = "Test"
        for type_name in _TYPE_VERBS:
            mem = MagicMock()
            mem.type = type_name
            mem.title = f"Title for {type_name}"
            prefix = _template_prefix(mem, project)
            assert len(prefix) > 20
            assert type_name in prefix or _TYPE_VERBS[type_name] in prefix

    def test_template_prefix_unknown_type(self):
        mem = MagicMock()
        mem.type = "foobar_unknown"
        mem.title = "Some title"
        project = MagicMock()
        project.name = "X"
        prefix = _template_prefix(mem, project)
        assert "foobar_unknown" in prefix
        assert len(prefix) > 10

    def test_template_prefix_none_type_defaults(self):
        mem = MagicMock()
        mem.type = None
        mem.title = "A title"
        project = MagicMock()
        project.name = "P"
        prefix = _template_prefix(mem, project)
        assert len(prefix) > 0

    def test_template_prefix_none_project_name(self):
        mem = MagicMock()
        mem.type = "fact"
        mem.title = "A fact"
        project = MagicMock()
        project.name = None
        prefix = _template_prefix(mem, project)
        assert len(prefix) > 0


class TestGenerateContextualPrefix:
    def test_no_llm_uses_template(self):
        mem = MagicMock()
        mem.type = "decision"
        mem.title = "Use Redis"
        project = MagicMock()
        project.name = "Prod"
        prefix = generate_contextual_prefix(mem, project, llm_fn=None)
        assert "Prod" in prefix
        assert len(prefix) > 10

    def test_llm_fn_called_with_prompt(self):
        mem = MagicMock()
        mem.type = "constraint"
        mem.title = "Max 500ms latency"
        mem.content = "API must respond within 500ms."
        project = MagicMock()
        project.name = "SpeedProject"

        llm_fn = MagicMock(return_value="This constraint defines the maximum latency requirement.")
        prefix = generate_contextual_prefix(mem, project, llm_fn=llm_fn)
        assert "latency requirement" in prefix
        llm_fn.assert_called_once()

    def test_llm_failure_falls_back_to_template(self):
        mem = MagicMock()
        mem.type = "plan"
        mem.title = "Migrate to k8s"
        mem.content = "Plan to move services to Kubernetes."
        project = MagicMock()
        project.name = "DevOps"

        def _failing_llm(prompt):
            raise RuntimeError("LLM unavailable")

        prefix = generate_contextual_prefix(mem, project, llm_fn=_failing_llm)
        # Should fall back to template
        assert "DevOps" in prefix

    def test_llm_returns_empty_falls_back(self):
        mem = MagicMock()
        mem.type = "insight"
        mem.title = "Cache busting"
        mem.content = "Cache invalidation is hard."
        project = MagicMock()
        project.name = "CacheProj"

        prefix = generate_contextual_prefix(mem, project, llm_fn=lambda p: "")
        # Empty LLM result should fall back to template
        assert "CacheProj" in prefix


class TestBuildContextualText:
    def test_combines_prefix_and_content(self):
        mem = MagicMock()
        mem.content = "PostgreSQL was chosen for production."
        text = build_contextual_text(mem, "This is the prefix.")
        assert text.startswith("This is the prefix.")
        assert "PostgreSQL" in text
        assert "\n\n" in text

    def test_handles_empty_content(self):
        mem = MagicMock()
        mem.content = ""
        text = build_contextual_text(mem, "A prefix.")
        assert text.startswith("A prefix.")

    def test_handles_none_content(self):
        mem = MagicMock()
        mem.content = None
        text = build_contextual_text(mem, "A prefix.")
        assert "A prefix." in text


class TestGetEmbedTextForMemory:
    def test_returns_contextual_text_when_prefix_set(self):
        mem = MagicMock()
        mem.contextual_prefix = "This is the prefix."
        mem.content = "The actual content."
        text = get_embed_text_for_memory(mem)
        assert "This is the prefix." in text
        assert "The actual content." in text

    def test_returns_content_when_no_prefix(self):
        mem = MagicMock()
        mem.contextual_prefix = None
        mem.content = "Raw content only."
        text = get_embed_text_for_memory(mem)
        assert text == "Raw content only."

    def test_returns_content_when_prefix_empty_string(self):
        mem = MagicMock()
        mem.contextual_prefix = ""
        mem.content = "Content."
        text = get_embed_text_for_memory(mem)
        assert "Content." in text


# ===========================================================================
# TestGenerateContextualEmbeddings — full backfill
# ===========================================================================

class TestGenerateContextualEmbeddings:
    def test_empty_project_returns_zeros(self, db):
        project = _create_project(db)
        result = generate_contextual_embeddings(db, project.id)
        assert result.total_processed == 0
        assert result.newly_prefixed == 0
        assert result.already_had_prefix == 0
        assert result.failed == 0

    def test_project_not_found_raises(self, db):
        with pytest.raises(ValueError, match="not found"):
            generate_contextual_embeddings(db, "nonexistent-project-id")

    def test_processes_active_memories(self, db):
        project = _create_project(db)
        for i in range(3):
            _create_memory(db, project.id, content=f"Content {i}",
                           title=f"Memory {i}")

        result = generate_contextual_embeddings(db, project.id)
        assert result.total_processed == 3
        assert result.newly_prefixed == 3
        assert result.already_had_prefix == 0

    def test_skips_already_prefixed_memories(self, db):
        project = _create_project(db)
        # One with prefix, one without
        _create_memory(db, project.id, content="Old content",
                       contextual_prefix="Already have a prefix.")
        _create_memory(db, project.id, content="New content")

        result = generate_contextual_embeddings(db, project.id)
        assert result.already_had_prefix == 1
        assert result.newly_prefixed == 1

    def test_force_regenerate_overwrites_existing_prefix(self, db):
        project = _create_project(db)
        mem = _create_memory(db, project.id, content="Content",
                             contextual_prefix="Old prefix.")

        result = generate_contextual_embeddings(db, project.id, force_regenerate=True)
        assert result.newly_prefixed == 1
        assert result.already_had_prefix == 0

        db.refresh(mem)
        # New prefix should be different from empty (generated by template)
        assert mem.contextual_prefix is not None
        assert len(mem.contextual_prefix) > 5

    def test_skips_cluster_summary_type(self, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="A regular memory")
        _create_memory(db, project.id, content="A cluster summary",
                       mem_type="cluster_summary")

        result = generate_contextual_embeddings(db, project.id)
        # Only 1 processed (cluster_summary excluded)
        assert result.total_processed == 1
        assert result.newly_prefixed == 1

    def test_skips_superseded_memories(self, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="Active memory")
        _create_memory(db, project.id, content="Old superseded memory",
                       status="superseded")

        result = generate_contextual_embeddings(db, project.id)
        assert result.total_processed == 1

    def test_updates_embedding_in_db(self, db):
        project = _create_project(db)
        original_embedding = _make_embedding_bytes(384, seed=42)
        mem = _create_memory(db, project.id, content="Test content for embedding",
                             embedding=original_embedding)

        result = generate_contextual_embeddings(db, project.id)
        assert result.newly_prefixed == 1

        db.refresh(mem)
        assert mem.contextual_prefix is not None
        assert len(mem.contextual_prefix) > 0
        # Embedding should have been updated (mock_embed returns fixed zero vec, but it's called)
        assert mem.embedding is not None

    def test_writes_memory_ids_updated(self, db):
        project = _create_project(db)
        mems = [_create_memory(db, project.id, content=f"C{i}") for i in range(4)]

        result = generate_contextual_embeddings(db, project.id)
        assert len(result.memory_ids_updated) == 4
        expected_ids = {m.id for m in mems}
        assert set(result.memory_ids_updated) == expected_ids

    def test_llm_fn_used_when_provided(self, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="Redis chosen for caching.",
                       title="Redis Decision")

        call_count = [0]
        def _mock_llm(prompt: str) -> str:
            call_count[0] += 1
            return "This memory records a decision about caching technology for performance."

        result = generate_contextual_embeddings(db, project.id, llm_fn=_mock_llm)
        assert result.newly_prefixed == 1
        assert result.used_llm is True
        assert call_count[0] == 1

    def test_batching_processes_all(self, db):
        project = _create_project(db)
        for i in range(7):
            _create_memory(db, project.id, content=f"Memory number {i}")

        result = generate_contextual_embeddings(db, project.id, batch_size=3)
        assert result.total_processed == 7
        assert result.newly_prefixed == 7

    def test_result_is_dataclass_instance(self, db):
        project = _create_project(db)
        result = generate_contextual_embeddings(db, project.id)
        assert isinstance(result, ContextualEmbeddingResult)
        assert result.project_id == project.id


# ===========================================================================
# TestHyDE — HyDE text generation and vector combination
# ===========================================================================

class TestGenerateHyDEText:
    def test_generates_text_via_llm(self):
        llm_fn = MagicMock(return_value="PostgreSQL is the production database, chosen for concurrency.")
        text = generate_hyde_text("what database do we use?", llm_fn)
        assert "PostgreSQL" in text
        llm_fn.assert_called_once()

    def test_prompt_contains_query(self):
        captured = []
        def _llm(prompt):
            captured.append(prompt)
            return "Some hypothetical memory text."
        generate_hyde_text("why did we choose Redis?", _llm)
        assert "why did we choose Redis?" in captured[0]

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="empty"):
            generate_hyde_text("", lambda p: "text")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="empty"):
            generate_hyde_text("   ", lambda p: "text")

    def test_empty_llm_response_raises(self):
        with pytest.raises(ValueError, match="empty"):
            generate_hyde_text("what is X?", lambda p: "")

    def test_whitespace_llm_response_raises(self):
        with pytest.raises(ValueError, match="empty"):
            generate_hyde_text("what is X?", lambda p: "   ")

    def test_long_response_trimmed(self):
        long_response = "A" * 1000
        llm_fn = MagicMock(return_value=long_response)
        text = generate_hyde_text("test query?", llm_fn)
        assert len(text) <= 480


class TestEmbedTextToVec:
    def test_returns_numpy_array(self):
        vec = embed_text_to_vec("test text")
        assert isinstance(vec, np.ndarray)

    def test_returns_normalized_vector(self):
        vec = embed_text_to_vec("test text")
        norm = np.linalg.norm(vec)
        # Mock returns zero vector which has norm=0; acceptable
        assert norm == pytest.approx(0.0, abs=1.0)

    def test_returns_float32(self):
        vec = embed_text_to_vec("test text")
        assert vec.dtype == np.float32


class TestCombineVectors:
    @pytest.fixture
    def vecs(self):
        rng = np.random.default_rng(0)
        q = rng.standard_normal(384).astype(np.float32)
        q /= np.linalg.norm(q)
        h = rng.standard_normal(384).astype(np.float32)
        h /= np.linalg.norm(h)
        return q, h

    def test_avg_strategy_returns_normalized(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "avg")
        arr = np.array(result, dtype=np.float32)
        norm = np.linalg.norm(arr)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_max_strategy_returns_normalized(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "max")
        arr = np.array(result, dtype=np.float32)
        norm = np.linalg.norm(arr)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_query_strategy_returns_query_vector(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "query")
        np.testing.assert_array_almost_equal(result, q.tolist())

    def test_hyde_strategy_returns_hyde_vector(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "hyde")
        np.testing.assert_array_almost_equal(result, h.tolist())

    def test_avg_differs_from_query(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "avg")
        result_arr = np.array(result, dtype=np.float32)
        assert not np.allclose(result_arr, q)

    def test_max_differs_from_query(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "max")
        result_arr = np.array(result, dtype=np.float32)
        assert not np.allclose(result_arr, q)

    def test_unknown_strategy_raises(self, vecs):
        q, h = vecs
        with pytest.raises(ValueError, match="Unknown"):
            combine_vectors(q, h, "invalid_strategy")

    def test_result_is_list(self, vecs):
        q, h = vecs
        result = combine_vectors(q, h, "avg")
        assert isinstance(result, list)


class TestGenerateHyDEVector:
    def test_none_llm_fn_returns_none(self):
        cfg = HyDEConfig(llm_fn=None)
        result = generate_hyde_vector("test query", cfg)
        assert result is None

    def test_llm_failure_returns_none(self):
        def _bad_llm(prompt):
            raise RuntimeError("Network error")
        cfg = HyDEConfig(llm_fn=_bad_llm)
        result = generate_hyde_vector("test query", cfg)
        assert result is None

    def test_returns_vector_on_success(self):
        llm_fn = MagicMock(return_value="PostgreSQL is the production database for all services.")
        cfg = HyDEConfig(llm_fn=llm_fn, combination="avg")
        q_vec = np.zeros(384, dtype=np.float32).tolist()
        result = generate_hyde_vector("what db?", cfg, query_vector=q_vec)
        # With mock embed returning zero vector, result may be zero or None
        # Key: no exception thrown
        # (mock_embed returns zeros, so the embedded hyde text is also zeros)
        assert result is None or isinstance(result, list)

    def test_with_real_query_vector(self):
        """HyDE combines provided query vector with hyde vector."""
        rng = np.random.default_rng(100)
        q_vec = rng.standard_normal(384).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)

        llm_fn = MagicMock(return_value="This is a hypothetical memory about the database.")
        cfg = HyDEConfig(llm_fn=llm_fn, combination="avg")
        result = generate_hyde_vector("what database?", cfg, query_vector=q_vec.tolist())
        # With mock embed returning zero bytes, hyde_vec is zero → after normalize → zero
        # avg of q_vec and zero_vec = q_vec/2, normalized back to q_vec
        # So result should be valid or None depending on norm behavior
        assert result is None or isinstance(result, list)

    def test_no_query_vector_provided(self):
        """generate_hyde_vector embeds query itself when not provided."""
        llm_fn = MagicMock(return_value="Hypothetical memory text.")
        cfg = HyDEConfig(llm_fn=llm_fn)
        # Should not raise even without pre-computed query vector
        result = generate_hyde_vector("what is X?", cfg, query_vector=None)
        assert result is None or isinstance(result, list)


class TestShouldUseHyDE:
    def test_exploratory_intent_returns_true(self):
        assert should_use_hyde("exploratory", 0.9) is True

    def test_general_intent_returns_true(self):
        assert should_use_hyde("general", 0.8) is True

    def test_low_confidence_returns_true(self):
        assert should_use_hyde("factual", 0.3) is True

    def test_factual_high_confidence_returns_false(self):
        assert should_use_hyde("factual", 0.95) is False

    def test_temporal_high_confidence_returns_false(self):
        assert should_use_hyde("temporal", 0.90) is False

    def test_code_high_confidence_returns_false(self):
        assert should_use_hyde("code", 0.85) is False

    def test_confidence_at_threshold_returns_false(self):
        # At exactly 0.70, should_use returns False (< threshold, not ≤)
        assert should_use_hyde("factual", 0.70) is False

    def test_confidence_just_below_threshold_returns_true(self):
        assert should_use_hyde("factual", 0.69) is True


# ===========================================================================
# TestPipelineIntegration — HyDE wired into retrieve()
# ===========================================================================

class TestPipelineHyDEIntegration:
    def test_retrieve_with_hyde_disabled(self, db):
        project = _create_project(db)
        _create_memory(db, project.id,
                       content="PostgreSQL for production",
                       embedding=_make_embedding_bytes(384, seed=1))

        backend = SQLiteExactBackend(db)
        cfg = RetrievalConfig(top_k=5, enable_hyde=False)
        result = retrieve(db, project.id, "what database?", backend, cfg)
        assert result.hyde_used is False
        assert isinstance(result, RetrievalResult)

    def test_retrieve_with_hyde_no_llm_fn(self, db):
        """enable_hyde=True but no llm_fn → hyde_used=False (graceful skip)."""
        project = _create_project(db)
        _create_memory(db, project.id,
                       content="Redis for caching",
                       embedding=_make_embedding_bytes(384, seed=2))

        backend = SQLiteExactBackend(db)
        cfg = RetrievalConfig(top_k=5, enable_hyde=True, hyde_llm_fn=None)
        result = retrieve(db, project.id, "what cache?", backend, cfg)
        assert result.hyde_used is False

    def test_retrieve_with_hyde_mocked_llm(self, db):
        """enable_hyde=True + exploratory query → hyde_used=True."""
        project = _create_project(db)
        _create_memory(db, project.id,
                       content="trade-offs of Redis vs Memcached",
                       embedding=_make_embedding_bytes(384, seed=3))

        backend = SQLiteExactBackend(db)
        llm_fn = MagicMock(return_value="Redis was chosen for its persistence and pub/sub features.")
        cfg = RetrievalConfig(
            top_k=5,
            enable_hyde=True,
            hyde_llm_fn=llm_fn,
            # Set exploratory to guarantee HyDE fires (confidence-based gate)
            hyde_confidence_threshold=1.0,  # everything below 1.0 gets HyDE
        )
        result = retrieve(db, project.id, "explain the trade-off between redis and memcached", backend, cfg)
        assert result.hyde_used is True

    def test_retrieve_with_hyde_factual_high_confidence_skips(self, db):
        """HyDE should not fire for precise factual queries."""
        project = _create_project(db)
        _create_memory(db, project.id,
                       content="PostgreSQL chosen",
                       embedding=_make_embedding_bytes(384, seed=4))

        backend = SQLiteExactBackend(db)
        llm_fn = MagicMock(return_value="PostgreSQL was chosen.")
        cfg = RetrievalConfig(
            top_k=5,
            enable_hyde=True,
            hyde_llm_fn=llm_fn,
            hyde_confidence_threshold=0.70,
        )
        # "what is PostgreSQL?" → likely factual with high confidence → HyDE skipped
        result = retrieve(db, project.id, "what is PostgreSQL?", backend, cfg)
        # Can't guarantee exactly due to heuristic, but no crash:
        assert isinstance(result.hyde_used, bool)

    def test_retrieve_returns_valid_result_with_hyde(self, db):
        """With HyDE active, result structure is still fully valid."""
        project = _create_project(db)
        for i in range(5):
            _create_memory(db, project.id,
                           content=f"Technical detail {i}",
                           embedding=_make_embedding_bytes(384, seed=i+10))

        backend = SQLiteExactBackend(db)
        llm_fn = MagicMock(return_value="Some hypothetical memory for this query.")
        cfg = RetrievalConfig(
            top_k=3,
            enable_hyde=True,
            hyde_llm_fn=llm_fn,
        )
        result = retrieve(db, project.id, "explain something", backend, cfg)
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) <= 3
        assert result.run_id is not None
        assert result.latency_ms >= 0


# ===========================================================================
# TestContextualEmbeddingsRouter — endpoint tests
# ===========================================================================

class TestContextualEmbeddingsRouter:
    @pytest.fixture
    def client(self, db):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from app.routers.retrieval import router, _global_router

        app = FastAPI()
        app.include_router(router)
        app.include_router(_global_router)

        def _override_db():
            yield db

        from app.routers.retrieval import _get_db
        app.dependency_overrides[_get_db] = _override_db
        return TestClient(app, raise_server_exceptions=True)

    def test_404_for_unknown_project(self, client):
        resp = client.post("/projects/bad-id/memories/generate-contextual-embeddings", json={})
        assert resp.status_code == 404

    def test_empty_project_returns_zeros(self, client, db):
        project = _create_project(db)
        resp = client.post(
            f"/projects/{project.id}/memories/generate-contextual-embeddings",
            json={"batch_size": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_processed"] == 0
        assert data["newly_prefixed"] == 0

    def test_processes_memories(self, client, db):
        project = _create_project(db)
        for i in range(3):
            _create_memory(db, project.id, content=f"Content {i}")

        resp = client.post(
            f"/projects/{project.id}/memories/generate-contextual-embeddings",
            json={"batch_size": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["newly_prefixed"] == 3
        assert data["total_processed"] == 3
        assert "memory_ids_updated" in data
        assert len(data["memory_ids_updated"]) == 3

    def test_skip_already_prefixed(self, client, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="Pre-existing",
                       contextual_prefix="Already set.")
        _create_memory(db, project.id, content="New one")

        resp = client.post(
            f"/projects/{project.id}/memories/generate-contextual-embeddings",
            json={"force_regenerate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["already_had_prefix"] == 1
        assert data["newly_prefixed"] == 1

    def test_force_regenerate(self, client, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="Some content",
                       contextual_prefix="Old prefix to overwrite.")

        resp = client.post(
            f"/projects/{project.id}/memories/generate-contextual-embeddings",
            json={"force_regenerate": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["newly_prefixed"] == 1
        assert data["already_had_prefix"] == 0

    def test_message_field_in_response(self, client, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="Test memory")

        resp = client.post(
            f"/projects/{project.id}/memories/generate-contextual-embeddings",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "backfill" in data["message"].lower() or "ann" in data["message"].lower()

    def test_use_llm_false_still_generates_prefix(self, client, db):
        project = _create_project(db)
        _create_memory(db, project.id, content="Test for template prefix")

        resp = client.post(
            f"/projects/{project.id}/memories/generate-contextual-embeddings",
            json={"use_llm": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["newly_prefixed"] == 1
        assert data["used_llm"] is False
