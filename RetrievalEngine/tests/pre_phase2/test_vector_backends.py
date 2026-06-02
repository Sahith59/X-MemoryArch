"""
Unit tests for SQLiteExactBackend.

Covers:
  - name property
  - search returns top-k sorted by cosine similarity
  - search respects allowed_ids filter
  - search with empty allowed_ids returns []
  - search with no embeddings in DB returns []
  - identical query and memory vectors → similarity ≈ 1.0
  - orthogonal vectors → similarity ≈ 0.0
  - upsert writes normalized bytes to Memory.embedding
  - delete clears embedding
  - batch_backfill stores all vectors
  - stats returns total_indexed count
  - VectorBackend Protocol satisfied
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
from app.services.vector_backends.base import VectorBackend


def _make_embedding(values: list[float]) -> bytes:
    """Create normalized float32 bytes embedding."""
    vec = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tobytes()


def _rand_embedding(dim: int = 384, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def _vec_from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_ml(mock_embed, mock_entities, mock_is_technical_true):
    pass


class TestSQLiteExactBackendName:
    def test_name(self, db):
        backend = SQLiteExactBackend(db)
        assert backend.name == "sqlite_exact"

    def test_satisfies_protocol(self, db):
        backend = SQLiteExactBackend(db)
        assert isinstance(backend, VectorBackend)


class TestSearch:
    def test_empty_db_returns_empty(self, db, project):
        backend = SQLiteExactBackend(db)
        vec = [0.0] * 384
        vec[0] = 1.0
        result = backend.search(vec, top_k=10, project_id=project.id)
        assert result == []

    def test_returns_top_k(self, db, project):
        from app import crud, schemas
        mems = []
        for i in range(5):
            vec = np.zeros(384, dtype=np.float32)
            vec[i] = 1.0
            emb = vec.tobytes()
            m = crud.create_memory(db, project.id, schemas.MemoryCreate(
                type=schemas.MemoryType.insight,
                title=f"Memory {i}",
                content=f"Content {i}",
                importance=3,
                tags=[], related_files=[], related_tools=[],
                status=schemas.MemoryStatus.active,
                source_type=schemas.SourceType.ai_session,
                privacy_level=schemas.PrivacyLevel.internal,
            ), embedding=emb)
            mems.append(m)

        backend = SQLiteExactBackend(db)
        query_vec = [0.0] * 384
        query_vec[0] = 1.0  # most similar to mems[0]
        result = backend.search(query_vec, top_k=3, project_id=project.id)
        assert len(result) == 3
        assert result[0][0] == mems[0].id  # highest similarity first

    def test_scores_in_descending_order(self, db, project):
        from app import crud, schemas
        for i in range(4):
            vec = np.zeros(384, dtype=np.float32)
            vec[i] = 1.0
            m = crud.create_memory(db, project.id, schemas.MemoryCreate(
                type=schemas.MemoryType.decision,
                title=f"M{i}", content=f"C{i}",
                importance=3, tags=[], related_files=[], related_tools=[],
                status=schemas.MemoryStatus.active,
                source_type=schemas.SourceType.ai_session,
                privacy_level=schemas.PrivacyLevel.internal,
            ), embedding=vec.tobytes())

        backend = SQLiteExactBackend(db)
        query_vec = list(np.zeros(384, dtype=np.float32))
        query_vec[0] = 1.0
        result = backend.search(query_vec, top_k=10, project_id=project.id)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_identical_vector_similarity_near_one(self, db, project):
        from app import crud, schemas
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        m = crud.create_memory(db, project.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight,
            title="T", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active,
            source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ), embedding=vec.tobytes())

        backend = SQLiteExactBackend(db)
        query = list(vec)
        result = backend.search(query, top_k=1, project_id=project.id)
        assert len(result) == 1
        assert result[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_near_zero(self, db, project):
        from app import crud, schemas
        vec_stored = np.zeros(384, dtype=np.float32)
        vec_stored[0] = 1.0
        m = crud.create_memory(db, project.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight,
            title="T", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active,
            source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ), embedding=vec_stored.tobytes())

        backend = SQLiteExactBackend(db)
        vec_query = np.zeros(384, dtype=np.float32)
        vec_query[1] = 1.0  # orthogonal to stored
        result = backend.search(list(vec_query), top_k=1, project_id=project.id)
        assert result[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_allowed_ids_filter(self, db, project):
        from app import crud, schemas
        mems = []
        for i in range(3):
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = 1.0
            m = crud.create_memory(db, project.id, schemas.MemoryCreate(
                type=schemas.MemoryType.insight, title=f"M{i}", content=f"C{i}",
                importance=3, tags=[], related_files=[], related_tools=[],
                status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
                privacy_level=schemas.PrivacyLevel.internal,
            ), embedding=vec.tobytes())
            mems.append(m)

        backend = SQLiteExactBackend(db)
        query = [0.0] * 384
        query[0] = 1.0
        # Only allow first memory
        result = backend.search(query, top_k=10, project_id=project.id, allowed_ids=[mems[0].id])
        ids_returned = {mid for mid, _ in result}
        assert ids_returned == {mems[0].id}

    def test_empty_allowed_ids_returns_empty(self, db, project):
        from app import crud, schemas
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        crud.create_memory(db, project.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight, title="T", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ), embedding=vec.tobytes())
        backend = SQLiteExactBackend(db)
        result = backend.search([1.0] + [0.0]*383, top_k=10, project_id=project.id, allowed_ids=[])
        assert result == []

    def test_no_cross_project_leakage(self, db):
        from app import crud, schemas
        p1 = crud.create_project(db, schemas.ProjectCreate(name="P1", description=""))
        p2 = crud.create_project(db, schemas.ProjectCreate(name="P2", description=""))
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        crud.create_memory(db, p1.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight, title="P1 Memory", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ), embedding=vec.tobytes())

        backend = SQLiteExactBackend(db)
        result = backend.search(list(vec), top_k=10, project_id=p2.id)
        assert result == []


class TestUpsertAndDelete:
    def test_upsert_writes_embedding(self, db, project):
        from app import crud, schemas
        m = crud.create_memory(db, project.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight, title="T", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ))
        assert m.embedding is None

        backend = SQLiteExactBackend(db)
        vec = [1.0] + [0.0] * 383
        backend.upsert(m.id, vec, project.id, "all-MiniLM-L6-v2", 384)
        db.refresh(m)
        assert m.embedding is not None
        stored = np.frombuffer(m.embedding, dtype=np.float32)
        assert stored[0] == pytest.approx(1.0, abs=1e-5)

    def test_delete_clears_embedding(self, db, project):
        from app import crud, schemas
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        m = crud.create_memory(db, project.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight, title="T", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ), embedding=vec.tobytes())

        backend = SQLiteExactBackend(db)
        backend.delete(m.id)
        db.refresh(m)
        assert m.embedding is None


class TestBatchBackfill:
    def test_backfill_stores_embeddings(self, db, project):
        from app import crud, schemas
        mems = []
        for i in range(3):
            m = crud.create_memory(db, project.id, schemas.MemoryCreate(
                type=schemas.MemoryType.insight, title=f"T{i}", content=f"C{i}",
                importance=3, tags=[], related_files=[], related_tools=[],
                status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
                privacy_level=schemas.PrivacyLevel.internal,
            ))
            mems.append(m)

        backend = SQLiteExactBackend(db)
        rows = [(m.id, [float(i)] + [0.0]*383, {}) for i, m in enumerate(mems)]
        backend.batch_backfill(rows)

        for m in mems:
            db.refresh(m)
            assert m.embedding is not None


class TestStats:
    def test_stats_returns_dict(self, db, project):
        backend = SQLiteExactBackend(db)
        s = backend.stats()
        assert isinstance(s, dict)
        assert "total_indexed" in s
        assert s["backend"] == "sqlite_exact"
        assert s["ann"] is False

    def test_stats_counts_embedded_memories(self, db, project):
        from app import crud, schemas
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        for _ in range(3):
            crud.create_memory(db, project.id, schemas.MemoryCreate(
                type=schemas.MemoryType.insight, title="T", content="C",
                importance=3, tags=[], related_files=[], related_tools=[],
                status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
                privacy_level=schemas.PrivacyLevel.internal,
            ), embedding=vec.tobytes())
        # One without embedding
        crud.create_memory(db, project.id, schemas.MemoryCreate(
            type=schemas.MemoryType.insight, title="NoEmb", content="C",
            importance=3, tags=[], related_files=[], related_tools=[],
            status=schemas.MemoryStatus.active, source_type=schemas.SourceType.ai_session,
            privacy_level=schemas.PrivacyLevel.internal,
        ))
        backend = SQLiteExactBackend(db)
        assert backend.stats()["total_indexed"] == 3
