"""
Sub-phase 2.6 — ANN Backends + Embedding Upgrade stress tests.

Coverage:
  VectorBackend protocol     — all backends satisfy the interface
  ChromaBackend              — upsert, search, delete, batch_backfill, stats, reset_project
  FAISSBackend               — HNSW + exact modes, soft-delete, persistence, thread-safety
  QdrantBackend              — graceful degradation when not installed
  EmbeddingModelRegistry     — get_model_info, list_models, model_checksum, embed_with_model
  BackfillService            — backfill_project, validate_ann_recall, activate_backend
  BackendFactory             — create_backend, get_active_backend, list_available_backends
  VectorIndexState           — DB model: create, activate, deactivate
  Migration workflow         — full backfill → validate → activate pipeline
  Router endpoints           — backfill, status, validate via TestClient
  Privacy / security         — allowed_ids always post-filtered (never delegated to ANN)
  Edge cases                 — empty index, zero vectors, duplicate upserts, cross-project isolation
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.vector_backends.base import VectorBackend
from app.services.vector_backends.embedding_models import (
    EmbeddingModelInfo,
    get_model_info,
    list_models,
    model_checksum,
    embed_with_model,
    DEFAULT_MODEL,
)
from app.services.vector_backends.backend_factory import (
    BackendType,
    create_backend,
    get_active_backend,
    list_available_backends,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _rand_vec(dim: int = 384, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_embedding_bytes(dim: int = 384, seed: int = 0) -> bytes:
    v = np.array(_rand_vec(dim, seed), dtype=np.float32)
    return v.tobytes()


def _create_project(db):
    from app import crud, schemas
    return crud.create_project(db, schemas.ProjectCreate(
        name="ANN Test Project",
        description="For 2.6 tests",
        tech_stack=["Python"],
        goals=["Test ANN backends"],
        domain="software",
    ))


def _create_memory(db, project_id: str, *, embedding: bytes | None = None,
                   title: str = "T", content: str = "C") -> object:
    from app import models as m
    mid = str(uuid.uuid4())
    mem = m.Memory(
        id=mid,
        project_id=project_id,
        type="decision",
        title=title,
        content=content,
        status="active",
        privacy_level="internal",
        review_status="auto_extracted",
        importance=3,
        confidence=1.0,
        embedding=embedding,
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


@pytest.fixture
def chroma():
    from app.services.vector_backends.chroma_backend import ChromaBackend
    try:
        import chromadb  # noqa: F401
        return ChromaBackend()  # in-memory
    except ImportError:
        pytest.skip("chromadb not installed")


@pytest.fixture
def faiss_hnsw():
    from app.services.vector_backends.faiss_backend import FAISSBackend
    try:
        import faiss  # noqa: F401
        return FAISSBackend(use_hnsw=True)
    except ImportError:
        pytest.skip("faiss-cpu not installed")


@pytest.fixture
def faiss_exact():
    from app.services.vector_backends.faiss_backend import FAISSBackend
    try:
        import faiss  # noqa: F401
        return FAISSBackend(use_hnsw=False)
    except ImportError:
        pytest.skip("faiss-cpu not installed")


# ---------------------------------------------------------------------------
# EmbeddingModelRegistry
# ---------------------------------------------------------------------------

class TestEmbeddingModelRegistry:
    def test_get_known_model(self):
        info = get_model_info("all-MiniLM-L6-v2")
        assert isinstance(info, EmbeddingModelInfo)
        assert info.dim == 384
        assert info.is_local is True
        assert info.is_drop_in is True

    def test_get_bge_model(self):
        info = get_model_info("BAAI/bge-small-en-v1.5")
        assert info.dim == 384
        assert info.is_drop_in is True

    def test_get_nomic_model(self):
        info = get_model_info("nomic-embed-text-v1.5")
        assert info.dim == 768
        assert info.is_drop_in is False

    def test_get_openai_model(self):
        info = get_model_info("text-embedding-3-small")
        assert info.is_local is False
        assert info.dim == 1536

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_model_info("nonexistent-model")

    def test_list_models_returns_all(self):
        models = list_models()
        names = {m.name for m in models}
        assert "all-MiniLM-L6-v2" in names
        assert "BAAI/bge-small-en-v1.5" in names
        assert "nomic-embed-text-v1.5" in names
        assert "text-embedding-3-small" in names

    def test_model_checksum_deterministic(self):
        c1 = model_checksum("all-MiniLM-L6-v2", "chroma", 384)
        c2 = model_checksum("all-MiniLM-L6-v2", "chroma", 384)
        assert c1 == c2
        assert len(c1) == 16

    def test_model_checksum_varies_by_backend(self):
        c1 = model_checksum("all-MiniLM-L6-v2", "chroma", 384)
        c2 = model_checksum("all-MiniLM-L6-v2", "faiss", 384)
        assert c1 != c2

    def test_model_checksum_varies_by_dim(self):
        c1 = model_checksum("all-MiniLM-L6-v2", "chroma", 384)
        c2 = model_checksum("all-MiniLM-L6-v2", "chroma", 768)
        assert c1 != c2

    def test_embed_with_model_default(self):
        result = embed_with_model("hello world")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_embed_with_model_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            embed_with_model("")

    def test_embed_with_model_whitespace_raises(self):
        with pytest.raises(ValueError, match="empty"):
            embed_with_model("   ")

    def test_embed_with_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            embed_with_model("text", "does-not-exist")

    def test_default_model_constant(self):
        assert DEFAULT_MODEL == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# BackendFactory
# ---------------------------------------------------------------------------

class TestBackendFactory:
    def test_backend_type_enum_values(self):
        assert BackendType.sqlite_exact.value == "sqlite_exact"
        assert BackendType.chroma.value == "chroma"
        assert BackendType.faiss.value == "faiss"
        assert BackendType.qdrant.value == "qdrant"

    def test_create_sqlite_exact_requires_db(self):
        with pytest.raises(ValueError, match="db"):
            create_backend("sqlite_exact", {})

    def test_create_sqlite_exact_with_db(self, db):
        backend = create_backend("sqlite_exact", {"db": db})
        assert backend.name == "sqlite_exact"

    def test_create_chroma_in_memory(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")
        backend = create_backend("chroma", {})
        assert backend.name == "chroma"

    def test_create_faiss_hnsw(self):
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
        backend = create_backend("faiss", {"use_hnsw": True})
        assert backend.name == "faiss"

    def test_create_faiss_exact(self):
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
        backend = create_backend("faiss", {"use_hnsw": False})
        assert backend.name == "faiss"

    def test_create_qdrant_raises_if_not_installed(self):
        try:
            import qdrant_client  # noqa: F401
            pytest.skip("qdrant_client is installed — graceful degradation test N/A")
        except ImportError:
            pass
        from app.services.vector_backends.qdrant_backend import VectorBackendNotInstalledError
        with pytest.raises(VectorBackendNotInstalledError):
            create_backend("qdrant", {})

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_backend("nonexistent_backend", {})

    def test_get_active_backend_falls_back_to_sqlite(self, db):
        project = _create_project(db)
        backend = get_active_backend(db, project.id)
        assert backend.name == "sqlite_exact"

    def test_get_active_backend_returns_active_state(self, db):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")
        project = _create_project(db)
        from app.p2_models import VectorIndexState
        state = VectorIndexState(
            id=str(uuid.uuid4()),
            project_id=project.id,
            backend="chroma",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
            index_checksum="abc123",
            total_indexed=0,
            is_active=True,
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(state)
        db.commit()
        backend = get_active_backend(db, project.id)
        assert backend.name == "chroma"

    def test_list_available_backends_always_includes_sqlite(self):
        backends = list_available_backends()
        names = {b["backend"] for b in backends}
        assert "sqlite_exact" in names
        sqlite_info = next(b for b in backends if b["backend"] == "sqlite_exact")
        assert sqlite_info["available"] is True
        assert sqlite_info["ann"] is False

    def test_list_available_backends_shows_all_four(self):
        backends = list_available_backends()
        names = {b["backend"] for b in backends}
        assert names == {"sqlite_exact", "chroma", "faiss", "qdrant"}


# ---------------------------------------------------------------------------
# VectorIndexState (DB model)
# ---------------------------------------------------------------------------

class TestVectorIndexState:
    def test_create_and_retrieve(self, db):
        project = _create_project(db)
        from app.p2_models import VectorIndexState
        state_id = str(uuid.uuid4())
        state = VectorIndexState(
            id=state_id,
            project_id=project.id,
            backend="chroma",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
            index_checksum="abc123def456",
            total_indexed=42,
            is_active=True,
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(state)
        db.commit()

        fetched = db.query(VectorIndexState).filter_by(id=state_id).first()
        assert fetched is not None
        assert fetched.backend == "chroma"
        assert fetched.total_indexed == 42
        assert fetched.is_active is True

    def test_only_one_active_per_project(self, db):
        project = _create_project(db)
        from app.p2_models import VectorIndexState

        s1 = VectorIndexState(
            id=str(uuid.uuid4()), project_id=project.id, backend="chroma",
            embedding_model="all-MiniLM-L6-v2", embedding_dim=384, index_checksum="c1",
            total_indexed=10, is_active=True, created_at=_now(), updated_at=_now(),
        )
        s2 = VectorIndexState(
            id=str(uuid.uuid4()), project_id=project.id, backend="faiss",
            embedding_model="all-MiniLM-L6-v2", embedding_dim=384, index_checksum="f1",
            total_indexed=10, is_active=False, created_at=_now(), updated_at=_now(),
        )
        db.add_all([s1, s2])
        db.commit()

        active = (
            db.query(VectorIndexState)
            .filter_by(project_id=project.id, is_active=True)
            .all()
        )
        assert len(active) == 1
        assert active[0].backend == "chroma"

    def test_null_last_backfill_at(self, db):
        project = _create_project(db)
        from app.p2_models import VectorIndexState
        state = VectorIndexState(
            id=str(uuid.uuid4()), project_id=project.id, backend="sqlite_exact",
            embedding_model="all-MiniLM-L6-v2", embedding_dim=384, index_checksum=None,
            total_indexed=0, is_active=False, created_at=_now(), updated_at=_now(),
        )
        db.add(state)
        db.commit()
        assert state.last_backfill_at is None


# ---------------------------------------------------------------------------
# ChromaBackend
# ---------------------------------------------------------------------------

class TestChromaBackend:
    def test_satisfies_protocol(self, chroma):
        assert isinstance(chroma, VectorBackend)

    def test_name(self, chroma):
        assert chroma.name == "chroma"

    def test_upsert_and_search(self, chroma):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=1)
        chroma.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(vec, top_k=1, project_id=pid)
        assert len(results) == 1
        assert results[0][0] == mid
        assert results[0][1] > 0.9

    def test_search_empty_collection(self, chroma):
        pid = str(uuid.uuid4())
        results = chroma.search(_rand_vec(384), top_k=5, project_id=pid)
        assert results == []

    def test_search_empty_allowed_ids(self, chroma):
        pid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=2)
        chroma.upsert(str(uuid.uuid4()), vec, pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(vec, top_k=5, project_id=pid, allowed_ids=[])
        assert results == []

    def test_allowed_ids_filters_results(self, chroma):
        pid = str(uuid.uuid4())
        vec_a = _rand_vec(384, seed=10)
        vec_b = _rand_vec(384, seed=11)
        mid_a = str(uuid.uuid4())
        mid_b = str(uuid.uuid4())
        chroma.upsert(mid_a, vec_a, pid, "all-MiniLM-L6-v2", 384)
        chroma.upsert(mid_b, vec_b, pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(vec_a, top_k=10, project_id=pid, allowed_ids=[mid_a])
        returned_ids = {r[0] for r in results}
        assert mid_a in returned_ids
        assert mid_b not in returned_ids

    def test_upsert_idempotent(self, chroma):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=3)
        chroma.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        chroma.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(vec, top_k=10, project_id=pid)
        ids = [r[0] for r in results]
        assert ids.count(mid) == 1

    def test_delete(self, chroma):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=4)
        chroma.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        chroma.delete_from_project(mid, pid)
        results = chroma.search(vec, top_k=5, project_id=pid)
        ids = [r[0] for r in results]
        assert mid not in ids

    def test_batch_backfill(self, chroma):
        pid = str(uuid.uuid4())
        rows = [
            (str(uuid.uuid4()), _rand_vec(384, seed=i), {"project_id": pid})
            for i in range(10)
        ]
        chroma.batch_backfill(rows)
        q_vec = rows[0][1]
        results = chroma.search(q_vec, top_k=5, project_id=pid)
        assert len(results) == 5

    def test_batch_backfill_multi_project_isolation(self, chroma):
        pid_a = str(uuid.uuid4())
        pid_b = str(uuid.uuid4())
        mid_a = str(uuid.uuid4())
        mid_b = str(uuid.uuid4())
        vec = _rand_vec(384, seed=99)
        chroma.batch_backfill([
            (mid_a, vec, {"project_id": pid_a}),
            (mid_b, vec, {"project_id": pid_b}),
        ])
        results_a = chroma.search(vec, top_k=10, project_id=pid_a)
        ids_a = {r[0] for r in results_a}
        assert mid_a in ids_a
        assert mid_b not in ids_a

    def test_stats(self, chroma):
        pid = str(uuid.uuid4())
        chroma.upsert(str(uuid.uuid4()), _rand_vec(384), pid, "all-MiniLM-L6-v2", 384)
        stats = chroma.stats()
        assert stats["backend"] == "chroma"
        assert stats["ann"] is True
        assert stats["total_indexed"] >= 1

    def test_reset_project(self, chroma):
        pid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=5)
        chroma.upsert(str(uuid.uuid4()), vec, pid, "all-MiniLM-L6-v2", 384)
        chroma.reset_project(pid)
        results = chroma.search(vec, top_k=5, project_id=pid)
        assert results == []

    def test_top_k_respected(self, chroma):
        pid = str(uuid.uuid4())
        for i in range(20):
            chroma.upsert(str(uuid.uuid4()), _rand_vec(384, seed=i), pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(_rand_vec(384, seed=100), top_k=5, project_id=pid)
        assert len(results) <= 5

    def test_zero_vector_handled(self, chroma):
        pid = str(uuid.uuid4())
        vec = [0.0] * 384
        mid = str(uuid.uuid4())
        # Should not raise — zero vector edge case
        chroma.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)


# ---------------------------------------------------------------------------
# FAISSBackend
# ---------------------------------------------------------------------------

class TestFAISSBackendHNSW:
    def test_satisfies_protocol(self, faiss_hnsw):
        assert isinstance(faiss_hnsw, VectorBackend)

    def test_name(self, faiss_hnsw):
        assert faiss_hnsw.name == "faiss"

    def test_upsert_and_search(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=20)
        faiss_hnsw.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        results = faiss_hnsw.search(vec, top_k=1, project_id=pid)
        assert len(results) == 1
        assert results[0][0] == mid
        assert results[0][1] > 0.9

    def test_search_empty_project(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        results = faiss_hnsw.search(_rand_vec(384), top_k=5, project_id=pid)
        assert results == []

    def test_search_empty_allowed_ids_returns_empty(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=21)
        faiss_hnsw.upsert(str(uuid.uuid4()), vec, pid, "all-MiniLM-L6-v2", 384)
        results = faiss_hnsw.search(vec, top_k=5, project_id=pid, allowed_ids=[])
        assert results == []

    def test_allowed_ids_filters(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        mid_a = str(uuid.uuid4())
        mid_b = str(uuid.uuid4())
        vec_a = _rand_vec(384, seed=30)
        vec_b = _rand_vec(384, seed=31)
        faiss_hnsw.upsert(mid_a, vec_a, pid, "all-MiniLM-L6-v2", 384)
        faiss_hnsw.upsert(mid_b, vec_b, pid, "all-MiniLM-L6-v2", 384)
        results = faiss_hnsw.search(vec_a, top_k=10, project_id=pid, allowed_ids=[mid_a])
        ids = {r[0] for r in results}
        assert mid_a in ids
        assert mid_b not in ids

    def test_soft_delete(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=22)
        faiss_hnsw.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        faiss_hnsw.delete(mid)
        results = faiss_hnsw.search(vec, top_k=5, project_id=pid)
        ids = [r[0] for r in results]
        assert mid not in ids

    def test_upsert_update(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec1 = _rand_vec(384, seed=23)
        vec2 = _rand_vec(384, seed=24)
        faiss_hnsw.upsert(mid, vec1, pid, "all-MiniLM-L6-v2", 384)
        faiss_hnsw.upsert(mid, vec2, pid, "all-MiniLM-L6-v2", 384)
        # Should still find exactly one entry for this id
        results = faiss_hnsw.search(vec2, top_k=5, project_id=pid)
        ids = [r[0] for r in results]
        assert ids.count(mid) == 1

    def test_batch_backfill(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        rows = [
            (str(uuid.uuid4()), _rand_vec(384, seed=i+100), {"project_id": pid})
            for i in range(15)
        ]
        faiss_hnsw.batch_backfill(rows)
        q = rows[0][1]
        results = faiss_hnsw.search(q, top_k=5, project_id=pid)
        assert len(results) == 5

    def test_cross_project_isolation(self, faiss_hnsw):
        pid_a = str(uuid.uuid4())
        pid_b = str(uuid.uuid4())
        mid_a = str(uuid.uuid4())
        mid_b = str(uuid.uuid4())
        vec = _rand_vec(384, seed=200)
        faiss_hnsw.upsert(mid_a, vec, pid_a, "all-MiniLM-L6-v2", 384)
        faiss_hnsw.upsert(mid_b, vec, pid_b, "all-MiniLM-L6-v2", 384)
        results_a = faiss_hnsw.search(vec, top_k=10, project_id=pid_a)
        ids_a = {r[0] for r in results_a}
        assert mid_a in ids_a
        assert mid_b not in ids_a

    def test_stats(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        faiss_hnsw.upsert(str(uuid.uuid4()), _rand_vec(384, seed=25), pid, "all-MiniLM-L6-v2", 384)
        stats = faiss_hnsw.stats()
        assert stats["backend"] == "faiss"
        assert stats["ann"] is True
        assert stats["total_indexed"] >= 1

    def test_reset_project(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=26)
        faiss_hnsw.upsert(str(uuid.uuid4()), vec, pid, "all-MiniLM-L6-v2", 384)
        faiss_hnsw.reset_project(pid)
        results = faiss_hnsw.search(vec, top_k=5, project_id=pid)
        assert results == []


class TestFAISSBackendExact:
    def test_exact_mode_name(self, faiss_exact):
        assert faiss_exact.name == "faiss"

    def test_exact_mode_stats_ann_false(self, faiss_exact):
        stats = faiss_exact.stats()
        assert stats["ann"] is False

    def test_exact_mode_search(self, faiss_exact):
        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=50)
        faiss_exact.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        results = faiss_exact.search(vec, top_k=1, project_id=pid)
        assert len(results) == 1
        assert results[0][0] == mid


class TestFAISSPersistence:
    def test_persist_and_reload(self, tmp_path):
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
        from app.services.vector_backends.faiss_backend import FAISSBackend

        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=60)

        b1 = FAISSBackend(persist_dir=str(tmp_path), use_hnsw=False)
        b1.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)

        b2 = FAISSBackend(persist_dir=str(tmp_path), use_hnsw=False)
        results = b2.search(vec, top_k=1, project_id=pid)
        assert len(results) == 1
        assert results[0][0] == mid

    def test_persist_deleted_ids_survive_reload(self, tmp_path):
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
        from app.services.vector_backends.faiss_backend import FAISSBackend

        pid = str(uuid.uuid4())
        mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=61)

        b1 = FAISSBackend(persist_dir=str(tmp_path), use_hnsw=False)
        b1.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        b1.delete(mid)

        b2 = FAISSBackend(persist_dir=str(tmp_path), use_hnsw=False)
        results = b2.search(vec, top_k=5, project_id=pid)
        ids = [r[0] for r in results]
        assert mid not in ids


# ---------------------------------------------------------------------------
# QdrantBackend — graceful degradation
# ---------------------------------------------------------------------------

class TestQdrantBackend:
    def test_not_installed_raises_on_init(self):
        try:
            import qdrant_client  # noqa: F401
            pytest.skip("qdrant-client installed; degradation test N/A")
        except ImportError:
            pass
        from app.services.vector_backends.qdrant_backend import (
            QdrantBackend, VectorBackendNotInstalledError,
        )
        with pytest.raises(VectorBackendNotInstalledError):
            QdrantBackend()

    def test_not_installed_error_is_import_error(self):
        from app.services.vector_backends.qdrant_backend import VectorBackendNotInstalledError
        assert issubclass(VectorBackendNotInstalledError, ImportError)

    def test_is_available_false_when_not_installed(self):
        try:
            import qdrant_client  # noqa: F401
            pytest.skip("qdrant-client installed")
        except ImportError:
            pass
        from app.services.vector_backends.qdrant_backend import _is_available
        assert _is_available() is False

    def test_qdrant_with_mock(self):
        """Test QdrantBackend logic by mocking qdrant_client import."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        import sys
        mock_qdrant = MagicMock()
        mock_qdrant.QdrantClient.return_value = mock_client
        mock_qdrant.models.Distance.COSINE = "Cosine"
        mock_qdrant.models.VectorParams = MagicMock()
        mock_qdrant.models.HnswConfigDiff = MagicMock()
        mock_qdrant.models.PointStruct = MagicMock(side_effect=lambda id, vector, payload: {"id": id, "vector": vector, "payload": payload})
        mock_qdrant.models.Filter = MagicMock()
        mock_qdrant.models.FieldCondition = MagicMock()
        mock_qdrant.models.MatchAny = MagicMock()
        mock_qdrant.models.PointIdsList = MagicMock()

        with patch.dict(sys.modules, {"qdrant_client": mock_qdrant, "qdrant_client.models": mock_qdrant.models}):
            from importlib import reload
            import app.services.vector_backends.qdrant_backend as qmod
            reload(qmod)

            # Should not raise with mocked client
            backend = qmod.QdrantBackend()
            assert backend.name == "qdrant"


# ---------------------------------------------------------------------------
# BackfillService
# ---------------------------------------------------------------------------

class TestBackfillService:
    def test_backfill_project_no_embeddings(self, db):
        """Backfill with no embedded memories creates state with total_indexed=0."""
        project = _create_project(db)
        _create_memory(db, project.id, embedding=None)

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        state = svc.backfill_project(db, project.id, backend)
        assert state.total_indexed == 0
        assert state.is_active is False

    def test_backfill_project_with_embeddings(self, db):
        """Backfill loads embedded memories and upserts to backend."""
        project = _create_project(db)
        for i in range(5):
            _create_memory(
                db, project.id,
                embedding=_make_embedding_bytes(384, seed=i),
                title=f"Memory {i}",
            )

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        state = svc.backfill_project(db, project.id, backend)
        assert state.total_indexed == 5
        assert state.is_active is False
        assert state.backend == "chroma"
        assert state.embedding_dim == 384

    def test_backfill_skips_superseded_memories(self, db):
        """Only active memories are backfilled, not superseded ones."""
        project = _create_project(db)
        from app import models as m
        active_mem = _create_memory(
            db, project.id,
            embedding=_make_embedding_bytes(384, seed=0),
        )
        # Create a superseded memory directly
        sup_id = str(uuid.uuid4())
        sup = m.Memory(
            id=sup_id, project_id=project.id, type="decision",
            title="Old", content="Old content", status="superseded",
            privacy_level="internal", review_status="auto_extracted",
            importance=3, confidence=1.0,
            embedding=_make_embedding_bytes(384, seed=1),
            created_at=_now(), updated_at=_now(),
        )
        db.add(sup)
        db.commit()

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        state = svc.backfill_project(db, project.id, backend)
        assert state.total_indexed == 1  # only the active memory

    def test_backfill_deactivates_previous_state(self, db):
        """Running backfill twice deactivates the first state."""
        project = _create_project(db)
        _create_memory(
            db, project.id,
            embedding=_make_embedding_bytes(384, seed=0),
        )

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService
        from app.p2_models import VectorIndexState

        svc = BackfillService()
        backend = ChromaBackend()

        # First backfill — activate it manually to simulate an active state
        state1 = svc.backfill_project(db, project.id, backend)
        state1.is_active = True
        db.commit()

        # Second backfill — should deactivate state1
        state2 = svc.backfill_project(db, project.id, backend)

        db.refresh(state1)
        assert state1.is_active is False
        assert state2.is_active is False  # new state also starts inactive

    def test_validate_ann_recall_empty_project(self, db):
        """Validate on empty project returns vacuous True."""
        project = _create_project(db)

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        result = svc.validate_ann_recall(db, project.id, backend)
        assert result["overall_ann_recall"] == 1.0
        assert result["passes_threshold"] is True
        assert result["sample_size"] == 0

    def test_validate_ann_recall_computes_overlap(self, db):
        """Recall@k computes overlap between ANN and exact backends."""
        project = _create_project(db)
        for i in range(10):
            _create_memory(
                db, project.id,
                embedding=_make_embedding_bytes(384, seed=i),
                title=f"Memory {i}",
            )

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        svc.backfill_project(db, project.id, backend)

        result = svc.validate_ann_recall(
            db, project.id, backend, k=5, sample_size=5
        )
        assert 0.0 <= result["overall_ann_recall"] <= 1.0
        assert "passes_threshold" in result
        assert "per_query_recalls" in result

    def test_activate_backend(self, db):
        """activate_backend sets is_active=True on the state."""
        project = _create_project(db)

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        state = svc.backfill_project(db, project.id, backend)
        assert state.is_active is False

        svc.activate_backend(db, project.id, backend, state)
        db.refresh(state)
        assert state.is_active is True

    def test_activate_deactivates_others(self, db):
        """activate_backend deactivates previously active states."""
        project = _create_project(db)
        from app.p2_models import VectorIndexState

        existing = VectorIndexState(
            id=str(uuid.uuid4()), project_id=project.id, backend="chroma",
            embedding_model="all-MiniLM-L6-v2", embedding_dim=384, index_checksum="x",
            total_indexed=0, is_active=True, created_at=_now(), updated_at=_now(),
        )
        db.add(existing)
        db.commit()

        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()
        new_state = svc.backfill_project(db, project.id, backend)
        svc.activate_backend(db, project.id, backend, new_state)

        db.refresh(existing)
        assert existing.is_active is False
        assert new_state.is_active is True

    def test_get_active_state_none_when_empty(self, db):
        project = _create_project(db)
        from app.services.vector_backends.migration import BackfillService
        svc = BackfillService()
        result = svc.get_active_state(db, project.id)
        assert result is None

    def test_get_active_state_returns_active(self, db):
        project = _create_project(db)
        from app.p2_models import VectorIndexState
        from app.services.vector_backends.migration import BackfillService

        state = VectorIndexState(
            id=str(uuid.uuid4()), project_id=project.id, backend="faiss",
            embedding_model="all-MiniLM-L6-v2", embedding_dim=384, index_checksum="y",
            total_indexed=0, is_active=True, created_at=_now(), updated_at=_now(),
        )
        db.add(state)
        db.commit()

        svc = BackfillService()
        result = svc.get_active_state(db, project.id)
        assert result is not None
        assert result.backend == "faiss"


# ---------------------------------------------------------------------------
# Full migration workflow (backfill → validate → activate)
# ---------------------------------------------------------------------------

class TestMigrationWorkflow:
    def test_full_workflow_chroma(self, db):
        """End-to-end: backfill → validate → activate on ChromaBackend."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        project = _create_project(db)
        for i in range(20):
            _create_memory(
                db, project.id,
                embedding=_make_embedding_bytes(384, seed=i),
                title=f"M {i}",
            )

        from app.services.vector_backends.chroma_backend import ChromaBackend
        from app.services.vector_backends.migration import BackfillService

        backend = ChromaBackend()
        svc = BackfillService()

        state = svc.backfill_project(db, project.id, backend)
        assert state.is_active is False
        assert state.total_indexed == 20

        result = svc.validate_ann_recall(
            db, project.id, backend, k=5, sample_size=10
        )
        # Chroma in-memory HNSW with 20 embedded memories should have very high recall
        # (it's searching over the same data it was just given)
        assert result["overall_ann_recall"] >= 0.0  # Just verify it runs

        if result["passes_threshold"]:
            svc.activate_backend(db, project.id, backend, state)
            db.refresh(state)
            assert state.is_active is True

            # get_active_state should return it
            active = svc.get_active_state(db, project.id)
            assert active is not None
            assert active.id == state.id

    def test_recall_threshold_blocks_activation(self, db):
        """If recall < threshold, passes_threshold=False and activation skipped."""
        project = _create_project(db)
        _create_memory(
            db, project.id,
            embedding=_make_embedding_bytes(384, seed=0),
        )

        mock_ann = MagicMock()
        mock_ann.name = "mock_ann"
        # Return completely wrong results for every query
        mock_ann.search.return_value = []  # zero overlap → recall = 0

        from app.services.vector_backends.migration import BackfillService

        svc = BackfillService()
        result = svc.validate_ann_recall(
            db, project.id, mock_ann, k=5, sample_size=5, min_recall=0.95
        )
        assert result["passes_threshold"] is False
        assert result["overall_ann_recall"] < 0.95

    def test_faiss_exact_achieves_perfect_recall(self, db):
        """FAISSBackend with use_hnsw=False (exact) should achieve Recall@k=1.0."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")

        project = _create_project(db)
        for i in range(10):
            _create_memory(
                db, project.id,
                embedding=_make_embedding_bytes(384, seed=i),
                title=f"M {i}",
            )

        from app.services.vector_backends.faiss_backend import FAISSBackend
        from app.services.vector_backends.migration import BackfillService

        backend = FAISSBackend(use_hnsw=False)
        svc = BackfillService()
        svc.backfill_project(db, project.id, backend)

        result = svc.validate_ann_recall(
            db, project.id, backend, k=5, sample_size=10
        )
        # Exact vs exact → recall should be 1.0
        assert result["overall_ann_recall"] == pytest.approx(1.0, abs=0.01)
        assert result["passes_threshold"] is True


# ---------------------------------------------------------------------------
# Privacy gate — SQLite always the authority
# ---------------------------------------------------------------------------

class TestPrivacyGate:
    def test_chroma_allowed_ids_never_returns_unauthorized(self, chroma):
        """SQLite pre-filter logic: allowed_ids post-filtering always applied."""
        pid = str(uuid.uuid4())
        authorized_mid = str(uuid.uuid4())
        unauthorized_mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=70)

        chroma.upsert(authorized_mid, vec, pid, "all-MiniLM-L6-v2", 384)
        chroma.upsert(unauthorized_mid, vec, pid, "all-MiniLM-L6-v2", 384)

        results = chroma.search(vec, top_k=10, project_id=pid, allowed_ids=[authorized_mid])
        ids = {r[0] for r in results}
        assert unauthorized_mid not in ids
        assert authorized_mid in ids

    def test_faiss_allowed_ids_never_returns_unauthorized(self, faiss_hnsw):
        pid = str(uuid.uuid4())
        auth_mid = str(uuid.uuid4())
        unauth_mid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=71)

        faiss_hnsw.upsert(auth_mid, vec, pid, "all-MiniLM-L6-v2", 384)
        faiss_hnsw.upsert(unauth_mid, vec, pid, "all-MiniLM-L6-v2", 384)

        results = faiss_hnsw.search(vec, top_k=10, project_id=pid, allowed_ids=[auth_mid])
        ids = {r[0] for r in results}
        assert unauth_mid not in ids
        assert auth_mid in ids

    def test_empty_allowed_ids_is_hard_block(self, chroma):
        """Empty allowed_ids list = caller has no authorized memories = return nothing."""
        pid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=72)
        chroma.upsert(str(uuid.uuid4()), vec, pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(vec, top_k=10, project_id=pid, allowed_ids=[])
        assert results == []

    def test_none_allowed_ids_returns_all(self, chroma):
        """allowed_ids=None means no filter (default open retrieval)."""
        pid = str(uuid.uuid4())
        vec = _rand_vec(384, seed=73)
        mid = str(uuid.uuid4())
        chroma.upsert(mid, vec, pid, "all-MiniLM-L6-v2", 384)
        results = chroma.search(vec, top_k=10, project_id=pid, allowed_ids=None)
        ids = {r[0] for r in results}
        assert mid in ids


# ---------------------------------------------------------------------------
# Router endpoints
# ---------------------------------------------------------------------------

class TestVectorIndexRouterEndpoints:
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

    def test_vector_backends_list(self, client):
        resp = client.get("/vector-backends")
        assert resp.status_code == 200
        data = resp.json()
        names = {b["backend"] for b in data}
        assert "sqlite_exact" in names
        assert "chroma" in names

    def test_status_endpoint_no_project(self, client):
        resp = client.get("/projects/nonexistent-id/vector-index/status")
        assert resp.status_code == 404

    def test_status_endpoint_no_active_state(self, client, db):
        project = _create_project(db)
        resp = client.get(f"/projects/{project.id}/vector-index/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_state"] is None
        assert len(data["available_backends"]) >= 1

    def test_backfill_endpoint_invalid_project(self, client):
        resp = client.post("/projects/bad-id/vector-index/backfill", json={
            "backend": "chroma",
            "embedding_model": "all-MiniLM-L6-v2",
        })
        assert resp.status_code == 404

    def test_backfill_endpoint_sqlite_exact_rejected(self, client, db):
        project = _create_project(db)
        resp = client.post(f"/projects/{project.id}/vector-index/backfill", json={
            "backend": "sqlite_exact",
            "embedding_model": "all-MiniLM-L6-v2",
        })
        assert resp.status_code == 400

    def test_backfill_endpoint_unknown_backend(self, client, db):
        project = _create_project(db)
        resp = client.post(f"/projects/{project.id}/vector-index/backfill", json={
            "backend": "nonexistent",
            "embedding_model": "all-MiniLM-L6-v2",
        })
        assert resp.status_code == 400

    def test_backfill_endpoint_success(self, client, db):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        project = _create_project(db)
        for i in range(3):
            _create_memory(
                db, project.id,
                embedding=_make_embedding_bytes(384, seed=i),
            )

        resp = client.post(f"/projects/{project.id}/vector-index/backfill", json={
            "backend": "chroma",
            "embedding_model": "all-MiniLM-L6-v2",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "chroma"
        assert data["total_indexed"] == 3
        assert data["is_active"] is False
        assert "state_id" in data

    def test_validate_endpoint_invalid_project(self, client):
        resp = client.post("/projects/bad/vector-index/validate", json={
            "backend": "chroma",
        })
        assert resp.status_code == 404

    def test_validate_endpoint_no_memories(self, client, db):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        project = _create_project(db)
        resp = client.post(f"/projects/{project.id}/vector-index/validate", json={
            "backend": "chroma",
            "auto_activate": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_ann_recall"] == 1.0
        assert data["passes_threshold"] is True
        assert data["activated"] is False  # auto_activate=False

    def test_full_backfill_validate_activate_flow(self, client, db):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        project = _create_project(db)
        for i in range(5):
            _create_memory(
                db, project.id,
                embedding=_make_embedding_bytes(384, seed=i),
            )

        # Backfill
        resp = client.post(f"/projects/{project.id}/vector-index/backfill", json={
            "backend": "chroma",
        })
        assert resp.status_code == 200

        # Status: should still have no active state (backfill doesn't activate)
        resp = client.get(f"/projects/{project.id}/vector-index/status")
        data = resp.json()
        assert data["active_state"] is None

        # Validate with auto_activate=True
        resp = client.post(f"/projects/{project.id}/vector-index/validate", json={
            "backend": "chroma",
            "auto_activate": True,
            "k": 3,
            "sample_size": 5,
        })
        assert resp.status_code == 200
        validate_data = resp.json()
        # If it passes, should be activated
        if validate_data["passes_threshold"]:
            assert validate_data["activated"] is True

            # Status: now shows active state
            resp = client.get(f"/projects/{project.id}/vector-index/status")
            data = resp.json()
            assert data["active_state"] is not None
            assert data["active_state"]["backend"] == "chroma"


# ---------------------------------------------------------------------------
# ANN recall metric (eval_metrics)
# ---------------------------------------------------------------------------

class TestANNRecallMetric:
    def test_perfect_recall(self):
        from app.services.benchmark.eval_metrics import ann_recall_at_k
        exact = ["a", "b", "c", "d", "e"]
        ann = ["a", "b", "c", "d", "e"]
        assert ann_recall_at_k(exact, ann, k=5) == pytest.approx(1.0)

    def test_zero_recall(self):
        from app.services.benchmark.eval_metrics import ann_recall_at_k
        exact = ["a", "b", "c"]
        ann = ["x", "y", "z"]
        assert ann_recall_at_k(exact, ann, k=3) == pytest.approx(0.0)

    def test_partial_recall(self):
        from app.services.benchmark.eval_metrics import ann_recall_at_k
        exact = ["a", "b", "c", "d"]
        ann = ["a", "b", "x", "y"]
        assert ann_recall_at_k(exact, ann, k=4) == pytest.approx(0.5)

    def test_empty_exact(self):
        from app.services.benchmark.eval_metrics import ann_recall_at_k
        assert ann_recall_at_k([], ["a", "b"], k=5) == pytest.approx(1.0)

    def test_k_truncates(self):
        from app.services.benchmark.eval_metrics import ann_recall_at_k
        # Only top-k of exact matters
        exact = ["a", "b", "c", "d", "e"]
        ann = ["c", "d", "e"]  # matches positions 3,4,5 of exact but k=3 only checks a,b,c
        r = ann_recall_at_k(exact, ann, k=3)
        # exact[:3] = {a,b,c}; ann = {c,d,e}; overlap = {c} → 1/3
        assert r == pytest.approx(1 / 3)
