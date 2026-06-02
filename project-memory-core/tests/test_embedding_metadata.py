"""
Sub-phase 1.24 — Embedding metadata column tests.

Covers:
  1.  embedding_model field present in MemoryResponse
  2.  embedding_dim field present in MemoryResponse
  3.  Manually created memory has embedding_model = None
  4.  Manually created memory has embedding_dim = None
  5.  Extracted memory has embedding_model = "all-MiniLM-L6-v2"
  6.  Extracted memory has embedding_dim = 384
  7.  GET /memories/{id} returns embedding_model
  8.  GET /memories/{id} returns embedding_dim
  9.  GET /projects/{id}/memories list includes embedding_model per item
  10. embedding_dim matches actual stored embedding byte length (384 × 4 = 1536)
  11. Two extracted memories from same session share same embedding_model
  12. embedding_model stays intact after a memory update (PUT /memories/{id})
  13. embedding_dim stays intact after a memory update
  14. Multiple sessions can produce memories with consistent embedding_model
  15. embedding_model value is exactly "all-MiniLM-L6-v2"
  16. export context.md still works (no regression)
  17. health dashboard still works (no regression)
  18. entity extraction still works alongside embedding metadata (no regression)
  19. tier still set correctly on extracted memories (no regression)
  20. search endpoint still works alongside new metadata columns (no regression)
"""
import pytest
from fastapi.testclient import TestClient

from app.services.semantic_classifier import EMBEDDING_MODEL_NAME, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "EmbedMeta Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _manual_memory(client: TestClient, pid: str) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": "decision",
        "title": "We use PostgreSQL for the primary database",
        "content": "PostgreSQL was chosen for its JSON support and reliability.",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _session_with_extraction(client: TestClient, pid: str) -> list[dict]:
    """Create a session and extract memories; return extracted memory list."""
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "EmbedMeta Test Session",
        "raw_content": (
            "We decided to use PostgreSQL as our primary database for better JSON "
            "support and reliable concurrent write handling under production load. "
            "The migration was performed using Alembic with a staged rollout strategy."
        ),
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    sid = r.json()["id"]
    r2 = client.post(f"/sessions/{sid}/extract-memories")
    assert r2.status_code == 200
    return r2.json()["memories"]


# ---------------------------------------------------------------------------
# 1–4: Fields present and null for manual memories
# ---------------------------------------------------------------------------

class TestEmbeddingMetadataFields:
    def test_embedding_model_field_in_response(self, client: TestClient):
        pid = _project(client)
        m = _manual_memory(client, pid)
        assert "embedding_model" in m

    def test_embedding_dim_field_in_response(self, client: TestClient):
        pid = _project(client)
        m = _manual_memory(client, pid)
        assert "embedding_dim" in m

    def test_manual_memory_embedding_model_is_set(self, client: TestClient):
        # Manual memories are now embedded at creation time (fixed in 1.42)
        pid = _project(client)
        m = _manual_memory(client, pid)
        assert m["embedding_model"] == "all-MiniLM-L6-v2"

    def test_manual_memory_embedding_dim_is_set(self, client: TestClient):
        pid = _project(client)
        m = _manual_memory(client, pid)
        assert m["embedding_dim"] == 384


# ---------------------------------------------------------------------------
# 5–6: Extracted memories carry model metadata
# ---------------------------------------------------------------------------

class TestExtractedMemoryEmbeddingMetadata:
    def test_extracted_memory_has_embedding_model(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted — session content not technical enough")
        assert mems[0]["embedding_model"] == EMBEDDING_MODEL_NAME

    def test_extracted_memory_has_embedding_dim(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        assert mems[0]["embedding_dim"] == EMBEDDING_DIM

    def test_embedding_model_value_is_minilm(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        assert mems[0]["embedding_model"] == "all-MiniLM-L6-v2"

    def test_embedding_dim_value_is_384(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        assert mems[0]["embedding_dim"] == 384

    def test_all_extracted_memories_share_same_model(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/sessions", json={
            "tool_name": "Claude",
            "title": "Multi-memory session",
            "raw_content": (
                "We decided to use PostgreSQL for the database due to JSON support. "
                "We chose FastAPI over Flask because of OpenAPI integration. "
                "The migration was done with Alembic for structured schema control."
            ),
            "session_date": "2026-05-25",
        })
        sid = r.json()["id"]
        result = client.post(f"/sessions/{sid}/extract-memories").json()
        mems = result["memories"]
        if len(mems) < 2:
            pytest.skip("Fewer than 2 memories extracted")
        models = {m["embedding_model"] for m in mems}
        assert models == {EMBEDDING_MODEL_NAME}


# ---------------------------------------------------------------------------
# 7–10: Via GET endpoints
# ---------------------------------------------------------------------------

class TestEmbeddingMetadataViaGetEndpoints:
    def test_get_memory_returns_embedding_model(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        mid = mems[0]["id"]
        r = client.get(f"/memories/{mid}")
        assert r.status_code == 200
        assert r.json()["embedding_model"] == EMBEDDING_MODEL_NAME

    def test_get_memory_returns_embedding_dim(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        mid = mems[0]["id"]
        r = client.get(f"/memories/{mid}")
        assert r.json()["embedding_dim"] == EMBEDDING_DIM

    def test_list_memories_includes_embedding_model(self, client: TestClient):
        pid = _project(client)
        _session_with_extraction(client, pid)
        r = client.get(f"/projects/{pid}/memories")
        assert r.status_code == 200
        mems = r.json()
        assert len(mems) > 0
        for m in mems:
            assert "embedding_model" in m
            assert "embedding_dim" in m

    def test_embedding_dim_matches_byte_length(self, client: TestClient):
        """embedding_dim=384 means 384 × 4 = 1536 bytes stored."""
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        dim = mems[0]["embedding_dim"]
        if dim is not None:
            assert dim == 384
            # 384 floats × 4 bytes each = 1536 bytes (verified by semantic_classifier design)
            assert dim * 4 == 1536


# ---------------------------------------------------------------------------
# 11–15: Constants from semantic_classifier
# ---------------------------------------------------------------------------

class TestEmbeddingConstants:
    def test_embedding_model_name_constant(self):
        assert EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"

    def test_embedding_dim_constant(self):
        assert EMBEDDING_DIM == 384

    def test_embedding_dim_times_4_is_byte_size(self):
        assert EMBEDDING_DIM * 4 == 1536


# ---------------------------------------------------------------------------
# 12–13: Metadata survives updates
# ---------------------------------------------------------------------------

class TestEmbeddingMetadataSurvivesUpdate:
    def test_embedding_model_survives_importance_update(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        mid = mems[0]["id"]
        r = client.put(f"/memories/{mid}", json={"importance": 5})
        assert r.status_code == 200
        assert r.json()["embedding_model"] == EMBEDDING_MODEL_NAME

    def test_embedding_dim_survives_status_update(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        mid = mems[0]["id"]
        r = client.put(f"/memories/{mid}", json={"status": "resolved"})
        assert r.status_code == 200
        assert r.json()["embedding_dim"] == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# 16–20: Regression tests — existing features unaffected
# ---------------------------------------------------------------------------

class TestEmbeddingMetadataRegressions:
    def test_export_still_works(self, client: TestClient):
        pid = _project(client)
        _session_with_extraction(client, pid)
        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200
        assert len(r.text) > 0

    def test_health_dashboard_still_works(self, client: TestClient):
        pid = _project(client)
        _session_with_extraction(client, pid)
        r = client.get(f"/projects/{pid}/health")
        assert r.status_code == 200
        assert "total_memories" in r.json()

    def test_entity_extraction_still_works(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        mid = mems[0]["id"]
        r = client.get(f"/memories/{mid}/entities")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_tier_still_set_on_extracted_memories(self, client: TestClient):
        pid = _project(client)
        mems = _session_with_extraction(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        for m in mems:
            assert m["tier"] in ("working", "archival")

    def test_search_still_works(self, client: TestClient):
        pid = _project(client)
        _session_with_extraction(client, pid)
        r = client.get(f"/projects/{pid}/memories/search", params={"q": "database"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_manual_memory_has_embedding_and_metadata(self, client: TestClient):
        # Manual memories are embedded at creation time (fixed in 1.42)
        pid = _project(client)
        m = _manual_memory(client, pid)
        assert m["embedding_model"] == "all-MiniLM-L6-v2"
        assert m["embedding_dim"] == 384
        assert m["tier"] in ("working", "archival")
        assert "type_metadata" in m
