"""
Sub-phase 1.44 — Memory deduplication tests.

find_near_duplicates() scans all embedded memories in a project and returns
those whose cosine similarity to the new embedding exceeds the threshold (0.97).

POST /projects/{id}/memories?check_dedup=true blocks creation with HTTP 409 if
a near-duplicate exists, returning the existing memory details in the response.

POST /projects/{id}/memories/find-duplicates is the non-blocking scan endpoint
that returns a DuplicateCheckResult without creating anything.

Covers:
  Unit (pure function):
  1.  find_near_duplicates returns empty list for empty project
  2.  find_near_duplicates returns empty when embedding is all-zeros
  3.  Identical embedding → similarity = 1.0 → returned as duplicate
  4.  Orthogonal embedding → similarity near 0 → not returned as duplicate
  5.  Result sorted by similarity descending
  6.  limit parameter caps the number of returned duplicates
  7.  threshold parameter is respected (lower threshold → more matches)
  8.  Memories without embeddings are ignored by find_near_duplicates
  9.  find_near_duplicates is scoped to project_id (cross-project isolation)
  10. find_near_duplicates returns (Memory, float) tuples

  API — check_dedup=true on create:
  11. Same title+content → check_dedup=true → 409 Conflict
  12. 409 body contains existing_id
  13. 409 body contains similarity score
  14. 409 body contains existing_title
  15. Different content → check_dedup=true → 201 created
  16. Default (check_dedup not set) → creates even if duplicate exists
  17. After 409, original memory is NOT replaced (still exists with old content)
  18. Cross-project isolation: same content in different project → 201 (no dedup cross-project)
  19. Empty project → check_dedup=true → 201 (no candidates to compare against)

  API — find-duplicates endpoint:
  20. POST /find-duplicates → 200 with DuplicateCheckResult
  21. find-duplicates returns is_duplicate=false for new content
  22. find-duplicates returns is_duplicate=true for identical content
  23. find-duplicates does NOT create a memory
  24. find-duplicates returns near_duplicates list
  25. find-duplicates near_duplicate has memory and similarity fields
  26. find-duplicates threshold param respected (lower threshold → more results)
  27. find-duplicates respects project isolation
  28. find-duplicates returns 404 for unknown project
  29. find-duplicates result sorted by similarity descending
  30. After find-duplicates, project memory count is unchanged
"""
from __future__ import annotations
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud, schemas
from app.services.semantic_classifier import embed_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Dedup Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _create(client: TestClient, pid: str, title: str, content: str,
            check_dedup: bool = False) -> tuple[int, dict]:
    params = {}
    if check_dedup:
        params["check_dedup"] = "true"
    r = client.post(f"/projects/{pid}/memories", params=params, json={
        "type": "decision",
        "title": title,
        "content": content,
        "importance": 3,
    })
    return r.status_code, r.json()


def _find_dupes(client: TestClient, pid: str, title: str, content: str,
                threshold: float | None = None) -> dict:
    params: dict = {}
    if threshold is not None:
        params["threshold"] = threshold
    r = client.post(f"/projects/{pid}/memories/find-duplicates", params=params, json={
        "type": "decision",
        "title": title,
        "content": content,
    })
    assert r.status_code == 200, r.text
    return r.json()


def _unit_vec(dim: int = 384, component: int = 0) -> bytes:
    v = np.zeros(dim, dtype=np.float32)
    v[component] = 1.0
    return v.tobytes()


# ---------------------------------------------------------------------------
# 1–10: Unit tests (find_near_duplicates pure function)
# ---------------------------------------------------------------------------

class TestFindNearDuplicatesUnit:
    def test_empty_project_returns_empty(self, db: Session, client: TestClient):
        pid = _project(client)
        embedding = embed_text("We chose PostgreSQL for its ACID guarantees.")
        result = crud.find_near_duplicates(db, pid, embedding)
        assert result == []

    def test_zero_embedding_returns_empty(self, db: Session, client: TestClient):
        pid = _project(client)
        # Create one memory first
        _create(client, pid, "PostgreSQL decision", "We use PostgreSQL.")
        zero_bytes = (np.zeros(384, dtype=np.float32)).tobytes()
        result = crud.find_near_duplicates(db, pid, zero_bytes)
        assert result == []

    def test_identical_embedding_returned_as_duplicate(self, db: Session, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL as our primary database for all persistent storage."
        _create(client, pid, "PostgreSQL choice", content)

        embedding = embed_text(f"PostgreSQL choice. {content}")
        result = crud.find_near_duplicates(db, pid, embedding, threshold=0.97)
        assert len(result) >= 1
        assert result[0][1] >= 0.97

    def test_orthogonal_topic_not_returned(self, db: Session, client: TestClient):
        pid = _project(client)
        _create(client, pid, "Deploy on AWS", "We host all production services on AWS EC2.")
        embedding = embed_text("We chose Redis for session caching, not PostgreSQL.")
        result = crud.find_near_duplicates(db, pid, embedding, threshold=0.97)
        # Very different content → similarity should be well below 0.97
        assert all(sim < 0.97 for _, sim in result)

    def test_results_sorted_by_similarity_descending(self, db: Session, client: TestClient):
        pid = _project(client)
        _create(client, pid, "Choice A", "We use PostgreSQL as our primary database for storage.")
        _create(client, pid, "Choice B", "We use Redis for in-memory session data caching.")
        # Exact copy of A's embedding target
        embedding = embed_text("Choice A. We use PostgreSQL as our primary database for storage.")
        result = crud.find_near_duplicates(db, pid, embedding, threshold=0.0, limit=10)
        assert len(result) >= 2
        sims = [r[1] for r in result]
        assert sims == sorted(sims, reverse=True)

    def test_limit_caps_results(self, db: Session, client: TestClient):
        pid = _project(client)
        for i in range(5):
            _create(client, pid, f"Same content {i}", "We use PostgreSQL as our primary database.")
        embedding = embed_text("Same content 0. We use PostgreSQL as our primary database.")
        result = crud.find_near_duplicates(db, pid, embedding, threshold=0.0, limit=3)
        assert len(result) <= 3

    def test_lower_threshold_returns_more_results(self, db: Session, client: TestClient):
        pid = _project(client)
        _create(client, pid, "PostgreSQL choice", "We decided to use PostgreSQL for all storage needs.")
        _create(client, pid, "Redis choice", "We use Redis for session caching in production.")
        embedding = embed_text("We use PostgreSQL as our primary database and Redis for caching.")
        high_thresh = crud.find_near_duplicates(db, pid, embedding, threshold=0.97)
        low_thresh = crud.find_near_duplicates(db, pid, embedding, threshold=0.50)
        assert len(low_thresh) >= len(high_thresh)

    def test_unembedded_memories_ignored(self, db: Session, client: TestClient):
        pid = _project(client)
        # Create a memory that has no embedding via crud directly
        mem_data = schemas.MemoryCreate(type="decision", title="Unembedded", content="No embedding here.")
        crud.create_memory(db, pid, mem_data, embedding=None)
        embedding = embed_text("No embedding here. Unembedded.")
        result = crud.find_near_duplicates(db, pid, embedding, threshold=0.0)
        # The unembedded memory should not appear in results
        assert all(m.embedding is not None for m, _ in result)

    def test_cross_project_isolation(self, db: Session, client: TestClient):
        pid_a = _project(client, "Project A")
        pid_b = _project(client, "Project B")
        content = "We decided to use PostgreSQL as our primary database for ACID compliance."
        _create(client, pid_a, "Postgres decision", content)
        # Search in project B — should find nothing even though A has this content
        embedding = embed_text(f"Postgres decision. {content}")
        result_b = crud.find_near_duplicates(db, pid_b, embedding, threshold=0.97)
        assert len(result_b) == 0

    def test_returns_memory_float_tuples(self, db: Session, client: TestClient):
        from app.models import Memory
        pid = _project(client)
        _create(client, pid, "PostgreSQL", "We use PostgreSQL as our primary database for storage.")
        embedding = embed_text("PostgreSQL. We use PostgreSQL as our primary database for storage.")
        result = crud.find_near_duplicates(db, pid, embedding, threshold=0.0)
        if result:
            mem, sim = result[0]
            assert isinstance(mem, Memory)
            assert isinstance(sim, float)
            assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# 11–19: API — check_dedup=true on create
# ---------------------------------------------------------------------------

class TestCreateWithDedup:
    def test_same_content_check_dedup_returns_409(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL as our primary database for ACID compliance and JSON support."
        _create(client, pid, "PostgreSQL decision", content)
        status, body = _create(client, pid, "PostgreSQL decision", content, check_dedup=True)
        assert status == 409

    def test_409_body_has_existing_id(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL as our primary database for all transactional storage."
        first_status, first_body = _create(client, pid, "PostgreSQL choice", content)
        assert first_status == 201
        existing_id = first_body["id"]

        status, body = _create(client, pid, "PostgreSQL choice", content, check_dedup=True)
        assert status == 409
        assert body["detail"]["existing_id"] == existing_id

    def test_409_body_has_similarity(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL for all our storage needs in production."
        _create(client, pid, "PG decision", content)
        status, body = _create(client, pid, "PG decision", content, check_dedup=True)
        assert status == 409
        assert "similarity" in body["detail"]
        assert body["detail"]["similarity"] >= 0.97

    def test_409_body_has_existing_title(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL for all our persistent data storage needs."
        _create(client, pid, "My PostgreSQL decision", content)
        status, body = _create(client, pid, "My PostgreSQL decision", content, check_dedup=True)
        assert status == 409
        assert body["detail"]["existing_title"] == "My PostgreSQL decision"

    def test_different_content_check_dedup_returns_201(self, client: TestClient):
        pid = _project(client)
        _create(client, pid, "PostgreSQL", "We use PostgreSQL for all persistent storage needs.")
        # Completely different content
        status, body = _create(
            client, pid,
            "Redis choice",
            "We chose Redis for distributed session caching because of its sub-millisecond latency.",
            check_dedup=True,
        )
        assert status == 201

    def test_default_no_check_dedup_creates_duplicate(self, client: TestClient):
        pid = _project(client)
        content = "We use PostgreSQL as the primary database for all persistent storage."
        _create(client, pid, "PostgreSQL", content)
        # Second create WITHOUT check_dedup should succeed (default is off)
        status, _ = _create(client, pid, "PostgreSQL", content, check_dedup=False)
        assert status == 201

    def test_409_does_not_replace_existing_memory(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL as our primary database for all storage."
        first_status, first_body = _create(client, pid, "PostgreSQL choice", content)
        assert first_status == 201
        original_id = first_body["id"]

        # Try to create a duplicate — should 409
        _create(client, pid, "PostgreSQL choice", content, check_dedup=True)

        # Original memory still exists unchanged
        r = client.get(f"/memories/{original_id}")
        assert r.status_code == 200
        assert r.json()["id"] == original_id

    def test_cross_project_no_dedup(self, client: TestClient):
        pid_a = _project(client, "Project Alpha")
        pid_b = _project(client, "Project Beta")
        content = "We decided to use PostgreSQL as our primary database for all persistent storage needs."
        _create(client, pid_a, "PostgreSQL decision", content)
        # Same content in DIFFERENT project → should succeed (no cross-project dedup)
        status, _ = _create(client, pid_b, "PostgreSQL decision", content, check_dedup=True)
        assert status == 201

    def test_empty_project_check_dedup_returns_201(self, client: TestClient):
        pid = _project(client)
        # Fresh project — no duplicates possible
        content = "We decided to use PostgreSQL as our primary database."
        status, body = _create(client, pid, "PostgreSQL", content, check_dedup=True)
        assert status == 201
        assert "id" in body


# ---------------------------------------------------------------------------
# 20–30: find-duplicates endpoint
# ---------------------------------------------------------------------------

class TestFindDuplicatesEndpoint:
    def test_find_duplicates_returns_200(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories/find-duplicates", json={
            "type": "decision",
            "title": "PostgreSQL choice",
            "content": "We use PostgreSQL for persistent storage.",
        })
        assert r.status_code == 200

    def test_find_duplicates_no_match_is_duplicate_false(self, client: TestClient):
        pid = _project(client)
        result = _find_dupes(client, pid, "Redis choice",
                             "We use Redis for session caching in memory.")
        assert result["is_duplicate"] is False

    def test_find_duplicates_exact_match_is_duplicate_true(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL as our primary database for ACID-compliant storage."
        _create(client, pid, "PostgreSQL decision", content)
        result = _find_dupes(client, pid, "PostgreSQL decision", content)
        assert result["is_duplicate"] is True

    def test_find_duplicates_does_not_create_memory(self, client: TestClient):
        pid = _project(client)
        before = len(client.get(f"/projects/{pid}/memories").json())
        _find_dupes(client, pid, "PostgreSQL decision",
                    "We decided to use PostgreSQL as our primary database for all persistent storage.")
        after = len(client.get(f"/projects/{pid}/memories").json())
        assert after == before

    def test_find_duplicates_returns_near_duplicates_list(self, client: TestClient):
        pid = _project(client)
        result = _find_dupes(client, pid, "Any title", "Any content for this memory check.")
        assert "near_duplicates" in result
        assert isinstance(result["near_duplicates"], list)

    def test_find_duplicates_near_duplicate_has_memory_and_similarity(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL as the primary database for all persistent storage needs."
        _create(client, pid, "PostgreSQL", content)
        result = _find_dupes(client, pid, "PostgreSQL", content)
        if result["near_duplicates"]:
            dup = result["near_duplicates"][0]
            assert "memory" in dup
            assert "similarity" in dup
            assert isinstance(dup["similarity"], float)

    def test_find_duplicates_threshold_param(self, client: TestClient):
        pid = _project(client)
        _create(client, pid, "PostgreSQL choice", "We use PostgreSQL for persistent storage.")
        _create(client, pid, "Redis choice", "We use Redis for session caching in memory.")

        # Very low threshold should find more duplicates
        r_low = _find_dupes(client, pid, "Cache decision", "We use a cache for session data.",
                            threshold=0.50)
        r_high = _find_dupes(client, pid, "Cache decision", "We use a cache for session data.",
                             threshold=0.97)
        assert len(r_low["near_duplicates"]) >= len(r_high["near_duplicates"])

    def test_find_duplicates_project_isolation(self, client: TestClient):
        pid_a = _project(client, "Project X")
        pid_b = _project(client, "Project Y")
        content = "We decided to use PostgreSQL as our primary database for ACID compliance."
        _create(client, pid_a, "PostgreSQL", content)
        # Check from project B — should find no duplicates
        result = _find_dupes(client, pid_b, "PostgreSQL", content)
        assert result["is_duplicate"] is False

    def test_find_duplicates_404_unknown_project(self, client: TestClient):
        r = client.post("/projects/nonexistent/memories/find-duplicates", json={
            "type": "decision",
            "title": "Test",
            "content": "Test content for unknown project.",
        })
        assert r.status_code == 404

    def test_find_duplicates_sorted_by_similarity_desc(self, client: TestClient):
        pid = _project(client)
        _create(client, pid, "PostgreSQL main",
                "We use PostgreSQL as our primary database for all storage needs.")
        _create(client, pid, "Redis sessions",
                "We use Redis for distributed session caching because of its latency.")
        result = _find_dupes(client, pid, "Data storage",
                             "We use PostgreSQL for storage and Redis for caching in production.",
                             threshold=0.0)
        if len(result["near_duplicates"]) >= 2:
            sims = [d["similarity"] for d in result["near_duplicates"]]
            assert sims == sorted(sims, reverse=True)

    def test_find_duplicates_count_unchanged_after_call(self, client: TestClient):
        pid = _project(client)
        _create(client, pid, "PostgreSQL decision",
                "We use PostgreSQL as our primary database for persistent storage.")
        before = len(client.get(f"/projects/{pid}/memories").json())
        _find_dupes(client, pid, "PostgreSQL", "We use PostgreSQL for persistent storage.")
        after = len(client.get(f"/projects/{pid}/memories").json())
        assert after == before
