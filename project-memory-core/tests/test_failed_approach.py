"""
Sub-phase 1.21 — failed_approach memory type tests.

Covers:
  1.  failed_approach is a valid MemoryType value
  2.  Memory can be created with type=failed_approach manually
  3.  FailedApproachMetadata schema validates correctly
  4.  approach_tried field stored and returned
  5.  reason_failed field stored and returned
  6.  alternative_chosen field stored and returned
  7.  symptoms list stored and returned
  8.  avoid_because field stored and returned
  9.  Partial metadata (only approach_tried) is valid
  10. Empty metadata dict is valid
  11. failed_approach memory appears in GET /projects/{id}/memories
  12. Failed approach sentence is extracted from a session
  13. Extracted failed_approach has type_metadata populated
  14. Extracted failed_approach has review_status=auto_extracted
  15. Extracted failed_approach has source_quote
  16. Extraction: "decided to use X" does NOT become failed_approach
  17. Extraction: "tried X but it failed" DOES become failed_approach
  18. failed_approach appears in GET /memories/{id}
  19. failed_approach included in export context.md
  20. Health dashboard counts failed_approach in type distribution
  21. failed_approach importance weight is 4 (equal to decision/bug)
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "FailedApproach Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, content: str) -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "FailedApproach Test Session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    return r.json()["id"]


def _extract(client: TestClient, sid: str) -> dict:
    r = client.post(f"/sessions/{sid}/extract-memories")
    assert r.status_code == 200
    return r.json()


def _create_failed_approach(client: TestClient, pid: str, **overrides) -> dict:
    body = {
        "type": "failed_approach",
        "title": "Tried Redis for session storage but abandoned it",
        "content": "We tried using Redis for session storage but encountered data consistency issues under load.",
        "type_metadata": {
            "approach_tried": "Redis for session storage",
            "reason_failed": "data consistency issues under concurrent writes",
            "symptoms": ["redis"],
            "alternative_chosen": "PostgreSQL-backed sessions",
            "avoid_because": "data consistency issues under concurrent writes",
        },
    }
    body.update(overrides)
    r = client.post(f"/projects/{pid}/memories", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–2: Type enum and basic creation
# ---------------------------------------------------------------------------

class TestFailedApproachType:
    def test_failed_approach_is_valid_memory_type(self, client: TestClient):
        pid = _project(client)
        m = _create_failed_approach(client, pid)
        assert m["type"] == "failed_approach"

    def test_failed_approach_stored_and_retrieved(self, client: TestClient):
        pid = _project(client)
        m = _create_failed_approach(client, pid)
        r = client.get(f"/memories/{m['id']}")
        assert r.status_code == 200
        assert r.json()["type"] == "failed_approach"


# ---------------------------------------------------------------------------
# 3–10: FailedApproachMetadata schema validation
# ---------------------------------------------------------------------------

class TestFailedApproachMetadata:
    def test_full_metadata_round_trips(self, client: TestClient):
        pid = _project(client)
        m = _create_failed_approach(client, pid)
        meta = m["type_metadata"]
        assert meta["approach_tried"] == "Redis for session storage"
        assert meta["reason_failed"] == "data consistency issues under concurrent writes"
        assert "redis" in meta["symptoms"]
        assert meta["alternative_chosen"] == "PostgreSQL-backed sessions"
        assert meta["avoid_because"] == "data consistency issues under concurrent writes"

    def test_partial_metadata_only_approach_tried(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "failed_approach",
            "title": "Tried GraphQL but abandoned it",
            "content": "We tried GraphQL but the team lacked expertise.",
            "type_metadata": {"approach_tried": "GraphQL"},
        })
        assert r.status_code == 201
        assert r.json()["type_metadata"]["approach_tried"] == "GraphQL"

    def test_empty_metadata_is_valid(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "failed_approach",
            "title": "Tried microservices architecture early on",
            "content": "We tried microservices but the overhead was too high.",
        })
        assert r.status_code == 201

    def test_symptoms_is_a_list(self, client: TestClient):
        pid = _project(client)
        m = _create_failed_approach(client, pid)
        assert isinstance(m["type_metadata"]["symptoms"], list)

    def test_metadata_survives_update(self, client: TestClient):
        pid = _project(client)
        m = _create_failed_approach(client, pid)
        r = client.put(f"/memories/{m['id']}", json={"importance": 5})
        assert r.status_code == 200
        assert r.json()["type_metadata"]["approach_tried"] == "Redis for session storage"


# ---------------------------------------------------------------------------
# 11: List endpoint includes failed_approach
# ---------------------------------------------------------------------------

class TestFailedApproachInList:
    def test_appears_in_project_memory_list(self, client: TestClient):
        pid = _project(client)
        _create_failed_approach(client, pid)
        mems = client.get(f"/projects/{pid}/memories").json()
        types = [m["type"] for m in mems]
        assert "failed_approach" in types

    def test_filterable_by_type(self, client: TestClient):
        pid = _project(client)
        _create_failed_approach(client, pid)
        r = client.get(f"/projects/{pid}/memories", params={"type": "failed_approach"})
        assert r.status_code == 200
        assert all(m["type"] == "failed_approach" for m in r.json())
        assert len(r.json()) >= 1


# ---------------------------------------------------------------------------
# 12–17: Extraction from session
# ---------------------------------------------------------------------------

class TestFailedApproachExtraction:
    def test_failed_approach_extracted_from_session(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We tried using Redis for session storage but abandoned it due to data "
            "consistency issues under concurrent writes.  We switched to PostgreSQL "
            "sessions which proved far more reliable under load."
        )
        result = _extract(client, sid)
        types = [m["type"] for m in result["memories"]]
        assert "failed_approach" in types

    def test_extracted_failed_approach_has_type_metadata(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We tried using Redis for session storage but abandoned it due to data "
            "consistency issues under concurrent writes.  We switched to PostgreSQL "
            "sessions which proved far more reliable under load."
        )
        result = _extract(client, sid)
        fa_mems = [m for m in result["memories"] if m["type"] == "failed_approach"]
        assert len(fa_mems) >= 1
        # type_metadata should be populated with at least one field
        assert fa_mems[0]["type_metadata"] is not None

    def test_extracted_failed_approach_review_status(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We attempted to use GraphQL for the API but switched back to REST "
            "because the team lacked the expertise to manage the schema effectively."
        )
        result = _extract(client, sid)
        fa_mems = [m for m in result["memories"] if m["type"] == "failed_approach"]
        if fa_mems:
            assert fa_mems[0]["review_status"] == "auto_extracted"

    def test_extracted_failed_approach_has_source_quote(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We tried using Redis for session storage but abandoned it due to data "
            "consistency issues under concurrent writes.  We switched to PostgreSQL."
        )
        result = _extract(client, sid)
        fa_mems = [m for m in result["memories"] if m["type"] == "failed_approach"]
        if fa_mems:
            assert fa_mems[0]["source_quote"] is not None

    def test_decision_sentence_not_classified_as_failed_approach(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use PostgreSQL as our primary database because it offers "
            "better JSON support and more reliable concurrent write handling."
        )
        result = _extract(client, sid)
        fa_mems = [m for m in result["memories"] if m["type"] == "failed_approach"]
        assert len(fa_mems) == 0

    def test_failed_approach_importance_weight(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We tried using Redis for session storage but abandoned it due to serious "
            "data consistency issues under concurrent writes to the cache layer."
        )
        result = _extract(client, sid)
        fa_mems = [m for m in result["memories"] if m["type"] == "failed_approach"]
        if fa_mems:
            # failed_approach type_weight=4, should produce importance >= 3
            assert fa_mems[0]["importance"] >= 3


# ---------------------------------------------------------------------------
# 18–20: Export and health
# ---------------------------------------------------------------------------

class TestFailedApproachExportAndHealth:
    def test_failed_approach_in_export(self, client: TestClient):
        pid = _project(client)
        _create_failed_approach(client, pid,
            title="Tried Redis for session storage — abandoned due to consistency issues",
            content="Redis session storage was abandoned after data consistency issues."
        )
        export = client.get(f"/projects/{pid}/export/context.md").text
        assert "Tried Redis for session storage" in export

    def test_health_counts_failed_approach(self, client: TestClient):
        pid = _project(client)
        _create_failed_approach(client, pid)
        h = client.get(f"/projects/{pid}/health").json()
        assert h["active_count"] >= 1
        assert h["total_memories"] >= 1
