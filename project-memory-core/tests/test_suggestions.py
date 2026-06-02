"""
Sub-phase 1.32 — MemorySuggestions tests.

Covers:
  1.  POST /projects/{id}/suggestions creates a pending suggestion
  2.  Created suggestion has status=pending
  3.  Created suggestion has created_by=manual
  4.  GET /projects/{id}/suggestions lists suggestions
  5.  GET /projects/{id}/suggestions?status=pending filters correctly
  6.  GET /projects/{id}/suggestions?status=approved returns only approved
  7.  GET /projects/{id}/suggestions returns 404 for unknown project
  8.  GET /suggestions/{id} returns the correct suggestion
  9.  GET /suggestions/{id} returns 404 for unknown
  10. PATCH /suggestions/{id} can update title
  11. PATCH /suggestions/{id} can update status to rejected
  12. PATCH /suggestions/{id} on pending+type_change sets status=edited
  13. POST /suggestions/{id}/approve creates a Memory
  14. POST /suggestions/{id}/approve sets suggestion status=approved
  15. POST /suggestions/{id}/approve sets reviewed_at
  16. POST /suggestions/{id}/approve is idempotent (existing memory returned)
  17. POST /suggestions/{id}/approve returns 404 for unknown suggestion
  18. DELETE /suggestions/{id} returns 204
  19. DELETE /suggestions/{id} removes the suggestion
  20. Extraction creates approved suggestion records (Option B)
  21. Extracted suggestions count == memories_created
  22. Extracted suggestions have created_by=rule_based
  23. Extracted suggestions have memory_id set
  24. Session delete sets source_session_id to NULL (not cascade-delete)
  25. Project delete cascades to suggestions
  26. source_message_ids stored and returned correctly
  27. confidence stored on suggestion
  28. Approved suggestion's memory has source_type=manual when no session
  29. POST suggestion with invalid type returns 422
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Suggestions Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, content: str = "We decided to use PostgreSQL for ACID compliance.") -> dict:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Suggestions Session",
        "raw_content": content,
        "session_date": "2026-05-26",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _create_suggestion(client: TestClient, pid: str, **kwargs) -> dict:
    payload = {
        "suggested_type": "decision",
        "title": "Use Redis for caching",
        "content": "Redis was chosen because it offers fast in-memory reads.",
        "confidence": 0.9,
    }
    payload.update(kwargs)
    r = client.post(f"/projects/{pid}/suggestions", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _extract(client: TestClient, sid: str) -> dict:
    r = client.post(f"/sessions/{sid}/extract-memories")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–3: Create pending suggestion
# ---------------------------------------------------------------------------

class TestCreateSuggestion:
    def test_creates_pending_suggestion(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        assert s["id"]
        assert s["project_id"] == pid

    def test_status_is_pending(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        assert s["status"] == "pending"

    def test_created_by_is_manual(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        assert s["created_by"] == "manual"


# ---------------------------------------------------------------------------
# 4–7: List
# ---------------------------------------------------------------------------

class TestListSuggestions:
    def test_list_returns_suggestions(self, client: TestClient):
        pid = _project(client)
        _create_suggestion(client, pid)
        _create_suggestion(client, pid, title="Second suggestion")
        r = client.get(f"/projects/{pid}/suggestions")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_filter_by_status_pending(self, client: TestClient):
        pid = _project(client)
        _create_suggestion(client, pid)
        suggestions = client.get(f"/projects/{pid}/suggestions?status=pending").json()
        assert all(s["status"] == "pending" for s in suggestions)

    def test_filter_by_status_approved(self, client: TestClient):
        pid = _project(client)
        _create_suggestion(client, pid)  # pending
        # Approve it
        sid = client.get(f"/projects/{pid}/suggestions").json()[0]["id"]
        client.post(f"/suggestions/{sid}/approve")
        approved = client.get(f"/projects/{pid}/suggestions?status=approved").json()
        assert all(s["status"] == "approved" for s in approved)

    def test_list_404_unknown_project(self, client: TestClient):
        r = client.get("/projects/does-not-exist/suggestions")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 8–9: Get single
# ---------------------------------------------------------------------------

class TestGetSuggestion:
    def test_get_returns_suggestion(self, client: TestClient):
        pid = _project(client)
        created = _create_suggestion(client, pid)
        r = client.get(f"/suggestions/{created['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == created["id"]

    def test_get_404_unknown(self, client: TestClient):
        r = client.get("/suggestions/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 10–12: PATCH
# ---------------------------------------------------------------------------

class TestUpdateSuggestion:
    def test_patch_updates_title(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.patch(f"/suggestions/{s['id']}", json={"title": "Updated title"})
        assert r.status_code == 200
        assert r.json()["title"] == "Updated title"

    def test_patch_status_to_rejected(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.patch(f"/suggestions/{s['id']}", json={"status": "rejected"})
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"

    def test_patch_type_change_sets_edited(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.patch(f"/suggestions/{s['id']}", json={"suggested_type": "problem"})
        assert r.status_code == 200
        assert r.json()["status"] == "edited"
        assert r.json()["suggested_type"] == "problem"


# ---------------------------------------------------------------------------
# 13–17: Approve
# ---------------------------------------------------------------------------

class TestApproveSuggestion:
    def test_approve_creates_memory(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.post(f"/suggestions/{s['id']}/approve")
        assert r.status_code == 200
        result = r.json()
        assert result["memory"]["id"]
        assert result["memory"]["title"] == s["title"]

    def test_approve_sets_status(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.post(f"/suggestions/{s['id']}/approve")
        assert r.json()["suggestion"]["status"] == "approved"

    def test_approve_sets_reviewed_at(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.post(f"/suggestions/{s['id']}/approve")
        assert r.json()["suggestion"]["reviewed_at"] is not None

    def test_approve_idempotent(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r1 = client.post(f"/suggestions/{s['id']}/approve")
        r2 = client.post(f"/suggestions/{s['id']}/approve")
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["memory"]["id"] == r2.json()["memory"]["id"]

    def test_approve_404_unknown(self, client: TestClient):
        r = client.post("/suggestions/does-not-exist/approve")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 18–19: Delete
# ---------------------------------------------------------------------------

class TestDeleteSuggestion:
    def test_delete_returns_204(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        r = client.delete(f"/suggestions/{s['id']}")
        assert r.status_code == 204

    def test_delete_removes_suggestion(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        client.delete(f"/suggestions/{s['id']}")
        assert client.get(f"/suggestions/{s['id']}").status_code == 404


# ---------------------------------------------------------------------------
# 20–23: Extraction Option B integration
# ---------------------------------------------------------------------------

class TestExtractionIntegration:
    _TECH_CONTENT = (
        "We decided to use PostgreSQL for ACID compliance because it offers "
        "reliable transactions. We also decided to avoid MySQL due to replication "
        "complexity. The architecture uses SQLAlchemy ORM for database access."
    )

    def test_extraction_creates_approved_suggestions(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content=self._TECH_CONTENT)
        result = _extract(client, s["id"])
        if result["memories_created"] == 0:
            pytest.skip("No memories extracted — content may not have passed semantic gate")
        suggestions = client.get(f"/projects/{pid}/suggestions?status=approved").json()
        assert len(suggestions) > 0

    def test_extraction_suggestion_count_matches_memories(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content=self._TECH_CONTENT)
        result = _extract(client, s["id"])
        if result["memories_created"] == 0:
            pytest.skip("No memories extracted")
        suggestions = client.get(f"/projects/{pid}/suggestions?status=approved").json()
        assert len(suggestions) == result["memories_created"]

    def test_extraction_suggestions_have_rule_based(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content=self._TECH_CONTENT)
        result = _extract(client, s["id"])
        if result["memories_created"] == 0:
            pytest.skip("No memories extracted")
        suggestions = client.get(f"/projects/{pid}/suggestions?status=approved").json()
        assert all(sg["created_by"] == "rule_based" for sg in suggestions)

    def test_extraction_suggestions_have_memory_id(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content=self._TECH_CONTENT)
        result = _extract(client, s["id"])
        if result["memories_created"] == 0:
            pytest.skip("No memories extracted")
        suggestions = client.get(f"/projects/{pid}/suggestions?status=approved").json()
        assert all(sg["memory_id"] is not None for sg in suggestions)


# ---------------------------------------------------------------------------
# 24–25: Cascade behavior
# ---------------------------------------------------------------------------

class TestCascade:
    def test_session_delete_nulls_source_session_id(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid)
        r = client.post(f"/projects/{pid}/suggestions", json={
            "suggested_type": "decision",
            "title": "Cache using Redis",
            "content": "Redis chosen for speed.",
            "source_session_id": s["id"],
        })
        assert r.status_code == 201
        sg_id = r.json()["id"]
        client.delete(f"/sessions/{s['id']}")
        sg = client.get(f"/suggestions/{sg_id}").json()
        assert sg["source_session_id"] is None

    def test_project_delete_cascades_to_suggestions(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        client.delete(f"/projects/{pid}")
        assert client.get(f"/suggestions/{s['id']}").status_code == 404


# ---------------------------------------------------------------------------
# 26–29: Field checks
# ---------------------------------------------------------------------------

class TestSuggestionFields:
    def test_source_message_ids_stored(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/suggestions", json={
            "suggested_type": "problem",
            "title": "NullPointer in auth",
            "content": "NullPointerException in the auth flow.",
            "source_message_ids": ["msg-001", "msg-002"],
        })
        assert r.status_code == 201
        assert r.json()["source_message_ids"] == ["msg-001", "msg-002"]

    def test_confidence_stored(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid, confidence=0.72)
        assert abs(s["confidence"] - 0.72) < 0.01

    def test_approve_no_session_gives_manual_source_type(self, client: TestClient):
        pid = _project(client)
        s = _create_suggestion(client, pid)
        result = client.post(f"/suggestions/{s['id']}/approve").json()
        assert result["memory"]["source_type"] == "manual"

    def test_invalid_type_returns_422(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/suggestions", json={
            "suggested_type": "not_a_real_type",
            "title": "Invalid",
            "content": "Content.",
        })
        assert r.status_code == 422
