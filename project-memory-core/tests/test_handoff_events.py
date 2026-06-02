"""
Sub-phase 1.34 — HandoffEvents tests.

Covers:
  1.  POST /projects/{id}/handoff-events creates event (201)
  2.  Correct project_id
  3.  Correct source_tool
  4.  Correct target_tool
  5.  Default status is pending
  6.  context_packet_id stored when provided
  7.  context_packet_id is None when not provided
  8.  note stored when provided
  9.  handoff_at is set on creation
  10. Custom handoff_at is respected
  11. GET /projects/{id}/handoff-events lists events
  12. GET /projects/{id}/handoff-events returns 404 for unknown project
  13. GET list filter by status=completed
  14. GET list filter by source_tool
  15. GET list filter by target_tool
  16. GET /handoff-events/{id} returns correct event
  17. GET /handoff-events/{id} returns 404 for unknown
  18. PATCH updates status to completed
  19. PATCH updates status to failed
  20. PATCH updates note
  21. DELETE returns 204
  22. DELETE removes event
  23. Project delete cascades to handoff events
  24. Context packet delete nulls context_packet_id on events
  25. Multiple events can coexist for same project
  26. Events ordered by handoff_at descending
  27. context_packet_id must belong to same project (404 if foreign)
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Handoff Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _packet(client: TestClient, pid: str) -> dict:
    r = client.post(f"/projects/{pid}/context-packets", json={
        "target_tool": "ChatGPT",
        "intent": "Transfer context",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _event(client: TestClient, pid: str, **kwargs) -> dict:
    payload = {"source_tool": "Claude", "target_tool": "ChatGPT"}
    payload.update(kwargs)
    r = client.post(f"/projects/{pid}/handoff-events", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–10: Create and field checks
# ---------------------------------------------------------------------------

class TestCreateHandoffEvent:
    def test_creates_event(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["id"]

    def test_correct_project_id(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["project_id"] == pid

    def test_correct_source_tool(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["source_tool"] == "Claude"

    def test_correct_target_tool(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["target_tool"] == "ChatGPT"

    def test_default_status_pending(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["status"] == "pending"

    def test_context_packet_id_stored(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        e = _event(client, pid, context_packet_id=p["id"])
        assert e["context_packet_id"] == p["id"]

    def test_context_packet_id_none_when_not_provided(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["context_packet_id"] is None

    def test_note_stored(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid, note="Switching to debug auth issue")
        assert e["note"] == "Switching to debug auth issue"

    def test_handoff_at_is_set(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        assert e["handoff_at"] is not None

    def test_custom_handoff_at_respected(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid, handoff_at="2026-01-15T10:00:00Z")
        assert "2026-01-15" in e["handoff_at"]


# ---------------------------------------------------------------------------
# 11–17: List and get
# ---------------------------------------------------------------------------

class TestListAndGet:
    def test_list_returns_events(self, client: TestClient):
        pid = _project(client)
        _event(client, pid)
        _event(client, pid, source_tool="Cursor", target_tool="Claude")
        r = client.get(f"/projects/{pid}/handoff-events")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_list_404_unknown_project(self, client: TestClient):
        r = client.get("/projects/does-not-exist/handoff-events")
        assert r.status_code == 404

    def test_filter_by_status_completed(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        client.patch(f"/handoff-events/{e['id']}", json={"status": "completed"})
        _event(client, pid)  # pending
        events = client.get(f"/projects/{pid}/handoff-events?status=completed").json()
        assert len(events) == 1
        assert events[0]["status"] == "completed"

    def test_filter_by_source_tool(self, client: TestClient):
        pid = _project(client)
        _event(client, pid, source_tool="Claude")
        _event(client, pid, source_tool="Cursor")
        events = client.get(f"/projects/{pid}/handoff-events?source_tool=Cursor").json()
        assert all(e["source_tool"] == "Cursor" for e in events)
        assert len(events) == 1

    def test_filter_by_target_tool(self, client: TestClient):
        pid = _project(client)
        _event(client, pid, target_tool="ChatGPT")
        _event(client, pid, target_tool="Gemini")
        events = client.get(f"/projects/{pid}/handoff-events?target_tool=Gemini").json()
        assert all(e["target_tool"] == "Gemini" for e in events)
        assert len(events) == 1

    def test_get_returns_event(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        r = client.get(f"/handoff-events/{e['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == e["id"]

    def test_get_404_unknown(self, client: TestClient):
        r = client.get("/handoff-events/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 18–20: PATCH
# ---------------------------------------------------------------------------

class TestUpdateHandoffEvent:
    def test_patch_status_completed(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        r = client.patch(f"/handoff-events/{e['id']}", json={"status": "completed"})
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_patch_status_failed(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        r = client.patch(f"/handoff-events/{e['id']}", json={"status": "failed"})
        assert r.json()["status"] == "failed"

    def test_patch_note(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        r = client.patch(f"/handoff-events/{e['id']}", json={"note": "Updated reason"})
        assert r.json()["note"] == "Updated reason"


# ---------------------------------------------------------------------------
# 21–22: Delete
# ---------------------------------------------------------------------------

class TestDeleteHandoffEvent:
    def test_delete_returns_204(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        r = client.delete(f"/handoff-events/{e['id']}")
        assert r.status_code == 204

    def test_delete_removes_event(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        client.delete(f"/handoff-events/{e['id']}")
        assert client.get(f"/handoff-events/{e['id']}").status_code == 404


# ---------------------------------------------------------------------------
# 23–27: Integration and edge cases
# ---------------------------------------------------------------------------

class TestHandoffIntegration:
    def test_project_delete_cascades_to_events(self, client: TestClient):
        pid = _project(client)
        e = _event(client, pid)
        client.delete(f"/projects/{pid}")
        assert client.get(f"/handoff-events/{e['id']}").status_code == 404

    def test_context_packet_delete_nulls_reference(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        e = _event(client, pid, context_packet_id=p["id"])
        assert e["context_packet_id"] == p["id"]
        client.delete(f"/context-packets/{p['id']}")
        updated = client.get(f"/handoff-events/{e['id']}").json()
        assert updated["context_packet_id"] is None

    def test_multiple_events_per_project(self, client: TestClient):
        pid = _project(client)
        for tool in ["ChatGPT", "Cursor", "Gemini"]:
            _event(client, pid, target_tool=tool)
        events = client.get(f"/projects/{pid}/handoff-events").json()
        assert len(events) == 3

    def test_events_ordered_by_handoff_at_desc(self, client: TestClient):
        pid = _project(client)
        _event(client, pid, handoff_at="2026-01-01T08:00:00Z")
        _event(client, pid, handoff_at="2026-01-03T08:00:00Z")
        _event(client, pid, handoff_at="2026-01-02T08:00:00Z")
        events = client.get(f"/projects/{pid}/handoff-events").json()
        dates = [e["handoff_at"] for e in events]
        assert dates == sorted(dates, reverse=True)

    def test_foreign_context_packet_returns_404(self, client: TestClient):
        pid = _project(client)
        other_pid = _project(client, name="Other Project")
        p = _packet(client, other_pid)
        r = client.post(f"/projects/{pid}/handoff-events", json={
            "source_tool": "Claude",
            "target_tool": "ChatGPT",
            "context_packet_id": p["id"],
        })
        assert r.status_code == 404
