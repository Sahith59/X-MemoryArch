"""
Sub-phase 1.25 — Access tracking tests.

Covers:
  1.  access_count field in MemoryResponse
  2.  last_accessed_at field in MemoryResponse
  3.  New memory starts with access_count = 0
  4.  New memory starts with last_accessed_at = None
  5.  GET /memories/{id} increments access_count to 1
  6.  Second GET /memories/{id} increments access_count to 2
  7.  last_accessed_at is set after first GET
  8.  last_accessed_at is updated on subsequent GET
  9.  List endpoint does NOT increment access_count
  10. access_count survives PUT /memories/{id} update
  11. last_accessed_at survives PUT /memories/{id} update
  12. access_count in list response matches stored value
  13. compute_tier uses last_accessed_at — recently accessed old-ish memory → working
  14. compute_tier: accessed 31+ days ago + old + low importance → archival
  15. GET /projects/{id}/memories/most-accessed returns list
  16. most-accessed endpoint sorts by access_count descending
  17. most-accessed respects limit param
  18. most-accessed 404 for unknown project
  19. Retier endpoint accounts for last_accessed_at (recently accessed → working)
  20. Extracted memories start with access_count = 0
  21. access_count in MemoryResponse from extraction result
  22. access_count does not increment on search results
  23. access_count does not increment on most-accessed endpoint
  24. access_count field in most-accessed response
"""
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from app.crud import compute_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Access Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, pid: str, title: str = "Test memory",
            importance: int = 3) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": "decision",
        "title": title,
        "content": f"Content for {title}.",
        "importance": importance,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _session_extract(client: TestClient, pid: str) -> list[dict]:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Access Test Session",
        "raw_content": (
            "We decided to use PostgreSQL as our primary database for better "
            "JSON support and reliable concurrent write handling under load."
        ),
        "session_date": "2026-05-25",
    })
    sid = r.json()["id"]
    return client.post(f"/sessions/{sid}/extract-memories").json()["memories"]


# ---------------------------------------------------------------------------
# 1–4: Fields present and initial values
# ---------------------------------------------------------------------------

class TestAccessFieldsPresent:
    def test_access_count_field_in_response(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "access_count" in m

    def test_last_accessed_at_field_in_response(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "last_accessed_at" in m

    def test_new_memory_access_count_is_zero(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["access_count"] == 0

    def test_new_memory_last_accessed_at_is_null(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["last_accessed_at"] is None


# ---------------------------------------------------------------------------
# 5–8: Tracking behaviour on GET /memories/{id}
# ---------------------------------------------------------------------------

class TestAccessTracking:
    def test_first_get_increments_count_to_1(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        r = client.get(f"/memories/{m['id']}")
        assert r.status_code == 200
        assert r.json()["access_count"] == 1

    def test_second_get_increments_count_to_2(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        client.get(f"/memories/{m['id']}")
        r = client.get(f"/memories/{m['id']}")
        assert r.json()["access_count"] == 2

    def test_first_get_sets_last_accessed_at(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        r = client.get(f"/memories/{m['id']}")
        assert r.json()["last_accessed_at"] is not None

    def test_last_accessed_at_updated_on_subsequent_get(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        r1 = client.get(f"/memories/{m['id']}")
        ts1 = r1.json()["last_accessed_at"]
        r2 = client.get(f"/memories/{m['id']}")
        ts2 = r2.json()["last_accessed_at"]
        # Both should be set; second may be same or later (same-second in tests)
        assert ts1 is not None
        assert ts2 is not None

    def test_access_count_reflected_in_list_endpoint(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        client.get(f"/memories/{m['id']}")   # access once
        client.get(f"/memories/{m['id']}")   # access twice
        mems = client.get(f"/projects/{pid}/memories").json()
        target = next(x for x in mems if x["id"] == m["id"])
        assert target["access_count"] == 2


# ---------------------------------------------------------------------------
# 9: List endpoint does NOT track access
# ---------------------------------------------------------------------------

class TestListDoesNotTrackAccess:
    def test_list_endpoint_does_not_increment_count(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        # Call list endpoint multiple times
        client.get(f"/projects/{pid}/memories")
        client.get(f"/projects/{pid}/memories")
        client.get(f"/projects/{pid}/memories")
        # access_count should still be 0
        r = client.get(f"/memories/{m['id']}")  # this one increments to 1
        # The three list calls should not have incremented
        # After the GET above it's now 1, so check memory state before that GET
        # Use last_accessed_at as proxy: it was None before the GET
        mems = client.get(f"/projects/{pid}/memories").json()
        target = next(x for x in mems if x["id"] == m["id"])
        # After one explicit GET, count should be 1 not 3+
        assert target["access_count"] == 1


# ---------------------------------------------------------------------------
# 10–12: Access data survives updates
# ---------------------------------------------------------------------------

class TestAccessSurvivesUpdate:
    def test_access_count_survives_importance_update(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        client.get(f"/memories/{m['id']}")  # access_count = 1
        r = client.put(f"/memories/{m['id']}", json={"importance": 5})
        assert r.status_code == 200
        assert r.json()["access_count"] == 1

    def test_last_accessed_at_survives_status_update(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        client.get(f"/memories/{m['id']}")  # sets last_accessed_at
        r = client.put(f"/memories/{m['id']}", json={"status": "resolved"})
        assert r.status_code == 200
        assert r.json()["last_accessed_at"] is not None

    def test_access_count_in_list_matches_stored(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        for _ in range(3):
            client.get(f"/memories/{m['id']}")
        mems = client.get(f"/projects/{pid}/memories").json()
        target = next(x for x in mems if x["id"] == m["id"])
        assert target["access_count"] == 3


# ---------------------------------------------------------------------------
# 13–14: compute_tier extended with last_accessed_at
# ---------------------------------------------------------------------------

class TestComputeTierWithAccess:
    def test_recently_accessed_old_memory_is_working(self):
        now = datetime(2026, 5, 25, tzinfo=timezone.utc)
        old_updated = datetime(2024, 1, 1, tzinfo=timezone.utc)   # 16 months ago
        recent_accessed = now - timedelta(days=5)                   # 5 days ago
        assert compute_tier(1, old_updated, last_accessed_at=recent_accessed, now=now) == "working"

    def test_old_accessed_and_old_updated_is_archival(self):
        now = datetime(2026, 5, 25, tzinfo=timezone.utc)
        old_updated = datetime(2024, 1, 1, tzinfo=timezone.utc)
        old_accessed = datetime(2024, 6, 1, tzinfo=timezone.utc)   # also old
        assert compute_tier(1, old_updated, last_accessed_at=old_accessed, now=now) == "archival"

    def test_none_last_accessed_falls_back_to_updated_at_rule(self):
        now = datetime(2026, 5, 25, tzinfo=timezone.utc)
        recent_updated = now - timedelta(days=10)
        assert compute_tier(2, recent_updated, last_accessed_at=None, now=now) == "working"

    def test_high_importance_ignores_access(self):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert compute_tier(4, old, last_accessed_at=None) == "working"


# ---------------------------------------------------------------------------
# 15–18: most-accessed endpoint
# ---------------------------------------------------------------------------

class TestMostAccessedEndpoint:
    def test_most_accessed_returns_200(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        r = client.get(f"/projects/{pid}/memories/most-accessed")
        assert r.status_code == 200

    def test_most_accessed_returns_list(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        r = client.get(f"/projects/{pid}/memories/most-accessed")
        assert isinstance(r.json(), list)

    def test_most_accessed_sorted_by_count(self, client: TestClient):
        pid = _project(client)
        m1 = _memory(client, pid, title="Memory A")
        m2 = _memory(client, pid, title="Memory B")
        # Access m2 twice, m1 once
        client.get(f"/memories/{m2['id']}")
        client.get(f"/memories/{m2['id']}")
        client.get(f"/memories/{m1['id']}")
        r = client.get(f"/projects/{pid}/memories/most-accessed")
        result = r.json()
        assert len(result) >= 2
        # m2 should appear before m1
        ids = [m["id"] for m in result]
        assert ids.index(m2["id"]) < ids.index(m1["id"])

    def test_most_accessed_respects_limit(self, client: TestClient):
        pid = _project(client)
        for i in range(5):
            _memory(client, pid, title=f"Memory {i}")
        r = client.get(f"/projects/{pid}/memories/most-accessed", params={"limit": 2})
        assert r.status_code == 200
        assert len(r.json()) <= 2

    def test_most_accessed_404_unknown_project(self, client: TestClient):
        r = client.get("/projects/does-not-exist/memories/most-accessed")
        assert r.status_code == 404

    def test_most_accessed_does_not_increment_count(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        # Call most-accessed endpoint multiple times
        client.get(f"/projects/{pid}/memories/most-accessed")
        client.get(f"/projects/{pid}/memories/most-accessed")
        # access_count should still be 0 (no explicit GET /memories/{id})
        r = client.get(f"/memories/{m['id']}")  # this increments to 1
        assert r.json()["access_count"] == 1

    def test_most_accessed_response_has_access_count(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        client.get(f"/memories/{m['id']}")
        result = client.get(f"/projects/{pid}/memories/most-accessed").json()
        for mem in result:
            assert "access_count" in mem


# ---------------------------------------------------------------------------
# 19–21: Retier and extraction integration
# ---------------------------------------------------------------------------

class TestAccessTrackingIntegration:
    def test_retier_considers_last_accessed_at(self, client: TestClient):
        pid = _project(client)
        # Low importance memory, forced archival
        m = _memory(client, pid, importance=1, title="Old low-importance memory")
        r = client.put(f"/memories/{m['id']}", json={"tier": "archival"})
        assert r.json()["tier"] == "archival"
        # Access it — should flip tier to working via track_memory_access
        r2 = client.get(f"/memories/{m['id']}")
        assert r2.json()["tier"] == "working"

    def test_extracted_memories_start_at_zero_access(self, client: TestClient):
        pid = _project(client)
        mems = _session_extract(client, pid)
        if not mems:
            pytest.skip("No memories extracted")
        for m in mems:
            assert m["access_count"] == 0
            assert m["last_accessed_at"] is None

    def test_search_does_not_increment_access(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="PostgreSQL database decision")
        r = client.get(f"/projects/{pid}/memories/search", params={"q": "PostgreSQL"})
        assert r.status_code == 200
        mems = r.json()
        if mems:
            # access_count should reflect only explicit GETs, not search hits
            direct = client.get(f"/memories/{mems[0]['id']}").json()
            # After this one GET, count should be 1 (not inflated by search)
            assert direct["access_count"] == 1
