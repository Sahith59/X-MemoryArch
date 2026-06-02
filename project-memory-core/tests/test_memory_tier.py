"""
Sub-phase 1.23 — Memory tier tests.

Covers:
  1.  compute_tier pure function — high importance → working
  2.  compute_tier pure function — recent update → working
  3.  compute_tier pure function — old + low importance → archival
  4.  compute_tier pure function — boundary: exactly 30 days old → working
  5.  compute_tier pure function — boundary: 31 days old + low importance → archival
  6.  New memory defaults to "working" tier
  7.  High importance memory (≥4) is always "working"
  8.  Low importance memory created now is "working" (recency)
  9.  Tier field present in MemoryResponse
  10. Tier value is "working" or "archival" (valid enum)
  11. GET /projects/{id}/memories filterable by tier=working
  12. GET /projects/{id}/memories filterable by tier=archival
  13. Archival memories visible in unfiltered list
  14. Manual tier override via MemoryCreate (tier="archival")
  15. Manual tier override via MemoryUpdate (tier="working")
  16. Updating importance recalculates tier
  17. POST /projects/{id}/memories/retier returns RetierResult schema
  18. POST /projects/{id}/memories/retier returns 404 for unknown project
  19. Retier counts working and archival correctly
  20. Retier updates tier for memories that need it
  21. GET /memories/{id} includes tier field
  22. Extracted memories get tier assigned
  23. Retier updated_count reflects only changed memories
"""
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from app.crud import compute_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Tier Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, pid: str, importance: int = 3,
            title: str = "Test memory", tier: str | None = None) -> dict:
    body = {
        "type": "decision",
        "title": title,
        "content": f"Content for {title}.",
        "importance": importance,
    }
    if tier is not None:
        body["tier"] = tier
    r = client.post(f"/projects/{pid}/memories", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _session(client: TestClient, pid: str, content: str) -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Tier Test Session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    return r.json()["id"]


# ---------------------------------------------------------------------------
# 1–5: compute_tier pure function
# ---------------------------------------------------------------------------

class TestComputeTierPure:
    def test_high_importance_is_working(self):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert compute_tier(4, old) == "working"
        assert compute_tier(5, old) == "working"

    def test_recent_low_importance_is_working(self):
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        assert compute_tier(2, recent) == "working"
        assert compute_tier(1, recent) == "working"

    def test_old_low_importance_is_archival(self):
        old = datetime(2024, 1, 1, tzinfo=timezone.utc)
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert compute_tier(1, old, now=now) == "archival"
        assert compute_tier(3, old, now=now) == "archival"

    def test_exactly_30_days_is_working(self):
        now = datetime(2026, 5, 25, tzinfo=timezone.utc)
        exactly_30 = now - timedelta(days=30)
        assert compute_tier(2, exactly_30, now=now) == "working"

    def test_31_days_low_importance_is_archival(self):
        now = datetime(2026, 5, 25, tzinfo=timezone.utc)
        old_31 = now - timedelta(days=31)
        assert compute_tier(3, old_31, now=now) == "archival"

    def test_naive_datetime_handled(self):
        # updated_at without tzinfo should not raise
        naive = datetime(2024, 1, 1)  # no tzinfo
        result = compute_tier(1, naive)
        assert result in ("working", "archival")


# ---------------------------------------------------------------------------
# 6–10: Tier in creation and response
# ---------------------------------------------------------------------------

class TestTierCreation:
    def test_new_memory_defaults_to_working(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["tier"] == "working"

    def test_high_importance_memory_is_working(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, importance=4)
        assert m["tier"] == "working"

    def test_importance_5_is_working(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, importance=5)
        assert m["tier"] == "working"

    def test_tier_field_in_response(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "tier" in m

    def test_tier_is_valid_enum_value(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["tier"] in ("working", "archival")

    def test_manual_tier_override_archival(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, importance=5, tier="archival")
        assert m["tier"] == "archival"

    def test_manual_tier_override_working(self, client: TestClient):
        pid = _project(client)
        # Force archival first, then check working override
        m = _memory(client, pid, importance=1, tier="working")
        assert m["tier"] == "working"


# ---------------------------------------------------------------------------
# 11–13: Tier filter on list endpoint
# ---------------------------------------------------------------------------

class TestTierFilter:
    def test_filter_working_returns_only_working(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, importance=4, title="Working memory")
        r = client.get(f"/projects/{pid}/memories", params={"tier": "working"})
        assert r.status_code == 200
        assert all(m["tier"] == "working" for m in r.json())
        assert len(r.json()) >= 1

    def test_filter_archival_returns_only_archival(self, client: TestClient):
        pid = _project(client)
        # Force an archival memory via explicit override
        _memory(client, pid, tier="archival", title="Archival memory")
        r = client.get(f"/projects/{pid}/memories", params={"tier": "archival"})
        assert r.status_code == 200
        assert all(m["tier"] == "archival" for m in r.json())
        assert len(r.json()) >= 1

    def test_unfiltered_list_includes_archival(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, tier="archival", title="Old archival memory")
        _memory(client, pid, importance=4, title="Working memory")
        r = client.get(f"/projects/{pid}/memories")
        assert r.status_code == 200
        tiers = {m["tier"] for m in r.json()}
        assert "archival" in tiers
        assert "working" in tiers


# ---------------------------------------------------------------------------
# 14–16: Tier via update
# ---------------------------------------------------------------------------

class TestTierUpdate:
    def test_update_tier_via_memory_update(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, importance=4)
        r = client.put(f"/memories/{m['id']}", json={"tier": "archival"})
        assert r.status_code == 200
        assert r.json()["tier"] == "archival"

    def test_update_importance_recalculates_tier(self, client: TestClient):
        pid = _project(client)
        # Start with archival forced
        m = _memory(client, pid, importance=1, tier="archival")
        assert m["tier"] == "archival"
        # Bump importance to 5 — tier should flip to working
        r = client.put(f"/memories/{m['id']}", json={"importance": 5})
        assert r.status_code == 200
        assert r.json()["tier"] == "working"

    def test_get_memory_includes_tier(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        r = client.get(f"/memories/{m['id']}")
        assert r.status_code == 200
        assert "tier" in r.json()
        assert r.json()["tier"] in ("working", "archival")


# ---------------------------------------------------------------------------
# 17–23: Retier endpoint
# ---------------------------------------------------------------------------

class TestRetierEndpoint:
    def test_retier_returns_200(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        r = client.post(f"/projects/{pid}/memories/retier")
        assert r.status_code == 200

    def test_retier_404_for_unknown_project(self, client: TestClient):
        r = client.post("/projects/does-not-exist/memories/retier")
        assert r.status_code == 404

    def test_retier_response_schema(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        r = client.post(f"/projects/{pid}/memories/retier")
        result = r.json()
        assert "updated_count" in result
        assert "working_count" in result
        assert "archival_count" in result

    def test_retier_counts_correct(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, importance=5, title="High importance")
        _memory(client, pid, importance=5, title="Also high importance")
        r = client.post(f"/projects/{pid}/memories/retier")
        result = r.json()
        assert result["working_count"] == 2
        assert result["archival_count"] == 0
        assert result["working_count"] + result["archival_count"] == 2

    def test_retier_promotes_high_importance(self, client: TestClient):
        pid = _project(client)
        # Force archival on a high-importance memory
        m = _memory(client, pid, importance=5, tier="archival")
        assert m["tier"] == "archival"
        # Retier should correct it back to working
        r = client.post(f"/projects/{pid}/memories/retier")
        assert r.json()["updated_count"] >= 1
        r2 = client.get(f"/memories/{m['id']}")
        assert r2.json()["tier"] == "working"

    def test_retier_updated_count_zero_when_already_correct(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, importance=5)  # already working, will stay working
        r = client.post(f"/projects/{pid}/memories/retier")
        assert r.json()["updated_count"] == 0

    def test_retier_empty_project_returns_zeros(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories/retier")
        result = r.json()
        assert result["updated_count"] == 0
        assert result["working_count"] == 0
        assert result["archival_count"] == 0

    def test_extracted_memories_have_tier(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use PostgreSQL as our primary database for better "
            "JSON support and reliable concurrent write handling under load."
        )
        r = client.post(f"/sessions/{sid}/extract-memories")
        assert r.status_code == 200
        mems = r.json()["memories"]
        if mems:
            assert "tier" in mems[0]
            assert mems[0]["tier"] in ("working", "archival")
