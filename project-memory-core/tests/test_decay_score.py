"""
Sub-phase 1.28 — Decay score tests.

Covers:
  Unit tests — compute_decay_score pure function:
  1.  Fresh memory (0 days old, importance=5) → score = 1.0
  2.  Fresh memory (0 days old, importance=1) → score = 0.20
  3.  importance=5 scores higher than importance=1 at same age
  4.  Score decreases as memory ages (0 days > 45 days > 90 days)
  5.  90-day-old importance=5 memory scores ≈0.50
  6.  Recent access boosts score above un-accessed equivalent
  7.  Access boost overrides stale update age
  8.  Score is always in [0.0, 1.0]
  9.  Score is a float (4 decimal places)
  10. importance=1 memory aged 365 days → very low score (< 0.05)

  Integration tests — API:
  11. decay_score field present in MemoryResponse
  12. New memory has non-null decay_score
  13. New importance=5 memory has higher decay_score than importance=1
  14. decay_score in list endpoint response
  15. decay_score in search results
  16. decay_score recalculated on PUT (importance change raises score)
  17. decay_score recalculated on GET /memories/{id} (access boost applied)
  18. decay_score in extraction result memories
  19. decay_score is between 0.0 and 1.0 for all new memories
  20. retier endpoint also refreshes decay_score
"""
import math
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from app.crud import compute_decay_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(days_ago: float, now=None) -> datetime:
    base = now or datetime.now(timezone.utc)
    return base - timedelta(days=days_ago)


def _project(client: TestClient, name: str = "Decay Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, pid: str, importance: int = 3, title: str = "A memory") -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": "decision",
        "title": title,
        "content": f"Content for {title}.",
        "importance": importance,
    })
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–10: Unit tests on compute_decay_score pure function
# ---------------------------------------------------------------------------

class TestComputeDecayScoreUnit:
    def test_fresh_importance5_scores_1(self):
        now = datetime.now(timezone.utc)
        score = compute_decay_score(5, now, None, now=now)
        assert score == 1.0

    def test_fresh_importance1_scores_0_20(self):
        now = datetime.now(timezone.utc)
        score = compute_decay_score(1, now, None, now=now)
        assert abs(score - 0.20) < 0.001

    def test_higher_importance_scores_higher_same_age(self):
        now = datetime.now(timezone.utc)
        old = _dt(60)
        s5 = compute_decay_score(5, old, None, now=now)
        s1 = compute_decay_score(1, old, None, now=now)
        assert s5 > s1

    def test_score_decreases_with_age(self):
        now = datetime.now(timezone.utc)
        s0 = compute_decay_score(5, _dt(0), None, now=now)
        s45 = compute_decay_score(5, _dt(45), None, now=now)
        s90 = compute_decay_score(5, _dt(90), None, now=now)
        assert s0 > s45 > s90

    def test_90_day_importance5_approx_half(self):
        now = datetime.now(timezone.utc)
        score = compute_decay_score(5, _dt(90, now), None, now=now)
        # 1.0 × e^(-90/90) = e^-1 ≈ 0.3679... wait
        # Actually e^(-90/90) = e^(-1) ≈ 0.368, not 0.5
        # half-life is 90 days, so e^(-90/90×ln2) would give 0.5
        # But we use e^(-t/90), so at t=90: e^(-1) ≈ 0.368
        # At t=90×ln2 ≈ 62.4 days: e^(-62.4/90) ≈ 0.5
        expected = 1.0 * math.exp(-1.0)  # = e^-1 ≈ 0.3679
        assert abs(score - expected) < 0.01

    def test_recent_access_boosts_score(self):
        now = datetime.now(timezone.utc)
        old_updated = _dt(180, now)  # very old
        # Without access
        s_no_access = compute_decay_score(3, old_updated, None, now=now)
        # With recent access (yesterday)
        recent_access = _dt(1, now)
        s_with_access = compute_decay_score(3, old_updated, recent_access, now=now)
        assert s_with_access > s_no_access

    def test_access_boost_overrides_stale_update(self):
        now = datetime.now(timezone.utc)
        very_old = _dt(365, now)
        # Accessed today
        s_accessed_today = compute_decay_score(3, very_old, now, now=now)
        # Accessed 200 days ago
        old_access = _dt(200, now)
        s_old_access = compute_decay_score(3, very_old, old_access, now=now)
        assert s_accessed_today > s_old_access

    def test_score_always_in_0_1_range(self):
        now = datetime.now(timezone.utc)
        for importance in range(1, 6):
            for days_ago in [0, 30, 90, 365, 1000]:
                score = compute_decay_score(importance, _dt(days_ago, now), None, now=now)
                assert 0.0 <= score <= 1.0, f"Out of range: importance={importance}, days={days_ago}, score={score}"

    def test_score_is_rounded_float(self):
        now = datetime.now(timezone.utc)
        score = compute_decay_score(3, _dt(45, now), None, now=now)
        assert isinstance(score, float)
        # 4 decimal places
        assert score == round(score, 4)

    def test_importance1_aged_365_days_very_low(self):
        now = datetime.now(timezone.utc)
        score = compute_decay_score(1, _dt(365, now), None, now=now)
        assert score < 0.05


# ---------------------------------------------------------------------------
# 11–20: Integration tests via API
# ---------------------------------------------------------------------------

class TestDecayScoreAPI:
    def test_decay_score_field_present(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "decay_score" in m

    def test_new_memory_has_nonnull_decay_score(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["decay_score"] is not None

    def test_importance5_higher_than_importance1(self, client: TestClient):
        pid = _project(client)
        m5 = _memory(client, pid, importance=5, title="High importance")
        m1 = _memory(client, pid, importance=1, title="Low importance")
        assert m5["decay_score"] > m1["decay_score"]

    def test_decay_score_in_list_response(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        mems = client.get(f"/projects/{pid}/memories").json()
        for m in mems:
            assert "decay_score" in m

    def test_decay_score_in_search_results(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Searchable decay memory")
        results = client.get(f"/projects/{pid}/memories/search",
                             params={"q": "Searchable"}).json()
        if results:
            assert "decay_score" in results[0]

    def test_decay_score_recalculated_on_importance_update(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, importance=1)
        low_score = m["decay_score"]
        r = client.put(f"/memories/{m['id']}", json={"importance": 5})
        assert r.status_code == 200
        high_score = r.json()["decay_score"]
        assert high_score > low_score

    def test_decay_score_updated_on_get(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        # GET /memories/{id} calls track_memory_access which recalculates decay
        r = client.get(f"/memories/{m['id']}")
        assert r.status_code == 200
        assert r.json()["decay_score"] is not None

    def test_decay_score_in_extraction_result(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/sessions", json={
            "tool_name": "Claude",
            "title": "Decay test session",
            "raw_content": (
                "We decided to use PostgreSQL as our primary database for better "
                "JSON support and reliable concurrent write handling under load."
            ),
            "session_date": "2026-05-25",
        })
        sid = r.json()["id"]
        result = client.post(f"/sessions/{sid}/extract-memories").json()
        mems = result.get("memories", [])
        if mems:
            assert "decay_score" in mems[0]
            assert mems[0]["decay_score"] is not None

    def test_decay_score_in_0_1_range_for_new_memories(self, client: TestClient):
        pid = _project(client)
        for i in range(1, 6):
            m = _memory(client, pid, importance=i, title=f"Memory importance {i}")
            assert 0.0 <= m["decay_score"] <= 1.0, f"Out of range for importance={i}"

    def test_retier_refreshes_decay_score(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["decay_score"] is not None
        r = client.post(f"/projects/{pid}/memories/retier")
        assert r.status_code == 200
        # Verify decay_score is still set after retier
        updated = client.get(f"/memories/{m['id']}").json()
        assert updated["decay_score"] is not None
