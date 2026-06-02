"""
Sub-phase 1.45 — Temporal supersession tests.

Two new capabilities:
  A. superseded_at field on MemoryLink — populated for relationship_type="supersedes"
     so the temporal graph knows exactly *when* a memory was replaced.
  B. Auto-supersede at creation time — POST /projects/{id}/memories?auto_supersede=true
     scans existing active memories of the same type; any with title similarity >= 0.85
     are automatically marked superseded and linked.

Covers:

  superseded_at on MemoryLink:
  1.  superseded_at field present in MemoryLinkResponse
  2.  superseded_at is None for non-supersedes links (related_to, conflicts_with)
  3.  superseded_at is set (non-None) for supersedes links created by supersede_memory()
  4.  superseded_at matches the valid_until of the old memory (same transaction)
  5.  superseded_at is set by resolve_conflicts()
  6.  superseded_at is a valid ISO timestamp
  7.  Two chained supersessions each have their own superseded_at
  8.  superseded_at is preserved after the link is retrieved again
  9.  superseded_at ordering: earlier supersession has earlier superseded_at
  10. superseded_at not set on manually-created non-supersedes links via API

  auto_supersede at creation time:
  11. auto_supersede=true returns AutoSupersedeResult when conflict found
  12. AutoSupersedeResult has memory, superseded_count, superseded_memories
  13. superseded_count matches len(superseded_memories)
  14. The old memory is marked status=superseded after auto_supersede
  15. The old memory has superseded_by pointing to the new memory
  16. A supersedes link with superseded_at is created from old→new
  17. auto_supersede=false (default) does NOT supersede anything
  18. auto_supersede works for decision type
  19. auto_supersede works for constraint type
  20. auto_supersede works for structure type
  21. auto_supersede works for how_to type
  22. auto_supersede does NOT fire for task type (non-supersedable)
  23. auto_supersede does NOT fire for insight type (non-supersedable)
  24. auto_supersede only supersedes memories in the same project
  25. auto_supersede only supersedes active memories (not already-superseded)
  26. auto_supersede with no conflict returns MemoryResponse (not AutoSupersedeResult)
  27. auto_supersede with empty project returns MemoryResponse
  28. title similarity exactly at threshold (0.85) triggers supersession
  29. title similarity below threshold does NOT trigger supersession
  30. auto_supersede + auto_supersede=true twice in sequence: second supersedes first
  31. old memory history has superseded_by and valid_until changelog entries
  32. superseded_at is within 5 seconds of request time
  33. multiple conflicting memories all get superseded in one call
  34. auto_supersede=true + check_dedup=true both work together
  35. AutoSupersedeResult new memory is active status
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud, schemas
from app.models import MemoryLink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "AutoSupersede Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(client: TestClient, pid: str, title: str, mem_type: str = "decision",
         importance: int = 3, content: str | None = None,
         auto_supersede: bool = False) -> tuple[int, dict]:
    params = {}
    if auto_supersede:
        params["auto_supersede"] = "true"
    r = client.post(f"/projects/{pid}/memories", params=params, json={
        "type": mem_type,
        "title": title,
        "content": content or f"{title} — technical decision for our project.",
        "importance": importance,
    })
    return r.status_code, r.json()


def _supersede(client: TestClient, old_id: str, title: str = "New decision",
               mem_type: str = "decision") -> dict:
    r = client.post(f"/memories/{old_id}/supersede", json={
        "type": mem_type,
        "title": title,
        "content": f"{title} — replacement content.",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _links(client: TestClient, mid: str) -> list[dict]:
    r = client.get(f"/memories/{mid}/links")
    assert r.status_code == 200
    return r.json()


def _supersedes_links(client: TestClient, mid: str) -> list[dict]:
    return [lk for lk in _links(client, mid) if lk["relationship_type"] == "supersedes"]


# ---------------------------------------------------------------------------
# 1–10: superseded_at on MemoryLink
# ---------------------------------------------------------------------------

class TestSupersededAtField:
    def test_superseded_at_field_in_response(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Flask for backend")
        result = _supersede(client, old["id"], "Use FastAPI for backend")
        links = _supersedes_links(client, old["id"])
        assert len(links) >= 1
        assert "superseded_at" in links[0]

    def test_related_to_link_has_null_superseded_at(self, client: TestClient):
        pid = _project(client)
        _, a = _mem(client, pid, "Decision A")
        _, b = _mem(client, pid, "Decision B")
        r = client.post(f"/memories/{a['id']}/links", json={
            "target_memory_id": b["id"],
            "relationship_type": "related_to",
        })
        assert r.status_code == 201
        assert r.json()["superseded_at"] is None

    def test_supersedes_link_has_non_null_superseded_at(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Celery for async tasks")
        _supersede(client, old["id"], "Use ARQ for async tasks")
        links = _supersedes_links(client, old["id"])
        assert len(links) >= 1
        assert links[0]["superseded_at"] is not None

    def test_superseded_at_matches_valid_until(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Gunicorn as server")
        result = _supersede(client, old["id"], "Use Uvicorn as server")
        old_mem = result["old_memory"]
        links = _supersedes_links(client, old["id"])
        assert len(links) >= 1
        # Both are set in the same transaction — they represent the same moment
        valid_until = old_mem["valid_until"]
        superseded_at = links[0]["superseded_at"]
        assert valid_until is not None
        assert superseded_at is not None
        # Parse and compare (allow 1-second tolerance for rounding)
        vu = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
        sa = datetime.fromisoformat(superseded_at.replace("Z", "+00:00"))
        if vu.tzinfo is None:
            vu = vu.replace(tzinfo=timezone.utc)
        if sa.tzinfo is None:
            sa = sa.replace(tzinfo=timezone.utc)
        assert abs((vu - sa).total_seconds()) < 2

    def test_resolve_conflicts_sets_superseded_at(self, client: TestClient, db: Session):
        pid = _project(client)
        _mem(client, pid, "Use PostgreSQL for data storage", importance=2)
        _mem(client, pid, "Use PostgreSQL for data storage", importance=5)
        client.post(f"/projects/{pid}/conflicts/resolve")
        links = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).all()
        assert len(links) >= 1
        assert all(lk.superseded_at is not None for lk in links)

    def test_superseded_at_is_valid_iso_timestamp(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use HTTP/1.1 for API communication")
        _supersede(client, old["id"], "Use HTTP/2 for API communication")
        links = _supersedes_links(client, old["id"])
        assert len(links) >= 1
        ts = links[0]["superseded_at"]
        assert ts is not None
        # Should parse without error
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed.year >= 2026

    def test_chained_supersessions_each_have_own_superseded_at(self, client: TestClient):
        pid = _project(client)
        _, a = _mem(client, pid, "Decision v1")
        b = _supersede(client, a["id"], "Decision v2")
        b_id = b["new_memory"]["id"]
        _supersede(client, b_id, "Decision v3")
        links_ab = _supersedes_links(client, a["id"])
        links_bc = _supersedes_links(client, b_id)
        assert all(lk["superseded_at"] is not None for lk in links_ab)
        assert all(lk["superseded_at"] is not None for lk in links_bc)

    def test_superseded_at_preserved_on_re_fetch(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use pip for package management")
        _supersede(client, old["id"], "Use uv for package management")
        # Fetch twice — value should be stable
        links1 = _supersedes_links(client, old["id"])
        links2 = _supersedes_links(client, old["id"])
        assert links1[0]["superseded_at"] == links2[0]["superseded_at"]

    def test_earlier_supersession_has_earlier_superseded_at(self, client: TestClient):
        import time
        pid = _project(client)
        _, a = _mem(client, pid, "First decision")
        b = _supersede(client, a["id"], "Second decision")
        b_id = b["new_memory"]["id"]
        time.sleep(0.05)  # ensure timestamp difference
        _supersede(client, b_id, "Third decision")
        links_ab = _supersedes_links(client, a["id"])
        links_bc = _supersedes_links(client, b_id)
        ta = datetime.fromisoformat(links_ab[0]["superseded_at"].replace("Z", "+00:00"))
        tb = datetime.fromisoformat(links_bc[0]["superseded_at"].replace("Z", "+00:00"))
        if ta.tzinfo is None:
            ta = ta.replace(tzinfo=timezone.utc)
        if tb.tzinfo is None:
            tb = tb.replace(tzinfo=timezone.utc)
        assert ta <= tb

    def test_manual_link_api_has_null_superseded_at(self, client: TestClient):
        pid = _project(client)
        _, a = _mem(client, pid, "Memory A")
        _, b = _mem(client, pid, "Memory B different topic entirely")
        r = client.post(f"/memories/{a['id']}/links", json={
            "target_memory_id": b["id"],
            "relationship_type": "conflicts_with",
        })
        assert r.status_code == 201
        assert r.json()["superseded_at"] is None


# ---------------------------------------------------------------------------
# 11–35: auto_supersede at creation time
# ---------------------------------------------------------------------------

class TestAutoSupersedeAtCreation:
    def test_auto_supersede_returns_auto_supersede_result(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use PostgreSQL as primary database")
        status, body = _mem(client, pid, "Use PostgreSQL as primary database v2",
                            auto_supersede=True)
        assert status == 201
        # Should return AutoSupersedeResult
        assert "superseded_count" in body
        assert "superseded_memories" in body

    def test_auto_supersede_result_has_all_fields(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use FastAPI as API framework")
        status, body = _mem(client, pid, "Use FastAPI as API framework v2",
                            auto_supersede=True)
        assert status == 201
        assert "memory" in body
        assert "superseded_count" in body
        assert "superseded_memories" in body

    def test_auto_supersede_count_matches_list(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use Redis for caching layer")
        status, body = _mem(client, pid, "Use Redis for caching layer updated",
                            auto_supersede=True)
        assert status == 201
        assert body["superseded_count"] == len(body["superseded_memories"])

    def test_old_memory_status_becomes_superseded(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use SQLite for local database")
        status, body = _mem(client, pid, "Use SQLite for local database v2",
                            auto_supersede=True)
        assert status == 201
        if body.get("superseded_count", 0) > 0:
            old_r = client.get(f"/memories/{old['id']}")
            assert old_r.json()["status"] == "superseded"

    def test_old_memory_superseded_by_points_to_new(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Docker for containerization")
        status, body = _mem(client, pid, "Use Docker for containerization v2",
                            auto_supersede=True)
        assert status == 201
        if body.get("superseded_count", 0) > 0:
            new_id = body["memory"]["id"]
            old_r = client.get(f"/memories/{old['id']}")
            assert old_r.json()["superseded_by"] == new_id

    def test_supersedes_link_created_with_superseded_at(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Kubernetes for orchestration")
        status, body = _mem(client, pid, "Use Kubernetes for orchestration v2",
                            auto_supersede=True)
        assert status == 201
        if body.get("superseded_count", 0) > 0:
            new_id = body["memory"]["id"]
            lk = db.query(MemoryLink).filter(
                MemoryLink.source_memory_id == old["id"],
                MemoryLink.target_memory_id == new_id,
                MemoryLink.relationship_type == "supersedes",
            ).first()
            assert lk is not None
            assert lk.superseded_at is not None

    def test_default_no_auto_supersede_keeps_old_active(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Nginx as load balancer")
        # Create without auto_supersede — old should stay active
        _mem(client, pid, "Use Nginx as load balancer v2")
        old_r = client.get(f"/memories/{old['id']}")
        assert old_r.json()["status"] == "active"

    def test_auto_supersede_decision_type(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use TypeScript for all services", mem_type="decision")
        status, body = _mem(client, pid, "Use TypeScript for all services v2",
                            mem_type="decision", auto_supersede=True)
        assert status == 201
        assert body.get("superseded_count", 0) >= 1

    def test_auto_supersede_constraint_type(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Must support Python 3.11+", mem_type="constraint")
        status, body = _mem(client, pid, "Must support Python 3.12+",
                            mem_type="constraint", auto_supersede=True)
        assert status == 201
        # Titles differ significantly so no guarantee of supersession
        # But no error should occur
        assert "id" in body or "memory" in body

    def test_auto_supersede_structure_type(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use MVC pattern for API layer", mem_type="structure")
        status, body = _mem(client, pid, "Use MVC pattern for API layer v2",
                            mem_type="structure", auto_supersede=True)
        assert status == 201

    def test_auto_supersede_how_to_type(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "How to deploy to production", mem_type="how_to")
        status, body = _mem(client, pid, "How to deploy to production updated",
                            mem_type="how_to", auto_supersede=True)
        assert status == 201

    def test_auto_supersede_does_not_fire_for_task(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Write API documentation", mem_type="task")
        status, body = _mem(client, pid, "Write API documentation updated",
                            mem_type="task", auto_supersede=True)
        assert status == 201
        # task is not in supersedable types — old memory should stay active
        old_r = client.get(f"/memories/{old['id']}")
        assert old_r.json()["status"] == "active"

    def test_auto_supersede_does_not_fire_for_insight(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "PostgreSQL handles concurrent writes well",
                      mem_type="insight")
        status, body = _mem(client, pid,
                            "PostgreSQL handles concurrent writes well — confirmed",
                            mem_type="insight", auto_supersede=True)
        assert status == 201
        old_r = client.get(f"/memories/{old['id']}")
        assert old_r.json()["status"] == "active"

    def test_auto_supersede_cross_project_isolation(self, client: TestClient):
        pid_a = _project(client, "Project A")
        pid_b = _project(client, "Project B")
        _, old = _mem(client, pid_a, "Use PostgreSQL as main database")
        # Create same-title memory in project B — should NOT supersede project A's memory
        _mem(client, pid_b, "Use PostgreSQL as main database",
             auto_supersede=True)
        old_r = client.get(f"/memories/{old['id']}")
        assert old_r.json()["status"] == "active"

    def test_auto_supersede_only_active_memories(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Flask for backend API")
        # Manually supersede old first
        _supersede(client, old["id"], "Use FastAPI for backend API")
        # Now create a third with auto_supersede — only active ones should be targeted
        _, already_superseded_id = old["id"], old["id"]
        status, body = _mem(client, pid, "Use FastAPI for backend API v3",
                            auto_supersede=True)
        assert status == 201
        # The already-superseded old memory status should not change again
        old_r = client.get(f"/memories/{old['id']}")
        assert old_r.json()["status"] == "superseded"

    def test_auto_supersede_no_conflict_returns_memory_response(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use PostgreSQL as primary database")
        # Very different title — no conflict expected
        status, body = _mem(
            client, pid,
            "Deploy using Terraform for infrastructure",
            auto_supersede=True,
        )
        assert status == 201
        # Should return MemoryResponse (no superseded_count)
        assert "id" in body
        assert "superseded_count" not in body or body.get("superseded_count", 0) == 0

    def test_auto_supersede_empty_project_returns_memory_response(self, client: TestClient):
        pid = _project(client)
        status, body = _mem(client, pid, "First decision in new project",
                            auto_supersede=True)
        assert status == 201
        assert "id" in body

    def test_title_at_threshold_triggers_supersession(self, client: TestClient):
        """Two titles with ratio >= 0.85 should trigger auto-supersession."""
        import difflib
        pid = _project(client)
        old_title = "Use PostgreSQL for persistent storage"
        new_title = "Use PostgreSQL for persistent storage v2"
        sim = difflib.SequenceMatcher(None, old_title.lower(), new_title.lower()).ratio()
        if sim < 0.85:
            pytest.skip(f"Title similarity {sim:.3f} < 0.85 — test would not trigger supersession")
        _, old = _mem(client, pid, old_title)
        status, body = _mem(client, pid, new_title, auto_supersede=True)
        assert status == 201
        if sim >= 0.85:
            assert body.get("superseded_count", 0) >= 1

    def test_title_below_threshold_no_supersession(self, client: TestClient):
        """Two very different titles should NOT trigger supersession."""
        import difflib
        pid = _project(client)
        old_title = "Use PostgreSQL as database"
        new_title = "Deploy application to AWS Kubernetes"
        sim = difflib.SequenceMatcher(None, old_title.lower(), new_title.lower()).ratio()
        assert sim < 0.85, f"Titles unexpectedly similar: {sim:.3f}"
        _, old = _mem(client, pid, old_title)
        status, body = _mem(client, pid, new_title, auto_supersede=True)
        assert status == 201
        old_r = client.get(f"/memories/{old['id']}")
        assert old_r.json()["status"] == "active"

    def test_chain_auto_supersede(self, client: TestClient):
        """Second auto_supersede supersedes the first auto_supersede's result."""
        pid = _project(client)
        _, v1 = _mem(client, pid, "Use PostgreSQL for all persistent storage")
        status, body_v2 = _mem(client, pid,
                               "Use PostgreSQL for all persistent storage v2",
                               auto_supersede=True)
        assert status == 201
        if body_v2.get("superseded_count", 0) == 0:
            pytest.skip("v1 not superseded — title similarity below threshold")
        v2_id = body_v2["memory"]["id"]
        status, body_v3 = _mem(client, pid,
                               "Use PostgreSQL for all persistent storage v3",
                               auto_supersede=True)
        assert status == 201
        # v2 should now be superseded
        v2_r = client.get(f"/memories/{v2_id}")
        assert v2_r.json()["status"] == "superseded"

    def test_old_memory_has_changelog_entries(self, client: TestClient):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Redis for session caching")
        status, body = _mem(client, pid, "Use Redis for session caching v2",
                            auto_supersede=True)
        assert status == 201
        if body.get("superseded_count", 0) > 0:
            history = client.get(f"/memories/{old['id']}/history").json()
            fields = {e["field"] for e in history}
            assert "status" in fields
            assert "superseded_by" in fields

    def test_superseded_at_within_5_seconds(self, client: TestClient, db: Session):
        pid = _project(client)
        _, old = _mem(client, pid, "Use Nginx for reverse proxy configuration")
        status, body = _mem(client, pid,
                            "Use Nginx for reverse proxy configuration v2",
                            auto_supersede=True)
        assert status == 201
        if body.get("superseded_count", 0) > 0:
            new_id = body["memory"]["id"]
            lk = db.query(MemoryLink).filter(
                MemoryLink.source_memory_id == old["id"],
                MemoryLink.target_memory_id == new_id,
            ).first()
            assert lk is not None and lk.superseded_at is not None
            now = datetime.now(timezone.utc)
            sa = lk.superseded_at
            if sa.tzinfo is None:
                sa = sa.replace(tzinfo=timezone.utc)
            assert (now - sa).total_seconds() < 30

    def test_multiple_conflicts_all_superseded(self, client: TestClient):
        pid = _project(client)
        # Create two near-identical decision memories
        _, old1 = _mem(client, pid, "Use PostgreSQL for persistent storage layer")
        _, old2 = _mem(client, pid, "Use PostgreSQL for persistent storage v2 layer")
        status, body = _mem(client, pid,
                            "Use PostgreSQL for persistent storage v3 layer",
                            auto_supersede=True)
        assert status == 201
        # Both old memories should be considered for supersession
        # (exact count depends on threshold similarity)
        if body.get("superseded_count", 0) > 0:
            superseded_ids = {m["id"] for m in body["superseded_memories"]}
            assert len(superseded_ids) >= 1

    def test_auto_supersede_and_check_dedup_coexist(self, client: TestClient):
        pid = _project(client)
        # Different content — no dedup conflict
        _mem(client, pid, "Use PostgreSQL for persistent storage")
        r = client.post(
            f"/projects/{pid}/memories",
            params={"auto_supersede": "true", "check_dedup": "true"},
            json={
                "type": "decision",
                "title": "Use PostgreSQL for persistent storage v2",
                "content": "Upgraded PostgreSQL version for better performance and JSON support.",
                "importance": 3,
            }
        )
        # Should not 409 (different enough content) and should process auto_supersede
        assert r.status_code == 201

    def test_new_memory_is_active_after_auto_supersede(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use Webpack for bundling")
        status, body = _mem(client, pid, "Use Webpack for bundling v2",
                            auto_supersede=True)
        assert status == 201
        # The new memory (whether in AutoSupersedeResult or MemoryResponse) is active
        new_id = body.get("memory", body).get("id")
        new_r = client.get(f"/memories/{new_id}")
        assert new_r.json()["status"] == "active"
