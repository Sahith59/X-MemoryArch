"""
Sub-phase 1.17 Stress Tests — Current Truth Resolver.

Tests every aspect of conflict detection and resolution:
  1.  No conflicts on empty project
  2.  No conflicts when memories are distinct
  3.  title_overlap detected when same type + titles >= 70% similar
  4.  title_overlap NOT detected for different types (even with similar titles)
  5.  content_overlap detected across types when content >= 85% similar
  6.  conflict report contains both memories and similarity score
  7.  resolve_conflicts marks the lower-quality memory as superseded
  8.  resolve_conflicts keeps the higher-importance memory when quality is tied
  9.  dry_run does not write to the database
  10. after resolve, the previously conflicting pair disappears from detect
  11. already-superseded memories are not flagged as conflicts
  12. resolve is idempotent — running twice produces same result
  13. 404 for detect on unknown project
  14. 404 for resolve on unknown project
  15. resolve returns changelog entries for the superseded memory
  16. multiple conflict pairs resolved in a single call
  17. similarity_score in conflict pair is between 0 and 1
  18. conflict_type is "title_overlap" or "content_overlap"
  19. resolve result contains resolved_count and pairs_resolved
  20. resolved memory has superseded_by pointing to the winner
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Conflict Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(
    client: TestClient,
    pid: str,
    *,
    title: str = "A test memory",
    content: str = "Content for conflict testing here.",
    mem_type: str = "decision",
    importance: int = 3,
    confidence: float = 0.80,
    file_path: str | None = None,
    source_type: str | None = None,
) -> dict:
    body: dict = {
        "type": mem_type,
        "title": title,
        "content": content,
        "importance": importance,
        "confidence": confidence,
    }
    for k, v in [("file_path", file_path), ("source_type", source_type)]:
        if v is not None:
            body[k] = v
    r = client.post(f"/projects/{pid}/memories", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _detect(client: TestClient, pid: str) -> dict:
    r = client.get(f"/projects/{pid}/conflicts")
    assert r.status_code == 200, r.text
    return r.json()


def _resolve(client: TestClient, pid: str, dry_run: bool = False) -> dict:
    params = {"dry_run": "true"} if dry_run else {}
    r = client.post(f"/projects/{pid}/conflicts/resolve", params=params)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Empty project
# ---------------------------------------------------------------------------

class TestEmptyProject:
    def test_no_conflicts_on_empty_project(self, client: TestClient):
        pid = _project(client)
        report = _detect(client, pid)
        assert report["total_conflicts"] == 0
        assert report["pairs"] == []

    def test_resolve_on_empty_project_is_noop(self, client: TestClient):
        pid = _project(client)
        result = _resolve(client, pid)
        assert result["resolved_count"] == 0
        assert result["pairs_resolved"] == []


# ---------------------------------------------------------------------------
# 2. No conflicts for distinct memories
# ---------------------------------------------------------------------------

class TestNoConflicts:
    def test_distinct_titles_different_types_no_conflict(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use FastAPI as web framework", content="Decision about framework choice.")
        _mem(client, pid, title="Auth bug in login endpoint", content="Bug in authentication flow.", mem_type="problem")
        _mem(client, pid, title="How to run the development server", content="Setup instruction for local dev.", mem_type="how_to")
        report = _detect(client, pid)
        assert report["total_conflicts"] == 0

    def test_similar_titles_different_types_no_title_conflict(self, client: TestClient):
        pid = _project(client)
        # Same title but different types — should NOT trigger title_overlap
        _mem(client, pid, title="Use Redis for caching", content="Decision to add Redis.", mem_type="decision")
        _mem(client, pid, title="Use Redis for caching setup", content="How to install and configure Redis.", mem_type="how_to")
        report = _detect(client, pid)
        # title_overlap requires same type, so these shouldn't trigger it
        # (may trigger content_overlap if content is similar enough, but they're different)
        for pair in report["pairs"]:
            assert pair["conflict_type"] != "title_overlap"


# ---------------------------------------------------------------------------
# 3. title_overlap detection
# ---------------------------------------------------------------------------

class TestTitleOverlapDetection:
    def test_same_type_similar_titles_detected(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use PostgreSQL as the production database", content="We chose PostgreSQL.")
        _mem(client, pid, title="Use PostgreSQL for production database", content="Slightly different content about PostgreSQL.")
        report = _detect(client, pid)
        assert report["total_conflicts"] >= 1
        title_pairs = [p for p in report["pairs"] if p["conflict_type"] == "title_overlap"]
        assert len(title_pairs) >= 1

    def test_title_overlap_has_high_similarity_score(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use PostgreSQL as the production database system", content="PostgreSQL decision.")
        _mem(client, pid, title="Use PostgreSQL as the production database choice", content="Different content entirely.")
        report = _detect(client, pid)
        for pair in report["pairs"]:
            if pair["conflict_type"] == "title_overlap":
                assert pair["similarity_score"] >= 0.70

    def test_both_memories_in_conflict_pair(self, client: TestClient):
        pid = _project(client)
        a = _mem(client, pid, title="Use Redis for session caching", content="Redis for caching decision.")
        b = _mem(client, pid, title="Use Redis for session storage", content="Different content for Redis.")
        report = _detect(client, pid)
        if report["total_conflicts"] >= 1:
            pair_ids = {
                frozenset({p["memory_a"]["id"], p["memory_b"]["id"]})
                for p in report["pairs"]
            }
            assert frozenset({a["id"], b["id"]}) in pair_ids


# ---------------------------------------------------------------------------
# 4. Different types — no title_overlap
# ---------------------------------------------------------------------------

class TestCrossTypeNoTitleOverlap:
    def test_different_types_no_title_overlap(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Configure Redis for session caching", content="How to configure Redis.", mem_type="how_to")
        _mem(client, pid, title="Configure Redis for session caching", content="Decision to configure Redis.", mem_type="decision")
        report = _detect(client, pid)
        for pair in report["pairs"]:
            assert pair["conflict_type"] != "title_overlap"


# ---------------------------------------------------------------------------
# 5. content_overlap detection (cross-type)
# ---------------------------------------------------------------------------

class TestContentOverlapDetection:
    def test_near_identical_content_different_types_detected(self, client: TestClient):
        pid = _project(client)
        content = "We use PostgreSQL as the primary database for all production data storage."
        _mem(client, pid, title="PostgreSQL database decision", content=content, mem_type="decision")
        _mem(client, pid, title="Note about PostgreSQL", content=content + " This is important.", mem_type="insight")
        report = _detect(client, pid)
        content_pairs = [p for p in report["pairs"] if p["conflict_type"] == "content_overlap"]
        assert len(content_pairs) >= 1

    def test_distinct_content_no_content_overlap(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="FastAPI decision", content="We chose FastAPI because of async support and type safety.", mem_type="decision")
        _mem(client, pid, title="SQLite insight", content="SQLite is excellent for local-first applications with low write volume.", mem_type="insight")
        report = _detect(client, pid)
        content_pairs = [p for p in report["pairs"] if p["conflict_type"] == "content_overlap"]
        assert len(content_pairs) == 0


# ---------------------------------------------------------------------------
# 6. Conflict pair structure
# ---------------------------------------------------------------------------

class TestConflictPairStructure:
    def test_conflict_pair_has_all_fields(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use FastAPI for the backend API server", content="FastAPI decision.")
        _mem(client, pid, title="Use FastAPI as backend API framework", content="Different FastAPI content.")
        report = _detect(client, pid)
        if report["pairs"]:
            pair = report["pairs"][0]
            assert "memory_a" in pair
            assert "memory_b" in pair
            assert "similarity_score" in pair
            assert "conflict_type" in pair
            assert 0.0 <= pair["similarity_score"] <= 1.0
            assert pair["conflict_type"] in ("title_overlap", "content_overlap")

    def test_conflict_pair_memories_are_full_responses(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Decision to use Docker for containers", content="Docker decision.")
        _mem(client, pid, title="Decision to use Docker for containers here", content="Docker content here.")
        report = _detect(client, pid)
        if report["pairs"]:
            pair = report["pairs"][0]
            for mem in [pair["memory_a"], pair["memory_b"]]:
                assert "id" in mem
                assert "title" in mem
                assert "quality_score" in mem
                assert "type" in mem


# ---------------------------------------------------------------------------
# 7 & 8. Resolve keeps higher-quality / higher-importance memory
# ---------------------------------------------------------------------------

class TestResolveStrategy:
    def test_resolve_keeps_higher_quality_memory(self, client: TestClient):
        pid = _project(client)
        # high quality: has file_path → quality_score higher
        high = _mem(
            client, pid,
            title="Use FastAPI as the primary backend web framework",
            content="FastAPI decision based on performance analysis.",
            file_path="app/main.py",
            source_type="manual",
        )
        low = _mem(
            client, pid,
            title="Use FastAPI as the backend web framework choice",
            content="FastAPI decision.",
        )
        assert high["quality_score"] > low["quality_score"]

        _resolve(client, pid)

        high_after = client.get(f"/memories/{high['id']}").json()
        low_after = client.get(f"/memories/{low['id']}").json()
        assert high_after["status"] == "active"
        assert low_after["status"] == "superseded"

    def test_resolve_keeps_higher_importance_when_quality_tied(self, client: TestClient):
        pid = _project(client)
        high_imp = _mem(
            client, pid,
            title="Architecture decision for service layer pattern",
            content="We use service layer pattern for all business logic.",
            importance=5,
        )
        low_imp = _mem(
            client, pid,
            title="Architecture service layer pattern decision",
            content="We use service layer for all business logic code.",
            importance=2,
        )
        _resolve(client, pid)

        high_after = client.get(f"/memories/{high_imp['id']}").json()
        low_after = client.get(f"/memories/{low_imp['id']}").json()
        assert high_after["status"] == "active"
        assert low_after["status"] == "superseded"

    def test_resolved_memory_has_superseded_by_pointing_to_winner(self, client: TestClient):
        pid = _project(client)
        winner = _mem(
            client, pid,
            title="We chose PostgreSQL as the production database system",
            content="PostgreSQL selected for production.",
            importance=5,
            file_path="app/models.py",
        )
        loser = _mem(
            client, pid,
            title="We chose PostgreSQL as production database system",
            content="PostgreSQL for production databases.",
            importance=1,
        )
        _resolve(client, pid)
        loser_after = client.get(f"/memories/{loser['id']}").json()
        assert loser_after["superseded_by"] == winner["id"]


# ---------------------------------------------------------------------------
# 9. dry_run does not modify the database
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_does_not_change_memory_status(self, client: TestClient, db: Session):
        pid = _project(client)
        a = _mem(client, pid, title="Use Docker for containerization of services", content="Docker decision.")
        b = _mem(client, pid, title="Use Docker to containerize all services here", content="Docker different content.")

        result = _resolve(client, pid, dry_run=True)

        # Memory statuses must not have changed
        a_after = client.get(f"/memories/{a['id']}").json()
        b_after = client.get(f"/memories/{b['id']}").json()
        assert a_after["status"] == "active"
        assert b_after["status"] == "active"

    def test_dry_run_reports_would_be_resolved_count(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use Kubernetes for container orchestration", content="K8s decision.")
        _mem(client, pid, title="Use Kubernetes to orchestrate all containers", content="K8s content different.")

        dry = _resolve(client, pid, dry_run=True)
        real = _resolve(client, pid)

        # Dry run should predict what real run actually does
        assert dry["resolved_count"] == real["resolved_count"]


# ---------------------------------------------------------------------------
# 10. After resolve, conflicts disappear
# ---------------------------------------------------------------------------

class TestPostResolveState:
    def test_no_more_conflicts_after_resolve(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use Redis as the session caching solution here", content="Redis decision.", importance=5)
        _mem(client, pid, title="Use Redis as the session caching approach here", content="Redis caching content.")

        before = _detect(client, pid)
        assert before["total_conflicts"] >= 1

        _resolve(client, pid)

        after = _detect(client, pid)
        assert after["total_conflicts"] == 0


# ---------------------------------------------------------------------------
# 11. Already-superseded memories not flagged
# ---------------------------------------------------------------------------

class TestSupersededNotFlagged:
    def test_superseded_memory_not_in_conflicts(self, client: TestClient):
        pid = _project(client)
        a = _mem(client, pid, title="Old Flask decision superseded by FastAPI", content="We chose Flask originally.")
        # Supersede manually via supersede endpoint
        client.post(f"/memories/{a['id']}/supersede", json={
            "type": "decision",
            "title": "New FastAPI decision replacing the old Flask choice",
            "content": "After performance testing we chose FastAPI over Flask entirely.",
        })
        # Create a second similar title (active)
        c = _mem(client, pid, title="New FastAPI decision replacing the old Flask one", content="FastAPI replaces Flask.")

        report = _detect(client, pid)
        # The superseded memory (a) should not appear in any conflict pair
        all_ids = set()
        for pair in report["pairs"]:
            all_ids.add(pair["memory_a"]["id"])
            all_ids.add(pair["memory_b"]["id"])
        assert a["id"] not in all_ids


# ---------------------------------------------------------------------------
# 12. Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_resolve_twice_same_result(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use Nginx as reverse proxy for the backend", content="Nginx decision.")
        _mem(client, pid, title="Use Nginx as our reverse proxy server here", content="Nginx content here.")

        first = _resolve(client, pid)
        second = _resolve(client, pid)

        # Second run should find no new conflicts (already resolved)
        assert second["resolved_count"] == 0


# ---------------------------------------------------------------------------
# 13 & 14. 404 errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_detect_404_for_unknown_project(self, client: TestClient):
        r = client.get("/projects/nonexistent-project/conflicts")
        assert r.status_code == 404

    def test_resolve_404_for_unknown_project(self, client: TestClient):
        r = client.post("/projects/nonexistent-project/conflicts/resolve")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 15. Resolve writes changelog entries for the superseded memory
# ---------------------------------------------------------------------------

class TestResolveChangelog:
    def test_resolve_writes_changelog_for_superseded(self, client: TestClient, db: Session):
        pid = _project(client)
        a = _mem(client, pid, title="Use SQLite for local-first database storage", content="SQLite decision.")
        b = _mem(client, pid, title="Use SQLite as local-first data storage here", content="SQLite different.", importance=5)

        _resolve(client, pid)

        # Whichever lost should have changelog entries
        a_after = client.get(f"/memories/{a['id']}").json()
        b_after = client.get(f"/memories/{b['id']}").json()
        loser_id = a["id"] if a_after["status"] == "superseded" else b["id"]

        history = client.get(f"/memories/{loser_id}/history").json()
        status_changes = [c for c in history if c["field"] == "status"]
        assert len(status_changes) >= 1
        assert any(c["new_value"] == "superseded" for c in status_changes)


# ---------------------------------------------------------------------------
# 16. Multiple pairs resolved in one call
# ---------------------------------------------------------------------------

class TestMultiplePairs:
    def test_multiple_conflicts_resolved_in_one_call(self, client: TestClient):
        pid = _project(client)
        # Pair 1: similar title decisions
        _mem(client, pid, title="Chose PostgreSQL as production database engine", content="PG choice.")
        _mem(client, pid, title="Chose PostgreSQL for production database system", content="PG other.")

        # Pair 2: similar title architectures
        _mem(client, pid, title="Microservices architecture for backend services", content="Microservices arch.", mem_type="structure")
        _mem(client, pid, title="Microservices architecture pattern for backend", content="Microservices different.", mem_type="structure")

        report = _detect(client, pid)
        assert report["total_conflicts"] >= 2

        result = _resolve(client, pid)
        assert result["resolved_count"] >= 2

    def test_resolved_count_equals_pairs_resolved_length(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use Docker for containerization of services", content="Docker containers.")
        _mem(client, pid, title="Use Docker to containerize all services here", content="Docker services.")

        result = _resolve(client, pid)
        assert result["resolved_count"] == len(result["pairs_resolved"])


# ---------------------------------------------------------------------------
# 17 & 18. Similarity score and conflict_type invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_similarity_score_between_0_and_1(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use FastAPI as backend framework for APIs", content="FastAPI choice.")
        _mem(client, pid, title="Use FastAPI as backend API framework system", content="Different FastAPI.")
        report = _detect(client, pid)
        for pair in report["pairs"]:
            assert 0.0 <= pair["similarity_score"] <= 1.0

    def test_conflict_type_is_valid_value(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use PostgreSQL for the production database engine", content="PostgreSQL.")
        _mem(client, pid, title="Use PostgreSQL as production database system here", content="PG different.")
        report = _detect(client, pid)
        for pair in report["pairs"]:
            assert pair["conflict_type"] in ("title_overlap", "content_overlap")
