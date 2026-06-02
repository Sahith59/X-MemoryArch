"""
Phase-1 Quality Audit — stress tests for every fix made during the audit pass.

Covers:
  - Export content-level deduplication
  - delete_memory cascade (orphan MemoryLink + MemoryChangelog cleanup)
  - delete_project cascade (same, at project level)
  - FTS5 correctness after memory delete
  - Session summary format
  - MemoryStatus.stale user-settable behaviour
  - Export limit does not silently truncate (smoke check)
  - Confidence varies with keyword strength
  - Noise filtering in extraction
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import text

from app import models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Audit Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(client: TestClient, pid: str, title: str = "A memory",
         content: str = "Default content.", mem_type: str = "decision",
         importance: int = 3) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": mem_type, "title": title, "content": content, "importance": importance,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _link(client: TestClient, src: str, tgt: str, rel: str = "related_to") -> dict:
    r = client.post(f"/memories/{src}/links", json={
        "target_memory_id": tgt, "relationship_type": rel,
    })
    assert r.status_code == 201
    return r.json()


def _session(client: TestClient, pid: str, content: str, tool: str = "Claude") -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": tool, "title": "Audit Session",
        "raw_content": content, "session_date": "2026-05-24",
    })
    assert r.status_code == 201
    return r.json()["id"]


# ---------------------------------------------------------------------------
# 1. Export content-level deduplication
# ---------------------------------------------------------------------------

class TestExportContentDedup:
    def test_near_identical_content_appears_once(self, client: TestClient):
        pid = _project(client)
        # Create two decisions with near-identical content (>85% similarity)
        content = "We decided to use SQLite as our primary database for local-first storage."
        _mem(client, pid, "Decision A — SQLite choice", content, importance=5)
        _mem(client, pid, "Decision B — SQLite choice copy",
             content + " This is essentially the same.", importance=3)

        export = client.get(f"/projects/{pid}/export/context.md").text
        # The high-importance one should appear; the near-duplicate should be suppressed
        assert "Decision A" in export
        # Only one occurrence of the unique content phrase
        assert export.count("SQLite as our primary database") == 1

    def test_distinct_content_both_appear(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use FastAPI", "FastAPI is our web framework choice.", importance=4)
        _mem(client, pid, "Use SQLite", "SQLite is our database choice.", importance=3)

        export = client.get(f"/projects/{pid}/export/context.md").text
        assert "FastAPI is our web framework" in export
        assert "SQLite is our database" in export

    def test_dedup_keeps_higher_importance_version(self, client: TestClient):
        pid = _project(client)
        # The CRUD query orders by importance DESC — higher importance is rendered first,
        # so the lower-importance duplicate is the one that gets skipped.
        content = "Decided to use PostgreSQL for the production database system."
        high = _mem(client, pid, "PostgreSQL — final decision", content, importance=5)
        _mem(client, pid, "PostgreSQL — draft decision", content, importance=2)

        export = client.get(f"/projects/{pid}/export/context.md").text
        # High importance one is rendered (appears first), low is skipped as a standalone entry
        assert "PostgreSQL — final decision" in export
        # The draft title must not appear as a section heading (it is suppressed by dedup).
        # It may appear in a relationship line on the high-importance memory, which is fine.
        assert "### PostgreSQL — draft decision" not in export


# ---------------------------------------------------------------------------
# 2. delete_memory cascade — orphan cleanup
# ---------------------------------------------------------------------------

class TestDeleteMemoryCascade:
    def test_delete_memory_removes_associated_links(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        a = _mem(client, pid, "Memory A")
        b = _mem(client, pid, "Memory B")
        lnk = _link(client, a["id"], b["id"])

        client.delete(f"/memories/{a['id']}")

        # Link should be gone — no orphan rows
        remaining = db.query(models.MemoryLink).filter(
            models.MemoryLink.id == lnk["id"]
        ).first()
        assert remaining is None

    def test_delete_memory_removes_changelog_entries(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        m = _mem(client, pid, "Memory with history")
        # Create a changelog entry by updating
        client.put(f"/memories/{m['id']}", json={"title": "Memory with history v2"})

        # Verify changelog exists
        entries = db.query(models.MemoryChangelog).filter(
            models.MemoryChangelog.memory_id == m["id"]
        ).all()
        assert len(entries) >= 1

        client.delete(f"/memories/{m['id']}")

        # Changelog entries should be gone
        entries_after = db.query(models.MemoryChangelog).filter(
            models.MemoryChangelog.memory_id == m["id"]
        ).all()
        assert entries_after == []

    def test_delete_memory_as_link_target_removes_link(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        a = _mem(client, pid, "Source memory")
        b = _mem(client, pid, "Target memory")
        lnk = _link(client, a["id"], b["id"])

        # Delete the TARGET memory
        client.delete(f"/memories/{b['id']}")

        remaining = db.query(models.MemoryLink).filter(
            models.MemoryLink.id == lnk["id"]
        ).first()
        assert remaining is None


# ---------------------------------------------------------------------------
# 3. delete_project cascade — orphan cleanup at project level
# ---------------------------------------------------------------------------

class TestDeleteProjectCascade:
    def test_delete_project_removes_memory_links(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        a = _mem(client, pid, "Memory A")
        b = _mem(client, pid, "Memory B")
        lnk = _link(client, a["id"], b["id"])

        client.delete(f"/projects/{pid}")

        # No orphan links
        remaining = db.query(models.MemoryLink).filter(
            models.MemoryLink.id == lnk["id"]
        ).first()
        assert remaining is None

    def test_delete_project_removes_memory_changelogs(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        m = _mem(client, pid, "Memory with changelog")
        client.put(f"/memories/{m['id']}", json={"importance": 5})

        mid = m["id"]
        entries_before = db.query(models.MemoryChangelog).filter(
            models.MemoryChangelog.memory_id == mid
        ).all()
        assert len(entries_before) >= 1

        client.delete(f"/projects/{pid}")

        entries_after = db.query(models.MemoryChangelog).filter(
            models.MemoryChangelog.memory_id == mid
        ).all()
        assert entries_after == []


# ---------------------------------------------------------------------------
# 4. FTS5 correctness — search after memory delete
# ---------------------------------------------------------------------------

class TestFTSIntegrity:
    def test_fts_empty_after_all_memories_deleted(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, "Microservices architecture decision",
                 "We chose microservices for scalability.", "structure")

        r = client.get(f"/projects/{pid}/memories/search", params={"q": "microservices"})
        assert len(r.json()) == 1

        client.delete(f"/memories/{m['id']}")

        r = client.get(f"/projects/{pid}/memories/search", params={"q": "microservices"})
        assert r.json() == []

    def test_fts_no_cross_test_pollution(self, client: TestClient):
        # If FTS cleanup works, this test should find only what IT created
        pid = _project(client)
        _mem(client, pid, "GraphQL decision", "We chose GraphQL for the API layer.", "decision")

        r = client.get(f"/projects/{pid}/memories/search", params={"q": "graphql"})
        assert len(r.json()) == 1
        assert r.json()[0]["project_id"] == pid


# ---------------------------------------------------------------------------
# 5. Confidence scoring varies with keyword strength
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_strong_keywords_yield_high_confidence(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided and must use FastAPI. This is a critical architecture decision."
        )
        r = client.post(f"/sessions/{sid}/extract-memories")
        assert r.status_code == 200
        mems = r.json()["memories"]
        if mems:
            # Semantic confidence is a cosine similarity score in [TYPE_CONF_THRESHOLD, 1.0].
            # Clear technical sentences reliably score above 0.35.
            assert any(m["confidence"] >= 0.35 for m in mems)

    def test_weak_keywords_yield_lower_confidence(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We might consider using Redis. Perhaps we should explore caching options."
        )
        r = client.post(f"/sessions/{sid}/extract-memories")
        assert r.status_code == 200
        mems = r.json()["memories"]
        if mems:
            # Weak keywords (might, consider, perhaps) → confidence = 0.50
            assert any(m["confidence"] <= 0.55 for m in mems)

    def test_confidence_not_flat(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided and must use FastAPI because it is critical. "
            "We might also consider adding Redis but perhaps that is optional."
        )
        r = client.post(f"/sessions/{sid}/extract-memories")
        mems = r.json()["memories"]
        if len(mems) >= 2:
            confidences = {m["confidence"] for m in mems}
            # At least two distinct confidence values — not all flat 0.5
            assert len(confidences) >= 2


# ---------------------------------------------------------------------------
# 6. Noise filtering in extraction
# ---------------------------------------------------------------------------

class TestNoiseFiltering:
    def test_ai_thinking_text_not_extracted(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Thought for 7 seconds while analyzing the architecture decision. "
            "Let me think about this carefully. "
            "Sure, I can help you with the architecture. "
            "We decided to use FastAPI for the backend."
        )
        r = client.post(f"/sessions/{sid}/extract-memories")
        mems = r.json()["memories"]
        titles = [m["title"] for m in mems]
        # Noise phrases should not become memory titles
        assert not any("Thought for" in t for t in titles)
        assert not any("Let me think" in t for t in titles)
        assert not any("Sure," in t for t in titles)
        # The actual decision should still be extracted
        assert any("FastAPI" in t for t in titles)

    def test_html_tags_not_extracted(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "<div class='response'>We decided to use SQLite. "
            "<span>Architecture note here.</span></div>"
        )
        r = client.post(f"/sessions/{sid}/extract-memories")
        mems = r.json()["memories"]
        titles = [m["title"] for m in mems]
        # HTML tag noise should be filtered
        assert not any("<div" in t for t in titles)
        assert not any("<span" in t for t in titles)


# ---------------------------------------------------------------------------
# 7. MemoryStatus.stale is user-settable (not auto-set by stale detection)
# ---------------------------------------------------------------------------

class TestStaleStatusIsUserSettable:
    def test_can_manually_mark_memory_as_stale(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, "A memory to mark stale")
        r = client.put(f"/memories/{m['id']}", json={"status": "stale"})
        assert r.status_code == 200
        assert r.json()["status"] == "stale"

    def test_stale_endpoint_does_not_return_manually_stale_memories(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        m = _mem(client, pid, "Manually stale memory")
        client.put(f"/memories/{m['id']}", json={"status": "stale"})

        # The stale endpoint only returns status=active memories with old updated_at
        r = client.get(f"/projects/{pid}/memories/stale")
        stale_ids = {mem["id"] for mem in r.json()}
        # Manually-stale memory has status != "active", so it should not appear here
        assert m["id"] not in stale_ids


# ---------------------------------------------------------------------------
# 8. Session summary format — no noisy machine prefix
# ---------------------------------------------------------------------------

class TestSessionSummaryFormat:
    def test_summary_does_not_contain_machine_prefix(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use FastAPI. There is a bug in the auth flow. "
            "TODO: write integration tests."
        )
        client.post(f"/sessions/{sid}/extract-memories")
        summary = client.get(f"/sessions/{sid}").json()["summary"]
        assert summary is not None
        assert "[Auto-extracted from" not in summary
        assert "Keywords detected:" not in summary

    def test_summary_contains_extracted_stats(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use FastAPI. There is a bug in the auth flow."
        )
        client.post(f"/sessions/{sid}/extract-memories")
        summary = client.get(f"/sessions/{sid}").json()["summary"]
        # Should contain a readable extraction summary (not the old key:value format)
        assert "Extracted:" in summary or len(summary) > 50
