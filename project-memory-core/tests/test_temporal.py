"""Sub-phase 1.14 — Temporal Invalidation + Audit Log + Stale Signals tests."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient) -> str:
    r = client.post("/projects", json={"name": "Temporal Test Project"})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(client, pid, title="Decision A", mem_type="decision", importance=3) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": mem_type,
        "title": title,
        "content": f"Content for {title}",
        "importance": importance,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _age_memory(db: Session, memory_id: str, days: int = 35) -> None:
    """Force a memory's updated_at to a past date to trigger stale detection."""
    past = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    db.execute(
        text("UPDATE memories SET updated_at = :dt WHERE id = :id"),
        {"dt": past, "id": memory_id},
    )
    db.commit()


# ---------------------------------------------------------------------------
# Test 1: supersede endpoint — atomic replacement
# ---------------------------------------------------------------------------

class TestSupersede:
    def test_supersede_creates_new_memory(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Flask for backend")

        r = client.post(f"/memories/{old['id']}/supersede", json={
            "type": "decision",
            "title": "Use FastAPI for backend",
            "content": "Switching to FastAPI for better performance.",
        })
        assert r.status_code == 201
        body = r.json()
        assert "old_memory" in body
        assert "new_memory" in body
        assert body["new_memory"]["title"] == "Use FastAPI for backend"

    def test_supersede_marks_old_as_superseded(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use SQLite for production")

        r = client.post(f"/memories/{old['id']}/supersede", json={
            "type": "decision",
            "title": "Use PostgreSQL for production",
            "content": "Scaling beyond SQLite limits.",
        })
        assert r.status_code == 201
        old_mem = r.json()["old_memory"]
        assert old_mem["status"] == "superseded"
        assert old_mem["valid_until"] is not None
        assert old_mem["superseded_by"] == r.json()["new_memory"]["id"]

    def test_old_memory_still_retrievable_after_supersede(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Old decision")

        r = client.post(f"/memories/{old['id']}/supersede", json={
            "type": "decision",
            "title": "New decision",
            "content": "Replacement decision content.",
        })
        old_id = r.json()["old_memory"]["id"]

        # Old memory still accessible — never deleted
        get_r = client.get(f"/memories/{old_id}")
        assert get_r.status_code == 200
        assert get_r.json()["status"] == "superseded"

    def test_supersede_nonexistent_memory_404(self, client: TestClient):
        r = client.post("/memories/nonexistent-id/supersede", json={
            "type": "decision",
            "title": "New decision",
            "content": "Content",
        })
        assert r.status_code == 404

    def test_new_memory_is_active_and_in_same_project(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Old arch decision")

        r = client.post(f"/memories/{old['id']}/supersede", json={
            "type": "structure",
            "title": "New arch decision",
            "content": "Updated architecture choice.",
        })
        new_mem = r.json()["new_memory"]
        assert new_mem["status"] == "active"
        assert new_mem["project_id"] == pid

    def test_supersede_writes_changelog_on_old_memory(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Decision to supersede")

        client.post(f"/memories/{old['id']}/supersede", json={
            "type": "decision",
            "title": "Replacement decision",
            "content": "New content.",
        })

        history = client.get(f"/memories/{old['id']}/history").json()
        fields_changed = {entry["field"] for entry in history}
        assert "status" in fields_changed
        assert "superseded_by" in fields_changed
        assert "valid_until" in fields_changed


# ---------------------------------------------------------------------------
# Test 2: changelog — update_memory writes audit entries
# ---------------------------------------------------------------------------

class TestChangelog:
    def test_update_title_creates_changelog_entry(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Original title")

        client.put(f"/memories/{mem['id']}", json={"title": "Updated title"})

        r = client.get(f"/memories/{mem['id']}/history")
        assert r.status_code == 200
        entries = r.json()
        assert len(entries) >= 1

        title_entry = next((e for e in entries if e["field"] == "title"), None)
        assert title_entry is not None
        assert title_entry["old_value"] == "Original title"
        assert title_entry["new_value"] == "Updated title"

    def test_update_status_logged(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Bug to resolve")
        client.put(f"/memories/{mem['id']}", json={"status": "resolved"})

        entries = client.get(f"/memories/{mem['id']}/history").json()
        status_entry = next((e for e in entries if e["field"] == "status"), None)
        assert status_entry is not None
        assert status_entry["old_value"] == "active"
        assert status_entry["new_value"] == "resolved"

    def test_update_importance_logged(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Memory to reprioritize", importance=2)
        client.put(f"/memories/{mem['id']}", json={"importance": 5})

        entries = client.get(f"/memories/{mem['id']}/history").json()
        imp_entry = next((e for e in entries if e["field"] == "importance"), None)
        assert imp_entry is not None
        assert imp_entry["old_value"] == "2"
        assert imp_entry["new_value"] == "5"

    def test_multiple_updates_accumulate_history(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Evolving decision")

        client.put(f"/memories/{mem['id']}", json={"title": "Evolving decision v2"})
        client.put(f"/memories/{mem['id']}", json={"title": "Evolving decision v3"})
        client.put(f"/memories/{mem['id']}", json={"importance": 5})

        entries = client.get(f"/memories/{mem['id']}/history").json()
        assert len(entries) >= 3

    def test_no_change_no_changelog(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Stable decision", importance=3)

        # Update with the same value — should not write changelog
        client.put(f"/memories/{mem['id']}", json={"importance": 3})

        entries = client.get(f"/memories/{mem['id']}/history").json()
        # No change happened, so no entries
        assert len(entries) == 0

    def test_history_ordered_chronologically(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Decision with history")

        client.put(f"/memories/{mem['id']}", json={"title": "Second version"})
        client.put(f"/memories/{mem['id']}", json={"title": "Third version"})

        entries = client.get(f"/memories/{mem['id']}/history").json()
        assert len(entries) >= 2
        # Entries should be oldest first
        times = [e["changed_at"] for e in entries]
        assert times == sorted(times)

    def test_history_memory_not_found_404(self, client: TestClient):
        r = client.get("/memories/nonexistent-id/history")
        assert r.status_code == 404

    def test_fresh_memory_has_empty_history(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, "Brand new memory")
        entries = client.get(f"/memories/{mem['id']}/history").json()
        assert entries == []


# ---------------------------------------------------------------------------
# Test 3: Stale memories endpoint
# ---------------------------------------------------------------------------

class TestStaleMemories:
    def test_stale_endpoint_returns_old_memories(self, client: TestClient, db: Session):
        pid = _project(client)
        old_mem = _mem(client, pid, "Old active decision")
        fresh_mem = _mem(client, pid, "Fresh active decision")

        # Age the first memory beyond the 30-day threshold
        _age_memory(db, old_mem["id"], days=40)

        r = client.get(f"/projects/{pid}/memories/stale")
        assert r.status_code == 200
        stale_ids = {m["id"] for m in r.json()}
        assert old_mem["id"] in stale_ids
        assert fresh_mem["id"] not in stale_ids

    def test_stale_only_returns_active_memories(self, client: TestClient, db: Session):
        pid = _project(client)
        active_old = _mem(client, pid, "Old active memory")
        resolved_old = _mem(client, pid, "Old resolved memory")

        # Age both
        _age_memory(db, active_old["id"], days=40)
        _age_memory(db, resolved_old["id"], days=40)

        # Mark one as resolved
        client.put(f"/memories/{resolved_old['id']}", json={"status": "resolved"})

        r = client.get(f"/projects/{pid}/memories/stale")
        stale_ids = {m["id"] for m in r.json()}
        assert active_old["id"] in stale_ids
        assert resolved_old["id"] not in stale_ids

    def test_custom_days_threshold(self, client: TestClient, db: Session):
        pid = _project(client)
        mem_7_days = _mem(client, pid, "Week old memory")
        mem_60_days = _mem(client, pid, "Two month old memory")

        _age_memory(db, mem_7_days["id"], days=7)
        _age_memory(db, mem_60_days["id"], days=60)

        # Default 30-day threshold: only 60-day memory is stale
        r30 = client.get(f"/projects/{pid}/memories/stale")
        stale_ids_30 = {m["id"] for m in r30.json()}
        assert mem_60_days["id"] in stale_ids_30
        assert mem_7_days["id"] not in stale_ids_30

        # Custom 5-day threshold: both are stale
        r5 = client.get(f"/projects/{pid}/memories/stale", params={"days": 5})
        stale_ids_5 = {m["id"] for m in r5.json()}
        assert mem_7_days["id"] in stale_ids_5
        assert mem_60_days["id"] in stale_ids_5

    def test_stale_returns_empty_when_all_fresh(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Fresh memory")
        r = client.get(f"/projects/{pid}/memories/stale")
        assert r.status_code == 200
        assert r.json() == []

    def test_stale_project_not_found(self, client: TestClient):
        r = client.get("/projects/nonexistent-id/memories/stale")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 4: Export shows stale signal + superseded section
# ---------------------------------------------------------------------------

class TestExportTemporalSignals:
    def test_export_shows_superseded_section(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Old database decision")

        client.post(f"/memories/{old['id']}/supersede", json={
            "type": "decision",
            "title": "New database decision",
            "content": "Updated database choice.",
        })

        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200
        content = r.text
        assert "Superseded" in content
        assert "Old database decision" in content

    def test_export_marks_stale_memory(self, client: TestClient, db: Session):
        pid = _project(client)
        stale = _mem(client, pid, "Stale architecture decision")
        _age_memory(db, stale["id"], days=45)

        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200
        assert "POSSIBLY STALE" in r.text

    def test_export_does_not_mark_fresh_memory_stale(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Fresh architecture decision")

        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200
        assert "POSSIBLY STALE" not in r.text


# ---------------------------------------------------------------------------
# Test 5: Full temporal lifecycle
# ---------------------------------------------------------------------------

class TestFullTemporalLifecycle:
    def test_full_supersede_lifecycle(self, client: TestClient):
        """Create → update (changelog) → supersede → verify history + export."""
        pid = _project(client)

        # Step 1: Create
        original = _mem(client, pid, "Use Flask for API", mem_type="decision")

        # Step 2: Update it (should log changelog)
        client.put(f"/memories/{original['id']}", json={
            "content": "Flask was our initial choice.",
            "importance": 4,
        })

        # Step 3: Supersede with FastAPI
        r = client.post(f"/memories/{original['id']}/supersede", json={
            "type": "decision",
            "title": "Use FastAPI for API",
            "content": "Switched to FastAPI for performance.",
        })
        assert r.status_code == 201
        new_id = r.json()["new_memory"]["id"]

        # Step 4: Old is superseded, new is active
        old_r = client.get(f"/memories/{original['id']}")
        assert old_r.json()["status"] == "superseded"
        assert old_r.json()["superseded_by"] == new_id

        new_r = client.get(f"/memories/{new_id}")
        assert new_r.json()["status"] == "active"

        # Step 5: History has multiple entries
        history = client.get(f"/memories/{original['id']}/history").json()
        assert len(history) >= 4  # content, importance, status, valid_until, superseded_by

        # Step 6: Export shows both active and superseded sections
        export = client.get(f"/projects/{pid}/export/context.md").text
        assert "Use FastAPI for API" in export
        assert "Superseded" in export
        assert "Use Flask for API" in export
