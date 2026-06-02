"""Sub-phase 1.10 — Enhanced Memory Schema tests."""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient) -> str:
    r = client.post("/projects", json={"name": "Schema Test Project"})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, project_id: str, **overrides) -> dict:
    payload = {
        "type": "decision",
        "title": "Base decision",
        "content": "We chose SQLite for local storage.",
        **overrides,
    }
    r = client.post(f"/projects/{project_id}/memories", json=payload)
    assert r.status_code == 201
    return r.json()


# ---------------------------------------------------------------------------
# Test 1: Structured bug memory — type_metadata round-trips correctly
# ---------------------------------------------------------------------------

class TestBugMetadata:
    def test_bug_memory_with_type_metadata(self, client: TestClient):
        pid = _project(client)
        metadata = {
            "error_message": "AttributeError: 'NoneType' object has no attribute 'id'",
            "root_cause": "Session not committed before accessing relationship",
            "fix_applied": "Added db.refresh() after db.commit()",
            "environment": "Python 3.12, SQLAlchemy 2.x",
            "affected_files": ["app/crud.py", "app/routers/sessions.py"],
        }
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "problem",
            "title": "NoneType on session relationship",
            "content": "Accessing memory.source_session after commit raised AttributeError.",
            "type_metadata": metadata,
        })
        assert r.status_code == 201
        body = r.json()
        assert body["type_metadata"] == metadata
        assert body["type_metadata"]["root_cause"] == "Session not committed before accessing relationship"
        assert body["type_metadata"]["affected_files"] == ["app/crud.py", "app/routers/sessions.py"]

    def test_bug_memory_metadata_persists_on_get(self, client: TestClient):
        pid = _project(client)
        metadata = {"error_message": "KeyError: 'token'", "root_cause": "Missing env var"}
        mem = _memory(client, pid, type="problem", type_metadata=metadata)
        r = client.get(f"/memories/{mem['id']}")
        assert r.status_code == 200
        assert r.json()["type_metadata"] == metadata


# ---------------------------------------------------------------------------
# Test 2: Structured decision memory — alternatives_considered as list
# ---------------------------------------------------------------------------

class TestDecisionMetadata:
    def test_decision_with_alternatives(self, client: TestClient):
        pid = _project(client)
        metadata = {
            "rationale": "SQLite requires no infra, perfect for local-first.",
            "alternatives_considered": ["PostgreSQL", "DuckDB", "TinyDB"],
            "consequences": "Limits concurrent write throughput.",
            "decision_status": "accepted",
        }
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Use SQLite as primary store",
            "content": "Chose SQLite over Postgres for zero-infra local deployment.",
            "type_metadata": metadata,
        })
        assert r.status_code == 201
        body = r.json()
        assert body["type_metadata"]["alternatives_considered"] == ["PostgreSQL", "DuckDB", "TinyDB"]
        assert body["type_metadata"]["decision_status"] == "accepted"


# ---------------------------------------------------------------------------
# Test 3: Code-anchored memory — file_path and commit_sha
# ---------------------------------------------------------------------------

class TestCodeAnchoring:
    def test_memory_with_file_path_and_commit_sha(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "structure",
            "title": "Memory model structure",
            "content": "The Memory ORM model lives in app/models.py.",
            "file_path": "app/models.py",
            "commit_sha": "abc123def456",
        })
        assert r.status_code == 201
        body = r.json()
        assert body["file_path"] == "app/models.py"
        assert body["commit_sha"] == "abc123def456"

    def test_memory_without_anchoring_fields(self, client: TestClient):
        pid = _project(client)
        mem = _memory(client, pid)
        assert mem["file_path"] is None
        assert mem["commit_sha"] is None

    def test_file_path_persists_on_get(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "problem",
            "title": "Import error in exports",
            "content": "Missing import for PlainTextResponse.",
            "file_path": "app/routers/exports.py",
            "commit_sha": "deadbeef",
        })
        mem_id = r.json()["id"]
        r2 = client.get(f"/memories/{mem_id}")
        assert r2.json()["file_path"] == "app/routers/exports.py"
        assert r2.json()["commit_sha"] == "deadbeef"


# ---------------------------------------------------------------------------
# Test 4: workflow_pattern type (new enum value)
# ---------------------------------------------------------------------------

class TestWorkflowPatternType:
    def test_workflow_pattern_memory(self, client: TestClient):
        pid = _project(client)
        metadata = {
            "trigger": "starting a new API endpoint",
            "steps": [
                "add alembic migration",
                "write failing test first",
                "implement router",
                "implement crud",
                "run tests",
            ],
            "tools_used": ["alembic", "pytest", "fastapi"],
        }
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "workflow_pattern",
            "title": "New endpoint development workflow",
            "content": "Standard process for adding endpoints to the API.",
            "type_metadata": metadata,
        })
        assert r.status_code == 201
        body = r.json()
        assert body["type"] == "workflow_pattern"
        assert body["type_metadata"]["steps"][0] == "add alembic migration"
        assert body["type_metadata"]["tools_used"] == ["alembic", "pytest", "fastapi"]


# ---------------------------------------------------------------------------
# Test 5: Temporal invalidation — superseded_by and valid_until
# ---------------------------------------------------------------------------

class TestTemporalInvalidation:
    def test_supersede_a_memory(self, client: TestClient):
        pid = _project(client)

        # Create the original decision
        mem_a = _memory(client, pid, title="Use Flask for web framework")
        mem_b = _memory(client, pid, title="Use FastAPI for web framework")

        now_iso = datetime.now(timezone.utc).isoformat()

        # Mark mem_a as superseded, pointing to mem_b
        r = client.put(f"/memories/{mem_a['id']}", json={
            "status": "superseded",
            "superseded_by": mem_b["id"],
            "valid_until": now_iso,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "superseded"
        assert body["superseded_by"] == mem_b["id"]
        assert body["valid_until"] is not None

    def test_superseded_memory_preserved_on_get(self, client: TestClient):
        pid = _project(client)
        mem_a = _memory(client, pid, title="Old architecture decision")
        mem_b = _memory(client, pid, title="New architecture decision")

        now_iso = datetime.now(timezone.utc).isoformat()
        client.put(f"/memories/{mem_a['id']}", json={
            "status": "superseded",
            "superseded_by": mem_b["id"],
            "valid_until": now_iso,
        })

        # Original memory still retrievable — never deleted
        r = client.get(f"/memories/{mem_a['id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "superseded"
        assert body["superseded_by"] == mem_b["id"]
        assert body["valid_until"] is not None

    def test_superseded_status_is_valid_enum(self, client: TestClient):
        pid = _project(client)
        mem = _memory(client, pid)
        r = client.put(f"/memories/{mem['id']}", json={"status": "superseded"})
        assert r.status_code == 200
        assert r.json()["status"] == "superseded"


# ---------------------------------------------------------------------------
# Test 6: Schema validation — type_metadata must be dict if provided
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_null_type_metadata_is_allowed(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "problem",
            "title": "No metadata bug",
            "content": "This bug has no structured metadata.",
        })
        assert r.status_code == 201
        assert r.json()["type_metadata"] is None

    def test_non_dict_type_metadata_rejected(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "problem",
            "title": "Bad metadata",
            "content": "This should fail.",
            "type_metadata": "just a string — not a dict",
        })
        assert r.status_code == 422

    def test_type_metadata_list_rejected(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Bad metadata",
            "content": "List instead of dict.",
            "type_metadata": ["item1", "item2"],
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Test 7: New fields absent means None (backwards compat with existing memories)
# ---------------------------------------------------------------------------

class TestDefaultsForNewFields:
    def test_all_new_fields_default_to_none(self, client: TestClient):
        pid = _project(client)
        mem = _memory(client, pid)
        assert mem["type_metadata"] is None
        assert mem["file_path"] is None
        assert mem["commit_sha"] is None
        assert mem["superseded_by"] is None
        assert mem["valid_until"] is None

    def test_all_new_fields_present_in_list_response(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        r = client.get(f"/projects/{pid}/memories")
        assert r.status_code == 200
        for mem in r.json():
            assert "type_metadata" in mem
            assert "file_path" in mem
            assert "commit_sha" in mem
            assert "superseded_by" in mem
            assert "valid_until" in mem
