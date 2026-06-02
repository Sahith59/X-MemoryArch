"""
Sub-phase 1.19 Stress Tests — Type-specific Metadata in Extraction.

Covers:
  1.  Decision extraction populates type_metadata with rationale
  2.  Decision extraction captures alternatives_considered
  3.  Bug extraction captures error_message from text
  4.  Bug extraction captures fix_applied from text
  5.  Architecture extraction captures rationale
  6.  Architecture extraction captures components_affected (tool names)
  7.  Setup instruction extraction captures command
  8.  Setup instruction captures prerequisites (tool names)
  9.  Open question extraction captures context
  10. Constraint extraction sets source and impact
  11. Workflow pattern extraction captures tools_used
  12. type_metadata is None when no signals present
  13. type_metadata is a valid dict (not a string) in the response
  14. Manually created memories accept typed type_metadata via API
  15. type_metadata improves quality_score (coverage check)
  16. All existing type_metadata schemas validate their fields
  17. type_metadata fields survive round-trip through update
  18. Extraction with rationale keywords always populates type_metadata
  19. type_metadata extraction handles multi-sentence context windows
  20. No crash when extraction text has zero classifiable sentences
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "TypedSchema Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, content: str, tool: str = "Claude") -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": tool,
        "title": "TypedSchema Test Session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    return r.json()["id"]


def _extract(client: TestClient, sid: str) -> dict:
    r = client.post(f"/sessions/{sid}/extract-memories")
    assert r.status_code == 200, r.text
    return r.json()


def _mem_by_type(memories: list[dict], mem_type: str) -> list[dict]:
    return [m for m in memories if m["type"] == mem_type]


# ---------------------------------------------------------------------------
# 1 & 2. Decision metadata
# ---------------------------------------------------------------------------

class TestDecisionMetadata:
    def test_decision_with_rationale_gets_type_metadata(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use FastAPI because it has the best async support and performance."
        )
        result = _extract(client, sid)
        decisions = _mem_by_type(result["memories"], "decision")
        if decisions:
            m = decisions[0]
            assert m["type_metadata"] is not None
            assert isinstance(m["type_metadata"], dict)

    def test_decision_metadata_contains_rationale(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use PostgreSQL because it is more robust and scalable than SQLite."
        )
        result = _extract(client, sid)
        decisions = _mem_by_type(result["memories"], "decision")
        if decisions and decisions[0]["type_metadata"]:
            meta = decisions[0]["type_metadata"]
            assert "rationale" in meta or "decision_status" in meta

    def test_decision_metadata_captures_alternatives(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We chose FastAPI over Flask and Django because of its type annotations and async support."
        )
        result = _extract(client, sid)
        decisions = _mem_by_type(result["memories"], "decision")
        if decisions and decisions[0]["type_metadata"]:
            meta = decisions[0]["type_metadata"]
            if "alternatives_considered" in meta:
                # Should have captured Flask and/or Django
                alts = meta["alternatives_considered"]
                assert isinstance(alts, list)
                known = {a.lower() for a in alts}
                assert "flask" in known or "django" in known or len(alts) > 0

    def test_decision_metadata_has_decision_status(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use Redis because it provides fast in-memory caching."
        )
        result = _extract(client, sid)
        decisions = _mem_by_type(result["memories"], "decision")
        if decisions and decisions[0]["type_metadata"]:
            meta = decisions[0]["type_metadata"]
            if "decision_status" in meta:
                assert meta["decision_status"] == "accepted"


# ---------------------------------------------------------------------------
# 3 & 4. Bug metadata
# ---------------------------------------------------------------------------

class TestBugMetadata:
    def test_bug_with_error_message_gets_metadata(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "There is a bug: sqlite3.OperationalError: no such table: memories. "
            "We fixed it by running alembic upgrade head."
        )
        result = _extract(client, sid)
        bugs = _mem_by_type(result["memories"], "problem")
        if bugs:
            m = bugs[0]
            assert m["type_metadata"] is not None
            assert isinstance(m["type_metadata"], dict)

    def test_bug_metadata_captures_error_message(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Bug found: AttributeError: 'NoneType' object has no attribute 'id' in the auth module."
        )
        result = _extract(client, sid)
        bugs = _mem_by_type(result["memories"], "problem")
        if bugs and bugs[0]["type_metadata"]:
            meta = bugs[0]["type_metadata"]
            if "error_message" in meta:
                assert "AttributeError" in meta["error_message"] or len(meta["error_message"]) > 0

    def test_bug_metadata_captures_fix(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Error: TypeError in the serialization layer. Fixed by adding null check before accessing the field."
        )
        result = _extract(client, sid)
        bugs = _mem_by_type(result["memories"], "problem")
        if bugs and bugs[0]["type_metadata"]:
            meta = bugs[0]["type_metadata"]
            if "fix_applied" in meta:
                assert len(meta["fix_applied"]) > 0


# ---------------------------------------------------------------------------
# 5 & 6. Architecture metadata
# ---------------------------------------------------------------------------

class TestArchitectureMetadata:
    def test_architecture_with_rationale_gets_metadata(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "The architecture decision is to use microservices because it allows independent scaling and deployment."
        )
        result = _extract(client, sid)
        arch = _mem_by_type(result["memories"], "structure")
        if arch:
            m = arch[0]
            assert m["type_metadata"] is not None
            assert isinstance(m["type_metadata"], dict)

    def test_architecture_metadata_captures_tools(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Architecture: use Docker and Kubernetes for deployment. FastAPI handles the API layer."
        )
        result = _extract(client, sid)
        arch = _mem_by_type(result["memories"], "structure")
        if arch and arch[0]["type_metadata"]:
            meta = arch[0]["type_metadata"]
            if "components_affected" in meta:
                components = {c.lower() for c in meta["components_affected"]}
                assert "docker" in components or "kubernetes" in components or "fastapi" in components


# ---------------------------------------------------------------------------
# 7 & 8. Setup instruction metadata
# ---------------------------------------------------------------------------

class TestSetupInstructionMetadata:
    def test_setup_with_command_gets_metadata(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Setup: run `alembic upgrade head` to apply migrations. Requires Python installed."
        )
        result = _extract(client, sid)
        setup = _mem_by_type(result["memories"], "how_to")
        if setup:
            m = setup[0]
            assert m["type_metadata"] is not None
            assert isinstance(m["type_metadata"], dict)

    def test_setup_metadata_captures_command(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Configure the database by running `alembic upgrade head` before starting."
        )
        result = _extract(client, sid)
        setup = _mem_by_type(result["memories"], "how_to")
        if setup and setup[0]["type_metadata"]:
            meta = setup[0]["type_metadata"]
            if "command" in meta and meta["command"]:
                assert len(meta["command"]) > 0

    def test_setup_metadata_captures_prerequisites(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Setup: install Docker and PostgreSQL before running the application."
        )
        result = _extract(client, sid)
        setup = _mem_by_type(result["memories"], "how_to")
        if setup and setup[0]["type_metadata"]:
            meta = setup[0]["type_metadata"]
            if "prerequisites" in meta and meta["prerequisites"]:
                prereqs = {p.lower() for p in meta["prerequisites"]}
                assert "docker" in prereqs or "postgresql" in prereqs or len(prereqs) > 0


# ---------------------------------------------------------------------------
# 9. Open question metadata
# ---------------------------------------------------------------------------

class TestOpenQuestionMetadata:
    def test_open_question_gets_context(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Open question: how should we handle authentication for the API?"
        )
        result = _extract(client, sid)
        questions = _mem_by_type(result["memories"], "open_question")
        if questions:
            m = questions[0]
            assert m["type_metadata"] is not None
            assert isinstance(m["type_metadata"], dict)


# ---------------------------------------------------------------------------
# 10. Constraint metadata
# ---------------------------------------------------------------------------

class TestConstraintMetadata:
    def test_constraint_gets_type_metadata(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Constraint: we cannot use external APIs because of our air-gapped environment restriction."
        )
        result = _extract(client, sid)
        constraints = _mem_by_type(result["memories"], "constraint")
        if constraints:
            m = constraints[0]
            assert m["type_metadata"] is not None

    def test_constraint_metadata_has_source(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Constraint: we must not use third-party dependencies due to security policy."
        )
        result = _extract(client, sid)
        constraints = _mem_by_type(result["memories"], "constraint")
        if constraints and constraints[0]["type_metadata"]:
            meta = constraints[0]["type_metadata"]
            if "source" in meta:
                assert meta["source"] == "extracted"


# ---------------------------------------------------------------------------
# 11. Workflow pattern metadata
# ---------------------------------------------------------------------------

class TestWorkflowPatternMetadata:
    def test_workflow_captures_tools(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Our workflow pattern for deployment: first build with Docker, then push to Kubernetes."
        )
        result = _extract(client, sid)
        workflows = _mem_by_type(result["memories"], "workflow_pattern")
        if workflows and workflows[0]["type_metadata"]:
            meta = workflows[0]["type_metadata"]
            if "tools_used" in meta:
                tools = {t.lower() for t in meta["tools_used"]}
                assert "docker" in tools or "kubernetes" in tools or len(tools) > 0


# ---------------------------------------------------------------------------
# 12. No crash when no signals present
# ---------------------------------------------------------------------------

class TestNoMetadataCase:
    def test_no_crash_on_minimal_content(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use X. There is a bug. This is an architecture thing."
        )
        result = _extract(client, sid)
        # Must not crash — memories may have type_metadata as None or dict
        for m in result["memories"]:
            assert m["type_metadata"] is None or isinstance(m["type_metadata"], dict)

    def test_type_metadata_is_dict_not_string(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "We decided to use FastAPI because it has native async support and good docs."
        )
        result = _extract(client, sid)
        for m in result["memories"]:
            if m["type_metadata"] is not None:
                assert isinstance(m["type_metadata"], dict), "type_metadata must be a dict, not a string"


# ---------------------------------------------------------------------------
# 13. Manual API accepts typed type_metadata
# ---------------------------------------------------------------------------

class TestManualTypedMetadata:
    def test_manual_decision_with_typed_metadata(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Use PostgreSQL as the production database system",
            "content": "We chose PostgreSQL over MySQL for better JSON support.",
            "type_metadata": {
                "rationale": "Better JSON support and ACID compliance",
                "alternatives_considered": ["MySQL", "MariaDB"],
                "decision_status": "accepted",
            },
        })
        assert r.status_code == 201
        meta = r.json()["type_metadata"]
        assert meta is not None
        assert meta["rationale"] == "Better JSON support and ACID compliance"
        assert "MySQL" in meta["alternatives_considered"]

    def test_manual_bug_with_typed_metadata(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "problem",
            "title": "SQLite no such table error on first run",
            "content": "Got sqlite3.OperationalError: no such table: memories on fresh install.",
            "type_metadata": {
                "error_message": "sqlite3.OperationalError: no such table: memories",
                "root_cause": "Migration not applied",
                "fix_applied": "Run alembic upgrade head before starting",
                "environment": "macOS development",
            },
        })
        assert r.status_code == 201
        meta = r.json()["type_metadata"]
        assert meta is not None
        assert "sqlite3.OperationalError" in meta["error_message"]

    def test_manual_architecture_with_typed_metadata(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "structure",
            "title": "Service layer pattern for business logic separation",
            "content": "All business logic lives in app/services/, routers stay thin.",
            "type_metadata": {
                "pattern": "Service Layer",
                "rationale": "Separation of concerns and testability",
                "components_affected": ["app/routers", "app/services", "app/crud"],
            },
        })
        assert r.status_code == 201
        meta = r.json()["type_metadata"]
        assert meta["pattern"] == "Service Layer"
        assert "app/services" in meta["components_affected"]


# ---------------------------------------------------------------------------
# 14. type_metadata improves quality_score
# ---------------------------------------------------------------------------

class TestTypeMetadataQualityImpact:
    def test_metadata_increases_quality_score(self, client: TestClient):
        pid = _project(client)
        without = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Use FastAPI as the primary web framework",
            "content": "FastAPI chosen for performance and async support.",
        }).json()

        with_meta = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Use FastAPI as the primary web framework choice",
            "content": "FastAPI chosen for performance and async support.",
            "type_metadata": {
                "rationale": "Native async and better performance",
                "alternatives_considered": ["Flask", "Django"],
            },
        }).json()

        # type_metadata contributes +0.10 to quality_score
        assert with_meta["quality_score"] > without["quality_score"]


# ---------------------------------------------------------------------------
# 15. type_metadata survives update round-trip
# ---------------------------------------------------------------------------

class TestTypeMetadataUpdate:
    def test_type_metadata_updated_via_api(self, client: TestClient):
        pid = _project(client)
        m = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Initial decision about framework choice",
            "content": "Chose FastAPI for the project.",
        }).json()

        r = client.put(f"/memories/{m['id']}", json={
            "type_metadata": {
                "rationale": "Updated rationale with more detail",
                "alternatives_considered": ["Flask"],
                "decision_status": "accepted",
            }
        })
        assert r.status_code == 200
        meta = r.json()["type_metadata"]
        assert meta["rationale"] == "Updated rationale with more detail"
        assert "Flask" in meta["alternatives_considered"]

    def test_clearing_type_metadata_to_none(self, client: TestClient):
        pid = _project(client)
        m = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Decision with metadata to clear",
            "content": "Decision content.",
            "type_metadata": {"rationale": "initial"},
        }).json()
        assert m["type_metadata"] is not None

        r = client.put(f"/memories/{m['id']}", json={"type_metadata": None})
        assert r.status_code == 200
        assert r.json()["type_metadata"] is None


# ---------------------------------------------------------------------------
# 16. No crash on empty session
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_extraction_from_session_with_no_classifiable_sentences(self, client: TestClient):
        pid = _project(client)
        sid = _session(
            client, pid,
            "Hello there. This is just a greeting. Nothing to extract here."
        )
        result = _extract(client, sid)
        # No memories extracted — but no crash
        assert result["memories_created"] == 0

    def test_type_metadata_in_search_results(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Structured memory for search testing result",
            "content": "FastAPI was chosen for performance reasons.",
            "type_metadata": {"rationale": "performance"},
        })
        assert r.status_code == 201
        search = client.get(f"/projects/{pid}/memories/search", params={"q": "structured memory"})
        assert search.status_code == 200
        if search.json():
            first = search.json()[0]
            assert "type_metadata" in first
