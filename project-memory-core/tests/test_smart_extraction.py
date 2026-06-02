"""Sub-phase 1.13 — Smart Extraction Engine tests."""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient) -> str:
    r = client.post("/projects", json={"name": "Extraction Test Project"})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, content: str, tool: str = "Claude") -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": tool,
        "title": "Test Session",
        "raw_content": content,
        "session_date": "2026-05-24",
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _extract(client: TestClient, sid: str) -> dict:
    r = client.post(f"/sessions/{sid}/extract-memories")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Test 1: Context window — content includes surrounding sentences
# ---------------------------------------------------------------------------

class TestContextWindow:
    def test_matched_sentence_includes_surrounding_context(self, client: TestClient):
        pid = _project(client)
        # Three clear sentences — decision is in the middle
        content = (
            "This is background context. "
            "We decided to use FastAPI for the backend API. "
            "This is follow-up context after the decision."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert result["memories_created"] >= 1
        decision_mem = next(
            (m for m in result["memories"] if m["type"] == "decision"), None
        )
        assert decision_mem is not None
        # Content should contain the sentence before AND after
        assert "background context" in decision_mem["content"]
        assert "follow-up context" in decision_mem["content"]

    def test_first_sentence_has_no_prev_context(self, client: TestClient):
        pid = _project(client)
        content = (
            "We decided to use SQLite as the primary database. "
            "SQLite requires no infrastructure setup."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        decision_mem = next(
            (m for m in result["memories"] if m["type"] == "decision"), None
        )
        assert decision_mem is not None
        # No sentence before — content starts with the decision sentence
        assert "decided" in decision_mem["content"].lower()
        # But includes the sentence after
        assert "infrastructure" in decision_mem["content"].lower()

    def test_last_sentence_has_no_next_context(self, client: TestClient):
        pid = _project(client)
        content = (
            "First sentence provides context. "
            "We decided to use Alembic for migrations."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        decision_mem = next(
            (m for m in result["memories"] if m["type"] == "decision"), None
        )
        assert decision_mem is not None
        # Includes sentence before
        assert "context" in decision_mem["content"].lower()


# ---------------------------------------------------------------------------
# Test 2: Section detection — markdown headers become tags
# ---------------------------------------------------------------------------

class TestSectionDetection:
    def test_header_becomes_tag(self, client: TestClient):
        pid = _project(client)
        content = (
            "## Architecture Decisions\n\n"
            "We decided to use a monorepo structure for the codebase."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert result["memories_created"] >= 1
        mem = result["memories"][0]
        assert "auto-extracted" in mem["tags"]
        # Header "Architecture Decisions" normalizes to "architecture decisions"
        assert any("architecture" in tag for tag in mem["tags"])

    def test_multiple_sections_different_tags(self, client: TestClient):
        pid = _project(client)
        content = (
            "## Bug Reports\n\n"
            "There is a critical error in the authentication module.\n\n"
            "## Tasks\n\n"
            "TODO: Add unit tests for the auth module."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert result["memories_created"] >= 2
        bug_mem = next((m for m in result["memories"] if m["type"] == "problem"), None)
        task_mem = next((m for m in result["memories"] if m["type"] == "task"), None)

        assert bug_mem is not None
        assert task_mem is not None
        # Bug is under "Bug Reports" header
        assert any("bug" in tag for tag in bug_mem["tags"])
        # Task is under "Tasks" header
        assert any("task" in tag for tag in task_mem["tags"])

    def test_no_header_means_no_section_tag(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL for the production database."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        if result["memories_created"] >= 1:
            mem = result["memories"][0]
            # Only the default "auto-extracted" tag — no section tag
            assert mem["tags"] == ["auto-extracted"]

    def test_h3_header_also_detected(self, client: TestClient):
        pid = _project(client)
        content = (
            "### Database Choices\n\n"
            "We decided to use Redis for session caching."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert result["memories_created"] >= 1
        mem = result["memories"][0]
        assert any("database" in tag for tag in mem["tags"])


# ---------------------------------------------------------------------------
# Test 3: Importance scoring — varies by type + keyword strength
# ---------------------------------------------------------------------------

class TestImportanceScoring:
    def test_decision_with_strong_keyword_scores_higher(self, client: TestClient):
        pid = _project(client)
        # "decided" is a strong keyword → importance should be 4
        content = "We decided to use FastAPI. This is a critical architecture choice."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        decision_mem = next(
            (m for m in result["memories"] if m["type"] == "decision"), None
        )
        assert decision_mem is not None
        # decision type_weight=4, strong keyword=1.0, position=1.0 → score=4
        assert decision_mem["importance"] >= 3

    def test_task_with_weak_keyword_scores_lower_than_decision(self, client: TestClient):
        pid = _project(client)
        content = (
            "We decided to use SQLite as the database. "
            "We should maybe consider adding an index later."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        decision_mem = next((m for m in result["memories"] if m["type"] == "decision"), None)
        task_mem = next((m for m in result["memories"] if m["type"] in ("task", "insight")), None)

        assert decision_mem is not None
        if task_mem:
            assert decision_mem["importance"] >= task_mem["importance"]

    def test_bug_with_strong_keyword_scores_high(self, client: TestClient):
        pid = _project(client)
        # "error" and "crash" are strong keywords
        content = "There is a critical error causing a crash in the auth module."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        bug_mem = next((m for m in result["memories"] if m["type"] == "problem"), None)
        assert bug_mem is not None
        # Bug type_weight=4; semantic confidence scales importance to 3–4 range.
        assert bug_mem["importance"] >= 3

    def test_importance_not_flat_two(self, client: TestClient):
        """All importances should NOT be the same flat value of 2."""
        pid = _project(client)
        content = (
            "We decided to use FastAPI. "
            "There is a critical error in the auth module. "
            "TODO: consider adding caching later."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        importances = [m["importance"] for m in result["memories"]]
        # At least one importance should differ from the old flat value of 2
        assert any(imp != 2 for imp in importances), f"All importances are flat: {importances}"

    def test_importance_within_valid_range(self, client: TestClient):
        pid = _project(client)
        content = (
            "We decided to use SQLite. "
            "There is a critical error and crash in the system. "
            "TODO: maybe consider caching."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        for mem in result["memories"]:
            assert 1 <= mem["importance"] <= 5, f"Importance out of range: {mem['importance']}"


# ---------------------------------------------------------------------------
# Test 4: Near-duplicate detection
# ---------------------------------------------------------------------------

class TestNearDuplicateDetection:
    def test_exact_duplicate_in_same_run_skipped(self, client: TestClient):
        pid = _project(client)
        # Same sentence repeated — second occurrence should be skipped
        content = (
            "We decided to use FastAPI for the API. "
            "Some context in the middle. "
            "We decided to use FastAPI for the API."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        # Should create 1 decision, not 2
        decision_mems = [m for m in result["memories"] if m["type"] == "decision"]
        assert len(decision_mems) == 1

    def test_near_duplicate_against_existing_memory_skipped(self, client: TestClient):
        pid = _project(client)
        # Create an existing memory that is very similar to what would be extracted
        client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "We decided to use FastAPI for the backend",
            "content": "FastAPI was chosen for performance.",
        })

        # Now extract a session with nearly the same decision
        content = "We decided to use FastAPI for the backend API layer."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        # Should be flagged as a duplicate and skipped
        assert result["skipped_duplicates"] >= 1
        assert result["memories_created"] == 0 or all(
            "fastapi" not in m["title"].lower() for m in result["memories"]
        )

    def test_skipped_duplicates_reported_in_result(self, client: TestClient):
        pid = _project(client)
        # Create an existing memory
        client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "We decided to use SQLite as the primary database",
            "content": "SQLite was chosen.",
        })

        # Extract a session with a nearly identical decision
        content = "We decided to use SQLite as the primary database store."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert "skipped_duplicates" in result
        assert "duplicate_titles" in result
        assert isinstance(result["duplicate_titles"], list)

    def test_dissimilar_memories_not_skipped(self, client: TestClient):
        pid = _project(client)
        # Existing memory about completely different topic
        client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "We decided to use Redis for caching",
            "content": "Redis was chosen.",
        })

        # Extract session about a different decision
        content = "We decided to use PostgreSQL for the production database."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        # Should NOT be skipped — too different
        assert result["memories_created"] >= 1
        assert result["skipped_duplicates"] == 0


# ---------------------------------------------------------------------------
# Test 5: File path extraction
# ---------------------------------------------------------------------------

class TestFilePathExtraction:
    def test_file_path_extracted_from_content(self, client: TestClient):
        pid = _project(client)
        content = (
            "There is a bug in app/routers/auth.py that causes a null pointer exception. "
            "The error occurs in the login handler."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        bug_mem = next((m for m in result["memories"] if m["type"] == "problem"), None)
        assert bug_mem is not None
        assert bug_mem["file_path"] == "app/routers/auth.py"
        assert "app/routers/auth.py" in bug_mem["related_files"]

    def test_multiple_file_paths_in_related_files(self, client: TestClient):
        pid = _project(client)
        content = (
            "There is a critical error across app/models.py and app/crud.py. "
            "Both files crash when the session is not refreshed."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        bug_mem = next((m for m in result["memories"] if m["type"] == "problem"), None)
        assert bug_mem is not None
        assert len(bug_mem["related_files"]) >= 2
        related = bug_mem["related_files"]
        assert any("models.py" in f for f in related)
        assert any("crud.py" in f for f in related)

    def test_no_file_path_when_none_present(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use FastAPI for the backend service."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        for mem in result["memories"]:
            assert mem["file_path"] is None
            assert mem["related_files"] == []

    def test_typescript_file_path_detected(self, client: TestClient):
        pid = _project(client)
        content = (
            "There is an error in src/components/Auth.tsx that crashes on invalid tokens. "
            "The exception is thrown in the login handler."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        bug_mem = next((m for m in result["memories"] if m["type"] == "problem"), None)
        assert bug_mem is not None
        assert bug_mem["file_path"] is not None
        assert ".tsx" in bug_mem["file_path"]


# ---------------------------------------------------------------------------
# Test 6: All features together — structured markdown session
# ---------------------------------------------------------------------------

class TestFullMarkdownSession:
    def test_structured_markdown_session(self, client: TestClient):
        pid = _project(client)
        content = """## Architecture Decisions

We decided to use a monorepo structure because it simplifies dependency management.
We chose FastAPI over Flask because of its performance and type safety.

## Bugs Found

There is a critical error in app/routers/sessions.py causing a crash on concurrent requests.
The exception occurs in the session middleware.

## Tasks

TODO: Add integration tests for the session router.
TODO: Fix the concurrent request handling in the session module."""

        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert result["memories_created"] >= 3

        # Check section tags are applied — header "Architecture" normalizes to "architecture"
        arch_mems = [m for m in result["memories"]
                     if any("architecture" in tag for tag in m["tags"])]
        assert len(arch_mems) >= 1

        bug_mems = [m for m in result["memories"] if m["type"] == "problem"]
        assert len(bug_mems) >= 1
        # Bug should have file path
        bugs_with_path = [m for m in bug_mems if m["file_path"] is not None]
        assert len(bugs_with_path) >= 1

        # Decisions should have higher importance than tasks
        decision_importances = [m["importance"] for m in result["memories"] if m["type"] == "decision"]
        task_importances = [m["importance"] for m in result["memories"] if m["type"] == "task"]
        if decision_importances and task_importances:
            assert max(decision_importances) >= max(task_importances)


# ---------------------------------------------------------------------------
# Test 7: Backward compatibility — existing extraction tests still work
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_basic_extraction_still_works(self, client: TestClient):
        pid = _project(client)
        content = (
            "We discussed the decision to use SQLite. "
            "There is a bug in the auth flow. "
            "TODO: write unit tests for the auth module."
        )
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert result["memories_created"] >= 1
        assert len(result["memories"]) == result["memories_created"]
        assert result["session_id"] == sid

    def test_result_has_all_required_fields(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use SQLite for local storage."
        sid = _session(client, pid, content)
        result = _extract(client, sid)

        assert "session_id" in result
        assert "memories_created" in result
        assert "summary" in result
        assert "memories" in result
        assert "skipped_duplicates" in result
        assert "duplicate_titles" in result

    def test_session_summary_updated(self, client: TestClient):
        pid = _project(client)
        content = "We discussed the decision to use SQLite. There is a bug in the auth flow."
        sid = _session(client, pid, content)
        _extract(client, sid)

        session = client.get(f"/sessions/{sid}").json()
        assert session["summary"] is not None
        assert len(session["summary"]) > 0
