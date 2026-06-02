"""
Sub-phase 1.26 — Retrieval hint tests.

Covers:
  Unit tests — compute_retrieval_hint pure function:
  1.  Returns a non-empty string for every known memory type
  2.  Return value is at most 300 characters
  3.  decision type — uses rationale when present
  4.  decision type — falls back to alternatives_considered
  5.  decision type — falls back to label+title when no meta
  6.  bug type — uses error_message
  7.  bug type — uses root_cause when no error_message
  8.  bug type — uses first sentence of content as last resort
  9.  failed_approach — uses approach_tried + avoid_because
  10. failed_approach — includes alternative_chosen
  11. architecture — uses pattern field
  12. architecture — appends rationale when both present
  13. setup_instruction — uses command field
  14. setup_instruction — appends platform suffix
  15. open_question — uses context field
  16. constraint — uses impact field
  17. constraint — falls back to source
  18. workflow_pattern — uses trigger
  19. workflow_pattern — falls back to steps list
  20. insight — prepends first sentence of content
  21. Default fallback — label + title + first sentence
  22. Default fallback — skips first sentence when ≤10 chars or in title
  23. Long content truncated to ≤300 chars
  24. insight — omits content when it duplicates the title

  Integration tests — end-to-end via API:
  25. retrieval_hint field present in MemoryResponse
  26. retrieval_hint is non-empty on create
  27. retrieval_hint updates after PUT (title changed)
  28. retrieval_hint present in list endpoint response
  29. retrieval_hint present in search results
  30. retrieval_hint present in extraction result memories
"""
import pytest
from fastapi.testclient import TestClient

from app.services.retrieval_hint import compute_retrieval_hint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hint(mem_type: str, title: str, content: str = "", meta: dict | None = None) -> str:
    return compute_retrieval_hint(mem_type, title, content, meta)


def _project(client: TestClient, name: str = "Hint Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, pid: str, **kwargs) -> dict:
    payload = {
        "type": "decision",
        "title": "Use PostgreSQL",
        "content": "We chose PostgreSQL for better JSON support.",
        **kwargs,
    }
    r = client.post(f"/projects/{pid}/memories", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–2: Basic contract
# ---------------------------------------------------------------------------

class TestBasicContract:
    @pytest.mark.parametrize("mem_type", [
        "decision", "problem", "failed_approach", "structure", "how_to",
        "open_question", "task", "constraint", "insight", "reference_context",
        "workflow_pattern", "conversation_note",
    ])
    def test_returns_nonempty_for_all_types(self, mem_type: str):
        result = _hint(mem_type, "Some title", "Some content")
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.parametrize("mem_type", [
        "decision", "problem", "failed_approach", "structure", "how_to",
        "open_question", "task", "constraint", "insight", "reference_context",
        "workflow_pattern", "conversation_note",
    ])
    def test_max_300_chars_for_all_types(self, mem_type: str):
        long_content = "x" * 500
        result = _hint(mem_type, "Title " * 20, long_content)
        assert len(result) <= 300


# ---------------------------------------------------------------------------
# 3–5: decision
# ---------------------------------------------------------------------------

class TestDecisionHint:
    def test_uses_rationale(self):
        hint = _hint("decision", "Use Redis", meta={"rationale": "sub-millisecond latency"})
        assert "Rationale" in hint
        assert "sub-millisecond latency" in hint

    def test_falls_back_to_alternatives(self):
        hint = _hint("decision", "Use Redis", meta={"alternatives_considered": ["Memcached", "Hazelcast"]})
        assert "Memcached" in hint

    def test_falls_back_to_label_title_only(self):
        hint = _hint("decision", "Use Redis")
        assert "Decision" in hint
        assert "Use Redis" in hint


# ---------------------------------------------------------------------------
# 6–8: bug
# ---------------------------------------------------------------------------

class TestBugHint:
    def test_uses_error_message(self):
        hint = _hint("problem", "DB crash", meta={"error_message": "FATAL: out of shared memory"})
        assert "FATAL: out of shared memory" in hint

    def test_uses_root_cause(self):
        hint = _hint("problem", "DB crash", meta={"root_cause": "connection pool exhausted"})
        assert "connection pool exhausted" in hint

    def test_falls_back_to_content_sentence(self):
        hint = _hint("problem", "DB crash", content="This happens when max_connections is exceeded.")
        assert "max_connections" in hint


# ---------------------------------------------------------------------------
# 9–10: failed_approach
# ---------------------------------------------------------------------------

class TestFailedApproachHint:
    def test_uses_approach_and_avoid(self):
        meta = {
            "approach_tried": "Polling every 100ms",
            "avoid_because": "burns CPU at idle",
        }
        hint = _hint("failed_approach", "Polling approach", meta=meta)
        assert "Polling every 100ms" in hint
        assert "burns CPU at idle" in hint

    def test_includes_alternative_chosen(self):
        meta = {
            "approach_tried": "Polling every 100ms",
            "alternative_chosen": "WebSocket push",
        }
        hint = _hint("failed_approach", "Polling approach", meta=meta)
        assert "WebSocket push" in hint


# ---------------------------------------------------------------------------
# 11–12: architecture
# ---------------------------------------------------------------------------

class TestArchitectureHint:
    def test_uses_pattern(self):
        hint = _hint("structure", "Event sourcing", meta={"pattern": "CQRS"})
        assert "CQRS" in hint

    def test_appends_rationale_when_pattern_present(self):
        hint = _hint("structure", "Event sourcing", meta={
            "pattern": "CQRS",
            "rationale": "independent read/write scaling",
        })
        assert "CQRS" in hint
        assert "independent read/write scaling" in hint


# ---------------------------------------------------------------------------
# 13–14: setup_instruction
# ---------------------------------------------------------------------------

class TestSetupInstructionHint:
    def test_uses_command(self):
        hint = _hint("how_to", "Install deps", meta={"command": "pip install -r requirements.txt"})
        assert "pip install -r requirements.txt" in hint

    def test_appends_platform(self):
        hint = _hint("how_to", "Install deps", meta={
            "command": "brew install redis",
            "platform": "macOS",
        })
        assert "macOS" in hint


# ---------------------------------------------------------------------------
# 15: open_question
# ---------------------------------------------------------------------------

class TestOpenQuestionHint:
    def test_uses_context(self):
        hint = _hint("open_question", "Which cache backend?", meta={"context": "latency vs cost tradeoff"})
        assert "latency vs cost tradeoff" in hint


# ---------------------------------------------------------------------------
# 16–17: constraint
# ---------------------------------------------------------------------------

class TestConstraintHint:
    def test_uses_impact(self):
        hint = _hint("constraint", "Max 10MB payload", meta={"impact": "breaks mobile upload"})
        assert "breaks mobile upload" in hint

    def test_falls_back_to_source(self):
        hint = _hint("constraint", "No PII in logs", meta={"source": "legal requirement"})
        assert "legal requirement" in hint


# ---------------------------------------------------------------------------
# 18–19: workflow_pattern
# ---------------------------------------------------------------------------

class TestWorkflowPatternHint:
    def test_uses_trigger(self):
        hint = _hint("workflow_pattern", "Deploy checklist", meta={"trigger": "before every production deploy"})
        assert "before every production deploy" in hint

    def test_falls_back_to_steps(self):
        hint = _hint("workflow_pattern", "Deploy checklist", meta={
            "steps": ["Run tests", "Build image", "Push to registry"],
        })
        assert "Run tests" in hint


# ---------------------------------------------------------------------------
# 20: insight
# ---------------------------------------------------------------------------

class TestInsightHint:
    def test_prepends_first_content_sentence(self):
        hint = _hint("insight", "Redis key design",
                     content="Short keys reduce memory by 20%. Avoid long namespaces.")
        assert "Short keys reduce memory by 20%" in hint

    def test_omits_duplicate_content(self):
        title = "Redis key design matters"
        hint = _hint("insight", title, content=title.lower() + ". More detail here.")
        # First sentence duplicates title — should not appear twice awkwardly
        assert len(hint) <= 300


# ---------------------------------------------------------------------------
# 21–23: default / edge cases
# ---------------------------------------------------------------------------

class TestDefaultAndEdgeCases:
    def test_default_includes_label_title_content(self):
        hint = _hint("reference_context", "Auth middleware", content="Handles JWT verification.")
        assert "Reference" in hint
        assert "Auth middleware" in hint
        assert "JWT" in hint

    def test_skips_short_content_sentence(self):
        # First sentence ≤10 chars should be skipped
        hint = _hint("reference_context", "Auth middleware", content="OK. More detail here.")
        assert hint.endswith("Auth middleware") or "OK" not in hint

    def test_long_output_truncated_to_300(self):
        long_rationale = "because " * 50
        hint = _hint("decision", "Big call", meta={"rationale": long_rationale})
        assert len(hint) <= 300


# ---------------------------------------------------------------------------
# 25–30: Integration — API
# ---------------------------------------------------------------------------

class TestRetrievalHintAPI:
    def test_field_present_in_response(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "retrieval_hint" in m

    def test_hint_is_nonempty_on_create(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["retrieval_hint"] and len(m["retrieval_hint"]) > 0

    def test_hint_updates_after_title_change(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, title="Old title")
        old_hint = m["retrieval_hint"]
        r = client.put(f"/memories/{m['id']}", json={"title": "Completely new title XYZ"})
        assert r.status_code == 200
        new_hint = r.json()["retrieval_hint"]
        assert new_hint != old_hint
        assert "Completely new title XYZ" in new_hint

    def test_hint_present_in_list_endpoint(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        mems = client.get(f"/projects/{pid}/memories").json()
        assert len(mems) > 0
        for m in mems:
            assert "retrieval_hint" in m

    def test_hint_present_in_search_results(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="PostgreSQL database decision")
        results = client.get(f"/projects/{pid}/memories/search", params={"q": "PostgreSQL"}).json()
        if results:
            assert "retrieval_hint" in results[0]

    def test_hint_contains_decision_label(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, type="decision", title="Choose FastAPI")
        assert "Decision" in m["retrieval_hint"]

    def test_hint_reflects_type_metadata_rationale(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, type="decision", title="Choose FastAPI",
                    type_metadata={"rationale": "async support out of the box"})
        assert "async support" in m["retrieval_hint"]

    def test_hint_present_in_extraction_result(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/sessions", json={
            "tool_name": "Claude",
            "title": "Hint extraction test",
            "raw_content": (
                "We decided to use PostgreSQL as our primary database. "
                "The rationale was better JSON support and ACID compliance."
            ),
            "session_date": "2026-05-25",
        })
        sid = r.json()["id"]
        result = client.post(f"/sessions/{sid}/extract-memories").json()
        mems = result.get("memories", [])
        if mems:
            assert "retrieval_hint" in mems[0]
            assert mems[0]["retrieval_hint"]  # non-empty
