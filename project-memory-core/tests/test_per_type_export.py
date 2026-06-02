"""
Sub-phase 1.36 — Per-type Markdown export stress tests.

Every test hits GET /projects/{id}/export/memories/{type}.md and verifies
structure, content, type-specific metadata rendering, and filter behavior.

Stress scenarios:
  1.  All 12 types return 200 (not 404/422)
  2.  Document header contains project name
  3.  Document header contains memory type
  4.  Exported_at date appears in header
  5.  Memory title appears in output
  6.  Memory content appears in output
  7.  Summary bar shows memory count
  8.  Summary bar shows avg importance
  9.  Summary bar shows avg confidence
  10. Importance stars rendered (★/☆)
  11. Confidence % rendered
  12. Status rendered
  13. Type-metadata decision: rationale rendered
  14. Type-metadata decision: alternatives_considered rendered
  15. Type-metadata decision: decision_status rendered
  16. Type-metadata bug: error_message rendered
  17. Type-metadata bug: fix_applied rendered
  18. Type-metadata bug: root_cause rendered
  19. Type-metadata architecture: pattern rendered
  20. Type-metadata architecture: components_affected rendered
  21. Type-metadata setup_instruction: command rendered
  22. Type-metadata setup_instruction: prerequisites rendered
  23. Type-metadata open_question: context rendered
  24. Type-metadata constraint: source rendered
  25. Type-metadata workflow_pattern: steps rendered
  26. Type-metadata failed_approach: approach_tried rendered
  27. Type-metadata failed_approach: avoid_because rendered
  28. Tags rendered when present
  29. Related files rendered when present
  30. Source quote rendered when present
  31. Retrieval hint (TL;DR) rendered when present
  32. Session attribution rendered (source session title)
  33. Code anchor (file_path) rendered when set
  34. Freshness bar rendered after decay computed
  35. status=active filter excludes resolved memories
  36. min_importance=4 filter excludes low-importance memories
  37. max_privacy_level=public excludes internal memories
  38. Memories sorted by importance desc within type
  39. Empty type returns "No memories found" message (not error)
  40. Unknown type returns 422
  41. Unknown project returns 404
  42. Cross-project isolation: other project's memories not included
  43. stat counts in summary bar are correct
  44. Multiple memories of same type all rendered
  45. Footer present at end of document
  46. Combined filters: status + importance + privacy all applied simultaneously
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TYPES = [
    "decision", "problem", "task", "insight", "structure",
    "reference_context", "how_to", "open_question",
    "constraint", "conversation_note", "workflow_pattern", "failed_approach",
]


def _project(client: TestClient, name: str = "TypeExport Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, title: str = "Session",
             content: str = "We decided to use PostgreSQL for ACID compliance.") -> dict:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude", "title": title, "raw_content": content,
        "session_date": "2026-05-26",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _memory(client: TestClient, pid: str, mem_type: str = "decision", **kwargs) -> dict:
    payload = {
        "type": mem_type,
        "title": f"Use Redis for {mem_type}",
        "content": f"Redis was chosen as the solution for {mem_type} reasons.",
        "importance": 3,
        "confidence": 0.85,
    }
    payload.update(kwargs)
    r = client.post(f"/projects/{pid}/memories", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _get_md(client: TestClient, pid: str, mem_type: str, **params) -> str:
    r = client.get(f"/projects/{pid}/export/memories/{mem_type}.md", params=params)
    assert r.status_code == 200, f"{r.status_code}: {r.text[:200]}"
    return r.text


# ---------------------------------------------------------------------------
# 1: All 12 types return 200
# ---------------------------------------------------------------------------

class TestAllTypesReturn200:
    @pytest.mark.parametrize("mem_type", _ALL_TYPES)
    def test_type_returns_200(self, client: TestClient, mem_type: str):
        pid = _project(client, name=f"Test {mem_type}")
        _memory(client, pid, mem_type=mem_type, title=f"Test {mem_type} memory")
        r = client.get(f"/projects/{pid}/export/memories/{mem_type}.md")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 2–12: Document structure and header
# ---------------------------------------------------------------------------

class TestDocumentStructure:
    def test_header_contains_project_name(self, client: TestClient):
        pid = _project(client, name="Alpha Project")
        _memory(client, pid, title="Some decision")
        md = _get_md(client, pid, "decision")
        assert "Alpha Project" in md

    def test_header_contains_type(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Bug memory", mem_type="problem")
        md = _get_md(client, pid, "problem")
        assert "problem" in md.lower()

    def test_header_contains_date(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Decision A")
        md = _get_md(client, pid, "decision")
        assert "2026" in md

    def test_memory_title_in_output(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Use PostgreSQL for production")
        md = _get_md(client, pid, "decision")
        assert "Use PostgreSQL for production" in md

    def test_memory_content_in_output(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Decision X",
                content="We specifically chose PostgreSQL because of ACID guarantees.")
        md = _get_md(client, pid, "decision")
        assert "We specifically chose PostgreSQL because of ACID guarantees." in md

    def test_summary_bar_shows_count(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="D1")
        _memory(client, pid, title="D2")
        md = _get_md(client, pid, "decision")
        assert "2 memories" in md

    def test_summary_bar_shows_avg_importance(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Imp2", importance=2)
        _memory(client, pid, title="Imp4", importance=4)
        md = _get_md(client, pid, "decision")
        assert "3.0/5" in md

    def test_summary_bar_shows_avg_confidence(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Conf80", confidence=0.80)
        md = _get_md(client, pid, "decision")
        assert "80%" in md

    def test_importance_stars_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, importance=4, title="High imp")
        md = _get_md(client, pid, "decision")
        assert "★★★★☆" in md

    def test_confidence_percent_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, confidence=0.92, title="High conf")
        md = _get_md(client, pid, "decision")
        assert "92%" in md

    def test_status_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Active decision", status="active")
        md = _get_md(client, pid, "decision")
        assert "active" in md

    def test_footer_present(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Decision F")
        md = _get_md(client, pid, "decision")
        # Footer ends with "_End of..."
        assert "_End of" in md


# ---------------------------------------------------------------------------
# 13–27: Type-specific metadata renderers
# ---------------------------------------------------------------------------

class TestTypeMetadataRendering:
    def test_decision_rationale(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="D",
                type_metadata={"rationale": "ACID compliance required", "alternatives_considered": []})
        assert "ACID compliance required" in _get_md(client, pid, "decision")

    def test_decision_alternatives(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="D",
                type_metadata={"alternatives_considered": ["MySQL", "SQLite"]})
        md = _get_md(client, pid, "decision")
        assert "MySQL" in md
        assert "SQLite" in md

    def test_decision_status_field(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="D",
                type_metadata={"decision_status": "accepted"})
        assert "accepted" in _get_md(client, pid, "decision")

    def test_bug_error_message(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="problem", title="B",
                type_metadata={"error_message": "NullPointerException at line 42"})
        assert "NullPointerException at line 42" in _get_md(client, pid, "problem")

    def test_bug_fix_applied(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="problem", title="B",
                type_metadata={"fix_applied": "Added null guard before access"})
        assert "Added null guard before access" in _get_md(client, pid, "problem")

    def test_bug_root_cause(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="problem", title="B",
                type_metadata={"root_cause": "Missing validation in controller"})
        assert "Missing validation in controller" in _get_md(client, pid, "problem")

    def test_architecture_pattern(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="structure", title="A",
                type_metadata={"pattern": "Hexagonal Architecture"})
        assert "Hexagonal Architecture" in _get_md(client, pid, "structure")

    def test_architecture_components(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="structure", title="A",
                type_metadata={"components_affected": ["api", "database"]})
        md = _get_md(client, pid, "structure")
        assert "api" in md and "database" in md

    def test_setup_instruction_command(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="how_to", title="S",
                type_metadata={"command": "pip install -r requirements.txt"})
        assert "pip install -r requirements.txt" in _get_md(client, pid, "how_to")

    def test_setup_instruction_prerequisites(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="how_to", title="S",
                type_metadata={"prerequisites": ["python3", "venv"]})
        md = _get_md(client, pid, "how_to")
        assert "python3" in md and "venv" in md

    def test_open_question_context(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="open_question", title="Q",
                type_metadata={"context": "Should we use gRPC or REST for internal services?"})
        assert "gRPC or REST" in _get_md(client, pid, "open_question")

    def test_constraint_source(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="constraint", title="C",
                type_metadata={"source": "legal", "impact": "Must encrypt PII at rest"})
        md = _get_md(client, pid, "constraint")
        assert "legal" in md

    def test_workflow_steps_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="workflow_pattern", title="W",
                type_metadata={"steps": ["Run tests", "Build image", "Deploy"]})
        md = _get_md(client, pid, "workflow_pattern")
        assert "Run tests" in md
        assert "Build image" in md
        assert "Deploy" in md

    def test_failed_approach_tried(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="failed_approach", title="F",
                type_metadata={"approach_tried": "Tried using MongoDB as primary DB"})
        assert "Tried using MongoDB as primary DB" in _get_md(client, pid, "failed_approach")

    def test_failed_approach_avoid_because(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, mem_type="failed_approach", title="F",
                type_metadata={"avoid_because": "Causes consistency issues under concurrent writes"})
        assert "Causes consistency issues under concurrent writes" in _get_md(client, pid, "failed_approach")


# ---------------------------------------------------------------------------
# 28–34: Extra fields rendering
# ---------------------------------------------------------------------------

class TestExtraFieldsRendering:
    def test_tags_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Tagged", tags=["performance", "database"])
        md = _get_md(client, pid, "decision")
        assert "performance" in md
        assert "database" in md

    def test_related_files_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="File mem", related_files=["app/database.py", "app/models.py"])
        md = _get_md(client, pid, "decision")
        assert "app/database.py" in md

    def test_source_quote_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Quoted",
                source_quote="We decided to use PostgreSQL because it offers ACID compliance.")
        md = _get_md(client, pid, "decision")
        assert "We decided to use PostgreSQL because it offers ACID compliance." in md

    def test_retrieval_hint_rendered(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, title="Hinted decision for caching")
        # Retrieval hint is auto-computed at write time — fetch the actual value
        actual_hint = client.get(f"/memories/{m['id']}").json()["retrieval_hint"]
        md = _get_md(client, pid, "decision")
        if actual_hint:
            assert actual_hint in md
        else:
            assert "TL;DR" not in md  # no hint = no TL;DR line rendered

    def test_session_attribution_rendered(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, title="Architecture Session")
        _memory(client, pid, title="Session mem", source_session_id=s["id"])
        md = _get_md(client, pid, "decision")
        assert "Architecture Session" in md

    def test_file_path_anchor_rendered(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Code mem", file_path="app/routers/memories.py")
        md = _get_md(client, pid, "decision")
        assert "app/routers/memories.py" in md

    def test_freshness_bar_after_decay(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Decay mem")
        client.post(f"/projects/{pid}/compute-decay")
        md = _get_md(client, pid, "decision")
        # After decay computation, freshness bar should appear
        assert "Freshness" in md or "█" in md


# ---------------------------------------------------------------------------
# 35–38: Filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_status_filter_excludes_resolved(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Active D", status="active")
        resolved = _memory(client, pid, title="Resolved D", status="resolved")
        md = _get_md(client, pid, "decision", status="active")
        assert "Resolved D" not in md

    def test_min_importance_excludes_low(self, client: TestClient):
        pid = _project(client)
        low = _memory(client, pid, title="Low imp decision", importance=1)
        _memory(client, pid, title="High imp decision", importance=5)
        md = _get_md(client, pid, "decision", min_importance=4)
        assert "Low imp decision" not in md
        assert "High imp decision" in md

    def test_privacy_filter_excludes_secret(self, client: TestClient):
        pid = _project(client)
        secret = _memory(client, pid, title="Secret decision", privacy_level="secret")
        _memory(client, pid, title="Public decision", privacy_level="public")
        md = _get_md(client, pid, "decision", max_privacy_level="public")
        assert "Secret decision" not in md
        assert "Public decision" in md

    def test_sorted_by_importance_desc(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Imp1", importance=1)
        _memory(client, pid, title="Imp5", importance=5)
        _memory(client, pid, title="Imp3", importance=3)
        md = _get_md(client, pid, "decision")
        pos1 = md.index("Imp5")
        pos3 = md.index("Imp3")
        pos5 = md.index("Imp1")
        assert pos1 < pos3 < pos5


# ---------------------------------------------------------------------------
# 39–46: Edge cases and integration
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_type_returns_no_memories_message(self, client: TestClient):
        pid = _project(client)
        # No bug memories — only decisions
        _memory(client, pid, mem_type="decision", title="Decision only")
        md = _get_md(client, pid, "problem")
        assert "No" in md and "problem" in md.lower()

    def test_unknown_type_returns_422(self, client: TestClient):
        pid = _project(client)
        r = client.get(f"/projects/{pid}/export/memories/not_a_type.md")
        assert r.status_code == 422

    def test_unknown_project_returns_404(self, client: TestClient):
        r = client.get("/projects/does-not-exist/export/memories/decision.md")
        assert r.status_code == 404

    def test_cross_project_isolation(self, client: TestClient):
        pid1 = _project(client, name="Project 1")
        pid2 = _project(client, name="Project 2")
        _memory(client, pid2, title="Other project's decision")
        md = _get_md(client, pid1, "decision")
        # If pid1 has no decisions, we get the empty message
        # Either way, pid2's memory must not appear
        assert "Other project's decision" not in md

    def test_summary_count_correct(self, client: TestClient):
        pid = _project(client)
        for i in range(4):
            _memory(client, pid, title=f"Decision {i}")
        md = _get_md(client, pid, "decision")
        assert "4 memories" in md

    def test_multiple_memories_all_rendered(self, client: TestClient):
        pid = _project(client)
        titles = ["Alpha decision", "Beta decision", "Gamma decision"]
        for t in titles:
            _memory(client, pid, title=t)
        md = _get_md(client, pid, "decision")
        for t in titles:
            assert t in md

    def test_combined_filters_applied(self, client: TestClient):
        pid = _project(client)
        # Excluded by each filter respectively
        _memory(client, pid, title="Wrong status", status="resolved",
                importance=5, privacy_level="public")
        _memory(client, pid, title="Low importance", status="active",
                importance=1, privacy_level="public")
        _memory(client, pid, title="Too private", status="active",
                importance=5, privacy_level="secret")
        # Should be included
        keeper = _memory(client, pid, title="The keeper", status="active",
                         importance=5, privacy_level="public")
        md = _get_md(client, pid, "decision",
                     status="active", min_importance=4, max_privacy_level="public")
        assert "The keeper" in md
        assert "Wrong status" not in md
        assert "Low importance" not in md
        assert "Too private" not in md

    @pytest.mark.parametrize("mem_type", _ALL_TYPES)
    def test_all_types_render_title(self, client: TestClient, mem_type: str):
        pid = _project(client, name=f"Round-trip {mem_type}")
        title = f"Round-trip title for {mem_type}"
        _memory(client, pid, mem_type=mem_type, title=title)
        md = _get_md(client, pid, mem_type)
        assert title in md
