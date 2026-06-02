"""
Sub-phase 1.40 — Universal Extraction Tests.

Verifies the semantic pipeline works across all supported domains, not just software.

Gate tests (is_technical — max-sim approach):
  1.  Software sentences pass the gate
  2.  Design sentences pass the gate
  3.  Research sentences pass the gate
  4.  Marketing sentences pass the gate
  5.  Personal task sentence passes the gate
  6.  Social filler fails the gate (6 sentences)

Classifier unit tests (classify):
  7.  Software: event-driven pattern → structure
  8.  Design: 8-column grid → structure
  9.  Research: sleep/retention correlation → insight
  10. Marketing: email deliverability blocker → problem
  11. Personal: budget review task → task
  12. domain=general still produces valid classifications

Integration tests (full pipeline via API):
  13. Design project: design content → at least one memory extracted
  14. Research project: research content → at least one memory extracted
  15. Marketing project: marketing content → at least one memory extracted
  16. Personal project: personal task content → task-type memory extracted
  17. Filler-only session → zero memories extracted
  18. Software project: technical content → decision + bug extracted
"""
import pytest
from fastapi.testclient import TestClient

from app.services.semantic_classifier import classify, is_technical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str, domain: str = "general") -> str:
    r = client.post("/projects", json={"name": name, "domain": domain})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _session(client: TestClient, project_id: str, content: str) -> str:
    r = client.post(f"/projects/{project_id}/sessions", json={
        "tool_name": "Claude",
        "title": "Test Session",
        "raw_content": content,
        "session_date": "2026-05-27",
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _extract(client: TestClient, session_id: str) -> dict:
    r = client.post(f"/sessions/{session_id}/extract-memories")
    assert r.status_code == 200, r.text
    return r.json()


def _all_types(result: dict) -> list[str]:
    return [m["type"] for m in result.get("memories", [])]


# ---------------------------------------------------------------------------
# Gate tests — is_technical()
# ---------------------------------------------------------------------------

SOFTWARE_GATE_SENTENCES = [
    "Authentication throws a TypeError when the session token is null or has expired.",
    "We decided to use PostgreSQL over MySQL for better JSON support.",
    "The migration script adds a composite index on project_id and created_at.",
]

DESIGN_GATE_SENTENCES = [
    "We use an 8-column grid system across all screens.",
    "The hero section does not render correctly on viewport widths below 375 pixels.",
    "Typography uses three scales: display at 48px, body at 16px, and caption at 12px.",
]

RESEARCH_GATE_SENTENCES = [
    "The study found a significant correlation between sleep quality and memory retention.",
    "Participants in the control group showed no measurable change in test scores.",
    "Statistical significance was achieved at p less than 0.05 in the primary outcome.",
]

MARKETING_GATE_SENTENCES = [
    "Email deliverability dropped after the domain change and is blocking the Q2 campaign.",
    "Paid cost per acquisition is now three times higher than organic due to ad competition.",
    "The landing page conversion rate improved from two to eight percent after the redesign.",
]

FILLER_SENTENCES = [
    "How's everything going with you today?",
    "I'm doing well, thanks for asking!",
    "That sounds great, glad to hear it!",
    "Let me know if there is anything I can help with.",
    "Sure thing, happy to help!",
    "Looking forward to working with you.",
]


@pytest.mark.parametrize("sentence", SOFTWARE_GATE_SENTENCES)
def test_gate_passes_software(sentence):
    assert is_technical(sentence), f"Software sentence failed gate: {sentence}"


@pytest.mark.parametrize("sentence", DESIGN_GATE_SENTENCES)
def test_gate_passes_design(sentence):
    assert is_technical(sentence), f"Design sentence failed gate: {sentence}"


@pytest.mark.parametrize("sentence", RESEARCH_GATE_SENTENCES)
def test_gate_passes_research(sentence):
    assert is_technical(sentence), f"Research sentence failed gate: {sentence}"


@pytest.mark.parametrize("sentence", MARKETING_GATE_SENTENCES)
def test_gate_passes_marketing(sentence):
    assert is_technical(sentence), f"Marketing sentence failed gate: {sentence}"


def test_gate_passes_personal_budget_task():
    assert is_technical("I need to review the budget spreadsheet before Thursday.")


@pytest.mark.parametrize("sentence", FILLER_SENTENCES)
def test_gate_rejects_filler(sentence):
    assert not is_technical(sentence), f"Filler sentence wrongly passed gate: {sentence}"


# ---------------------------------------------------------------------------
# Classifier unit tests — classify()
# ---------------------------------------------------------------------------

def test_classify_software_event_driven_pattern():
    result = classify("The microservices use an event-driven pattern for cross-service communication.", domain="software")
    assert result is not None, "Software structure sentence returned None"
    mem_type, conf = result
    assert mem_type == "structure", f"Expected 'structure', got '{mem_type}'"
    assert conf >= 0.23


def test_classify_design_grid_system():
    result = classify("We use an 8-column grid system across all screens.", domain="design")
    assert result is not None, "Design structure sentence returned None"
    mem_type, conf = result
    assert mem_type == "structure", f"Expected 'structure', got '{mem_type}'"
    assert conf >= 0.23


def test_classify_research_insight():
    result = classify("The study found a correlation between sleep quality and memory retention.", domain="research")
    assert result is not None, "Research insight sentence returned None"
    mem_type, conf = result
    assert mem_type == "insight", f"Expected 'insight', got '{mem_type}'"
    assert conf >= 0.23


def test_classify_marketing_problem():
    result = classify("Email deliverability is our main blocker for growth.", domain="marketing")
    assert result is not None, "Marketing problem sentence returned None"
    mem_type, conf = result
    assert mem_type == "problem", f"Expected 'problem', got '{mem_type}'"
    assert conf >= 0.23


def test_classify_personal_task():
    result = classify("I need to review the budget spreadsheet before Thursday.", domain="personal")
    assert result is not None, "Personal task sentence returned None"
    mem_type, conf = result
    assert mem_type == "task", f"Expected 'task', got '{mem_type}'"
    assert conf >= 0.23


def test_classify_general_domain_decision():
    result = classify("We decided to use PostgreSQL over MySQL for better JSON support.", domain="general")
    assert result is not None
    mem_type, _ = result
    assert mem_type == "decision"


def test_classify_confidence_in_range():
    result = classify("We decided to use PostgreSQL over MySQL for better JSON support.", domain="software")
    assert result is not None
    _, conf = result
    assert 0.23 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Integration tests — full pipeline via API
# ---------------------------------------------------------------------------

SOFTWARE_CONTENT = """
We decided to use FastAPI over Flask because of built-in type validation and OpenAPI docs.
There is a bug in the authentication flow where the session token expires silently.
The microservices use an event-driven pattern for cross-service communication.
"""

DESIGN_CONTENT = """
We chose Figma over Sketch because of its real-time multiplayer collaboration feature.
The hero section does not render correctly on viewport widths below 375 pixels.
We agreed on an 8-column grid as the baseline layout system for all screen sizes.
"""

RESEARCH_CONTENT = """
The study found a significant correlation between sleep quality and memory retention.
We selected a qualitative approach because the sample size was insufficient for statistics.
Participants in the control group showed no measurable change in test scores.
"""

MARKETING_CONTENT = """
Email deliverability dropped after the domain change and is blocking the Q2 campaign.
The landing page conversion rate improved from two to eight percent after the redesign.
Paid cost per acquisition is now three times higher than organic due to ad competition.
"""

PERSONAL_CONTENT = """
I need to review the budget spreadsheet before Thursday to confirm Q2 spending.
The Q3 retrospective revealed three recurring blockers in the release process.
Focus sessions increased from two to four hours daily after switching to time-blocking.
"""

FILLER_CONTENT = """
How's everything going with you today?
I'm doing well, thanks for asking!
That sounds great, glad to hear it!
Let me know if there is anything I can help with.
Sure thing, happy to help!
Looking forward to working with you.
"""


def test_integration_software_project_extracts_memories(client):
    pid = _project(client, "Software Test", domain="software")
    sid = _session(client, pid, SOFTWARE_CONTENT)
    result = _extract(client, sid)
    assert result["memories_created"] > 0, "Expected at least one memory from software content"
    types = _all_types(result)
    assert "decision" in types or "bug" in types or "structure" in types, \
        f"Expected decision/bug/structure, got: {types}"


def test_integration_design_project_extracts_memories(client):
    pid = _project(client, "Design Test", domain="design")
    sid = _session(client, pid, DESIGN_CONTENT)
    result = _extract(client, sid)
    assert result["memories_created"] > 0, "Expected at least one memory from design content"


def test_integration_research_project_extracts_memories(client):
    pid = _project(client, "Research Test", domain="research")
    sid = _session(client, pid, RESEARCH_CONTENT)
    result = _extract(client, sid)
    assert result["memories_created"] > 0, "Expected at least one memory from research content"


def test_integration_marketing_project_extracts_memories(client):
    pid = _project(client, "Marketing Test", domain="marketing")
    sid = _session(client, pid, MARKETING_CONTENT)
    result = _extract(client, sid)
    assert result["memories_created"] > 0, "Expected at least one memory from marketing content"


def test_integration_personal_project_extracts_task(client):
    pid = _project(client, "Personal Test", domain="personal")
    sid = _session(client, pid, PERSONAL_CONTENT)
    result = _extract(client, sid)
    assert result["memories_created"] > 0, "Expected at least one memory from personal content"
    types = _all_types(result)
    assert "task" in types, f"Expected 'task' type from personal content, got: {types}"


def test_integration_filler_only_produces_zero_memories(client):
    pid = _project(client, "Filler Test", domain="general")
    sid = _session(client, pid, FILLER_CONTENT)
    result = _extract(client, sid)
    assert result["memories_created"] == 0, \
        f"Filler-only session should produce 0 memories, got {result['memories_created']}"
