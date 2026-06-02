"""
Sub-phase 1.41 — Low-confidence queue + manual classification tests.

Coverage:
  Unit
    1.  classify_best() always returns (type, score) — never None
    2.  classify_best() score is in [0, 1]
    3.  classify_best() returns "unclassified" as fallback only when truly empty

  Schema / API
    4.  ExtractionResult response includes low_confidence_queued field (even if 0)
    5.  POST /memories accepts type=unclassified
    6.  Unclassified memory defaults to review_status=needs_review when created with that status
    7.  POST /memories/{id}/classify assigns the given type and sets review_status=verified
    8.  POST /memories/{id}/classify on a non-existent ID returns 404
    9.  POST /memories/{id}/classify rejects an invalid type (422)
   10.  Classify endpoint works for every valid confirmed type
   11.  Context export includes "Review Queue" section when unclassified memories exist
   12.  Context export omits Review Queue when no unclassified memories exist
   13.  Classifying an unclassified memory removes it from the Review Queue in export
"""
import pytest
from fastapi.testclient import TestClient

from app.services.semantic_classifier import classify_best
from tests.conftest import make_project, make_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Test Project", domain: str = "general") -> str:
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


def _make_unclassified(client: TestClient, project_id: str, title: str = "Unclear sentence about infrastructure") -> dict:
    r = client.post(f"/projects/{project_id}/memories", json={
        "type": "unclassified",
        "title": title,
        "content": f"{title}. This needs manual review.",
        "importance": 2,
        "confidence": 0.21,
        "tags": ["auto-extracted", "needs-review"],
        "related_tools": ["Claude"],
        "status": "active",
        "review_status": "needs_review",
        "type_metadata": {
            "suggested_type": "insight",
            "classifier_confidence": 0.21,
        },
    })
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Unit tests — classify_best()
# ---------------------------------------------------------------------------

def test_classify_best_always_returns_tuple():
    result = classify_best("Something that is sort of technical but ambiguous.")
    assert isinstance(result, tuple), "classify_best() must return a tuple"
    assert len(result) == 2, "classify_best() must return (type, score)"


def test_classify_best_score_in_range():
    _, score = classify_best("Some technical-ish statement.")
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_classify_best_type_is_string():
    mem_type, _ = classify_best("We are migrating the database to improve performance.")
    assert isinstance(mem_type, str), f"Expected str type, got {type(mem_type)}"
    assert len(mem_type) > 0


def test_classify_best_returns_something_for_clear_sentence():
    mem_type, score = classify_best(
        "We decided to use FastAPI over Flask for better async support.",
        domain="software"
    )
    assert mem_type != "unclassified", f"Clear decision sentence should not be unclassified, got {mem_type}"
    assert score > 0.23


# ---------------------------------------------------------------------------
# Schema / extraction tests
# ---------------------------------------------------------------------------

def test_extraction_result_has_low_confidence_field(client):
    """ExtractionResult always includes low_confidence_queued — even when 0."""
    pid = _project(client, "LC Queue Test")
    sid = _session(client, pid,
        "We decided to use FastAPI over Flask because of type validation and OpenAPI support. "
        "There is a bug in the authentication flow where the session token expires silently."
    )
    result = _extract(client, sid)
    assert "low_confidence_queued" in result, (
        "ExtractionResult must include low_confidence_queued field"
    )
    assert isinstance(result["low_confidence_queued"], int)


def test_extraction_low_confidence_count_nonnegative(client):
    pid = _project(client, "LC Count Test")
    sid = _session(client, pid,
        "We decided to use PostgreSQL over MySQL for better JSON support and indexing performance. "
        "The authentication bug causes the session token to expire without notifying the user."
    )
    result = _extract(client, sid)
    assert result["low_confidence_queued"] >= 0


# ---------------------------------------------------------------------------
# Unclassified memory creation
# ---------------------------------------------------------------------------

def test_create_unclassified_memory(client):
    pid = _project(client)
    mem = _make_unclassified(client, pid)
    assert mem["type"] == "unclassified"
    assert mem["review_status"] == "needs_review"


def test_unclassified_memory_has_suggested_type_metadata(client):
    pid = _project(client)
    mem = _make_unclassified(client, pid)
    meta = mem.get("type_metadata") or {}
    assert "suggested_type" in meta, "type_metadata must include suggested_type"
    assert "classifier_confidence" in meta, "type_metadata must include classifier_confidence"


def test_unclassified_memory_appears_in_project_list(client):
    pid = _project(client)
    _make_unclassified(client, pid, title="Ambiguous infrastructure note")
    r = client.get(f"/projects/{pid}/memories", params={"type": "unclassified"})
    assert r.status_code == 200
    memories = r.json()
    assert len(memories) == 1
    assert memories[0]["type"] == "unclassified"


# ---------------------------------------------------------------------------
# Manual classify endpoint
# ---------------------------------------------------------------------------

def test_classify_endpoint_assigns_type(client):
    pid = _project(client)
    mem = _make_unclassified(client, pid)
    r = client.post(f"/memories/{mem['id']}/classify", json={"type": "decision"})
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["type"] == "decision"


def test_classify_endpoint_sets_verified(client):
    pid = _project(client)
    mem = _make_unclassified(client, pid)
    r = client.post(f"/memories/{mem['id']}/classify", json={"type": "insight"})
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["review_status"] == "verified"


def test_classify_endpoint_404_on_missing_memory(client):
    r = client.post("/memories/nonexistent-id/classify", json={"type": "decision"})
    assert r.status_code == 404


def test_classify_endpoint_422_on_invalid_type(client):
    pid = _project(client)
    mem = _make_unclassified(client, pid)
    r = client.post(f"/memories/{mem['id']}/classify", json={"type": "not_a_real_type"})
    assert r.status_code == 422


@pytest.mark.parametrize("mem_type", [
    "decision", "problem", "task", "insight", "structure",
    "reference_context", "how_to", "open_question", "constraint",
    "workflow_pattern", "failed_approach", "conversation_note",
])
def test_classify_endpoint_accepts_all_valid_types(client, mem_type):
    pid = _project(client, name=f"Classify Test {mem_type}")
    mem = _make_unclassified(client, pid, title=f"Ambiguous note about {mem_type}")
    r = client.post(f"/memories/{mem['id']}/classify", json={"type": mem_type})
    assert r.status_code == 200, f"classify failed for type={mem_type}: {r.text}"
    assert r.json()["type"] == mem_type


def test_classify_persists_across_get(client):
    """GET after classify should reflect the updated type and review_status."""
    pid = _project(client)
    mem = _make_unclassified(client, pid)
    client.post(f"/memories/{mem['id']}/classify", json={"type": "constraint"})
    r = client.get(f"/memories/{mem['id']}")
    assert r.status_code == 200
    fetched = r.json()
    assert fetched["type"] == "constraint"
    assert fetched["review_status"] == "verified"


# ---------------------------------------------------------------------------
# Context export
# ---------------------------------------------------------------------------

def test_export_includes_review_queue_section_when_unclassified_exist(client):
    pid = _project(client)
    _make_unclassified(client, pid, title="An unclear infrastructure sentence")
    r = client.get(f"/projects/{pid}/export/context.md")
    assert r.status_code == 200
    text = r.text
    assert "Review Queue" in text, "Export must include Review Queue section when unclassified memories exist"


def test_export_omits_review_queue_when_no_unclassified(client):
    pid = _project(client)
    make_memory(client, pid, type="decision", title="Use PostgreSQL")
    r = client.get(f"/projects/{pid}/export/context.md")
    assert r.status_code == 200
    assert "Review Queue" not in r.text


def test_export_review_queue_absent_after_classification(client):
    """After classifying the only unclassified memory, Review Queue should disappear from export."""
    pid = _project(client)
    mem = _make_unclassified(client, pid, title="Ambiguous infra statement")
    # Classify it
    client.post(f"/memories/{mem['id']}/classify", json={"type": "insight"})
    r = client.get(f"/projects/{pid}/export/context.md")
    assert r.status_code == 200
    assert "Review Queue" not in r.text, (
        "Review Queue section should disappear once all unclassified memories are classified"
    )
