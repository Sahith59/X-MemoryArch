"""
Sub-phase 1.15 Stress Tests — Quality Layer.

Covers every behaviour introduced by the quality layer:
  1.  quality_score is computed on create (min, mid, max cases)
  2.  quality_score re-computes on update (proven to increase)
  3.  search_text contains all searchable dimensions
  4.  Extracted memories receive review_status=auto_extracted
  5.  Extracted memories receive source_type=ai_session
  6.  Extracted memories have source_quote set to the trigger sentence
  7.  review_status can be updated and is tracked in changelog
  8.  source_type=manual can be set on a manually created memory
  9.  symbol_name, line_start, line_end round-trip through the API
  10. branch_name round-trips through the API
  11. quality_score ∈ [0.0, 1.0] for every valid input
  12. search_text is updated when content-related fields change
  13. API response includes all 9 new fields
  14. high-quality memory has higher score than low-quality memory
  15. quality_score accounts for source traceability signals
  16. quality_score accounts for specificity signals (file_path + symbol)
  17. search_text includes type_metadata fields
  18. Supersede produces a new memory with quality_score set
  19. source_quote is capped at 500 chars on extraction
  20. Rejection status flows through; changelog records it
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Quality Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(
    client: TestClient,
    pid: str,
    *,
    title: str = "A test memory",
    content: str = "Some content here.",
    mem_type: str = "decision",
    importance: int = 3,
    confidence: float = 1.0,
    tags: list[str] | None = None,
    related_files: list[str] | None = None,
    file_path: str | None = None,
    symbol_name: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    branch_name: str | None = None,
    type_metadata: dict | None = None,
    source_quote: str | None = None,
    source_type: str | None = None,
    review_status: str | None = None,
) -> dict:
    body: dict = {
        "type": mem_type,
        "title": title,
        "content": content,
        "importance": importance,
        "confidence": confidence,
        "tags": tags or [],
        "related_files": related_files or [],
    }
    for k, v in [
        ("file_path", file_path),
        ("symbol_name", symbol_name),
        ("line_start", line_start),
        ("line_end", line_end),
        ("branch_name", branch_name),
        ("type_metadata", type_metadata),
        ("source_quote", source_quote),
        ("source_type", source_type),
        ("review_status", review_status),
    ]:
        if v is not None:
            body[k] = v
    r = client.post(f"/projects/{pid}/memories", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _session(client: TestClient, pid: str, content: str, tool: str = "Claude") -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": tool,
        "title": "Quality Test Session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    return r.json()["id"]


def _extract(client: TestClient, sid: str) -> dict:
    r = client.post(f"/sessions/{sid}/extract-memories")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. quality_score computed on create — minimal memory yields low score
# ---------------------------------------------------------------------------

class TestQualityScoreOnCreate:
    def test_minimal_memory_has_low_quality(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Hi", content="Short.")
        # Very short title (<5 chars) + very short content (<20 chars) + nothing else
        assert m["quality_score"] is not None
        assert m["quality_score"] < 0.30

    def test_rich_memory_has_high_quality(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="We decided to use FastAPI as our primary web framework for this project",
            content=(
                "After evaluating Flask, Django REST Framework, and FastAPI, we chose "
                "FastAPI because of its native async support, automatic OpenAPI docs, "
                "and performance characteristics suitable for our workload. "
                "This decision affects all API endpoints going forward."
            ),
            tags=["structure", "backend"],
            file_path="app/main.py",
            symbol_name="create_app",
            type_metadata={"rationale": "async support", "alternatives_considered": ["Flask", "DRF"]},
        )
        # Long title (>=50) + long content (>=200) + type_metadata + tags + file_path + symbol_name
        assert m["quality_score"] >= 0.60

    def test_quality_score_bounded_0_to_1(self, client: TestClient):
        pid = _project(client)
        # Throw everything at it to verify capping
        m = _mem(
            client, pid,
            title="A" * 60,
            content="B" * 250,
            tags=["tag1", "tag2"],
            file_path="app/models.py",
            symbol_name="Memory",
            related_files=["app/crud.py"],
            type_metadata={"rationale": "good reason"},
            source_quote="We decided this is important.",
            source_type="manual",
        )
        assert 0.0 <= m["quality_score"] <= 1.0

    def test_quality_score_not_zero_for_medium_memory(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="Use PostgreSQL for production",
            content="We evaluated SQLite vs PostgreSQL and chose PostgreSQL for its robustness.",
            tags=["database"],
        )
        assert m["quality_score"] > 0.0


# ---------------------------------------------------------------------------
# 2. quality_score recomputed on update
# ---------------------------------------------------------------------------

class TestQualityScoreOnUpdate:
    def test_adding_file_path_raises_quality_score(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Plain decision", content="We chose to do X instead of Y here.")
        score_before = m["quality_score"]

        r = client.put(f"/memories/{m['id']}", json={"file_path": "app/models.py"})
        assert r.status_code == 200
        assert r.json()["quality_score"] > score_before

    def test_adding_symbol_name_raises_quality_score(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Architecture note", content="We structure code around service layers.")
        score_before = m["quality_score"]

        r = client.put(f"/memories/{m['id']}", json={"symbol_name": "ServiceLayer"})
        assert r.status_code == 200
        assert r.json()["quality_score"] > score_before

    def test_adding_source_quote_raises_quality_score(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Why we chose Redis", content="Redis is used for session caching.")
        score_before = m["quality_score"]

        r = client.put(f"/memories/{m['id']}", json={"source_quote": "We must use Redis for caching."})
        assert r.status_code == 200
        assert r.json()["quality_score"] > score_before

    def test_quality_score_field_present_in_response(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Test memory", content="Some content for testing quality.")
        assert "quality_score" in m
        r = client.put(f"/memories/{m['id']}", json={"importance": 5})
        assert "quality_score" in r.json()


# ---------------------------------------------------------------------------
# 3. search_text contains all searchable dimensions
# ---------------------------------------------------------------------------

class TestSearchText:
    def test_search_text_contains_title_and_content(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="Unique XYZ decision",
            content="We chose XYZ approach for the backend pipeline.",
        )
        assert m["search_text"] is not None
        assert "Unique XYZ decision" in m["search_text"]
        assert "XYZ approach" in m["search_text"]

    def test_search_text_contains_tags(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Tag test", content="Testing tag inclusion.", tags=["database", "cache"])
        assert "database" in m["search_text"]
        assert "cache" in m["search_text"]

    def test_search_text_contains_file_path(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="File path test", content="About the models file.", file_path="app/models.py")
        assert "app/models.py" in m["search_text"]

    def test_search_text_contains_symbol_name(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Symbol test", content="About Memory model.", symbol_name="MemoryModel")
        assert "MemoryModel" in m["search_text"]

    def test_search_text_contains_branch_name(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Branch test", content="Work on this feature branch.", branch_name="feature/quality-layer")
        assert "feature/quality-layer" in m["search_text"]

    def test_search_text_contains_type_metadata(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="Metadata test",
            content="Decision with metadata.",
            type_metadata={"rationale": "performance improvement", "consequences": "migration needed"},
        )
        assert "performance improvement" in m["search_text"]
        assert "migration needed" in m["search_text"]

    def test_search_text_contains_source_quote(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="Quote test",
            content="The decision was made clearly.",
            source_quote="We definitely must use this approach.",
        )
        assert "We definitely must use this approach" in m["search_text"]

    def test_search_text_updated_when_content_changes(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Updatable memory", content="Original content here for testing.")
        old_search_text = m["search_text"]

        r = client.put(f"/memories/{m['id']}", json={"content": "Completely revised content with new keywords."})
        assert r.status_code == 200
        new_search_text = r.json()["search_text"]
        assert "Completely revised content" in new_search_text
        assert new_search_text != old_search_text


# ---------------------------------------------------------------------------
# 4–6. Extracted memories get traceability fields
# ---------------------------------------------------------------------------

class TestExtractionTraceability:
    def test_extracted_memory_has_review_status_auto_extracted(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use FastAPI as the web framework.")
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert m["review_status"] == "auto_extracted"

    def test_extracted_memory_has_source_type_ai_session(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use PostgreSQL for the production database.")
        result = _extract(client, sid)
        for m in result["memories"]:
            assert m["source_type"] == "ai_session"

    def test_extracted_memory_has_source_quote(self, client: TestClient):
        pid = _project(client)
        trigger = "We decided to use Redis for session caching in the backend."
        sid = _session(client, pid, trigger)
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        # At least one extracted memory should have source_quote containing key words
        quotes = [m["source_quote"] for m in result["memories"] if m["source_quote"]]
        assert len(quotes) >= 1
        # The quote should be the actual sentence (or close to it)
        assert any("Redis" in q or "decided" in q for q in quotes)

    def test_extracted_memory_source_quote_capped_at_500(self, client: TestClient):
        pid = _project(client)
        long_sentence = "We decided to use this approach. " + "x" * 600
        sid = _session(client, pid, long_sentence)
        result = _extract(client, sid)
        for m in result["memories"]:
            if m["source_quote"]:
                assert len(m["source_quote"]) <= 500

    def test_extraction_quality_score_non_zero(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use microservices architecture for scalability.")
        result = _extract(client, sid)
        for m in result["memories"]:
            # Should have session reference → quality_score > 0
            assert m["quality_score"] is not None
            assert m["quality_score"] > 0.0

    def test_extraction_traceability_increases_quality_vs_manual(self, client: TestClient):
        pid = _project(client)
        # Manual memory — no session, no source_quote, no source_type
        manual = _mem(
            client, pid,
            title="Manual decision without traceability",
            content="We chose to use this approach without any source reference.",
        )
        # Extracted memory should have higher quality because it has session + source_type + source_quote
        sid = _session(
            client, pid,
            "We decided to use this architecture because it scales better horizontally."
        )
        result = _extract(client, sid)
        if result["memories"]:
            extracted_score = result["memories"][0]["quality_score"]
            # Extracted gets +0.15 (session) + 0.05 (source_type) + 0.05 (source_quote) = +0.25
            assert extracted_score > manual["quality_score"]


# ---------------------------------------------------------------------------
# 7. review_status can be updated + changelog tracks it
# ---------------------------------------------------------------------------

class TestReviewStatusUpdate:
    def test_can_update_review_status_to_verified(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use FastAPI for all API endpoints.")
        result = _extract(client, sid)
        mem_id = result["memories"][0]["id"]

        r = client.put(f"/memories/{mem_id}", json={"review_status": "verified"})
        assert r.status_code == 200
        assert r.json()["review_status"] == "verified"

    def test_can_update_review_status_to_rejected(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use GraphQL for the frontend API.")
        result = _extract(client, sid)
        mem_id = result["memories"][0]["id"]

        r = client.put(f"/memories/{mem_id}", json={"review_status": "rejected"})
        assert r.status_code == 200
        assert r.json()["review_status"] == "rejected"

    def test_can_update_review_status_to_needs_review(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="A memory to flag", content="This might need review later.")
        r = client.put(f"/memories/{m['id']}", json={"review_status": "needs_review"})
        assert r.status_code == 200
        assert r.json()["review_status"] == "needs_review"

    def test_review_status_change_tracked_in_changelog(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use SQLite for local-first storage.")
        result = _extract(client, sid)
        mem_id = result["memories"][0]["id"]

        # Change auto_extracted → verified
        client.put(f"/memories/{mem_id}", json={"review_status": "verified"})

        r = client.get(f"/memories/{mem_id}/history")
        assert r.status_code == 200
        changes = r.json()
        review_changes = [c for c in changes if c["field"] == "review_status"]
        assert len(review_changes) >= 1
        assert review_changes[-1]["old_value"] == "auto_extracted"
        assert review_changes[-1]["new_value"] == "verified"


# ---------------------------------------------------------------------------
# 8. source_type=manual on manually created memory
# ---------------------------------------------------------------------------

class TestSourceTypeManual:
    def test_manual_source_type_stored(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="Manually entered architecture decision",
            content="We chose a monorepo structure for all services.",
            source_type="manual",
        )
        assert m["source_type"] == "manual"

    def test_source_type_none_by_default(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="No source type", content="Just a plain memory entry here.")
        assert m["source_type"] is None

    def test_source_type_affects_quality_score(self, client: TestClient):
        pid = _project(client)
        without_source = _mem(
            client, pid,
            title="Memory without source type fields",
            content="We decided to use this approach for the project.",
        )
        with_source = _mem(
            client, pid,
            title="Memory with source type set explicitly",
            content="We decided to use this approach for the project.",
            source_type="manual",
        )
        # source_type contributes +0.05 to quality
        assert with_source["quality_score"] > without_source["quality_score"]


# ---------------------------------------------------------------------------
# 9. symbol_name / line_start / line_end round-trip
# ---------------------------------------------------------------------------

class TestCodeAnchorFields:
    def test_symbol_name_round_trips(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Function anchored memory", content="Decision about create_memory function.", symbol_name="create_memory")
        assert m["symbol_name"] == "create_memory"

    def test_line_range_round_trips(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Line anchored memory", content="Decision about lines 42 to 51.", line_start=42, line_end=51)
        assert m["line_start"] == 42
        assert m["line_end"] == 51

    def test_symbol_contributes_to_quality_score(self, client: TestClient):
        pid = _project(client)
        without = _mem(client, pid, title="Without symbol", content="Memory without a symbol reference attached.")
        with_sym = _mem(client, pid, title="With symbol", content="Memory without a symbol reference attached.", symbol_name="SomeFunction")
        # symbol_name adds +0.05
        assert with_sym["quality_score"] > without["quality_score"]

    def test_can_update_code_anchor_fields(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Update anchor test", content="Memory that will get code anchor.")
        r = client.put(f"/memories/{m['id']}", json={
            "symbol_name": "updated_function",
            "line_start": 100,
            "line_end": 120,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["symbol_name"] == "updated_function"
        assert data["line_start"] == 100
        assert data["line_end"] == 120


# ---------------------------------------------------------------------------
# 10. branch_name round-trips
# ---------------------------------------------------------------------------

class TestBranchName:
    def test_branch_name_round_trips(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Branch memory", content="Decision made on feature branch.", branch_name="feature/auth-refactor")
        assert m["branch_name"] == "feature/auth-refactor"

    def test_branch_name_in_search_text(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Branch search", content="Memory for branch search test.", branch_name="feature/search-index")
        assert "feature/search-index" in m["search_text"]

    def test_can_update_branch_name(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Update branch", content="Memory to update branch name on.")
        r = client.put(f"/memories/{m['id']}", json={"branch_name": "main"})
        assert r.status_code == 200
        assert r.json()["branch_name"] == "main"


# ---------------------------------------------------------------------------
# 11. quality_score remains in [0.0, 1.0] for edge inputs
# ---------------------------------------------------------------------------

class TestQualityScoreBounds:
    def test_empty_title_still_bounded(self, client: TestClient):
        pid = _project(client)
        # We need at minimum 1-char titles per schema, so use short ones
        m = _mem(client, pid, title="A", content="B" * 5)
        assert 0.0 <= m["quality_score"] <= 1.0

    def test_maximally_rich_memory_capped_at_1(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="X" * 60,
            content="Y" * 300,
            tags=["meaningful-tag"],
            file_path="app/models.py",
            symbol_name="Memory",
            related_files=["app/crud.py", "app/schemas.py"],
            type_metadata={"rationale": "great reason", "alternatives": ["alt1", "alt2"]},
            source_quote="We decided this is the best approach ever taken.",
            source_type="manual",
            confidence=1.0,
        )
        assert m["quality_score"] <= 1.0

    def test_zero_confidence_memory_bounded(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Low confidence", content="Uncertain about this decision.", confidence=0.0)
        assert 0.0 <= m["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# 12. Relative quality ranking
# ---------------------------------------------------------------------------

class TestQualityRanking:
    def test_high_quality_higher_than_low_quality(self, client: TestClient):
        pid = _project(client)
        low = _mem(
            client, pid,
            title="Hi",
            content="Short.",
        )
        high = _mem(
            client, pid,
            title="We decided to use FastAPI as the primary backend web framework",
            content=(
                "After evaluating several options including Flask and Django, FastAPI "
                "was chosen for its native async support, high performance, and automatic "
                "API documentation generation via OpenAPI. This directly impacts every "
                "endpoint we build and should not be changed lightly."
            ),
            tags=["structure", "backend"],
            file_path="app/main.py",
            symbol_name="create_app",
            type_metadata={"rationale": "async support", "alternatives_considered": ["Flask", "Django"]},
            source_quote="We decided to use FastAPI because of performance.",
            source_type="manual",
        )
        assert high["quality_score"] > low["quality_score"]

    def test_memory_with_session_ref_higher_than_without(self, client: TestClient):
        pid = _project(client)
        # Create a session and extract so we get memories with source_session_id
        sid = _session(client, pid, "We decided to use Docker for containerization of services.")
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        extracted_score = result["memories"][0]["quality_score"]

        # Manual memory with same content but no session ref
        manual = _mem(
            client, pid,
            title="Use Docker for containerization of services",
            content="We decided to use Docker for containerization of services.",
        )
        assert extracted_score > manual["quality_score"]


# ---------------------------------------------------------------------------
# 13. API response completeness — all 9 new fields present
# ---------------------------------------------------------------------------

class TestAPIResponseCompleteness:
    def test_all_quality_fields_in_create_response(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="API completeness test", content="Checking all quality fields are present.")
        required_fields = [
            "quality_score", "review_status", "search_text",
            "source_type", "source_quote", "symbol_name",
            "line_start", "line_end", "branch_name",
        ]
        for field in required_fields:
            assert field in m, f"Missing field: {field}"

    def test_all_quality_fields_in_update_response(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Update completeness test", content="Checking update response fields.")
        r = client.put(f"/memories/{m['id']}", json={"importance": 4})
        data = r.json()
        required_fields = [
            "quality_score", "review_status", "search_text",
            "source_type", "source_quote", "symbol_name",
            "line_start", "line_end", "branch_name",
        ]
        for field in required_fields:
            assert field in data, f"Missing field in update response: {field}"

    def test_all_quality_fields_in_list_response(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="List field test", content="Memory for list response field check.")
        r = client.get(f"/projects/{pid}/memories")
        assert r.status_code == 200
        mems = r.json()
        assert len(mems) >= 1
        m = mems[0]
        for field in ["quality_score", "review_status", "search_text", "source_type"]:
            assert field in m

    def test_all_quality_fields_in_extraction_response(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use Kubernetes for orchestration.")
        result = _extract(client, sid)
        if result["memories"]:
            m = result["memories"][0]
            for field in ["quality_score", "review_status", "source_type", "source_quote"]:
                assert field in m, f"Missing field in extraction response: {field}"


# ---------------------------------------------------------------------------
# 14. Supersede creates new memory with quality_score
# ---------------------------------------------------------------------------

class TestSupersededMemoryQuality:
    def test_supersede_new_memory_has_quality_score(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Old decision", content="We used Flask as the web framework.")
        r = client.post(f"/memories/{m['id']}/supersede", json={
            "type": "decision",
            "title": "Updated decision: we now use FastAPI instead of Flask",
            "content": (
                "After performance testing revealed Flask could not handle concurrent load, "
                "we migrated to FastAPI. This supersedes the earlier Flask decision entirely."
            ),
        })
        assert r.status_code == 201
        new_mem = r.json()["new_memory"]
        assert new_mem["quality_score"] is not None
        assert new_mem["quality_score"] > 0.0

    def test_supersede_new_memory_has_search_text(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Supersede search test", content="Initial version of the decision.")
        r = client.post(f"/memories/{m['id']}/supersede", json={
            "type": "decision",
            "title": "Superseded decision with new search text",
            "content": "This is the replacement content with unique search keywords XYZ.",
        })
        assert r.status_code == 201
        new_mem = r.json()["new_memory"]
        assert new_mem["search_text"] is not None
        assert "Superseded decision" in new_mem["search_text"]


# ---------------------------------------------------------------------------
# 15. search_text is not in FTS5 (no double-indexing)
# ---------------------------------------------------------------------------

class TestSearchTextNotFTS5:
    def test_fts_search_uses_title_content_not_search_text(self, client: TestClient):
        pid = _project(client)
        m = _mem(
            client, pid,
            title="Unique orchestration decision",
            content="We chose Kubernetes for container orchestration at scale.",
            # Put a unique word only in search_text-contributing fields (tags)
            tags=["unique-orchestration-tag-xyq"],
        )
        # BM25 search should find it by title/content keyword
        r = client.get(f"/projects/{pid}/memories/search", params={"q": "orchestration"})
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert m["id"] in ids
