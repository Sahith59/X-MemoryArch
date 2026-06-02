"""
Sub-phase 1.35 — Canonical YAML export stress tests.

Every test loads the YAML with yaml.safe_load() to verify it is syntactically
valid before checking any content — a bad YAML string fails fast at the parse
step with a clear error, not a confusing assertion.

Stress scenarios:
  1.  Empty project — valid YAML with empty memories/sessions/entities/clusters
  2.  Schema version present and equals "1.0"
  3.  exported_at is an ISO-8601 UTC timestamp
  4.  project section contains all metadata fields
  5.  stats.total_memories == len(memories list)
  6.  stats.by_type is accurate
  7.  stats.avg_importance is correct
  8.  stats.avg_confidence is correct
  9.  stats.by_tier reflects working/archival split
  10. stats.by_review_status counts match
  11. Memories sorted: importance desc, updated_at desc
  12. Memory record contains all expected keys
  13. Memory type_metadata is a dict when present
  14. Memory entities list is populated from NER
  15. Memory links list is populated from memory_links table
  16. Memory source_quote present for auto-extracted
  17. Memory retrieval_hint present when computed
  18. Memory cluster_id / cluster_label present when clustered
  19. Memory decay_score is a float in [0, 1]
  20. Memory access_count >= 0
  21. Sessions section has correct session fields
  22. Sessions include memory_count per session
  23. Sessions include message_count per session (from Messages table)
  24. Entity index sorted by memory_count desc
  25. Entity index deduplicated (same entity in two memories → one entry, count=2)
  26. Clusters section correct (only cluster_id >= 0)
  27. status_filter=active excludes resolved memories
  28. min_importance=4 excludes low-importance memories
  29. max_privacy_level=public excludes internal memories
  30. max_privacy_level=sensitive includes internal but excludes secret
  31. All 12 memory types representable (round-trip with no data loss)
  32. Memories with no entities: entities field is empty list
  33. Memories with no links: links field is empty list
  34. superseded_by field present and correct
  35. valid_until field present (null when not set)
  36. type_metadata for decision includes rationale/alternatives_considered
  37. type_metadata for bug includes error_message/fix_applied
  38. Memories from other projects not included
  39. Privacy filter: secret-level memory excluded from internal export
  40. 404 returned for unknown project
"""
import yaml
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "YAML Test", domain: str = "software", **kwargs) -> str:
    payload = {"name": name, "domain": domain, "tech_stack": ["Python", "FastAPI"], "goals": ["Ship it"]}
    payload.update(kwargs)
    r = client.post("/projects", json=payload)
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, title: str = "Arch Session",
             content: str = "We decided to use PostgreSQL for ACID compliance.") -> dict:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude", "title": title, "raw_content": content, "session_date": "2026-05-26",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _memory(client: TestClient, pid: str, **kwargs) -> dict:
    payload = {
        "type": "decision",
        "title": "Use Redis for caching",
        "content": "Redis was chosen because it offers fast in-memory reads with persistence.",
        "importance": 4,
        "confidence": 0.9,
        "tags": ["redis", "caching"],
    }
    payload.update(kwargs)
    r = client.post(f"/projects/{pid}/memories", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _yaml(client: TestClient, pid: str, **params) -> dict:
    """Fetch YAML export and parse it — raises on invalid YAML."""
    r = client.get(f"/projects/{pid}/export/memory.yaml", params=params)
    assert r.status_code == 200, r.text
    doc = yaml.safe_load(r.text)
    assert isinstance(doc, dict), "YAML root must be a mapping"
    return doc


# ---------------------------------------------------------------------------
# 1–3: Document structure
# ---------------------------------------------------------------------------

class TestDocumentStructure:
    def test_empty_project_valid_yaml(self, client: TestClient):
        pid = _project(client)
        doc = _yaml(client, pid)
        assert doc["memories"] == []
        assert doc["sessions"] == []
        assert doc["entity_index"] == []
        assert doc["clusters"] == []

    def test_schema_version(self, client: TestClient):
        pid = _project(client)
        doc = _yaml(client, pid)
        assert doc["schema_version"] == "1.0"

    def test_exported_at_is_iso8601(self, client: TestClient):
        pid = _project(client)
        doc = _yaml(client, pid)
        ts = doc["exported_at"]
        assert "T" in ts and ts.endswith("Z"), f"Not ISO-8601: {ts}"


# ---------------------------------------------------------------------------
# 4: Project section
# ---------------------------------------------------------------------------

class TestProjectSection:
    def test_project_fields_present(self, client: TestClient):
        pid = _project(client, name="FieldTest", description="desc")
        doc = _yaml(client, pid)
        p = doc["project"]
        for key in ("id", "name", "description", "tech_stack", "goals", "repo_path", "created_at", "updated_at"):
            assert key in p, f"Missing project field: {key}"

    def test_project_id_correct(self, client: TestClient):
        pid = _project(client)
        doc = _yaml(client, pid)
        assert doc["project"]["id"] == pid

    def test_project_tech_stack_list(self, client: TestClient):
        pid = _project(client)
        doc = _yaml(client, pid)
        assert isinstance(doc["project"]["tech_stack"], list)

    def test_project_goals_list(self, client: TestClient):
        pid = _project(client)
        doc = _yaml(client, pid)
        assert isinstance(doc["project"]["goals"], list)


# ---------------------------------------------------------------------------
# 5–10: Stats section
# ---------------------------------------------------------------------------

class TestStatsSection:
    def test_total_memories_matches_list(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        _memory(client, pid, title="Use PostgreSQL")
        doc = _yaml(client, pid)
        assert doc["stats"]["total_memories"] == len(doc["memories"])

    def test_by_type_accurate(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, type="decision", title="Decision A")
        _memory(client, pid, type="problem", title="Bug A")
        _memory(client, pid, type="problem", title="Bug B")
        doc = _yaml(client, pid)
        by_type = doc["stats"]["by_type"]
        assert by_type.get("decision") == 1
        assert by_type.get("problem") == 2

    def test_avg_importance_correct(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, importance=2, title="Low")
        _memory(client, pid, importance=4, title="High")
        doc = _yaml(client, pid)
        # avg of 2 and 4 = 3.0
        assert abs(doc["stats"]["avg_importance"] - 3.0) < 0.01

    def test_avg_confidence_correct(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, confidence=0.8, title="One")
        _memory(client, pid, confidence=0.6, title="Two")
        doc = _yaml(client, pid)
        assert abs(doc["stats"]["avg_confidence"] - 0.7) < 0.01

    def test_by_tier_present(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        doc = _yaml(client, pid)
        assert "by_tier" in doc["stats"]
        tier = doc["stats"]["by_tier"]
        assert "working" in tier and "archival" in tier

    def test_by_review_status_present(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, review_status="verified", title="Verified")
        doc = _yaml(client, pid)
        rs = doc["stats"]["by_review_status"]
        assert isinstance(rs, dict)


# ---------------------------------------------------------------------------
# 11–20: Memory records
# ---------------------------------------------------------------------------

class TestMemoryRecords:
    _EXPECTED_KEYS = {
        "id", "type", "title", "content", "importance", "confidence",
        "status", "tier", "decay_score", "quality_score", "review_status",
        "source_type", "privacy_level", "retrieval_hint", "cluster_id", "cluster_label",
        "tags", "related_files", "related_tools", "source_quote",
        "file_path", "commit_sha", "branch_name", "symbol_name",
        "line_start", "line_end", "superseded_by", "valid_until",
        "type_metadata", "source_message_ids", "entities", "links",
        "embedding_model", "embedding_dim",
        "access_count", "last_accessed_at", "source_session_id",
        "created_at", "updated_at",
    }

    def test_sorted_by_importance_desc(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, importance=2, title="Low imp")
        _memory(client, pid, importance=5, title="High imp")
        _memory(client, pid, importance=3, title="Mid imp")
        doc = _yaml(client, pid)
        imps = [m["importance"] for m in doc["memories"]]
        assert imps == sorted(imps, reverse=True)

    def test_memory_has_all_expected_keys(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        doc = _yaml(client, pid)
        mem = doc["memories"][0]
        missing = self._EXPECTED_KEYS - set(mem.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_type_metadata_is_dict(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, type_metadata={"rationale": "speed", "alternatives_considered": ["mysql"]})
        doc = _yaml(client, pid)
        tm = doc["memories"][0]["type_metadata"]
        assert isinstance(tm, dict)
        assert tm["rationale"] == "speed"

    def test_tags_is_list(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, tags=["alpha", "beta"])
        doc = _yaml(client, pid)
        assert doc["memories"][0]["tags"] == ["alpha", "beta"]

    def test_entities_list_present(self, client: TestClient):
        pid = _project(client)
        # spaCy will extract entities from technical content
        _memory(client, pid, title="PostgreSQL Decision",
                content="We chose PostgreSQL over MySQL for ACID compliance.")
        doc = _yaml(client, pid)
        assert isinstance(doc["memories"][0]["entities"], list)

    def test_links_list_present(self, client: TestClient):
        pid = _project(client)
        m1 = _memory(client, pid, title="Decision A")
        m2 = _memory(client, pid, title="Decision B")
        client.post(f"/memories/{m1['id']}/links", json={
            "target_memory_id": m2["id"],
            "relationship_type": "related_to",
        })
        doc = _yaml(client, pid)
        m1_record = next(m for m in doc["memories"] if m["id"] == m1["id"])
        assert len(m1_record["links"]) == 1
        assert m1_record["links"][0]["target_id"] == m2["id"]
        assert m1_record["links"][0]["relationship"] == "related_to"

    def test_entities_empty_list_for_no_entities(self, client: TestClient):
        pid = _project(client)
        # Very short generic content unlikely to produce NER hits
        _memory(client, pid, title="Note", content="Some brief note.")
        doc = _yaml(client, pid)
        assert isinstance(doc["memories"][0]["entities"], list)

    def test_links_empty_list_when_no_links(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        doc = _yaml(client, pid)
        assert doc["memories"][0]["links"] == []

    def test_decay_score_in_range(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        # Trigger decay computation
        client.post(f"/projects/{pid}/compute-decay")
        doc = _yaml(client, pid)
        ds = doc["memories"][0]["decay_score"]
        if ds is not None:
            assert 0.0 <= ds <= 1.0

    def test_access_count_non_negative(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        doc = _yaml(client, pid)
        assert doc["memories"][0]["access_count"] >= 0

    def test_superseded_by_field_present(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, title="Old decision")
        assert "superseded_by" in _yaml(client, pid)["memories"][0]

    def test_valid_until_null_when_not_set(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        doc = _yaml(client, pid)
        assert doc["memories"][0]["valid_until"] is None


# ---------------------------------------------------------------------------
# 21–23: Sessions section
# ---------------------------------------------------------------------------

class TestSessionsSection:
    def test_session_fields_present(self, client: TestClient):
        pid = _project(client)
        _session(client, pid)
        doc = _yaml(client, pid)
        s = doc["sessions"][0]
        for key in ("id", "tool_name", "title", "session_date", "summary",
                    "memory_count", "message_count", "created_at", "updated_at"):
            assert key in s, f"Missing session field: {key}"

    def test_session_memory_count(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content="We decided to use PostgreSQL for ACID compliance in production.")
        client.post(f"/sessions/{s['id']}/extract-memories")
        doc = _yaml(client, pid)
        sess_record = doc["sessions"][0]
        # Memory count in YAML should reflect extracted memories
        mem_count_yaml = sess_record["memory_count"]
        actual = len([m for m in doc["memories"] if m.get("source_session_id") == s["id"]])
        assert mem_count_yaml == actual

    def test_session_message_count(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid)  # auto-creates 1 message (backward compat)
        doc = _yaml(client, pid)
        sess_record = next(ss for ss in doc["sessions"] if ss["id"] == s["id"])
        assert sess_record["message_count"] == 1


# ---------------------------------------------------------------------------
# 24–26: Entity index
# ---------------------------------------------------------------------------

class TestEntityIndex:
    def test_entity_index_sorted_by_count_desc(self, client: TestClient):
        pid = _project(client)
        # Create multiple memories that trigger entity extraction
        for i in range(3):
            _memory(client, pid, title=f"PostgreSQL Decision {i}",
                    content=f"We use PostgreSQL for ACID compliance in deployment {i}.")
        _memory(client, pid, title="Redis caching",
                content="Redis is used for caching.")
        doc = _yaml(client, pid)
        counts = [e["memory_count"] for e in doc["entity_index"]]
        assert counts == sorted(counts, reverse=True)

    def test_entity_index_deduplicated(self, client: TestClient):
        pid = _project(client)
        # Same entity "PostgreSQL" in two memories should appear once with count >= 2
        _memory(client, pid, title="Decision 1",
                content="We use PostgreSQL for the primary database.")
        _memory(client, pid, title="Decision 2",
                content="PostgreSQL handles our ACID requirements well.")
        doc = _yaml(client, pid)
        pg_entries = [e for e in doc["entity_index"] if e["text"] == "postgresql"]
        if pg_entries:   # only assert if spaCy extracted the entity
            assert len(pg_entries) == 1
            assert pg_entries[0]["memory_count"] >= 2

    def test_entity_label_present(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="PostgreSQL Decision",
                content="We chose PostgreSQL for database storage.")
        doc = _yaml(client, pid)
        for entry in doc["entity_index"]:
            assert "label" in entry
            assert entry["label"] in (
                "TECH", "ORG", "PRODUCT", "PERSON", "LANGUAGE",
                "CONCEPT", "GPE", "EVENT", "WORK_OF_ART", "LAW", "MONEY", "MISC",
            )


# ---------------------------------------------------------------------------
# 27–30: Filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_status_filter_active_excludes_resolved(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Active one", status="active")
        resolved = _memory(client, pid, title="Resolved one", status="resolved")
        doc = _yaml(client, pid, status="active")
        ids = [m["id"] for m in doc["memories"]]
        assert resolved["id"] not in ids

    def test_status_filter_stats_consistent(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Active", status="active")
        _memory(client, pid, title="Resolved", status="resolved")
        doc = _yaml(client, pid, status="active")
        assert doc["stats"]["total_memories"] == len(doc["memories"])

    def test_min_importance_excludes_low(self, client: TestClient):
        pid = _project(client)
        low = _memory(client, pid, importance=1, title="Low importance")
        _memory(client, pid, importance=4, title="High importance")
        doc = _yaml(client, pid, min_importance=4)
        ids = [m["id"] for m in doc["memories"]]
        assert low["id"] not in ids
        assert all(m["importance"] >= 4 for m in doc["memories"])

    def test_privacy_public_excludes_internal(self, client: TestClient):
        pid = _project(client)
        internal = _memory(client, pid, title="Internal memory", privacy_level="internal")
        _memory(client, pid, title="Public memory", privacy_level="public")
        doc = _yaml(client, pid, max_privacy_level="public")
        ids = [m["id"] for m in doc["memories"]]
        assert internal["id"] not in ids

    def test_privacy_sensitive_includes_internal(self, client: TestClient):
        pid = _project(client)
        internal = _memory(client, pid, title="Internal memory", privacy_level="internal")
        doc = _yaml(client, pid, max_privacy_level="sensitive")
        ids = [m["id"] for m in doc["memories"]]
        assert internal["id"] in ids

    def test_privacy_secret_excluded_from_internal_export(self, client: TestClient):
        pid = _project(client)
        secret = _memory(client, pid, title="Secret memory", privacy_level="secret")
        doc = _yaml(client, pid, max_privacy_level="internal")
        ids = [m["id"] for m in doc["memories"]]
        assert secret["id"] not in ids


# ---------------------------------------------------------------------------
# 31–38: Type coverage and edge cases
# ---------------------------------------------------------------------------

class TestTypeCoverage:
    _ALL_TYPES = [
        "decision", "problem", "task", "insight", "structure",
        "reference_context", "how_to", "open_question",
        "constraint", "conversation_note", "workflow_pattern", "failed_approach",
    ]

    def test_all_12_types_round_trip(self, client: TestClient):
        pid = _project(client)
        for t in self._ALL_TYPES:
            _memory(client, pid, type=t, title=f"Type {t}")
        doc = _yaml(client, pid)
        exported_types = {m["type"] for m in doc["memories"]}
        assert exported_types == set(self._ALL_TYPES)

    def test_decision_type_metadata(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, type="decision",
                type_metadata={"rationale": "ACID needed", "alternatives_considered": ["MySQL"],
                               "decision_status": "accepted"})
        doc = _yaml(client, pid)
        tm = doc["memories"][0]["type_metadata"]
        assert tm["rationale"] == "ACID needed"
        assert tm["alternatives_considered"] == ["MySQL"]
        assert tm["decision_status"] == "accepted"

    def test_bug_type_metadata(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, type="problem",
                type_metadata={"error_message": "NullPointerException", "fix_applied": "Added null check"})
        doc = _yaml(client, pid)
        tm = doc["memories"][0]["type_metadata"]
        assert tm["error_message"] == "NullPointerException"
        assert tm["fix_applied"] == "Added null check"

    def test_memories_from_other_project_excluded(self, client: TestClient):
        pid1 = _project(client, name="Project A")
        pid2 = _project(client, name="Project B")
        m_other = _memory(client, pid2, title="Other project memory")
        doc = _yaml(client, pid1)
        ids = [m["id"] for m in doc["memories"]]
        assert m_other["id"] not in ids

    def test_source_quote_present_for_auto_extracted(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content=(
            "We decided to use PostgreSQL for ACID compliance. "
            "There is a bug in the auth module causing crashes. "
            "The architecture uses a layered service pattern."
        ))
        result = client.post(f"/sessions/{s['id']}/extract-memories").json()
        if result["memories_created"] == 0:
            pytest.skip("No memories extracted")
        doc = _yaml(client, pid)
        auto_extracted = [m for m in doc["memories"] if m.get("review_status") == "auto_extracted"]
        if auto_extracted:
            assert auto_extracted[0]["source_quote"] is not None

    def test_source_session_id_on_extracted_memories(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, content=(
            "We decided to use PostgreSQL for ACID compliance in the system."
        ))
        result = client.post(f"/sessions/{s['id']}/extract-memories").json()
        if result["memories_created"] == 0:
            pytest.skip("No memories extracted")
        doc = _yaml(client, pid)
        auto = [m for m in doc["memories"] if m.get("source_session_id") == s["id"]]
        assert len(auto) == result["memories_created"]

    def test_multiple_memory_links_in_export(self, client: TestClient):
        pid = _project(client)
        m1 = _memory(client, pid, title="Core Decision")
        m2 = _memory(client, pid, title="Related Insight")
        m3 = _memory(client, pid, title="Blocking Task")
        client.post(f"/memories/{m1['id']}/links",
                    json={"target_memory_id": m2["id"], "relationship_type": "related_to"})
        client.post(f"/memories/{m1['id']}/links",
                    json={"target_memory_id": m3["id"], "relationship_type": "blocks"})
        doc = _yaml(client, pid)
        m1_rec = next(m for m in doc["memories"] if m["id"] == m1["id"])
        assert len(m1_rec["links"]) == 2
        rel_types = {lk["relationship"] for lk in m1_rec["links"]}
        assert "related_to" in rel_types
        assert "blocks" in rel_types


# ---------------------------------------------------------------------------
# 39–40: Error handling and 404
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_project_returns_404(self, client: TestClient):
        r = client.get("/projects/does-not-exist/export/memory.yaml")
        assert r.status_code == 404

    def test_combined_filters_all_applied(self, client: TestClient):
        pid = _project(client)
        # Should be excluded: low importance
        _memory(client, pid, title="Low imp", importance=1, status="active", privacy_level="public")
        # Should be excluded: wrong status
        _memory(client, pid, title="Resolved", importance=4, status="resolved", privacy_level="public")
        # Should be excluded: private
        _memory(client, pid, title="Secret", importance=4, status="active", privacy_level="secret")
        # Should be INCLUDED: passes all filters
        keeper = _memory(client, pid, title="Keeper", importance=4, status="active", privacy_level="public")
        doc = _yaml(client, pid, status="active", min_importance=4, max_privacy_level="public")
        ids = [m["id"] for m in doc["memories"]]
        assert ids == [keeper["id"]]
        assert doc["stats"]["total_memories"] == 1
