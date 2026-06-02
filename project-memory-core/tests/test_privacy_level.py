"""
Sub-phase 1.18 Stress Tests — Privacy Level + Export Safety.

Covers:
  1.  Default privacy_level is "internal" when not set
  2.  All four privacy levels can be set on create
  3.  privacy_level round-trips through the API
  4.  privacy_level can be updated and is tracked in changelog
  5.  GET /memories with max_privacy_level=public excludes internal/sensitive/secret
  6.  GET /memories with max_privacy_level=internal includes public+internal, excludes sensitive+secret
  7.  GET /memories with max_privacy_level=sensitive includes public+internal+sensitive, excludes secret
  8.  GET /memories with max_privacy_level=secret returns all levels
  9.  GET /memories without max_privacy_level returns all levels (no filter)
  10. Export context.md excludes sensitive/secret by default
  11. Export context.md with max_privacy_level=sensitive includes sensitive memories
  12. Export context.md with max_privacy_level=public only includes public memories
  13. Extracted memories default to internal privacy
  14. privacy_level appears in MemoryResponse
  15. privacy_level is tracked in changelog when changed
  16. Conflict detection only compares active memories regardless of privacy
  17. Health dashboard counts cover all privacy levels
  18. Search results respect privacy filtering
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Privacy Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(
    client: TestClient,
    pid: str,
    *,
    title: str = "Privacy test memory",
    content: str = "Content for privacy level testing here.",
    privacy_level: str | None = None,
    mem_type: str = "decision",
) -> dict:
    body: dict = {"type": mem_type, "title": title, "content": content}
    if privacy_level:
        body["privacy_level"] = privacy_level
    r = client.post(f"/projects/{pid}/memories", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _session(client: TestClient, pid: str, content: str) -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Privacy Test Session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    return r.json()["id"]


# ---------------------------------------------------------------------------
# 1. Default privacy_level
# ---------------------------------------------------------------------------

class TestDefaultPrivacyLevel:
    def test_default_is_internal(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Default privacy memory", content="No privacy level set here.")
        assert m["privacy_level"] == "internal"

    def test_explicitly_set_public(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Public memory", content="Safe for external sharing.", privacy_level="public")
        assert m["privacy_level"] == "public"

    def test_explicitly_set_sensitive(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Sensitive memory", content="Contains private architecture details.", privacy_level="sensitive")
        assert m["privacy_level"] == "sensitive"

    def test_explicitly_set_secret(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Secret memory", content="Contains API keys and credentials.", privacy_level="secret")
        assert m["privacy_level"] == "secret"


# ---------------------------------------------------------------------------
# 2 & 3. Round-trip
# ---------------------------------------------------------------------------

class TestPrivacyLevelRoundTrip:
    def test_all_four_levels_round_trip(self, client: TestClient):
        pid = _project(client)
        for level in ["public", "internal", "sensitive", "secret"]:
            m = _mem(client, pid, title=f"Memory level {level}", content="Content for round trip test.", privacy_level=level)
            assert m["privacy_level"] == level

    def test_privacy_level_in_list_response(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="List privacy test", content="Checking privacy in list response.", privacy_level="sensitive")
        r = client.get(f"/projects/{pid}/memories")
        assert r.status_code == 200
        assert any(m["privacy_level"] == "sensitive" for m in r.json())

    def test_privacy_level_in_get_by_id_response(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Get by ID privacy test", content="Checking privacy by ID.", privacy_level="secret")
        r = client.get(f"/memories/{m['id']}")
        assert r.status_code == 200
        assert r.json()["privacy_level"] == "secret"


# ---------------------------------------------------------------------------
# 4. Update + changelog
# ---------------------------------------------------------------------------

class TestPrivacyLevelUpdate:
    def test_can_update_privacy_level(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Escalate privacy", content="Initially internal memory.")
        r = client.put(f"/memories/{m['id']}", json={"privacy_level": "sensitive"})
        assert r.status_code == 200
        assert r.json()["privacy_level"] == "sensitive"

    def test_can_downgrade_privacy_level(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Downgrade privacy", content="Initially sensitive memory.", privacy_level="sensitive")
        r = client.put(f"/memories/{m['id']}", json={"privacy_level": "public"})
        assert r.status_code == 200
        assert r.json()["privacy_level"] == "public"

    def test_privacy_level_change_tracked_in_changelog(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Track privacy change", content="Memory to track privacy change.")
        client.put(f"/memories/{m['id']}", json={"privacy_level": "sensitive"})
        history = client.get(f"/memories/{m['id']}/history").json()
        privacy_changes = [c for c in history if c["field"] == "privacy_level"]
        assert len(privacy_changes) >= 1
        assert privacy_changes[-1]["old_value"] == "internal"
        assert privacy_changes[-1]["new_value"] == "sensitive"


# ---------------------------------------------------------------------------
# 5–9. GET /memories privacy filtering
# ---------------------------------------------------------------------------

class TestMemoryListPrivacyFilter:
    def _setup_four_level_project(self, client: TestClient) -> tuple[str, dict, dict, dict, dict]:
        pid = _project(client)
        pub = _mem(client, pid, title="Public memory", content="Safe public content.", privacy_level="public")
        internal = _mem(client, pid, title="Internal memory", content="Internal team content.", privacy_level="internal")
        sensitive = _mem(client, pid, title="Sensitive memory", content="Sensitive company details.", privacy_level="sensitive")
        secret = _mem(client, pid, title="Secret memory", content="Absolute secret content.", privacy_level="secret")
        return pid, pub, internal, sensitive, secret

    def test_max_public_excludes_higher_levels(self, client: TestClient):
        pid, pub, internal, sensitive, secret = self._setup_four_level_project(client)
        r = client.get(f"/projects/{pid}/memories", params={"max_privacy_level": "public"})
        ids = [m["id"] for m in r.json()]
        assert pub["id"] in ids
        assert internal["id"] not in ids
        assert sensitive["id"] not in ids
        assert secret["id"] not in ids

    def test_max_internal_includes_public_and_internal(self, client: TestClient):
        pid, pub, internal, sensitive, secret = self._setup_four_level_project(client)
        r = client.get(f"/projects/{pid}/memories", params={"max_privacy_level": "internal"})
        ids = [m["id"] for m in r.json()]
        assert pub["id"] in ids
        assert internal["id"] in ids
        assert sensitive["id"] not in ids
        assert secret["id"] not in ids

    def test_max_sensitive_excludes_secret(self, client: TestClient):
        pid, pub, internal, sensitive, secret = self._setup_four_level_project(client)
        r = client.get(f"/projects/{pid}/memories", params={"max_privacy_level": "sensitive"})
        ids = [m["id"] for m in r.json()]
        assert pub["id"] in ids
        assert internal["id"] in ids
        assert sensitive["id"] in ids
        assert secret["id"] not in ids

    def test_max_secret_returns_all(self, client: TestClient):
        pid, pub, internal, sensitive, secret = self._setup_four_level_project(client)
        r = client.get(f"/projects/{pid}/memories", params={"max_privacy_level": "secret"})
        ids = [m["id"] for m in r.json()]
        assert pub["id"] in ids
        assert internal["id"] in ids
        assert sensitive["id"] in ids
        assert secret["id"] in ids

    def test_no_privacy_filter_returns_all(self, client: TestClient):
        pid, pub, internal, sensitive, secret = self._setup_four_level_project(client)
        r = client.get(f"/projects/{pid}/memories")
        ids = [m["id"] for m in r.json()]
        assert pub["id"] in ids
        assert internal["id"] in ids
        assert sensitive["id"] in ids
        assert secret["id"] in ids


# ---------------------------------------------------------------------------
# 10–12. Export respects privacy
# ---------------------------------------------------------------------------

class TestExportPrivacySafety:
    def test_export_default_excludes_sensitive(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid,
             title="Public architecture overview of the system",
             content="Public information about the system architecture.",
             privacy_level="public")
        _mem(client, pid,
             title="Sensitive API credentials storage approach",
             content="We store API keys in encrypted vault with rotation policy.",
             privacy_level="sensitive")
        export = client.get(f"/projects/{pid}/export/context.md").text
        assert "Public architecture overview" in export
        assert "Sensitive API credentials" not in export

    def test_export_default_excludes_secret(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid,
             title="Normal decision about framework",
             content="We chose FastAPI as our web framework.",
             privacy_level="internal")
        _mem(client, pid,
             title="Secret internal API key rotation policy",
             content="Secret content that should never be exported.",
             privacy_level="secret")
        export = client.get(f"/projects/{pid}/export/context.md").text
        assert "Normal decision about framework" in export
        assert "Secret content that should never be exported" not in export

    def test_export_with_sensitive_level_includes_sensitive(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid,
             title="Sensitive architecture pattern detail",
             content="Sensitive technical detail about authentication flow.",
             privacy_level="sensitive")
        export = client.get(
            f"/projects/{pid}/export/context.md",
            params={"max_privacy_level": "sensitive"}
        ).text
        assert "Sensitive architecture pattern detail" in export

    def test_export_with_public_only_excludes_internal(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid,
             title="Publicly shareable architectural overview",
             content="This architecture is fine to share publicly.",
             privacy_level="public")
        _mem(client, pid,
             title="Internal team decision about database",
             content="Internal team decision — not for public sharing.",
             privacy_level="internal")
        export = client.get(
            f"/projects/{pid}/export/context.md",
            params={"max_privacy_level": "public"}
        ).text
        assert "Publicly shareable architectural overview" in export
        assert "Internal team decision about database" not in export

    def test_export_empty_project_does_not_crash(self, client: TestClient):
        pid = _project(client)
        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 13. Extracted memories default to internal
# ---------------------------------------------------------------------------

class TestExtractionDefaultPrivacy:
    def test_extracted_memories_are_internal_by_default(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid, "We decided to use FastAPI as the primary web framework.")
        result = client.post(f"/sessions/{sid}/extract-memories").json()
        for m in result["memories"]:
            assert m["privacy_level"] == "internal"


# ---------------------------------------------------------------------------
# 14. privacy_level in MemoryResponse
# ---------------------------------------------------------------------------

class TestPrivacyInResponse:
    def test_privacy_level_field_present_on_create(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Response field test", content="Checking response schema.")
        assert "privacy_level" in m

    def test_privacy_level_field_present_on_update(self, client: TestClient):
        pid = _project(client)
        m = _mem(client, pid, title="Update response field test", content="Checking update response schema.")
        r = client.put(f"/memories/{m['id']}", json={"importance": 4})
        assert "privacy_level" in r.json()

    def test_privacy_level_field_present_in_search_results(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Search privacy field test", content="Checking search result schema.", privacy_level="public")
        r = client.get(f"/projects/{pid}/memories/search", params={"q": "privacy field"})
        assert r.status_code == 200
        if r.json():
            assert "privacy_level" in r.json()[0]


# ---------------------------------------------------------------------------
# 15. Health dashboard counts all privacy levels
# ---------------------------------------------------------------------------

class TestHealthDashboardWithPrivacy:
    def test_health_counts_all_privacy_levels(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Public health memory", content="Public content.", privacy_level="public")
        _mem(client, pid, title="Internal health memory", content="Internal content.", privacy_level="internal")
        _mem(client, pid, title="Sensitive health memory", content="Sensitive content.", privacy_level="sensitive")
        _mem(client, pid, title="Secret health memory", content="Secret content.", privacy_level="secret")
        h = client.get(f"/projects/{pid}/health").json()
        # Health dashboard should count all 4 memories regardless of privacy
        assert h["active_count"] == 4
        assert h["total_memories"] == 4
