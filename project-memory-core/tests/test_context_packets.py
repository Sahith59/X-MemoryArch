"""
Sub-phase 1.33 — ContextPackets tests.

Covers:
  1.  POST /projects/{id}/context-packets creates a packet (201)
  2.  Packet has correct project_id
  3.  Packet has correct target_tool
  4.  Packet has correct intent
  5.  Packet has non-empty content field
  6.  token_estimate > 0 when content is generated
  7.  included_memory_ids stored and returned correctly
  8.  included_session_ids stored and returned correctly
  9.  Content includes memory titles when memory_ids provided
  10. Content includes session titles when session_ids provided
  11. Memory content appears in packet (retrieval_hint section)
  12. GET /projects/{id}/context-packets lists all packets
  13. GET /projects/{id}/context-packets returns 404 for unknown project
  14. GET /context-packets/{id} returns the correct packet
  15. GET /context-packets/{id} returns 404 for unknown
  16. DELETE /context-packets/{id} returns 204
  17. DELETE removes the packet
  18. Project delete cascades to context packets
  19. Multiple packets can coexist for one project
  20. Empty memory_ids and session_ids produces valid packet with header only
  21. Packet with memories only (no sessions) works
  22. Packet with sessions only (no memories) works
  23. Memory IDs not in the project are silently excluded (no 422)
  24. token_estimate scales with content size
  25. Memories grouped by type in content
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Packet Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, title: str = "Test Session", content: str = "We decided to use PostgreSQL.") -> dict:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": title,
        "raw_content": content,
        "session_date": "2026-05-26",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _memory(client: TestClient, pid: str, title: str = "Use Redis", mem_type: str = "decision") -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": mem_type,
        "title": title,
        "content": f"Content for {title}.",
        "importance": 4,
        "confidence": 0.9,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _packet(client: TestClient, pid: str, **kwargs) -> dict:
    payload = {"target_tool": "ChatGPT", "intent": "Continue working on auth"}
    payload.update(kwargs)
    r = client.post(f"/projects/{pid}/context-packets", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–8: Create and field checks
# ---------------------------------------------------------------------------

class TestCreatePacket:
    def test_creates_packet(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        assert p["id"]

    def test_correct_project_id(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        assert p["project_id"] == pid

    def test_correct_target_tool(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        assert p["target_tool"] == "ChatGPT"

    def test_correct_intent(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        assert p["intent"] == "Continue working on auth"

    def test_content_non_empty(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        assert len(p["content"]) > 0

    def test_token_estimate_positive(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        p = _packet(client, pid, included_memory_ids=[m["id"]])
        assert p["token_estimate"] > 0

    def test_memory_ids_stored(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        p = _packet(client, pid, included_memory_ids=[m["id"]])
        assert m["id"] in p["included_memory_ids"]

    def test_session_ids_stored(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid)
        p = _packet(client, pid, included_session_ids=[s["id"]])
        assert s["id"] in p["included_session_ids"]


# ---------------------------------------------------------------------------
# 9–11: Content assembly
# ---------------------------------------------------------------------------

class TestPacketContent:
    def test_content_includes_memory_title(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, title="Use PostgreSQL for ACID compliance")
        p = _packet(client, pid, included_memory_ids=[m["id"]])
        assert "Use PostgreSQL for ACID compliance" in p["content"]

    def test_content_includes_session_title(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid, title="Database Architecture Session")
        p = _packet(client, pid, included_session_ids=[s["id"]])
        assert "Database Architecture Session" in p["content"]

    def test_content_includes_memory_body(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid, title="Switch to Redis")
        p = _packet(client, pid, included_memory_ids=[m["id"]])
        assert "Content for Switch to Redis" in p["content"]

    def test_memories_grouped_by_type(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Avoid MySQL", mem_type="decision")
        _memory(client, pid, title="NullPointer in auth", mem_type="problem")
        mids = [m["id"] for m in client.get(f"/projects/{pid}/memories").json()]
        p = _packet(client, pid, included_memory_ids=mids)
        # Both type headers should appear
        assert "Decision" in p["content"]
        assert "Problem" in p["content"]

    def test_target_tool_in_content(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid, target_tool="Cursor")
        assert "Cursor" in p["content"]

    def test_intent_in_content(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid, intent="Debug the payment flow")
        assert "Debug the payment flow" in p["content"]


# ---------------------------------------------------------------------------
# 12–15: List and get
# ---------------------------------------------------------------------------

class TestListAndGet:
    def test_list_returns_packets(self, client: TestClient):
        pid = _project(client)
        _packet(client, pid)
        _packet(client, pid, target_tool="Gemini")
        r = client.get(f"/projects/{pid}/context-packets")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_list_404_unknown_project(self, client: TestClient):
        r = client.get("/projects/does-not-exist/context-packets")
        assert r.status_code == 404

    def test_get_returns_packet(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        r = client.get(f"/context-packets/{p['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == p["id"]

    def test_get_404_unknown(self, client: TestClient):
        r = client.get("/context-packets/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 16–17: Delete
# ---------------------------------------------------------------------------

class TestDeletePacket:
    def test_delete_returns_204(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        r = client.delete(f"/context-packets/{p['id']}")
        assert r.status_code == 204

    def test_delete_removes_packet(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        client.delete(f"/context-packets/{p['id']}")
        assert client.get(f"/context-packets/{p['id']}").status_code == 404


# ---------------------------------------------------------------------------
# 18–25: Edge cases and integration
# ---------------------------------------------------------------------------

class TestPacketIntegration:
    def test_project_delete_cascades_to_packets(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid)
        client.delete(f"/projects/{pid}")
        assert client.get(f"/context-packets/{p['id']}").status_code == 404

    def test_multiple_packets_per_project(self, client: TestClient):
        pid = _project(client)
        for tool in ["Claude", "ChatGPT", "Cursor"]:
            _packet(client, pid, target_tool=tool)
        packets = client.get(f"/projects/{pid}/context-packets").json()
        assert len(packets) == 3

    def test_empty_ids_produces_valid_packet(self, client: TestClient):
        pid = _project(client)
        p = _packet(client, pid, included_memory_ids=[], included_session_ids=[])
        assert p["id"]
        assert len(p["content"]) > 0  # header still generated
        assert p["included_memory_ids"] == []
        assert p["included_session_ids"] == []

    def test_memories_only_no_sessions(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        p = _packet(client, pid, included_memory_ids=[m["id"]], included_session_ids=[])
        assert m["id"] in p["included_memory_ids"]
        assert p["included_session_ids"] == []

    def test_sessions_only_no_memories(self, client: TestClient):
        pid = _project(client)
        s = _session(client, pid)
        p = _packet(client, pid, included_memory_ids=[], included_session_ids=[s["id"]])
        assert s["id"] in p["included_session_ids"]
        assert p["included_memory_ids"] == []

    def test_foreign_memory_ids_silently_excluded(self, client: TestClient):
        pid = _project(client)
        other_pid = _project(client, name="Other Project")
        other_mem = _memory(client, other_pid)
        # Providing a memory ID from another project — should not raise 422
        p = _packet(client, pid, included_memory_ids=[other_mem["id"]])
        assert p["id"]
        # The foreign memory ID is stored in included_memory_ids (as given)
        # but content assembler silently excludes it (project_id filter)
        assert other_mem["title"] not in p["content"]

    def test_token_estimate_scales_with_content(self, client: TestClient):
        pid = _project(client)
        # Packet with no memories: smaller
        p_small = _packet(client, pid)
        # Packet with many memories: larger
        for i in range(5):
            _memory(client, pid, title=f"Decision {i}: Use tool {i} for performance reasons")
        mids = [m["id"] for m in client.get(f"/projects/{pid}/memories").json()]
        p_large = _packet(client, pid, included_memory_ids=mids)
        assert p_large["token_estimate"] >= p_small["token_estimate"]
