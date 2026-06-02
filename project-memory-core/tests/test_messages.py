"""
Sub-phase 1.31 — Messages tests.

Covers:
  1.  Session creation auto-creates 1 message (backward compat)
  2.  Auto-created message has role=unknown and content_type=mixed
  3.  Auto-created message content matches session raw_content
  4.  Auto-created message token_estimate > 0
  5.  Session creation with structured messages stores each individually
  6.  Structured messages respect role and content_type
  7.  message_index ordering preserved
  8.  GET /sessions/{id}/messages returns messages in index order
  9.  GET /sessions/{id}/messages returns 404 for unknown session
  10. POST /sessions/{id}/messages adds a new message
  11. Added message gets correct message_index (next in sequence)
  12. GET /messages/{id} returns correct message
  13. GET /messages/{id} returns 404 for unknown message
  14. DELETE /messages/{id} removes the message
  15. DELETE /messages/{id} returns 204
  16. has_code detected from code blocks in content
  17. has_error detected from traceback/Error in content
  18. language detected from fenced code block
  19. token_estimate scales with content length
  20. extra_metadata (metadata field) stored and returned correctly
  21. Session cascade delete removes messages
  22. message_count on SessionResponse reflects actual message count
  23. Structured messages: 3 messages → message_count = 3
  24. source_message_ids on MemoryResponse defaults to empty list
  25. GET /projects/{id}/sessions returns message_count per session
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as DBSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Msg Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session_raw(client: TestClient, pid: str, content: str = "We decided to use Redis.") -> dict:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Test Session",
        "raw_content": content,
        "session_date": "2026-05-26",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _session_structured(client: TestClient, pid: str, messages: list[dict]) -> dict:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Structured Session",
        "raw_content": " ".join(m["content"] for m in messages),
        "session_date": "2026-05-26",
        "messages": messages,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _messages(client: TestClient, sid: str) -> list[dict]:
    r = client.get(f"/sessions/{sid}/messages")
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------------
# 1–4: Backward compat — raw_content auto-message
# ---------------------------------------------------------------------------

class TestAutoMessageCreation:
    def test_raw_session_creates_one_message(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        msgs = _messages(client, s["id"])
        assert len(msgs) == 1

    def test_auto_message_role_is_unknown(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        msg = _messages(client, s["id"])[0]
        assert msg["role"] == "unknown"

    def test_auto_message_content_type_is_mixed(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        msg = _messages(client, s["id"])[0]
        assert msg["content_type"] == "mixed"

    def test_auto_message_content_matches_raw(self, client: TestClient):
        pid = _project(client)
        content = "We decided to use PostgreSQL for the database."
        s = _session_raw(client, pid, content=content)
        msg = _messages(client, s["id"])[0]
        assert msg["content"] == content

    def test_auto_message_token_estimate_nonzero(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid, content="We use Redis for caching.")
        msg = _messages(client, s["id"])[0]
        assert msg["token_estimate"] > 0


# ---------------------------------------------------------------------------
# 5–7: Structured messages
# ---------------------------------------------------------------------------

class TestStructuredMessages:
    def test_structured_messages_stored_individually(self, client: TestClient):
        pid = _project(client)
        s = _session_structured(client, pid, [
            {"role": "user", "content": "Which database should we use?"},
            {"role": "assistant", "content": "I recommend PostgreSQL for ACID compliance."},
        ])
        msgs = _messages(client, s["id"])
        assert len(msgs) == 2

    def test_structured_messages_preserve_roles(self, client: TestClient):
        pid = _project(client)
        s = _session_structured(client, pid, [
            {"role": "user", "content": "Should we use Redis?"},
            {"role": "assistant", "content": "Yes, Redis is great for caching."},
        ])
        msgs = _messages(client, s["id"])
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_structured_messages_ordered_by_index(self, client: TestClient):
        pid = _project(client)
        s = _session_structured(client, pid, [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ])
        msgs = _messages(client, s["id"])
        assert [m["message_index"] for m in msgs] == [0, 1, 2]
        assert msgs[0]["content"] == "First message"
        assert msgs[2]["content"] == "Third message"


# ---------------------------------------------------------------------------
# 8–9: List endpoint
# ---------------------------------------------------------------------------

class TestListMessages:
    def test_list_returns_messages_in_order(self, client: TestClient):
        pid = _project(client)
        s = _session_structured(client, pid, [
            {"role": "user", "content": "Alpha"},
            {"role": "assistant", "content": "Beta"},
        ])
        msgs = _messages(client, s["id"])
        assert msgs[0]["content"] == "Alpha"
        assert msgs[1]["content"] == "Beta"

    def test_list_404_unknown_session(self, client: TestClient):
        r = client.get("/sessions/does-not-exist/messages")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 10–15: Add / Get / Delete
# ---------------------------------------------------------------------------

class TestMessageCRUD:
    def test_add_message_to_session(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        r = client.post(f"/sessions/{s['id']}/messages", json={
            "role": "user",
            "content": "Actually, let's reconsider and use MySQL.",
            "content_type": "text",
        })
        assert r.status_code == 201
        assert r.json()["role"] == "user"

    def test_added_message_gets_next_index(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)  # starts with index 0
        r = client.post(f"/sessions/{s['id']}/messages", json={
            "role": "assistant",
            "content": "Follow-up message.",
        })
        assert r.json()["message_index"] == 1

    def test_get_message_by_id(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid, content="We chose FastAPI.")
        msg_id = _messages(client, s["id"])[0]["id"]
        r = client.get(f"/messages/{msg_id}")
        assert r.status_code == 200
        assert r.json()["id"] == msg_id

    def test_get_message_404(self, client: TestClient):
        r = client.get("/messages/does-not-exist")
        assert r.status_code == 404

    def test_delete_message(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        r = client.post(f"/sessions/{s['id']}/messages", json={
            "role": "user",
            "content": "Extra message to delete.",
        })
        msg_id = r.json()["id"]
        del_r = client.delete(f"/messages/{msg_id}")
        assert del_r.status_code == 204
        assert client.get(f"/messages/{msg_id}").status_code == 404

    def test_delete_message_returns_204(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        msg_id = _messages(client, s["id"])[0]["id"]
        r = client.delete(f"/messages/{msg_id}")
        assert r.status_code == 204


# ---------------------------------------------------------------------------
# 16–20: Content signal detection
# ---------------------------------------------------------------------------

class TestContentSignals:
    def test_has_code_detected(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid, content="Here is the fix:\n```python\nprint('hello')\n```")
        msg = _messages(client, s["id"])[0]
        assert msg["has_code"] is True

    def test_has_error_detected(self, client: TestClient):
        pid = _project(client)
        content = "We got this error:\nTraceback (most recent call last):\n  File app.py\nValueError: invalid input"
        s = _session_raw(client, pid, content=content)
        msg = _messages(client, s["id"])[0]
        assert msg["has_error"] is True

    def test_language_detected_from_fence(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid, content="Example:\n```typescript\nconst x = 1;\n```")
        msg = _messages(client, s["id"])[0]
        assert msg["language"] == "typescript"

    def test_token_estimate_scales_with_length(self, client: TestClient):
        pid = _project(client)
        short = _session_raw(client, pid, content="Hi.")
        long_content = "We decided to use PostgreSQL " * 50
        s2 = _session_raw(client, _project(client), content=long_content)
        short_tokens = _messages(client, short["id"])[0]["token_estimate"]
        long_tokens = _messages(client, s2["id"])[0]["token_estimate"]
        assert long_tokens > short_tokens

    def test_extra_metadata_stored_and_returned(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        r = client.post(f"/sessions/{s['id']}/messages", json={
            "role": "tool",
            "content": "Tool output here.",
            "metadata": {"tool_name": "web_search", "query": "redis vs memcached"},
        })
        assert r.status_code == 201
        assert r.json()["metadata"]["tool_name"] == "web_search"
        assert r.json()["metadata"]["query"] == "redis vs memcached"


# ---------------------------------------------------------------------------
# 21–25: Cascade, counts, traceability
# ---------------------------------------------------------------------------

class TestMessageIntegration:
    def test_session_delete_cascades_to_messages(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        msg_id = _messages(client, s["id"])[0]["id"]
        client.delete(f"/sessions/{s['id']}")
        assert client.get(f"/messages/{msg_id}").status_code == 404

    def test_message_count_on_session_response(self, client: TestClient):
        pid = _project(client)
        s = _session_raw(client, pid)
        assert s["message_count"] == 1

    def test_structured_three_messages_count(self, client: TestClient):
        pid = _project(client)
        s = _session_structured(client, pid, [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Two"},
            {"role": "user", "content": "Three"},
        ])
        assert s["message_count"] == 3

    def test_source_message_ids_defaults_empty(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Use Redis",
            "content": "Redis chosen for caching.",
        })
        assert r.status_code == 201
        assert r.json()["source_message_ids"] == []

    def test_list_sessions_includes_message_count(self, client: TestClient):
        pid = _project(client)
        _session_raw(client, pid)
        sessions = client.get(f"/projects/{pid}/sessions").json()
        assert len(sessions) == 1
        assert sessions[0]["message_count"] == 1
