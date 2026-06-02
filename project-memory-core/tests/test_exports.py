from tests.conftest import make_project, make_session, make_memory


def test_export_empty_project(client):
    p = make_project(client, name="Export Test")
    r = client.get(f"/projects/{p['id']}/export/context.md")
    assert r.status_code == 200
    body = r.text
    assert "# Project Context: Export Test" in body
    assert "## Goals" in body
    assert "## Tech Stack" in body
    assert "## Recent AI Sessions" in body


def test_export_contains_memories(client):
    p = make_project(client, name="Memory Project")
    make_memory(client, p["id"], type="decision", title="Use FastAPI")
    make_memory(client, p["id"], type="problem", title="Auth flow broken")
    make_memory(client, p["id"], type="task", title="Write tests")

    r = client.get(f"/projects/{p['id']}/export/context.md")
    assert r.status_code == 200
    body = r.text

    assert "## Active Decisions" in body
    assert "Use FastAPI" in body
    assert "## Problems" in body
    assert "Auth flow broken" in body
    assert "## Current Tasks" in body
    assert "Write tests" in body


def test_export_contains_session_summaries(client):
    p = make_project(client)
    s = make_session(client, p["id"], title="Architecture Session")
    # Add summary to session
    client.put(f"/sessions/{s['id']}", json={"summary": "We decided to use SQLite."})

    r = client.get(f"/projects/{p['id']}/export/context.md")
    body = r.text
    assert "Architecture Session" in body
    assert "We decided to use SQLite." in body


def test_export_omits_resolved_memories(client):
    p = make_project(client)
    m = make_memory(client, p["id"], type="problem", title="Fixed bug")
    client.put(f"/memories/{m['id']}", json={"status": "resolved"})

    r = client.get(f"/projects/{p['id']}/export/context.md")
    body = r.text
    # Resolved memories are not active, should not appear in export
    assert "Fixed bug" not in body


def test_export_project_not_found(client):
    r = client.get("/projects/nonexistent/export/context.md")
    assert r.status_code == 404


def test_export_includes_tech_stack_and_goals(client):
    r = client.post("/projects", json={
        "name": "Stack Test",
        "tech_stack": ["Python", "FastAPI", "SQLite"],
        "goals": ["Build Phase 1", "Ship by June"],
    })
    p = r.json()
    export = client.get(f"/projects/{p['id']}/export/context.md").text
    assert "Python" in export
    assert "FastAPI" in export
    assert "Build Phase 1" in export
    assert "Ship by June" in export


def test_extract_memories_from_session(client):
    p = make_project(client)
    s = client.post(f"/projects/{p['id']}/sessions", json={
        "tool_name": "Claude",
        "title": "Design session",
        "raw_content": (
            "We decided to use SQLite for local storage. "
            "There is a bug in the authentication flow that causes a crash. "
            "TODO: write integration tests for the memory endpoint. "
            "The architecture follows a layered service pattern. "
            "Setup requires installing Python 3.11 and running pip install."
        ),
    }).json()

    r = client.post(f"/sessions/{s['id']}/extract-memories")
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == s["id"]
    assert data["memories_created"] > 0
    assert data["summary"] != ""
    # Each returned memory should reference the session
    for m in data["memories"]:
        assert m["source_session_id"] == s["id"]
        assert "auto-extracted" in m["tags"]


def test_extract_memories_updates_session_summary(client):
    p = make_project(client)
    s = make_session(client, p["id"])
    client.post(f"/sessions/{s['id']}/extract-memories")
    updated = client.get(f"/sessions/{s['id']}").json()
    assert updated["summary"] is not None
    assert len(updated["summary"]) > 0


def test_extract_memories_session_not_found(client):
    r = client.post("/sessions/bad-id/extract-memories")
    assert r.status_code == 404
