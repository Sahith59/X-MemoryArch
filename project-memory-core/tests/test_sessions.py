from tests.conftest import make_project, make_session


def test_create_session(client):
    p = make_project(client)
    r = client.post(f"/projects/{p['id']}/sessions", json={
        "tool_name": "ChatGPT",
        "title": "Design discussion",
        "raw_content": "We decided to use SQLite.",
        "session_date": "2026-05-24",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["project_id"] == p["id"]
    assert data["tool_name"] == "ChatGPT"
    assert data["session_date"] == "2026-05-24"
    assert data["summary"] is None


def test_create_session_project_not_found(client):
    r = client.post("/projects/bad-id/sessions", json={
        "tool_name": "Claude", "title": "T", "raw_content": "raw"
    })
    assert r.status_code == 404


def test_list_sessions(client):
    p = make_project(client)
    make_session(client, p["id"], title="S1")
    make_session(client, p["id"], title="S2")
    r = client.get(f"/projects/{p['id']}/sessions")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_list_sessions_scoped_to_project(client):
    p1 = make_project(client, "P1")
    p2 = make_project(client, "P2")
    make_session(client, p1["id"])
    make_session(client, p2["id"])
    r = client.get(f"/projects/{p1['id']}/sessions")
    assert len(r.json()) == 1
    assert r.json()[0]["project_id"] == p1["id"]


def test_get_session(client):
    p = make_project(client)
    s = make_session(client, p["id"])
    r = client.get(f"/sessions/{s['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == s["id"]


def test_get_session_not_found(client):
    r = client.get("/sessions/nonexistent")
    assert r.status_code == 404


def test_update_session(client):
    p = make_project(client)
    s = make_session(client, p["id"])
    r = client.put(f"/sessions/{s['id']}", json={"summary": "Quick summary"})
    assert r.status_code == 200
    assert r.json()["summary"] == "Quick summary"
    assert r.json()["title"] == s["title"]  # unchanged


def test_update_session_not_found(client):
    r = client.put("/sessions/bad-id", json={"title": "X"})
    assert r.status_code == 404


def test_delete_session(client):
    p = make_project(client)
    s = make_session(client, p["id"])
    r = client.delete(f"/sessions/{s['id']}")
    assert r.status_code == 204
    assert client.get(f"/sessions/{s['id']}").status_code == 404


def test_delete_session_not_found(client):
    r = client.delete("/sessions/bad-id")
    assert r.status_code == 404
