from tests.conftest import make_project, make_memory, make_session


def test_create_memory(client):
    p = make_project(client)
    r = client.post(f"/projects/{p['id']}/memories", json={
        "type": "decision",
        "title": "Use SQLite",
        "content": "Chose SQLite for local simplicity",
        "importance": 4,
        "confidence": 0.95,
        "tags": ["db", "structure"],
        "related_tools": ["Claude"],
        "status": "active",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["type"] == "decision"
    assert data["tags"] == ["db", "structure"]
    assert data["related_tools"] == ["Claude"]
    assert data["importance"] == 4
    assert data["status"] == "active"


def test_create_memory_invalid_type(client):
    p = make_project(client)
    r = client.post(f"/projects/{p['id']}/memories", json={
        "type": "not_a_real_type",
        "title": "T", "content": "C", "importance": 3,
        "confidence": 1.0, "tags": [], "related_tools": [], "status": "active",
    })
    assert r.status_code == 422


def test_create_memory_importance_out_of_range(client):
    p = make_project(client)
    r = client.post(f"/projects/{p['id']}/memories", json={
        "type": "problem", "title": "T", "content": "C",
        "importance": 10,  # invalid
        "confidence": 1.0, "tags": [], "related_tools": [], "status": "active",
    })
    assert r.status_code == 422


def test_create_memory_project_not_found(client):
    r = client.post("/projects/bad-id/memories", json={
        "type": "problem", "title": "T", "content": "C", "importance": 3,
        "confidence": 1.0, "tags": [], "related_tools": [], "status": "active",
    })
    assert r.status_code == 404


def test_list_memories(client):
    p = make_project(client)
    make_memory(client, p["id"], type="decision", title="D1")
    make_memory(client, p["id"], type="problem", title="B1")
    r = client.get(f"/projects/{p['id']}/memories")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_filter_by_type(client):
    p = make_project(client)
    make_memory(client, p["id"], type="decision")
    make_memory(client, p["id"], type="problem")
    r = client.get(f"/projects/{p['id']}/memories?type=decision")
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["type"] == "decision"


def test_filter_by_status(client):
    p = make_project(client)
    m = make_memory(client, p["id"])
    # Update one to resolved
    client.put(f"/memories/{m['id']}", json={"status": "resolved"})
    make_memory(client, p["id"], title="Active one")

    r = client.get(f"/projects/{p['id']}/memories?status=resolved")
    assert len(r.json()) == 1
    r = client.get(f"/projects/{p['id']}/memories?status=active")
    assert len(r.json()) == 1


def test_filter_by_importance_min(client):
    p = make_project(client)
    make_memory(client, p["id"], importance=2)
    make_memory(client, p["id"], importance=4)
    make_memory(client, p["id"], importance=5)

    r = client.get(f"/projects/{p['id']}/memories?importance_min=4")
    assert len(r.json()) == 2
    r = client.get(f"/projects/{p['id']}/memories?importance_min=5")
    assert len(r.json()) == 1


def test_filter_by_tag(client):
    p = make_project(client)
    client.post(f"/projects/{p['id']}/memories", json={
        "type": "problem", "title": "Tagged", "content": "c", "importance": 3,
        "confidence": 1.0, "tags": ["auth", "security"], "related_tools": [], "status": "active",
    })
    client.post(f"/projects/{p['id']}/memories", json={
        "type": "task", "title": "Untagged", "content": "c", "importance": 3,
        "confidence": 1.0, "tags": ["db"], "related_tools": [], "status": "active",
    })

    r = client.get(f"/projects/{p['id']}/memories?tag=auth")
    assert len(r.json()) == 1
    assert r.json()[0]["title"] == "Tagged"


def test_get_memory(client):
    p = make_project(client)
    m = make_memory(client, p["id"])
    r = client.get(f"/memories/{m['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == m["id"]


def test_get_memory_not_found(client):
    r = client.get("/memories/nonexistent")
    assert r.status_code == 404


def test_update_memory(client):
    p = make_project(client)
    m = make_memory(client, p["id"])
    r = client.put(f"/memories/{m['id']}", json={"importance": 5, "status": "resolved"})
    assert r.status_code == 200
    data = r.json()
    assert data["importance"] == 5
    assert data["status"] == "resolved"
    assert data["title"] == m["title"]  # unchanged


def test_delete_memory(client):
    p = make_project(client)
    m = make_memory(client, p["id"])
    r = client.delete(f"/memories/{m['id']}")
    assert r.status_code == 204
    assert client.get(f"/memories/{m['id']}").status_code == 404


def test_memories_scoped_to_project(client):
    p1 = make_project(client, "P1")
    p2 = make_project(client, "P2")
    make_memory(client, p1["id"])
    make_memory(client, p2["id"])
    r = client.get(f"/projects/{p1['id']}/memories")
    assert len(r.json()) == 1
    assert r.json()[0]["project_id"] == p1["id"]


def test_memory_with_session_link(client):
    p = make_project(client)
    s = make_session(client, p["id"])
    r = client.post(f"/projects/{p['id']}/memories", json={
        "type": "insight", "title": "Insight from session", "content": "c",
        "importance": 3, "confidence": 0.8, "tags": [], "related_tools": [],
        "status": "active", "source_session_id": s["id"],
    })
    assert r.status_code == 201
    assert r.json()["source_session_id"] == s["id"]
