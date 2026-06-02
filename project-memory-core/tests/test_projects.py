from tests.conftest import make_project


def test_create_project(client):
    r = client.post("/projects", json={"name": "My Project", "tech_stack": ["Python"]})
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "My Project"
    assert data["tech_stack"] == ["Python"]
    assert "id" in data


def test_create_project_missing_name(client):
    r = client.post("/projects", json={"description": "no name"})
    assert r.status_code == 422


def test_list_projects_empty(client):
    r = client.get("/projects")
    assert r.status_code == 200
    assert r.json() == []


def test_list_projects(client):
    make_project(client, "P1")
    make_project(client, "P2")
    r = client.get("/projects")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_project(client):
    p = make_project(client)
    r = client.get(f"/projects/{p['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == p["id"]


def test_get_project_not_found(client):
    r = client.get("/projects/nonexistent-id")
    assert r.status_code == 404


def test_update_project(client):
    p = make_project(client)
    r = client.put(f"/projects/{p['id']}", json={"name": "Renamed"})
    assert r.status_code == 200
    assert r.json()["name"] == "Renamed"
    # Other fields unchanged
    assert r.json()["tech_stack"] == p["tech_stack"]


def test_update_project_tech_stack(client):
    p = make_project(client)
    r = client.put(f"/projects/{p['id']}", json={"tech_stack": ["Go", "Postgres"]})
    assert r.status_code == 200
    assert r.json()["tech_stack"] == ["Go", "Postgres"]


def test_delete_project(client):
    p = make_project(client)
    r = client.delete(f"/projects/{p['id']}")
    assert r.status_code == 204
    r = client.get(f"/projects/{p['id']}")
    assert r.status_code == 404


def test_delete_project_not_found(client):
    r = client.delete("/projects/bad-id")
    assert r.status_code == 404


def test_delete_project_cascades(client):
    p = make_project(client)
    # Add a session and memory
    client.post(f"/projects/{p['id']}/sessions", json={
        "tool_name": "Claude", "title": "S", "raw_content": "raw"
    })
    client.post(f"/projects/{p['id']}/memories", json={
        "type": "problem", "title": "B", "content": "c", "importance": 3,
        "confidence": 1.0, "tags": [], "related_tools": [], "status": "active"
    })
    client.delete(f"/projects/{p['id']}")
    # Verify project is gone
    assert client.get(f"/projects/{p['id']}").status_code == 404
