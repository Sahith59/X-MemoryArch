"""Sub-phase 1.11 — Memory Relationships (Graph-Lite) tests."""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient) -> str:
    r = client.post("/projects", json={"name": "Link Test Project"})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, project_id: str, title: str = "A memory", mem_type: str = "decision") -> str:
    r = client.post(f"/projects/{project_id}/memories", json={
        "type": mem_type,
        "title": title,
        "content": f"Content of: {title}",
    })
    assert r.status_code == 201
    return r.json()["id"]


def _link(client: TestClient, source_id: str, target_id: str, rel: str = "related_to", note: str | None = None) -> dict:
    payload = {"target_memory_id": target_id, "relationship_type": rel}
    if note:
        payload["note"] = note
    r = client.post(f"/memories/{source_id}/links", json=payload)
    assert r.status_code == 201
    return r.json()


# ---------------------------------------------------------------------------
# Test 1: Create a link between two memories
# ---------------------------------------------------------------------------

class TestCreateLink:
    def test_create_link_basic(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Decision A")
        b = _memory(client, pid, "Bug B", mem_type="problem")

        lnk = _link(client, a, b, rel="influenced_by")

        assert lnk["source_memory_id"] == a
        assert lnk["target_memory_id"] == b
        assert lnk["relationship_type"] == "influenced_by"
        assert lnk["note"] is None
        assert "id" in lnk
        assert "created_at" in lnk

    def test_create_link_with_note(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Architecture choice")
        b = _memory(client, pid, "Bug caused by it", mem_type="problem")

        lnk = _link(client, a, b, rel="derived_from", note="This bug came from the architecture decision.")
        assert lnk["note"] == "This bug came from the architecture decision."

    def test_all_relationship_types_accepted(self, client: TestClient):
        pid = _project(client)
        rel_types = ["supersedes", "related_to", "conflicts_with", "influenced_by", "derived_from", "blocks", "resolves"]
        memories = [_memory(client, pid, f"Memory {i}") for i in range(len(rel_types) + 1)]

        for i, rel in enumerate(rel_types):
            lnk = _link(client, memories[i], memories[i + 1], rel=rel)
            assert lnk["relationship_type"] == rel

    def test_invalid_relationship_type_rejected(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Memory A")
        b = _memory(client, pid, "Memory B")

        r = client.post(f"/memories/{a}/links", json={
            "target_memory_id": b,
            "relationship_type": "invented_type",
        })
        assert r.status_code == 422

    def test_self_link_rejected(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Self-referencing memory")

        r = client.post(f"/memories/{a}/links", json={
            "target_memory_id": a,
            "relationship_type": "related_to",
        })
        assert r.status_code == 400

    def test_duplicate_link_rejected(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Memory A")
        b = _memory(client, pid, "Memory B")
        _link(client, a, b, rel="related_to")

        r = client.post(f"/memories/{a}/links", json={
            "target_memory_id": b,
            "relationship_type": "related_to",
        })
        assert r.status_code == 409

    def test_same_pair_different_types_allowed(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Memory A")
        b = _memory(client, pid, "Memory B")
        _link(client, a, b, rel="related_to")
        lnk2 = _link(client, a, b, rel="influences_by" if False else "influenced_by")
        assert lnk2["relationship_type"] == "influenced_by"

    def test_source_not_found(self, client: TestClient):
        pid = _project(client)
        b = _memory(client, pid, "Memory B")
        r = client.post(f"/memories/nonexistent-id/links", json={
            "target_memory_id": b,
            "relationship_type": "related_to",
        })
        assert r.status_code == 404

    def test_target_not_found(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Memory A")
        r = client.post(f"/memories/{a}/links", json={
            "target_memory_id": "nonexistent-id",
            "relationship_type": "related_to",
        })
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 2: Get links for a memory (bidirectional)
# ---------------------------------------------------------------------------

class TestGetLinks:
    def test_list_links_as_source(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Decision A")
        b = _memory(client, pid, "Bug B", mem_type="problem")
        c = _memory(client, pid, "Task C", mem_type="task")
        _link(client, a, b, rel="blocks")
        _link(client, a, c, rel="related_to")

        r = client.get(f"/memories/{a}/links")
        assert r.status_code == 200
        links = r.json()
        assert len(links) == 2
        rel_types = {lnk["relationship_type"] for lnk in links}
        assert rel_types == {"blocks", "related_to"}

    def test_list_links_bidirectional(self, client: TestClient):
        """A memory should see links whether it is source or target."""
        pid = _project(client)
        a = _memory(client, pid, "Memory A")
        b = _memory(client, pid, "Memory B")
        _link(client, a, b, rel="resolves")

        # b is the target — should still see the link
        r = client.get(f"/memories/{b}/links")
        assert r.status_code == 200
        links = r.json()
        assert len(links) == 1
        assert links[0]["source_memory_id"] == a
        assert links[0]["target_memory_id"] == b

    def test_list_links_empty(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Isolated memory")
        r = client.get(f"/memories/{a}/links")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_links_memory_not_found(self, client: TestClient):
        r = client.get("/memories/nonexistent-id/links")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 3: Delete a link
# ---------------------------------------------------------------------------

class TestDeleteLink:
    def test_delete_link(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Memory A")
        b = _memory(client, pid, "Memory B")
        lnk = _link(client, a, b, rel="conflicts_with")

        r = client.delete(f"/memory-links/{lnk['id']}")
        assert r.status_code == 204

        # Link should be gone
        links = client.get(f"/memories/{a}/links").json()
        assert links == []

    def test_delete_nonexistent_link(self, client: TestClient):
        r = client.delete("/memory-links/nonexistent-id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 4: Export includes relationships
# ---------------------------------------------------------------------------

class TestExportWithLinks:
    def test_export_shows_relationship(self, client: TestClient):
        pid = _project(client)
        a = _memory(client, pid, "Use FastAPI")
        b = _memory(client, pid, "Performance issue found", mem_type="problem")
        _link(client, b, a, rel="derived_from")

        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200
        content = r.text
        # The relationship should appear in the export
        assert "derived from" in content
        assert "Use FastAPI" in content or "Performance issue found" in content

    def test_export_no_links_no_relationship_line(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, "Standalone decision")
        r = client.get(f"/projects/{pid}/export/context.md")
        assert r.status_code == 200
        assert "Relationships:" not in r.text


# ---------------------------------------------------------------------------
# Test 5: Multiple link chains (graph traversal via list endpoint)
# ---------------------------------------------------------------------------

class TestLinkChains:
    def test_chain_of_three_memories(self, client: TestClient):
        """A → influenced_by → B → resolves → C"""
        pid = _project(client)
        a = _memory(client, pid, "Architecture Decision")
        b = _memory(client, pid, "API Design", mem_type="structure")
        c = _memory(client, pid, "API Bug Fixed", mem_type="problem")

        _link(client, a, b, rel="influenced_by")
        _link(client, b, c, rel="resolves")

        a_links = client.get(f"/memories/{a}/links").json()
        b_links = client.get(f"/memories/{b}/links").json()
        c_links = client.get(f"/memories/{c}/links").json()

        assert len(a_links) == 1
        assert len(b_links) == 2  # source in one, target in another
        assert len(c_links) == 1

    def test_many_links_on_one_memory(self, client: TestClient):
        pid = _project(client)
        hub = _memory(client, pid, "Core Architecture Decision")
        others = [_memory(client, pid, f"Memory {i}") for i in range(5)]

        for other_id in others:
            _link(client, hub, other_id, rel="related_to")
            # Different type going the other direction
            _link(client, other_id, hub, rel="influenced_by")

        hub_links = client.get(f"/memories/{hub}/links").json()
        # 5 outgoing related_to + 5 incoming influenced_by = 10 total
        assert len(hub_links) == 10
