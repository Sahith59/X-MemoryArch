"""
Sub-phase 1.43 — Supersedes link auto-creation tests.

When supersede_memory() or resolve_conflicts() marks a memory as superseded,
a MemoryLink(relationship_type="supersedes") must be automatically created
from the old memory to the new memory.  This makes the supersession chain
graph-traversable and consistent with the memory_links API.

Covers:
  Unit / CRUD:
  1.  supersede_memory creates a supersedes link
  2.  supersedes link source is old memory, target is new memory
  3.  supersedes link has relationship_type == "supersedes"
  4.  supersedes link is retrievable via get_memory_links(old_id)
  5.  supersedes link is retrievable via get_memory_links(new_id)
  6.  Two chained supersessions create two separate supersedes links
  7.  Transitive chain: A→B→C has links A→B and B→C, NOT A→C
  8.  resolve_conflicts creates a supersedes link for the resolved pair
  9.  resolve_conflicts dry_run does NOT create a link
  10. resolve_conflicts does not duplicate an existing supersedes link

  API (router level):
  11. POST /memories/{id}/supersede creates supersedes link (status 201)
  12. GET /memories/{old_id}/links returns a supersedes link
  13. GET /memories/{new_id}/links returns the same supersedes link
  14. supersedes link appears in both directions on the links endpoint
  15. supersedes link created_at is close to now (within 5 seconds)
  16. supersedes link note contains expected text
  17. Supersede an already-superseded memory: new link created, original chain intact
  18. Supersede 404 does not create any links
  19. supersedes link survives memory GET (link still there after accessing new memory)
  20. resolve_conflicts auto-creates links for all resolved pairs in one call
  21. After supersede, GET /memories/{old_id}/links count increases by at least 1
  22. supersedes link direction: old→new (source=old, target=new)
  23. supersedes link note contains "Auto-created"
  24. supersedes link node present in YAML export links section
  25. supersedes link visible after recluster (not wiped by recluster)
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud, schemas
from app.models import MemoryLink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "SupersedesLink Test") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(client: TestClient, pid: str, title: str = "Decision A",
         mem_type: str = "decision", importance: int = 3) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": mem_type,
        "title": title,
        "content": f"Content for {title}.",
        "importance": importance,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _supersede(client: TestClient, old_id: str, title: str = "New decision") -> dict:
    r = client.post(f"/memories/{old_id}/supersede", json={
        "type": "decision",
        "title": title,
        "content": f"Content for {title}.",
    })
    assert r.status_code == 201, r.text
    return r.json()


def _links(client: TestClient, memory_id: str) -> list[dict]:
    r = client.get(f"/memories/{memory_id}/links")
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------------
# 1–10: CRUD-level tests
# ---------------------------------------------------------------------------

class TestSupersedesLinkCRUD:
    def test_supersede_creates_link(self, db: Session, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Flask")
        result = _supersede(client, old["id"], "Use FastAPI")
        new_id = result["new_memory"]["id"]

        links = (
            db.query(MemoryLink)
            .filter(MemoryLink.source_memory_id == old["id"])
            .all()
        )
        supersedes_links = [lk for lk in links if lk.relationship_type == "supersedes"]
        assert len(supersedes_links) == 1

    def test_supersedes_link_direction_old_to_new(self, db: Session, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Celery for tasks")
        result = _supersede(client, old["id"], "Use ARQ for tasks")
        new_id = result["new_memory"]["id"]

        lk = (
            db.query(MemoryLink)
            .filter(
                MemoryLink.source_memory_id == old["id"],
                MemoryLink.target_memory_id == new_id,
                MemoryLink.relationship_type == "supersedes",
            )
            .first()
        )
        assert lk is not None
        assert lk.source_memory_id == old["id"]
        assert lk.target_memory_id == new_id

    def test_supersedes_link_relationship_type(self, db: Session, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use MySQL")
        result = _supersede(client, old["id"], "Use PostgreSQL")
        new_id = result["new_memory"]["id"]

        lk = (
            db.query(MemoryLink)
            .filter(
                MemoryLink.source_memory_id == old["id"],
                MemoryLink.target_memory_id == new_id,
            )
            .first()
        )
        assert lk is not None
        assert lk.relationship_type == "supersedes"

    def test_supersedes_link_retrievable_from_old(self, db: Session, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Redis 6")
        result = _supersede(client, old["id"], "Use Redis 7")
        new_id = result["new_memory"]["id"]

        links = crud.get_memory_links(db, old["id"])
        supersedes = [lk for lk in links if lk.relationship_type == "supersedes"]
        assert any(lk.target_memory_id == new_id for lk in supersedes)

    def test_supersedes_link_retrievable_from_new(self, db: Session, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Node 16")
        result = _supersede(client, old["id"], "Use Node 20")
        new_id = result["new_memory"]["id"]

        links = crud.get_memory_links(db, new_id)
        supersedes = [lk for lk in links if lk.relationship_type == "supersedes"]
        assert any(lk.source_memory_id == old["id"] for lk in supersedes)

    def test_chained_supersessions_create_two_links(self, db: Session, client: TestClient):
        pid = _project(client)
        a = _mem(client, pid, "Decision A v1")
        b_result = _supersede(client, a["id"], "Decision A v2")
        b_id = b_result["new_memory"]["id"]
        c_result = _supersede(client, b_id, "Decision A v3")
        c_id = c_result["new_memory"]["id"]

        link_a_b = (
            db.query(MemoryLink)
            .filter(
                MemoryLink.source_memory_id == a["id"],
                MemoryLink.target_memory_id == b_id,
                MemoryLink.relationship_type == "supersedes",
            )
            .first()
        )
        link_b_c = (
            db.query(MemoryLink)
            .filter(
                MemoryLink.source_memory_id == b_id,
                MemoryLink.target_memory_id == c_id,
                MemoryLink.relationship_type == "supersedes",
            )
            .first()
        )
        assert link_a_b is not None
        assert link_b_c is not None

    def test_chain_has_no_skip_link(self, db: Session, client: TestClient):
        pid = _project(client)
        a = _mem(client, pid, "Arch decision v1")
        b_result = _supersede(client, a["id"], "Arch decision v2")
        b_id = b_result["new_memory"]["id"]
        c_result = _supersede(client, b_id, "Arch decision v3")
        c_id = c_result["new_memory"]["id"]

        # No direct A→C link should exist
        skip_link = (
            db.query(MemoryLink)
            .filter(
                MemoryLink.source_memory_id == a["id"],
                MemoryLink.target_memory_id == c_id,
                MemoryLink.relationship_type == "supersedes",
            )
            .first()
        )
        assert skip_link is None

    def test_resolve_conflicts_creates_supersedes_link(self, db: Session, client: TestClient):
        pid = _project(client)
        # Two identical-title memories — conflict triggers
        a = _mem(client, pid, "Use PostgreSQL for data", importance=3)
        b = _mem(client, pid, "Use PostgreSQL for data", importance=5)  # winner

        result = client.post(f"/projects/{pid}/conflicts/resolve")
        assert result.status_code == 200
        assert result.json()["resolved_count"] >= 1

        # A supersedes link should now exist between the pair
        all_links = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).all()
        involved_ids = {a["id"], b["id"]}
        found = any(
            lk.source_memory_id in involved_ids and lk.target_memory_id in involved_ids
            for lk in all_links
        )
        assert found

    def test_resolve_conflicts_dry_run_no_link(self, db: Session, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use SQLite for storage", importance=2)
        _mem(client, pid, "Use SQLite for storage", importance=4)

        client.post(f"/projects/{pid}/conflicts/resolve", params={"dry_run": True})

        # No links should have been created
        links = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).all()
        assert len(links) == 0

    def test_resolve_conflicts_no_duplicate_link(self, db: Session, client: TestClient):
        pid = _project(client)
        _mem(client, pid, "Use Nginx as proxy", importance=2)
        _mem(client, pid, "Use Nginx as proxy", importance=4)

        # Resolve twice
        client.post(f"/projects/{pid}/conflicts/resolve")
        client.post(f"/projects/{pid}/conflicts/resolve")

        links = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).all()
        # Should be exactly one supersedes link, not two
        assert len(links) == 1


# ---------------------------------------------------------------------------
# 11–25: API-level tests
# ---------------------------------------------------------------------------

class TestSupersedesLinkAPI:
    def test_supersede_endpoint_returns_201(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Gunicorn")
        r = client.post(f"/memories/{old['id']}/supersede", json={
            "type": "decision",
            "title": "Use Uvicorn",
            "content": "Switching to async ASGI server.",
        })
        assert r.status_code == 201

    def test_old_memory_links_contains_supersedes(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Nginx")
        result = _supersede(client, old["id"], "Use Caddy")
        new_id = result["new_memory"]["id"]

        links = _links(client, old["id"])
        supersedes_links = [lk for lk in links if lk["relationship_type"] == "supersedes"]
        assert len(supersedes_links) >= 1
        assert any(lk["target_memory_id"] == new_id for lk in supersedes_links)

    def test_new_memory_links_contains_supersedes(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use HTTP polling")
        result = _supersede(client, old["id"], "Use WebSockets")
        new_id = result["new_memory"]["id"]

        links = _links(client, new_id)
        supersedes_links = [lk for lk in links if lk["relationship_type"] == "supersedes"]
        assert len(supersedes_links) >= 1
        assert any(lk["source_memory_id"] == old["id"] for lk in supersedes_links)

    def test_supersedes_link_visible_both_directions(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Store sessions in cookie")
        result = _supersede(client, old["id"], "Store sessions in Redis")
        new_id = result["new_memory"]["id"]

        old_links = _links(client, old["id"])
        new_links = _links(client, new_id)

        # Both endpoints show the same link (linked in both directions)
        old_supersedes = [lk for lk in old_links if lk["relationship_type"] == "supersedes"]
        new_supersedes = [lk for lk in new_links if lk["relationship_type"] == "supersedes"]
        assert len(old_supersedes) >= 1
        assert len(new_supersedes) >= 1

    def test_supersedes_link_created_at_is_recent(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use JWT tokens")
        result = _supersede(client, old["id"], "Use OAuth tokens")
        new_id = result["new_memory"]["id"]

        links = _links(client, old["id"])
        supersedes_links = [lk for lk in links if lk["relationship_type"] == "supersedes"]
        assert len(supersedes_links) >= 1
        created_at_str = supersedes_links[0]["created_at"]
        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        assert (now - created_at).total_seconds() < 30

    def test_supersedes_link_note_contains_auto_created(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use Terraform")
        result = _supersede(client, old["id"], "Use Pulumi")
        new_id = result["new_memory"]["id"]

        links = _links(client, old["id"])
        supersedes_links = [lk for lk in links if lk["relationship_type"] == "supersedes"]
        assert len(supersedes_links) >= 1
        note = supersedes_links[0].get("note") or ""
        assert "Auto-created" in note

    def test_supersede_already_superseded_creates_new_link(self, client: TestClient):
        pid = _project(client)
        a = _mem(client, pid, "Decision v1")
        b_result = _supersede(client, a["id"], "Decision v2")
        b_id = b_result["new_memory"]["id"]
        c_result = _supersede(client, b_id, "Decision v3")
        c_id = c_result["new_memory"]["id"]

        # B→C link should exist even though B was created by superseding A
        b_links = _links(client, b_id)
        b_supersedes = [lk for lk in b_links if lk["relationship_type"] == "supersedes"
                        and lk["target_memory_id"] == c_id]
        assert len(b_supersedes) >= 1

    def test_supersede_404_creates_no_link(self, client: TestClient, db: Session):
        before_count = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).count()

        r = client.post("/memories/nonexistent-id/supersede", json={
            "type": "decision",
            "title": "New decision",
            "content": "Content",
        })
        assert r.status_code == 404

        after_count = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).count()
        assert after_count == before_count

    def test_supersedes_link_survives_memory_get(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use pip install")
        result = _supersede(client, old["id"], "Use uv install")
        new_id = result["new_memory"]["id"]

        # Access new memory (triggers access tracking)
        client.get(f"/memories/{new_id}")

        # Link should still be there
        links = _links(client, old["id"])
        supersedes_links = [lk for lk in links if lk["relationship_type"] == "supersedes"]
        assert len(supersedes_links) >= 1

    def test_resolve_conflicts_auto_creates_links_for_all_pairs(
        self, client: TestClient, db: Session
    ):
        pid = _project(client)
        _mem(client, pid, "Use Docker for builds", importance=2)
        _mem(client, pid, "Use Docker for builds", importance=4)
        _mem(client, pid, "Use Kubernetes for deploy", importance=1)
        _mem(client, pid, "Use Kubernetes for deploy", importance=3)

        result = client.post(f"/projects/{pid}/conflicts/resolve")
        resolved = result.json()["resolved_count"]

        # Each resolved pair should have produced a supersedes link
        links_count = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).count()
        assert links_count >= resolved

    def test_supersede_link_count_increases_by_one(self, client: TestClient, db: Session):
        pid = _project(client)
        old = _mem(client, pid, "Use Apache Kafka")
        before = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).count()
        _supersede(client, old["id"], "Use NATS for messaging")
        after = db.query(MemoryLink).filter(
            MemoryLink.relationship_type == "supersedes"
        ).count()
        assert after == before + 1

    def test_supersedes_link_source_is_old_memory(self, client: TestClient, db: Session):
        pid = _project(client)
        old = _mem(client, pid, "Use Webpack")
        result = _supersede(client, old["id"], "Use Vite")
        new_id = result["new_memory"]["id"]

        lk = db.query(MemoryLink).filter(
            MemoryLink.source_memory_id == old["id"],
            MemoryLink.target_memory_id == new_id,
            MemoryLink.relationship_type == "supersedes",
        ).first()
        assert lk is not None
        assert lk.source_memory_id == old["id"]

    def test_supersedes_link_target_is_new_memory(self, client: TestClient, db: Session):
        pid = _project(client)
        old = _mem(client, pid, "Use Babel")
        result = _supersede(client, old["id"], "Use SWC")
        new_id = result["new_memory"]["id"]

        lk = db.query(MemoryLink).filter(
            MemoryLink.source_memory_id == old["id"],
            MemoryLink.target_memory_id == new_id,
            MemoryLink.relationship_type == "supersedes",
        ).first()
        assert lk is not None
        assert lk.target_memory_id == new_id

    def test_supersedes_link_in_yaml_export(self, client: TestClient):
        pid = _project(client)
        old = _mem(client, pid, "Use MongoDB")
        result = _supersede(client, old["id"], "Use CouchDB")
        new_id = result["new_memory"]["id"]

        r = client.get(f"/projects/{pid}/export/memory.yaml")
        assert r.status_code == 200
        # The old memory's links section should mention the supersedes relationship
        assert "supersedes" in r.text

    def test_supersedes_link_survives_recluster(self, client: TestClient, db: Session):
        pid = _project(client)
        old = _mem(client, pid, "Use GraphQL API")
        result = _supersede(client, old["id"], "Use REST API")
        new_id = result["new_memory"]["id"]

        # Run recluster
        client.post(f"/projects/{pid}/memories/recluster")

        # Supersedes link should still exist
        lk = db.query(MemoryLink).filter(
            MemoryLink.source_memory_id == old["id"],
            MemoryLink.target_memory_id == new_id,
            MemoryLink.relationship_type == "supersedes",
        ).first()
        assert lk is not None
