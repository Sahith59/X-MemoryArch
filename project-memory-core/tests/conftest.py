import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.search import setup_fts

# StaticPool ensures all connections share the same in-memory DB instance.
# Without it each new connection would get a blank database.
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function", autouse=False)
def db():
    Base.metadata.create_all(bind=engine)
    setup_fts(engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop FTS5 virtual table first (not in Base.metadata, not cleaned by drop_all).
        # Without this, ghost BM25 entries from the previous test pollute scoring
        # for the next test since StaticPool shares one in-memory DB across all tests.
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS memories_fts"))
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Reusable data factories
# ---------------------------------------------------------------------------

def make_project(client, name="Test Project", tech_stack=None, goals=None):
    payload = {
        "name": name,
        "description": "A test project",
        "tech_stack": tech_stack or ["Python", "FastAPI"],
        "goals": goals or ["Ship Phase 1"],
    }
    r = client.post("/projects", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def make_session(client, project_id, tool="Claude", title="Test Session"):
    payload = {
        "tool_name": tool,
        "title": title,
        "raw_content": "We discussed the decision to use SQLite. There is a bug in the auth flow. TODO: write tests.",
        "session_date": "2026-05-24",
    }
    r = client.post(f"/projects/{project_id}/sessions", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def make_memory(client, project_id, type="decision", title="Use SQLite", importance=4):
    payload = {
        "type": type,
        "title": title,
        "content": f"Content for {title}",
        "importance": importance,
        "confidence": 0.9,
        "tags": ["test", type],
        "related_tools": ["Claude"],
        "status": "active",
    }
    r = client.post(f"/projects/{project_id}/memories", json=payload)
    assert r.status_code == 201, r.text
    return r.json()
