"""Sub-phase 1.12 — Hybrid FTS5 Search tests."""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient) -> str:
    r = client.post("/projects", json={"name": "Search Test Project"})
    assert r.status_code == 201
    return r.json()["id"]


def _mem(client, pid, *, title, content, mem_type="decision", importance=3, tags=None, file_path=None):
    payload = {
        "type": mem_type,
        "title": title,
        "content": content,
        "importance": importance,
        "tags": tags or [],
    }
    if file_path:
        payload["file_path"] = file_path
    r = client.post(f"/projects/{pid}/memories", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _search(client, pid, q, **params):
    return client.get(f"/projects/{pid}/memories/search", params={"q": q, **params})


# ---------------------------------------------------------------------------
# Test 1: Basic match — exact term in title
# ---------------------------------------------------------------------------

class TestBasicSearch:
    def test_exact_term_in_title(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Use PostgreSQL for main store", content="We chose Postgres.")
        _mem(client, pid, title="Redis for caching layer", content="Redis handles cache.")

        r = _search(client, pid, "PostgreSQL")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1
        assert any("PostgreSQL" in res["title"] for res in results)

    def test_exact_term_in_content(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Auth decision", content="We decided to use JWT tokens for auth.")
        _mem(client, pid, title="Unrelated memory", content="This is about something else entirely.")

        r = _search(client, pid, "JWT")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1
        assert any("JWT" in res["content"] for res in results)

    def test_term_in_tags(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Performance tuning",
             content="Optimized database queries.",
             tags=["sqlalchemy", "performance"])

        r = _search(client, pid, "sqlalchemy")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1

    def test_term_in_file_path(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Bug in auth router",
             content="Null pointer in the auth module.",
             mem_type="problem",
             file_path="app/routers/auth.py")

        r = _search(client, pid, "auth")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1

    def test_no_match_returns_empty(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="SQLite decision", content="Using SQLite for local storage.")

        r = _search(client, pid, "zephyr")
        assert r.status_code == 200
        assert r.json() == []

    def test_empty_query_rejected(self, client: TestClient):
        pid = _project(client)
        r = _search(client, pid, "")
        assert r.status_code == 422

    def test_project_not_found(self, client: TestClient):
        r = _search(client, "nonexistent-id", "anything")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 2: Prefix matching
# ---------------------------------------------------------------------------

class TestPrefixSearch:
    def test_prefix_match(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Authentication middleware", content="Handles auth flow.")

        # "authen" should prefix-match "authentication" and "auth"
        r = _search(client, pid, "authen")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1

    def test_multi_token_and_match(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="SQLite migration strategy",
             content="Using Alembic for SQLite migrations.")
        _mem(client, pid, title="SQLite basics",
             content="SQLite is a local database.")
        _mem(client, pid, title="Migration guide",
             content="General migration approach using Alembic.")

        # Both "sqlite" AND "migration" must appear — only first memory matches
        r = _search(client, pid, "sqlite migration")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1
        titles = [res["title"] for res in results]
        assert "SQLite migration strategy" in titles


# ---------------------------------------------------------------------------
# Test 3: Relevance score is present in response
# ---------------------------------------------------------------------------

class TestRelevanceScore:
    def test_relevance_score_in_response(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="FastAPI routing decision",
             content="We chose FastAPI for its speed and type safety.")

        r = _search(client, pid, "FastAPI")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1
        assert "relevance_score" in results[0]
        assert isinstance(results[0]["relevance_score"], float)
        assert results[0]["relevance_score"] > 0

    def test_results_sorted_by_relevance_desc(self, client: TestClient):
        pid = _project(client)
        # High importance decision mentioning "database" multiple times in title+content
        _mem(client, pid, title="Database architecture decision",
             content="The database choice is critical. We evaluated database options carefully.",
             mem_type="decision",
             importance=5)
        # Low importance note with one mention
        _mem(client, pid, title="Quick database note",
             content="Noted: database might need tuning later.",
             mem_type="conversation_note",
             importance=1)

        r = _search(client, pid, "database")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 2
        scores = [res["relevance_score"] for res in results]
        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)
        # High importance decision should rank first
        assert results[0]["importance"] == 5


# ---------------------------------------------------------------------------
# Test 4: Type weight — decisions/bugs rank higher than notes for same term
# ---------------------------------------------------------------------------

class TestTypeWeighting:
    def test_decision_outranks_conversation_note(self, client: TestClient):
        pid = _project(client)
        # Same keyword "caching" — decision should rank higher than note
        _mem(client, pid, title="Caching strategy decision",
             content="We decided on Redis for caching all API responses.",
             mem_type="decision",
             importance=3)
        _mem(client, pid, title="Caching mentioned",
             content="We briefly talked about caching as an option.",
             mem_type="conversation_note",
             importance=3)

        r = _search(client, pid, "caching")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 2
        # Decision (type_weight=1.4) should outrank conversation_note (type_weight=1.0)
        assert results[0]["type"] == "decision"

    def test_bug_outranks_insight(self, client: TestClient):
        pid = _project(client)
        # Bug has "timeout" in title + three times in content → higher BM25 + type_weight=1.3
        _mem(client, pid, title="Timeout bug in API",
             content="The timeout error occurs repeatedly. Every request hits a timeout. Timeout is critical.",
             mem_type="problem",
             importance=3)
        # Insight has "timeout" only in title → lower BM25, type_weight=1.0
        _mem(client, pid, title="Timeout is interesting",
             content="Worth noting for future analysis.",
             mem_type="insight",
             importance=3)

        r = _search(client, pid, "timeout")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 2
        assert results[0]["type"] == "problem"


# ---------------------------------------------------------------------------
# Test 5: Filters — type, status, importance_min combined with search
# ---------------------------------------------------------------------------

class TestSearchFilters:
    def test_filter_by_type(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Memory leak bug",
             content="Detected a memory leak in the session handler.",
             mem_type="problem")
        _mem(client, pid, title="Memory architecture decision",
             content="We designed the memory layer as local-first.",
             mem_type="decision")

        r = _search(client, pid, "memory", type="problem")
        assert r.status_code == 200
        results = r.json()
        assert all(res["type"] == "problem" for res in results)

    def test_filter_by_status(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Active database decision",
             content="SQLite is our database choice and it is active.")
        # Create resolved memory
        r = client.post(f"/projects/{pid}/memories", json={
            "type": "decision",
            "title": "Resolved database question",
            "content": "Old decision about database now resolved.",
            "status": "resolved",
        })
        assert r.status_code == 201

        r = _search(client, pid, "database", status="resolved")
        assert r.status_code == 200
        results = r.json()
        assert all(res["status"] == "resolved" for res in results)

    def test_filter_by_importance_min(self, client: TestClient):
        pid = _project(client)
        _mem(client, pid, title="Critical auth decision",
             content="Authentication is the most critical component.",
             importance=5)
        _mem(client, pid, title="Minor auth note",
             content="Authentication was briefly mentioned.",
             importance=1)

        r = _search(client, pid, "auth", importance_min=4)
        assert r.status_code == 200
        results = r.json()
        assert all(res["importance"] >= 4 for res in results)

    def test_limit_respected(self, client: TestClient):
        pid = _project(client)
        for i in range(10):
            _mem(client, pid, title=f"Deployment step {i}",
                 content="Deploying to production requires careful steps.")

        r = _search(client, pid, "deploy", limit=3)
        assert r.status_code == 200
        assert len(r.json()) <= 3


# ---------------------------------------------------------------------------
# Test 6: FTS5 trigger sync — update and delete kept in sync
# ---------------------------------------------------------------------------

class TestFTSTriggers:
    def test_updated_memory_is_searchable_by_new_content(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, title="Initial title", content="Original content about nothing.")

        # Before update — search for "kubernetes" should return nothing
        r = _search(client, pid, "kubernetes")
        assert r.json() == []

        # Update the memory to mention kubernetes
        client.put(f"/memories/{mem['id']}", json={
            "content": "We decided to use Kubernetes for container orchestration."
        })

        # Now it should be found
        r = _search(client, pid, "kubernetes")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_deleted_memory_no_longer_searchable(self, client: TestClient):
        pid = _project(client)
        mem = _mem(client, pid, title="Temporary microservices note",
                   content="We considered using microservices architecture.")

        r = _search(client, pid, "microservices")
        assert len(r.json()) >= 1

        client.delete(f"/memories/{mem['id']}")

        r = _search(client, pid, "microservices")
        assert r.json() == []

    def test_project_isolation(self, client: TestClient):
        """Search should not return memories from a different project."""
        pid1 = _project(client)
        pid2 = (lambda r: r.json()["id"])(
            client.post("/projects", json={"name": "Other Project"})
        )
        _mem(client, pid1, title="GraphQL decision", content="Using GraphQL for the API.")
        _mem(client, pid2, title="GraphQL in other project",
             content="GraphQL is also used here.")

        r = _search(client, pid1, "GraphQL")
        assert r.status_code == 200
        for res in r.json():
            assert res["project_id"] == pid1
