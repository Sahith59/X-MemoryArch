"""
Sub-phase 1.27 — DBSCAN clustering tests.

Covers:
  Unit tests — compute_clusters pure function:
  1.  Empty input → empty result
  2.  Single embedded memory → noise (-1)
  3.  Two identical embeddings → same cluster (cluster_id >= 0)
  4.  Two orthogonal embeddings → both noise (cosine distance = 1.0 > eps)
  5.  Three similar embeddings → all in same cluster
  6.  Mixed: one cluster + one outlier → cluster_id 0 for group, -1 for outlier
  7.  cluster_label is non-empty string for real clusters
  8.  Noise memories have cluster_label=None
  9.  Memories without embeddings are excluded from result
  10. cluster_id values are integers

  Integration tests — API:
  11. cluster_id field present in MemoryResponse
  12. cluster_label field present in MemoryResponse
  13. New memory has cluster_id=None
  14. New memory has cluster_label=None
  15. POST /projects/{id}/memories/recluster returns 200
  16. recluster returns ClusterResult with correct fields
  17. recluster 404 for unknown project
  18. recluster with empty project → all counts zero, clusters=[]
  19. recluster with only non-embedded memories → total_unclustered == total
  20. GET /projects/{id}/clusters returns list
  21. GET /projects/{id}/clusters 404 for unknown project
  22. GET /projects/{id}/clusters with no clusters → empty list
  23. cluster_id present in list endpoint response
  24. cluster_id present in search results
  25. total_clustered + total_noise + total_unclustered == total memories
  26. Extraction auto-triggers recluster (embedded memories have cluster_id set)
  27. Recluster is idempotent (calling twice gives same counts)
  28. ClusterInfo has correct memory_count
  29. Memory with cluster_id >= 0 appears in GET /clusters response
  30. After recluster, cluster_label appears in memory response
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.clustering import ClusterAssignment, compute_clusters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(component: int, ndim: int = 384) -> bytes:
    """Unit vector pointing in the given component direction."""
    v = np.zeros(ndim, dtype=np.float32)
    v[component] = 1.0
    return v.tobytes()


def _near_vec(component: int, noise: float = 0.01, ndim: int = 384) -> bytes:
    """Near-unit vector in component direction with tiny noise — still clusters with base."""
    v = np.zeros(ndim, dtype=np.float32)
    v[component] = 1.0
    v[(component + 1) % ndim] = noise
    norm = np.linalg.norm(v)
    v = v / norm
    return v.tobytes()


class _FakeMem:
    """Minimal stand-in for a Memory ORM object — only needs id, title, embedding."""
    def __init__(self, id: str, title: str, embedding: bytes | None = None):
        self.id = id
        self.title = title
        self.embedding = embedding


def _project(client: TestClient, name: str = "Cluster Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _memory(client: TestClient, pid: str, title: str = "A memory", **kwargs) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": "decision",
        "title": title,
        "content": f"Content for {title}.",
        **kwargs,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _session_extract(client: TestClient, pid: str, content: str) -> list[dict]:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Cluster test session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    sid = r.json()["id"]
    return client.post(f"/sessions/{sid}/extract-memories").json()["memories"]


# ---------------------------------------------------------------------------
# 1–10: Unit tests on compute_clusters pure function
# ---------------------------------------------------------------------------

class TestComputeClustersUnit:
    def test_empty_input_returns_empty(self):
        result = compute_clusters([])
        assert result == []

    def test_single_embedded_memory_is_noise(self):
        mems = [_FakeMem("a", "Title A", _unit_vec(0))]
        result = compute_clusters(mems)
        assert len(result) == 1
        assert result[0].cluster_id == -1

    def test_two_identical_embeddings_form_cluster(self):
        vec = _unit_vec(0)
        mems = [
            _FakeMem("a", "Database choice", vec),
            _FakeMem("b", "Database decision", vec),
        ]
        result = compute_clusters(mems)
        assert len(result) == 2
        assert result[0].cluster_id == result[1].cluster_id
        assert result[0].cluster_id >= 0

    def test_orthogonal_embeddings_are_noise(self):
        # cosine distance between orthogonal unit vectors = 1.0 > eps (0.40)
        mems = [
            _FakeMem("a", "Alpha topic", _unit_vec(0)),
            _FakeMem("b", "Beta topic", _unit_vec(1)),
        ]
        result = compute_clusters(mems)
        assert all(r.cluster_id == -1 for r in result)

    def test_three_similar_embeddings_same_cluster(self):
        mems = [
            _FakeMem("a", "Cache decision", _unit_vec(0)),
            _FakeMem("b", "Cache choice", _near_vec(0, noise=0.01)),
            _FakeMem("c", "Cache approach", _near_vec(0, noise=0.005)),
        ]
        result = compute_clusters(mems)
        ids = {r.cluster_id for r in result}
        # All three should be in the same cluster
        assert len(ids) == 1
        assert list(ids)[0] >= 0

    def test_mixed_cluster_and_outlier(self):
        # Two near-identical vecs (cluster) + one orthogonal (noise)
        mems = [
            _FakeMem("a", "Database storage", _unit_vec(0)),
            _FakeMem("b", "Database backend", _near_vec(0, noise=0.01)),
            _FakeMem("c", "Completely different", _unit_vec(1)),
        ]
        result = compute_clusters(mems)
        cluster_ids = {r.cluster_id for r in result}
        assert -1 in cluster_ids   # outlier present
        assert any(r.cluster_id >= 0 for r in result)  # at least one cluster

    def test_cluster_label_nonempty_for_real_cluster(self):
        vec = _unit_vec(0)
        mems = [
            _FakeMem("a", "PostgreSQL database choice", vec),
            _FakeMem("b", "PostgreSQL backend decision", vec),
        ]
        result = compute_clusters(mems)
        clustered = [r for r in result if r.cluster_id >= 0]
        assert all(r.cluster_label and len(r.cluster_label) > 0 for r in clustered)

    def test_noise_memory_has_none_label(self):
        mems = [_FakeMem("a", "Solo memory", _unit_vec(0))]
        result = compute_clusters(mems)
        assert result[0].cluster_label is None

    def test_memories_without_embeddings_excluded(self):
        mems = [
            _FakeMem("a", "Has embedding", _unit_vec(0)),
            _FakeMem("b", "No embedding", None),
        ]
        result = compute_clusters(mems)
        ids = {r.memory_id for r in result}
        assert "a" in ids
        assert "b" not in ids

    def test_cluster_id_is_integer(self):
        vec = _unit_vec(0)
        mems = [_FakeMem("a", "Title A", vec), _FakeMem("b", "Title B", vec)]
        result = compute_clusters(mems)
        for r in result:
            assert isinstance(r.cluster_id, int)


# ---------------------------------------------------------------------------
# 11–14: Fields in MemoryResponse
# ---------------------------------------------------------------------------

class TestClusterFieldsPresent:
    def test_cluster_id_field_present(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "cluster_id" in m

    def test_cluster_label_field_present(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert "cluster_label" in m

    def test_new_memory_cluster_id_is_null(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["cluster_id"] is None

    def test_new_memory_cluster_label_is_null(self, client: TestClient):
        pid = _project(client)
        m = _memory(client, pid)
        assert m["cluster_label"] is None


# ---------------------------------------------------------------------------
# 15–19: Recluster endpoint
# ---------------------------------------------------------------------------

class TestReclusterEndpoint:
    def test_recluster_returns_200(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories/recluster")
        assert r.status_code == 200

    def test_recluster_returns_cluster_result_schema(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories/recluster").json()
        assert "project_id" in r
        assert "total_clustered" in r
        assert "total_noise" in r
        assert "total_unclustered" in r
        assert "clusters" in r
        assert isinstance(r["clusters"], list)

    def test_recluster_404_unknown_project(self, client: TestClient):
        r = client.post("/projects/does-not-exist/memories/recluster")
        assert r.status_code == 404

    def test_recluster_empty_project_all_zeros(self, client: TestClient):
        pid = _project(client)
        r = client.post(f"/projects/{pid}/memories/recluster").json()
        assert r["total_clustered"] == 0
        assert r["total_noise"] == 0
        assert r["total_unclustered"] == 0
        assert r["clusters"] == []

    def test_recluster_two_memories_produce_no_clusters(self, client: TestClient):
        # Manual memories are now embedded at creation time (fixed in 1.42).
        # Two memories are too few for DBSCAN (min_samples=2 needs 2 *neighbours*),
        # so they land as noise (cluster_id=-1), not unclustered.
        pid = _project(client)
        _memory(client, pid, title="Manual memory A about authentication")
        _memory(client, pid, title="Manual memory B about authentication")
        r = client.post(f"/projects/{pid}/memories/recluster").json()
        assert r["total_unclustered"] == 0
        assert r["total_clustered"] + r["total_noise"] == 2


# ---------------------------------------------------------------------------
# 20–22: GET /projects/{id}/clusters endpoint
# ---------------------------------------------------------------------------

class TestGetClustersEndpoint:
    def test_get_clusters_returns_list(self, client: TestClient):
        pid = _project(client)
        r = client.get(f"/projects/{pid}/clusters")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_clusters_404_unknown_project(self, client: TestClient):
        r = client.get("/projects/does-not-exist/clusters")
        assert r.status_code == 404

    def test_get_clusters_empty_when_no_clusters(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)   # no embedding → no cluster
        client.post(f"/projects/{pid}/memories/recluster")
        r = client.get(f"/projects/{pid}/clusters").json()
        assert r == []


# ---------------------------------------------------------------------------
# 23–25: cluster fields visible in list and search
# ---------------------------------------------------------------------------

class TestClusterFieldsInListAndSearch:
    def test_cluster_id_in_list_response(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        mems = client.get(f"/projects/{pid}/memories").json()
        for m in mems:
            assert "cluster_id" in m

    def test_cluster_id_in_search_results(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Searchable clustering memory")
        results = client.get(f"/projects/{pid}/memories/search",
                             params={"q": "Searchable"}).json()
        if results:
            assert "cluster_id" in results[0]

    def test_total_counts_sum_to_total_memories(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid, title="Memory One")
        _memory(client, pid, title="Memory Two")
        _memory(client, pid, title="Memory Three")
        r = client.post(f"/projects/{pid}/memories/recluster").json()
        total = r["total_clustered"] + r["total_noise"] + r["total_unclustered"]
        assert total == 3


# ---------------------------------------------------------------------------
# 26–30: Integration — extraction, idempotency, cluster info
# ---------------------------------------------------------------------------

class TestClusteringIntegration:
    def test_extraction_triggers_recluster(self, client: TestClient):
        pid = _project(client)
        mems = _session_extract(client, pid, (
            "We decided to use PostgreSQL as our primary database for better "
            "JSON support and reliable concurrent writes under load. "
            "The architecture uses event sourcing with CQRS pattern for "
            "independent read and write scaling."
        ))
        if not mems:
            pytest.skip("No memories extracted")
        # After extraction (which triggers recluster), check via GET /memories
        project_mems = client.get(f"/projects/{pid}/memories").json()
        embedded = [m for m in project_mems if m.get("embedding_model")]
        # All embedded memories should have cluster_id set (not None)
        for m in embedded:
            assert m["cluster_id"] is not None

    def test_recluster_is_idempotent(self, client: TestClient):
        pid = _project(client)
        _memory(client, pid)
        r1 = client.post(f"/projects/{pid}/memories/recluster").json()
        r2 = client.post(f"/projects/{pid}/memories/recluster").json()
        assert r1["total_clustered"] == r2["total_clustered"]
        assert r1["total_noise"] == r2["total_noise"]
        assert r1["total_unclustered"] == r2["total_unclustered"]

    def test_cluster_info_has_correct_memory_count(self, client: TestClient):
        pid = _project(client)
        # Extract multiple memories and recluster to see if ClusterInfo counts are correct
        _session_extract(client, pid, (
            "We decided to adopt PostgreSQL for data persistence. "
            "The system uses PostgreSQL transactions for ACID guarantees. "
            "All writes go through PostgreSQL stored procedures for audit logging."
        ))
        r = client.post(f"/projects/{pid}/memories/recluster").json()
        for cluster in r["clusters"]:
            assert cluster["memory_count"] == len(cluster["memory_ids"])

    def test_clustered_memory_appears_in_get_clusters(self, client: TestClient):
        pid = _project(client)
        _session_extract(client, pid, (
            "We decided to use Redis for caching session data. "
            "Redis was chosen for its sub-millisecond latency requirements. "
            "Our caching layer relies on Redis pub/sub for real-time invalidation."
        ))
        recluster_result = client.post(f"/projects/{pid}/memories/recluster").json()
        if recluster_result["total_clustered"] == 0:
            pytest.skip("No clusters formed — not enough similar embeddings")
        clusters = client.get(f"/projects/{pid}/clusters").json()
        assert len(clusters) > 0
        assert all("cluster_id" in c for c in clusters)
        assert all("cluster_label" in c for c in clusters)
        assert all("memory_count" in c for c in clusters)
        assert all("memory_ids" in c for c in clusters)

    def test_cluster_label_appears_in_memory_response_after_recluster(self, client: TestClient):
        pid = _project(client)
        _session_extract(client, pid, (
            "We decided to use FastAPI for the REST API layer. "
            "FastAPI was chosen for its async support and automatic OpenAPI docs. "
            "The backend API runs on FastAPI with Uvicorn as the ASGI server."
        ))
        client.post(f"/projects/{pid}/memories/recluster")
        project_mems = client.get(f"/projects/{pid}/memories").json()
        clustered = [m for m in project_mems if m.get("cluster_id") is not None and m["cluster_id"] >= 0]
        # All clustered memories should have a non-empty cluster_label
        for m in clustered:
            assert m["cluster_label"] and len(m["cluster_label"]) > 0
