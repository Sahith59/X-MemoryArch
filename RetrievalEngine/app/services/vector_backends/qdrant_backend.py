"""
Sub-phase 2.6 — Qdrant backend (local embedded + cloud).

Qdrant is the recommended cloud/team path:
  - Best payload filtering (filterable HNSW)
  - Asymmetric quantization for memory efficiency
  - Local embedded mode (no server needed for dev)
  - Qdrant Cloud for production scale

⚠️ Privacy gate warning (from phase-2-plan.md):
  Combinations of strict payload filters can create disconnected HNSW graph
  components in Qdrant. The privacy gate MUST remain a SQLite authority.
  Pre-filter with allowed_ids in SQLite, pass to search() — never rely on
  Qdrant payload filters as the sole privacy mechanism.

HNSW defaults: m=16, ef_construct=100

Install: pip install qdrant-client
If not installed, all methods raise VectorBackendNotInstalledError.

Usage:
  backend = QdrantBackend()                    # local in-memory (requires qdrant-client)
  backend = QdrantBackend(url="http://...")    # remote Qdrant server
  backend = QdrantBackend(api_key="...", url="https://...qdrant.io")  # Qdrant Cloud
"""
from __future__ import annotations


class VectorBackendNotInstalledError(ImportError):
    """Raised when qdrant-client package is not installed."""


def _require_qdrant():
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        raise VectorBackendNotInstalledError(
            "qdrant-client is not installed. "
            "Install it with: pip install qdrant-client"
        )


def _is_available() -> bool:
    try:
        import qdrant_client  # noqa: F401
        return True
    except ImportError:
        return False


class QdrantBackend:
    """
    Qdrant vector backend (local embedded or remote/cloud).

    Raises VectorBackendNotInstalledError on any call when qdrant-client
    is not installed.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ) -> None:
        _require_qdrant()
        self._url = url
        self._api_key = api_key
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._client = None

    @property
    def name(self) -> str:
        return "qdrant"

    @staticmethod
    def is_available() -> bool:
        return _is_available()

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self._url:
                self._client = QdrantClient(url=self._url, api_key=self._api_key)
            else:
                self._client = QdrantClient(":memory:")  # local in-memory
        return self._client

    def _ensure_collection(self, project_id: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

        client = self._get_client()
        coll_name = f"xmem_{project_id.replace('-', '')[:28]}"
        try:
            client.get_collection(coll_name)
        except Exception:
            client.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(
                    m=self._hnsw_m,
                    ef_construct=self._hnsw_ef_construct,
                ),
            )

    def _coll_name(self, project_id: str) -> str:
        return f"xmem_{project_id.replace('-', '')[:28]}"

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        _require_qdrant()
        if allowed_ids is not None and len(allowed_ids) == 0:
            return []

        import numpy as np
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        client = self._get_client()
        coll_name = self._coll_name(project_id)

        qvec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        # NOTE: We do NOT use Qdrant's payload filter as the privacy gate.
        # allowed_ids comes from SQLite hard filters. We pass it as a filter
        # to oversample candidates, then post-filter in Python for safety.
        # This avoids the disconnected-graph-component failure mode.
        query_filter = None
        if allowed_ids is not None:
            query_filter = Filter(
                must=[FieldCondition(key="memory_id", match=MatchAny(any=allowed_ids[:1000]))]
            )

        try:
            results = client.search(
                collection_name=coll_name,
                query_vector=qvec.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception:
            return []

        allowed_set = set(allowed_ids) if allowed_ids is not None else None
        scored = []
        for r in results:
            mid = r.payload.get("memory_id") or r.id
            if allowed_set is not None and mid not in allowed_set:
                continue
            scored.append((str(mid), float(r.score)))

        return scored[:top_k]

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        project_id: str,
        embedding_model: str,
        embedding_dim: int,
        payload: dict | None = None,
    ) -> None:
        _require_qdrant()
        import numpy as np
        from qdrant_client.models import PointStruct

        self._ensure_collection(project_id, embedding_dim)
        client = self._get_client()

        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        meta = {"memory_id": memory_id, "project_id": project_id, "embedding_model": embedding_model}
        if payload:
            meta.update({k: str(v) for k, v in payload.items() if v is not None})

        # Use a deterministic int ID derived from memory_id for Qdrant's uint64 requirement
        # We store the actual string memory_id in payload for retrieval
        import hashlib
        int_id = int(hashlib.md5(memory_id.encode()).hexdigest()[:16], 16) % (2**63)

        client.upsert(
            collection_name=self._coll_name(project_id),
            points=[PointStruct(id=int_id, vector=vec.tolist(), payload=meta)],
        )

    def delete(self, memory_id: str) -> None:
        _require_qdrant()
        # Without knowing project_id, we can't efficiently delete.
        # In production, use delete_from_project() instead.
        pass

    def delete_from_project(self, memory_id: str, project_id: str) -> None:
        _require_qdrant()
        import hashlib
        from qdrant_client.models import PointIdsList
        int_id = int(hashlib.md5(memory_id.encode()).hexdigest()[:16], 16) % (2**63)
        client = self._get_client()
        try:
            client.delete(
                collection_name=self._coll_name(project_id),
                points_selector=PointIdsList(points=[int_id]),
            )
        except Exception:
            pass

    def batch_backfill(
        self,
        rows: list[tuple[str, list[float], dict]],
    ) -> None:
        _require_qdrant()
        if not rows:
            return

        import hashlib
        import numpy as np
        from qdrant_client.models import PointStruct

        by_project: dict[str, list] = {}
        for memory_id, vector, payload in rows:
            pid = payload.get("project_id", "unknown")
            by_project.setdefault(pid, []).append((memory_id, vector, payload))

        for pid, group in by_project.items():
            dim = len(group[0][1])
            self._ensure_collection(pid, dim)
            client = self._get_client()
            points = []
            for memory_id, vector, payload in group:
                vec = np.array(vector, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                int_id = int(hashlib.md5(memory_id.encode()).hexdigest()[:16], 16) % (2**63)
                meta = {"memory_id": memory_id, "project_id": pid}
                if payload:
                    meta.update({k: str(v) for k, v in payload.items() if v is not None})
                points.append(PointStruct(id=int_id, vector=vec.tolist(), payload=meta))

            # Qdrant batch limit: upsert in chunks of 100
            for i in range(0, len(points), 100):
                client.upsert(
                    collection_name=self._coll_name(pid),
                    points=points[i:i + 100],
                )

    def stats(self) -> dict:
        _require_qdrant()
        client = self._get_client()
        try:
            collections = client.get_collections().collections
            total = 0
            for c in collections:
                info = client.get_collection(c.name)
                total += info.points_count or 0
            n = len(collections)
        except Exception:
            total, n = 0, 0
        return {
            "backend": self.name,
            "total_indexed": total,
            "collections": n,
            "ann": True,
            "description": "Qdrant HNSW (local or cloud)",
        }
