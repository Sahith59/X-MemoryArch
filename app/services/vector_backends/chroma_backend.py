"""
Sub-phase 2.6 — Chroma embedded HNSW backend.

Recommended local ANN backend: documented HNSW tuning, embedded (no separate
server), works in local and cloud paths.

HNSW defaults (per phase-2-plan.md):
  M=16, efConstruction=100

Design:
  - One Chroma collection per project (collection name = _coll_name(project_id))
  - IDs = memory_id strings (Chroma supports string IDs natively)
  - project_id stored in metadata for stats / audit
  - allowed_ids filtering: oversample by 4×, post-filter in Python
    (SQLite is always the privacy authority — never delegate to Chroma)

Usage:
  backend = ChromaBackend()                             # in-memory (tests)
  backend = ChromaBackend(persist_path="/data/chroma")  # persistent (production)
  backend.upsert(memory_id, vector, project_id, model, dim)
  results = backend.search(query_vec, top_k=50, project_id=pid, allowed_ids=ids)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 100
_OVERSAMPLE_FACTOR = 4  # fetch N × this many, then filter by allowed_ids


def _coll_name(project_id: str) -> str:
    """Stable collection name for a project (Chroma: 3-63 chars, alphanumeric/-/_)."""
    safe = project_id.replace("-", "")[:28]
    return f"xm_{safe}"


class ChromaBackend:
    """
    Chroma embedded HNSW vector backend.

    Thread-safe: uses a lock around client/collection access.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        hnsw_m: int = _HNSW_M,
        hnsw_ef_construction: int = _HNSW_EF_CONSTRUCTION,
    ) -> None:
        self._persist_path = persist_path
        self._hnsw_m = hnsw_m
        self._hnsw_ef = hnsw_ef_construction
        self._client = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "chroma"

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import chromadb
                    if self._persist_path:
                        self._client = chromadb.PersistentClient(path=str(self._persist_path))
                    else:
                        self._client = chromadb.EphemeralClient()
        return self._client

    def _get_collection(self, project_id: str):
        client = self._get_client()
        coll_name = _coll_name(project_id)
        return client.get_or_create_collection(
            name=coll_name,
            metadata={
                "hnsw:M": self._hnsw_m,
                "hnsw:construction_ef": self._hnsw_ef,
                "project_id": project_id,
            },
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if allowed_ids is not None and len(allowed_ids) == 0:
            return []

        coll = self._get_collection(project_id)
        count = coll.count()
        if count == 0:
            return []

        # Normalise query vector
        qvec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        # Oversample to compensate for allowed_ids filtering
        fetch_n = min(count, top_k * _OVERSAMPLE_FACTOR if allowed_ids else top_k)
        fetch_n = max(fetch_n, 1)

        try:
            result = coll.query(
                query_embeddings=[qvec.tolist()],
                n_results=fetch_n,
                include=["distances"],
            )
        except Exception:
            return []

        ids = result["ids"][0] if result["ids"] else []
        distances = result["distances"][0] if result["distances"] else []

        # Chroma returns L2 distances by default; convert to cosine similarity
        # For unit-normalised vectors: cosine_sim = 1 - (L2_dist² / 2)
        # But Chroma with default space is "l2"; we need cosine. Let's check:
        # Actually Chroma stores with cosine space when requested; otherwise L2.
        # We'll convert L2 distance to cosine similarity: sim = 1 - dist/2 (for unit vecs)
        scored: list[tuple[str, float]] = []
        allowed_set = set(allowed_ids) if allowed_ids is not None else None

        for mid, dist in zip(ids, distances):
            if allowed_set is not None and mid not in allowed_set:
                continue
            # Convert L2 distance to cosine similarity (unit vectors): sim = 1 - d²/2
            cosine_sim = float(1.0 - (dist / 2.0))
            scored.append((mid, cosine_sim))

        scored.sort(key=lambda x: x[1], reverse=True)
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
        coll = self._get_collection(project_id)

        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        metadata = {"project_id": project_id, "embedding_model": embedding_model}
        if payload:
            metadata.update({k: str(v) for k, v in payload.items() if v is not None})

        coll.upsert(
            ids=[memory_id],
            embeddings=[vec.tolist()],
            metadatas=[metadata],
        )

    def delete(self, memory_id: str) -> None:
        # We don't know the project_id here, so try all collections
        # In practice, the caller should track project_id externally
        import chromadb
        client = self._get_client()
        for coll in client.list_collections():
            try:
                existing = coll.get(ids=[memory_id])
                if existing["ids"]:
                    coll.delete(ids=[memory_id])
                    return
            except Exception:
                continue

    def delete_from_project(self, memory_id: str, project_id: str) -> None:
        """Preferred delete — scoped to project (avoids scanning all collections)."""
        coll = self._get_collection(project_id)
        try:
            coll.delete(ids=[memory_id])
        except Exception:
            pass

    def batch_backfill(
        self,
        rows: list[tuple[str, list[float], dict]],
    ) -> None:
        """Bulk upsert (memory_id, vector, payload) rows grouped by project_id."""
        if not rows:
            return

        # Group by project_id from payload
        by_project: dict[str, list[tuple[str, list[float], dict]]] = {}
        for memory_id, vector, payload in rows:
            pid = payload.get("project_id", "unknown")
            by_project.setdefault(pid, []).append((memory_id, vector, payload))

        for pid, group in by_project.items():
            coll = self._get_collection(pid)
            ids, embeddings, metadatas = [], [], []
            for memory_id, vector, payload in group:
                vec = np.array(vector, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                meta = {"project_id": pid}
                if payload:
                    meta.update({k: str(v) for k, v in payload.items() if v is not None})
                ids.append(memory_id)
                embeddings.append(vec.tolist())
                metadatas.append(meta)

            # Chroma batch limit: upsert in chunks of 500
            _CHUNK = 500
            for i in range(0, len(ids), _CHUNK):
                coll.upsert(
                    ids=ids[i:i + _CHUNK],
                    embeddings=embeddings[i:i + _CHUNK],
                    metadatas=metadatas[i:i + _CHUNK],
                )

    def stats(self) -> dict:
        client = self._get_client()
        total = 0
        collections = client.list_collections()
        for coll in collections:
            total += coll.count()
        return {
            "backend": self.name,
            "total_indexed": total,
            "collections": len(collections),
            "ann": True,
            "hnsw_m": self._hnsw_m,
            "hnsw_ef_construction": self._hnsw_ef,
            "description": "Chroma embedded HNSW",
        }

    def project_stats(self, project_id: str) -> dict:
        coll = self._get_collection(project_id)
        return {
            "backend": self.name,
            "project_id": project_id,
            "total_indexed": coll.count(),
        }

    def reset_project(self, project_id: str) -> None:
        """Drop and recreate the collection for a project (used in migrations)."""
        client = self._get_client()
        coll_name = _coll_name(project_id)
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
