"""
Sub-phase 2.6 — FAISS backend (HNSW + exact fallback).

Best raw ANN speed on CPU. No built-in persistence or filtering — this
wrapper adds both.

HNSW defaults (per phase-2-plan.md): M=16, efSearch tunable.

Design:
  - In-memory per-project index (loaded from disk on startup if persist_dir set)
  - FAISS uses int64 IDs internally; this wrapper maintains str↔int mapping
  - Soft-deletion via a per-project deleted-ID set (FAISS HNSW doesn't support
    native deletion without full rebuild)
  - allowed_ids filtering: oversample, post-filter in Python
    (SQLite is always the privacy authority)
  - Persistence: index saved to {persist_dir}/{project_id}.index + id_map.json

Backend modes:
  use_hnsw=True (default)  — IndexHNSWFlat (ANN, fast)
  use_hnsw=False           — IndexFlatIP   (exact, correctness baseline)

Usage:
  backend = FAISSBackend()                              # in-memory HNSW
  backend = FAISSBackend(persist_dir="/data/faiss")     # persistent
  backend = FAISSBackend(use_hnsw=False)                # exact mode
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_HNSW_M = 16
_HNSW_EF_SEARCH = 64
_OVERSAMPLE_FACTOR = 4


@dataclass
class _ProjectIndex:
    """Per-project in-memory state."""
    index: Any                          # faiss.Index
    id_to_str: dict[int, str] = field(default_factory=dict)   # faiss int → memory_id
    str_to_id: dict[str, int] = field(default_factory=dict)   # memory_id → faiss int
    deleted: set[str] = field(default_factory=set)            # soft-deleted ids
    next_id: int = 0
    dim: int = 0


class FAISSBackend:
    """
    FAISS HNSW vector backend with soft-delete and optional persistence.

    Thread-safe via per-project locks.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        use_hnsw: bool = True,
        hnsw_m: int = _HNSW_M,
        hnsw_ef_search: int = _HNSW_EF_SEARCH,
    ) -> None:
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._use_hnsw = use_hnsw
        self._hnsw_m = hnsw_m
        self._hnsw_ef_search = hnsw_ef_search
        self._projects: dict[str, _ProjectIndex] = {}
        self._mem_to_project: dict[str, str] = {}   # global reverse map for delete()
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "faiss"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lock(self, project_id: str) -> threading.Lock:
        with self._global_lock:
            if project_id not in self._locks:
                self._locks[project_id] = threading.Lock()
        return self._locks[project_id]

    def _build_index(self, dim: int):
        import faiss
        if self._use_hnsw:
            # Use METRIC_INNER_PRODUCT so scores are cosine similarities for unit vectors
            index = faiss.IndexHNSWFlat(dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self._hnsw_ef_search
        else:
            index = faiss.IndexFlatIP(dim)
        return index

    def _get_project(self, project_id: str, dim: int | None = None) -> _ProjectIndex:
        """Get or create the in-memory project index. Loads from disk if available."""
        if project_id not in self._projects:
            loaded = self._load_from_disk(project_id) if self._persist_dir else None
            if loaded is not None:
                self._projects[project_id] = loaded
            elif dim is not None:
                idx = self._build_index(dim)
                self._projects[project_id] = _ProjectIndex(index=idx, dim=dim)
            else:
                raise ValueError(f"No existing index for project {project_id} and no dim given")
        return self._projects[project_id]

    def _index_path(self, project_id: str) -> Path:
        return self._persist_dir / f"{project_id}.index"

    def _map_path(self, project_id: str) -> Path:
        return self._persist_dir / f"{project_id}_map.json"

    def _save_to_disk(self, project_id: str) -> None:
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        proj = self._projects[project_id]
        import faiss
        faiss.write_index(proj.index, str(self._index_path(project_id)))
        map_data = {
            "id_to_str": {str(k): v for k, v in proj.id_to_str.items()},
            "deleted": list(proj.deleted),
            "next_id": proj.next_id,
            "dim": proj.dim,
        }
        self._map_path(project_id).write_text(json.dumps(map_data))

    def _load_from_disk(self, project_id: str) -> _ProjectIndex | None:
        idx_path = self._index_path(project_id)
        map_path = self._map_path(project_id)
        if not idx_path.exists() or not map_path.exists():
            return None
        import faiss
        index = faiss.read_index(str(idx_path))
        data = json.loads(map_path.read_text())
        id_to_str = {int(k): v for k, v in data["id_to_str"].items()}
        str_to_id = {v: int(k) for k, v in id_to_str.items()}
        return _ProjectIndex(
            index=index,
            id_to_str=id_to_str,
            str_to_id=str_to_id,
            deleted=set(data.get("deleted", [])),
            next_id=data["next_id"],
            dim=data["dim"],
        )

    # ------------------------------------------------------------------
    # VectorBackend interface
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        try:
            import faiss  # noqa: F401
            return True
        except ImportError:
            return False

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if allowed_ids is not None and len(allowed_ids) == 0:
            return []

        with self._get_lock(project_id):
            if project_id not in self._projects:
                if self._persist_dir:
                    loaded = self._load_from_disk(project_id)
                    if loaded is None:
                        return []
                    self._projects[project_id] = loaded
                    for mid in loaded.str_to_id:
                        self._mem_to_project[mid] = project_id
                else:
                    return []
            proj = self._projects[project_id]
            if proj.index.ntotal == 0:
                return []

            qvec = np.array(query_vector, dtype=np.float32)
            norm = np.linalg.norm(qvec)
            if norm > 0:
                qvec = qvec / norm
            qvec = qvec.reshape(1, -1)

            # Oversample to compensate for post-filtering
            n_alive = proj.index.ntotal - len(proj.deleted)
            if n_alive <= 0:
                return []

            fetch_n = min(
                proj.index.ntotal,
                top_k * _OVERSAMPLE_FACTOR if allowed_ids or proj.deleted else top_k
            )
            fetch_n = max(fetch_n, 1)

            try:
                distances, int_ids = proj.index.search(qvec, fetch_n)
            except Exception:
                return []

            allowed_set = set(allowed_ids) if allowed_ids is not None else None
            scored: list[tuple[str, float]] = []

            for dist, iid in zip(distances[0], int_ids[0]):
                if iid < 0:  # FAISS sentinel for "no result"
                    continue
                memory_id = proj.id_to_str.get(int(iid))
                if memory_id is None:
                    continue
                if memory_id in proj.deleted:
                    continue
                if allowed_set is not None and memory_id not in allowed_set:
                    continue
                # dist = inner product score (higher = more similar for unit vecs)
                scored.append((memory_id, float(dist)))
                if len(scored) >= top_k:
                    break

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
        with self._get_lock(project_id):
            proj = self._get_project(project_id, dim=embedding_dim)

            vec = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec = vec.reshape(1, -1)

            if memory_id in proj.str_to_id:
                # Update: soft-delete old, re-add new
                proj.deleted.discard(memory_id)
                # Remove from deleted set before re-adding
                old_id = proj.str_to_id[memory_id]
                proj.id_to_str.pop(old_id, None)

            # Assign new FAISS int ID
            new_int_id = proj.next_id
            proj.next_id += 1
            proj.id_to_str[new_int_id] = memory_id
            proj.str_to_id[memory_id] = new_int_id

            # HNSW needs ID mapping — use IndexIDMap wrapper approach or just add
            # For IndexFlatIP / IndexHNSWFlat, we maintain our own id mapping
            proj.index.add(vec)
            # Note: FAISS adds in sequence; our int ID == index position for FlatIP
            # For HNSW, same sequential assignment. This is safe for insert-only workloads.

            self._mem_to_project[memory_id] = project_id
            if self._persist_dir:
                self._save_to_disk(project_id)

    def delete(self, memory_id: str) -> None:
        project_id = self._mem_to_project.get(memory_id)
        if not project_id:
            return
        with self._get_lock(project_id):
            if project_id not in self._projects:
                return
            proj = self._projects[project_id]
            proj.deleted.add(memory_id)
            if self._persist_dir:
                self._save_to_disk(project_id)

    def batch_backfill(
        self,
        rows: list[tuple[str, list[float], dict]],
    ) -> None:
        if not rows:
            return

        # Group by project_id
        by_project: dict[str, list[tuple[str, list[float]]]] = {}
        dim_by_project: dict[str, int] = {}
        for memory_id, vector, payload in rows:
            pid = payload.get("project_id", "unknown")
            by_project.setdefault(pid, []).append((memory_id, vector))
            dim_by_project[pid] = len(vector)

        for pid, group in by_project.items():
            dim = dim_by_project[pid]
            with self._get_lock(pid):
                proj = self._get_project(pid, dim=dim)
                ids_to_add, vecs_to_add = [], []
                for memory_id, vector in group:
                    vec = np.array(vector, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm

                    if memory_id not in proj.str_to_id:
                        new_int_id = proj.next_id
                        proj.next_id += 1
                        proj.id_to_str[new_int_id] = memory_id
                        proj.str_to_id[memory_id] = new_int_id
                        ids_to_add.append(memory_id)
                        vecs_to_add.append(vec)
                        self._mem_to_project[memory_id] = pid

                if vecs_to_add:
                    matrix = np.vstack(vecs_to_add)
                    proj.index.add(matrix)

                if self._persist_dir:
                    self._save_to_disk(pid)

    def stats(self) -> dict:
        total = sum(
            p.index.ntotal - len(p.deleted)
            for p in self._projects.values()
        )
        return {
            "backend": self.name,
            "total_indexed": total,
            "projects": len(self._projects),
            "ann": self._use_hnsw,
            "hnsw_m": self._hnsw_m,
            "hnsw_ef_search": self._hnsw_ef_search,
            "description": "FAISS HNSW" if self._use_hnsw else "FAISS exact (IndexFlatIP)",
        }

    def reset_project(self, project_id: str) -> None:
        """Drop the in-memory (and on-disk) index for a project."""
        with self._get_lock(project_id):
            if project_id in self._projects:
                proj = self._projects.pop(project_id)
                for mid in list(proj.str_to_id.keys()):
                    self._mem_to_project.pop(mid, None)
            if self._persist_dir:
                self._index_path(project_id).unlink(missing_ok=True)
                self._map_path(project_id).unlink(missing_ok=True)
