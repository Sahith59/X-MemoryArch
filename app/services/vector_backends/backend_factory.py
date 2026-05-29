"""
Sub-phase 2.6 — Backend factory.

Resolves a BackendType string to the correct VectorBackend implementation.
Also provides get_active_backend() to look up the active VectorIndexState
for a project and return the matching backend instance.

Usage:
  from app.services.vector_backends.backend_factory import BackendType, create_backend
  backend = create_backend(BackendType.chroma, {"persist_path": "/data/chroma"})
  backend = create_backend("faiss", {"persist_dir": "/data/faiss"})
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.services.vector_backends.base import VectorBackend


class BackendType(str, Enum):
    sqlite_exact = "sqlite_exact"
    chroma = "chroma"
    faiss = "faiss"
    qdrant = "qdrant"


def create_backend(
    backend_type: "BackendType | str",
    config: dict | None = None,
) -> "VectorBackend":
    """
    Instantiate a VectorBackend from a type name and optional config dict.

    Config keys per backend:
      sqlite_exact: db (Session) — required
      chroma:       persist_path (str|None), hnsw_m (int), hnsw_ef_construction (int)
      faiss:        persist_dir (str|None), use_hnsw (bool), hnsw_m (int), hnsw_ef_search (int)
      qdrant:       url (str|None), api_key (str|None), hnsw_m (int), hnsw_ef_construct (int)
    """
    cfg = config or {}
    bt = BackendType(backend_type)

    if bt == BackendType.sqlite_exact:
        from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
        db = cfg.get("db")
        if db is None:
            raise ValueError("sqlite_exact backend requires config['db'] (SQLAlchemy Session)")
        return SQLiteExactBackend(db)

    if bt == BackendType.chroma:
        from app.services.vector_backends.chroma_backend import ChromaBackend
        return ChromaBackend(
            persist_path=cfg.get("persist_path"),
            hnsw_m=cfg.get("hnsw_m", 16),
            hnsw_ef_construction=cfg.get("hnsw_ef_construction", 100),
        )

    if bt == BackendType.faiss:
        from app.services.vector_backends.faiss_backend import FAISSBackend
        return FAISSBackend(
            persist_dir=cfg.get("persist_dir"),
            use_hnsw=cfg.get("use_hnsw", True),
            hnsw_m=cfg.get("hnsw_m", 16),
            hnsw_ef_search=cfg.get("hnsw_ef_search", 64),
        )

    if bt == BackendType.qdrant:
        from app.services.vector_backends.qdrant_backend import QdrantBackend
        return QdrantBackend(
            url=cfg.get("url"),
            api_key=cfg.get("api_key"),
            hnsw_m=cfg.get("hnsw_m", 16),
            hnsw_ef_construct=cfg.get("hnsw_ef_construct", 100),
        )

    raise ValueError(f"Unknown backend type: {backend_type!r}")


def get_active_backend(
    db: "Session",
    project_id: str,
) -> "VectorBackend":
    """
    Look up the active VectorIndexState for a project and return the matching backend.

    Falls back to SQLiteExactBackend if no active state is found (safe default).
    """
    from app.p2_models import VectorIndexState

    state = (
        db.query(VectorIndexState)
        .filter(
            VectorIndexState.project_id == project_id,
            VectorIndexState.is_active == True,  # noqa: E712
        )
        .first()
    )

    if state is None:
        from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
        return SQLiteExactBackend(db)

    return create_backend(state.backend, {"db": db})


def list_available_backends() -> list[dict]:
    """Return metadata for all backend types, including availability."""
    results = []

    # sqlite_exact is always available
    results.append({
        "backend": "sqlite_exact",
        "available": True,
        "ann": False,
        "description": "Exact cosine via SQLite float32 blobs (always available)",
    })

    # Chroma
    try:
        import chromadb  # noqa: F401
        chroma_ok = True
    except ImportError:
        chroma_ok = False
    results.append({
        "backend": "chroma",
        "available": chroma_ok,
        "ann": True,
        "description": "Chroma embedded HNSW (pip install chromadb)",
    })

    # FAISS
    try:
        import faiss  # noqa: F401
        faiss_ok = True
    except ImportError:
        faiss_ok = False
    results.append({
        "backend": "faiss",
        "available": faiss_ok,
        "ann": True,
        "description": "FAISS HNSW with soft-delete (pip install faiss-cpu)",
    })

    # Qdrant
    try:
        import qdrant_client  # noqa: F401
        qdrant_ok = True
    except ImportError:
        qdrant_ok = False
    results.append({
        "backend": "qdrant",
        "available": qdrant_ok,
        "ann": True,
        "description": "Qdrant filterable HNSW — local or cloud (pip install qdrant-client)",
    })

    return results
