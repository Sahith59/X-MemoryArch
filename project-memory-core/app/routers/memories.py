from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.search import search_memories as _search_memories
from app.services import entity_extractor, semantic_classifier

router = APIRouter(tags=["memories"])


@router.post(
    "/projects/{project_id}/memories",
    status_code=201,
)
def create_memory(
    project_id: str,
    data: schemas.MemoryCreate,
    check_dedup: bool = Query(
        default=False,
        description="Block creation if a near-duplicate exists (similarity >= 0.97). "
                    "Returns 409 with the existing memory details.",
    ),
    auto_supersede: bool = Query(
        default=False,
        description="Automatically supersede existing active memories of the same type "
                    "whose title is >= 85% similar. Returns AutoSupersedeResult when any "
                    "memories are superseded, MemoryResponse otherwise.",
    ),
    db: Session = Depends(get_db),
):
    project = crud.get_project(db, project_id)
    embedding_bytes = semantic_classifier.embed_text(f"{data.title}. {data.content}")

    if check_dedup:
        dupes = crud.find_near_duplicates(db, project_id, embedding_bytes)
        if dupes:
            existing, sim = dupes[0]
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Near-duplicate memory exists",
                    "existing_id": existing.id,
                    "existing_title": existing.title,
                    "similarity": sim,
                    "hint": "Remove check_dedup=true to create anyway, or review the existing memory.",
                },
            )

    memory = crud.create_memory(db, project_id, data, embedding=embedding_bytes)
    entities = entity_extractor.extract_entities_for_memory(
        memory.title, memory.content, domain=project.domain or "general"
    )
    crud.create_memory_entities(db, memory.id, project_id, entities)
    crud.autolink_by_entities(db, project_id, memory.id)

    if auto_supersede:
        superseded = crud.auto_supersede_on_conflict(db, memory)
        if superseded:
            return schemas.AutoSupersedeResult(
                memory=schemas.MemoryResponse.model_validate(memory),
                superseded_count=len(superseded),
                superseded_memories=[
                    schemas.MemoryResponse.model_validate(m) for m in superseded
                ],
            )

    return schemas.MemoryResponse.model_validate(memory)


@router.get(
    "/projects/{project_id}/memories/search",
    response_model=list[schemas.SearchResult],
)
def search_memories(
    project_id: str,
    q: str = Query(..., min_length=1, description="Search query — all terms must match (prefix)"),
    type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    importance_min: Optional[int] = Query(default=None, ge=1, le=5),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    crud.get_project(db, project_id)  # 404 if project missing
    results = _search_memories(
        db, project_id, q,
        mem_type=type, status=status, importance_min=importance_min, limit=limit,
    )
    out = []
    for m, score in results:
        base = schemas.MemoryResponse.model_validate(m).model_dump()
        base["relevance_score"] = round(score, 6)
        out.append(schemas.SearchResult.model_validate(base))
    return out


@router.get(
    "/projects/{project_id}/memories/stale",
    response_model=list[schemas.MemoryResponse],
)
def stale_memories(
    project_id: str,
    days: int = Query(default=30, ge=1),
    db: Session = Depends(get_db),
):
    return crud.get_stale_memories(db, project_id, days=days)


@router.get(
    "/projects/{project_id}/memories",
    response_model=list[schemas.MemoryResponse],
)
def list_memories(
    project_id: str,
    type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    importance_min: Optional[int] = Query(default=None, ge=1, le=5),
    tag: Optional[str] = Query(default=None),
    max_privacy_level: Optional[str] = Query(default=None, description="Filter: only return memories at or below this privacy level (public < internal < sensitive < secret)"),
    tier: Optional[str] = Query(default=None, description="Filter by tier: working | archival"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return crud.get_memories(
        db, project_id,
        type=type, status=status, importance_min=importance_min, tag=tag,
        max_privacy_level=max_privacy_level, tier=tier,
        skip=skip, limit=limit,
    )


@router.post(
    "/projects/{project_id}/memories/embed-all",
    response_model=schemas.EmbedAllResult,
)
def embed_all_memories(project_id: str, db: Session = Depends(get_db)):
    """Backfill embeddings for every memory in the project that doesn't have one.
    Safe to call multiple times — skips already-embedded memories."""
    return crud.backfill_embeddings(db, project_id)


@router.post(
    "/projects/{project_id}/memories/find-duplicates",
    response_model=schemas.DuplicateCheckResult,
)
def find_duplicate_memories(
    project_id: str,
    data: schemas.MemoryCreate,
    threshold: float = Query(default=0.92, ge=0.0, le=1.0, description="Cosine similarity threshold"),
    db: Session = Depends(get_db),
):
    """Check whether a memory would be a near-duplicate of existing memories.

    Does NOT create a memory. Use this before creation to preview duplicates.
    Returns is_duplicate=True and the matching memories if any similarity >= threshold.
    """
    crud.get_project(db, project_id)
    embedding_bytes = semantic_classifier.embed_text(f"{data.title}. {data.content}")
    dupes = crud.find_near_duplicates(db, project_id, embedding_bytes, threshold=threshold)
    near_dupes = [
        schemas.NearDuplicate(
            memory=schemas.MemoryResponse.model_validate(m),
            similarity=sim,
        )
        for m, sim in dupes
    ]
    return schemas.DuplicateCheckResult(
        is_duplicate=len(near_dupes) > 0,
        threshold=threshold,
        near_duplicates=near_dupes,
    )


@router.post(
    "/projects/{project_id}/memories/retier",
    response_model=schemas.RetierResult,
)
def retier_memories(project_id: str, db: Session = Depends(get_db)):
    """Recompute tier for all memories in the project based on current rules."""
    return crud.retier_project_memories(db, project_id)


@router.post(
    "/projects/{project_id}/memories/autolink",
    response_model=schemas.AutolinkResult,
)
def autolink_memories(project_id: str, db: Session = Depends(get_db)):
    """Create related_to links between memories sharing high-value entities (TECH/ORG/PRODUCT)."""
    return crud.autolink_project_memories(db, project_id)


@router.post(
    "/projects/{project_id}/memories/recluster",
    response_model=schemas.ClusterResult,
)
def recluster_memories(project_id: str, db: Session = Depends(get_db)):
    """Run DBSCAN on all embedded memories and update cluster_id + cluster_label."""
    return crud.recluster_project_memories(db, project_id)


@router.get(
    "/projects/{project_id}/clusters",
    response_model=list[schemas.ClusterInfo],
)
def get_clusters(project_id: str, db: Session = Depends(get_db)):
    """Return all non-noise clusters in the project with member counts and IDs."""
    return crud.get_project_clusters(db, project_id)


@router.get(
    "/projects/{project_id}/memories/most-accessed",
    response_model=list[schemas.MemoryResponse],
)
def most_accessed_memories(
    project_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return memories ranked by access_count — useful for Phase-2 retrieval prioritisation."""
    return crud.get_most_accessed_memories(db, project_id, limit=limit)


@router.get("/memories/{memory_id}", response_model=schemas.MemoryResponse)
def get_memory(memory_id: str, db: Session = Depends(get_db)):
    memory = crud.get_memory(db, memory_id)
    crud.track_memory_access(db, memory)
    return memory


@router.put("/memories/{memory_id}", response_model=schemas.MemoryResponse)
def update_memory(
    memory_id: str, data: schemas.MemoryUpdate, db: Session = Depends(get_db)
):
    return crud.update_memory(db, memory_id, data)


@router.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str, db: Session = Depends(get_db)):
    crud.delete_memory(db, memory_id)


@router.post(
    "/memories/{memory_id}/supersede",
    response_model=schemas.SupersedeResult,
    status_code=201,
)
def supersede_memory(
    memory_id: str,
    data: schemas.MemoryCreate,
    db: Session = Depends(get_db),
):
    old_mem, new_mem = crud.supersede_memory(db, memory_id, data)
    return schemas.SupersedeResult(
        old_memory=schemas.MemoryResponse.model_validate(old_mem),
        new_memory=schemas.MemoryResponse.model_validate(new_mem),
    )


@router.post(
    "/memories/{memory_id}/classify",
    response_model=schemas.MemoryResponse,
)
def manually_classify_memory(
    memory_id: str,
    data: schemas.ManualClassifyRequest,
    db: Session = Depends(get_db),
):
    """Assign a confirmed type to an unclassified memory and mark it verified."""
    crud.get_memory(db, memory_id)  # 404 if missing
    update = schemas.MemoryUpdate(
        type=data.type,
        review_status=schemas.ReviewStatus.verified,
    )
    return crud.update_memory(db, memory_id, update)


@router.get(
    "/memories/{memory_id}/history",
    response_model=list[schemas.MemoryChangelogResponse],
)
def memory_history(memory_id: str, db: Session = Depends(get_db)):
    return crud.get_memory_history(db, memory_id)


# ---------------------------------------------------------------------------
# Entity endpoints (Sub-phase 1.22)
# ---------------------------------------------------------------------------

@router.get(
    "/memories/{memory_id}/entities",
    response_model=list[schemas.MemoryEntityResponse],
)
def get_memory_entities(memory_id: str, db: Session = Depends(get_db)):
    crud.get_memory(db, memory_id)  # 404 if missing
    return crud.get_entities_for_memory(db, memory_id)


@router.get(
    "/projects/{project_id}/entities",
    response_model=list[schemas.ProjectEntitySummary],
)
def get_project_entities(
    project_id: str,
    label: Optional[str] = Query(default=None, description="Filter by entity label: ORG, PRODUCT, PERSON, LANGUAGE, TECH"),
    db: Session = Depends(get_db),
):
    crud.get_project(db, project_id)  # 404 if missing
    return crud.get_entities_for_project(db, project_id, label=label)
