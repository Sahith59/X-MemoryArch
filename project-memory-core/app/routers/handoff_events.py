"""Sub-phase 1.34 — HandoffEvents router."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DBSession

from app import crud, schemas
from app.database import get_db

router = APIRouter(tags=["handoff_events"])


@router.post(
    "/projects/{project_id}/handoff-events",
    response_model=schemas.HandoffEventResponse,
    status_code=201,
)
def create_handoff_event(
    project_id: str,
    data: schemas.HandoffEventCreate,
    db: DBSession = Depends(get_db),
):
    return crud.create_handoff_event(db, project_id, data)


@router.get(
    "/projects/{project_id}/handoff-events",
    response_model=list[schemas.HandoffEventResponse],
)
def list_handoff_events(
    project_id: str,
    status: str | None = Query(default=None),
    source_tool: str | None = Query(default=None),
    target_tool: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: DBSession = Depends(get_db),
):
    return crud.get_handoff_events(
        db, project_id,
        status=status,
        source_tool=source_tool,
        target_tool=target_tool,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/handoff-events/{event_id}",
    response_model=schemas.HandoffEventResponse,
)
def get_handoff_event(
    event_id: str,
    db: DBSession = Depends(get_db),
):
    return crud.get_handoff_event(db, event_id)


@router.patch(
    "/handoff-events/{event_id}",
    response_model=schemas.HandoffEventResponse,
)
def update_handoff_event(
    event_id: str,
    data: schemas.HandoffEventUpdate,
    db: DBSession = Depends(get_db),
):
    return crud.update_handoff_event(db, event_id, data)


@router.delete(
    "/handoff-events/{event_id}",
    status_code=204,
)
def delete_handoff_event(
    event_id: str,
    db: DBSession = Depends(get_db),
):
    crud.delete_handoff_event(db, event_id)
