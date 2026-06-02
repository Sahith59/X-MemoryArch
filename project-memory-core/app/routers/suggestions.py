"""Sub-phase 1.32 — MemorySuggestions router."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DBSession

from app import crud, schemas
from app.database import get_db

router = APIRouter(tags=["suggestions"])


# ---------------------------------------------------------------------------
# Per-project endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/projects/{project_id}/suggestions",
    response_model=schemas.SuggestionResponse,
    status_code=201,
)
def create_suggestion(
    project_id: str,
    data: schemas.SuggestionCreate,
    db: DBSession = Depends(get_db),
):
    crud.get_project(db, project_id)  # 404 guard
    return crud.create_suggestion(db, project_id, data)


@router.get(
    "/projects/{project_id}/suggestions",
    response_model=list[schemas.SuggestionResponse],
)
def list_suggestions(
    project_id: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: DBSession = Depends(get_db),
):
    return crud.get_suggestions(db, project_id, status=status, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Per-suggestion endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/suggestions/{suggestion_id}",
    response_model=schemas.SuggestionResponse,
)
def get_suggestion(
    suggestion_id: str,
    db: DBSession = Depends(get_db),
):
    return crud.get_suggestion(db, suggestion_id)


@router.patch(
    "/suggestions/{suggestion_id}",
    response_model=schemas.SuggestionResponse,
)
def update_suggestion(
    suggestion_id: str,
    data: schemas.SuggestionUpdate,
    db: DBSession = Depends(get_db),
):
    return crud.update_suggestion(db, suggestion_id, data)


@router.post(
    "/suggestions/{suggestion_id}/approve",
    response_model=schemas.ApproveResult,
    status_code=200,
)
def approve_suggestion(
    suggestion_id: str,
    db: DBSession = Depends(get_db),
):
    suggestion, memory = crud.approve_suggestion(db, suggestion_id)
    return schemas.ApproveResult(
        suggestion=schemas.SuggestionResponse.model_validate(suggestion),
        memory=schemas.MemoryResponse.model_validate(memory),
    )


@router.delete(
    "/suggestions/{suggestion_id}",
    status_code=204,
)
def delete_suggestion(
    suggestion_id: str,
    db: DBSession = Depends(get_db),
):
    crud.delete_suggestion(db, suggestion_id)
