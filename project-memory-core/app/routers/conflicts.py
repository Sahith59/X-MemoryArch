from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(tags=["conflicts"])


@router.get(
    "/projects/{project_id}/conflicts",
    response_model=schemas.ConflictReport,
)
def detect_conflicts(project_id: str, db: Session = Depends(get_db)):
    return crud.detect_conflicts(db, project_id)


@router.post(
    "/projects/{project_id}/conflicts/resolve",
    response_model=schemas.ResolveConflictsResult,
)
def resolve_conflicts(
    project_id: str,
    dry_run: bool = Query(default=False, description="If true, report without writing"),
    db: Session = Depends(get_db),
):
    return crud.resolve_conflicts(db, project_id, dry_run=dry_run)
