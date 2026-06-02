from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=schemas.ProjectResponse, status_code=201)
def create_project(data: schemas.ProjectCreate, db: Session = Depends(get_db)):
    return crud.create_project(db, data)


@router.get("", response_model=list[schemas.ProjectResponse])
def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_projects(db, skip=skip, limit=limit)


@router.get("/{project_id}", response_model=schemas.ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    return crud.get_project(db, project_id)


@router.put("/{project_id}", response_model=schemas.ProjectResponse)
def update_project(
    project_id: str, data: schemas.ProjectUpdate, db: Session = Depends(get_db)
):
    return crud.update_project(db, project_id, data)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    crud.delete_project(db, project_id)


@router.get("/{project_id}/health", response_model=schemas.ProjectHealthReport)
def get_project_health(project_id: str, db: Session = Depends(get_db)):
    return crud.get_project_health(db, project_id)
