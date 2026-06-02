from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(tags=["memory-links"])


@router.post(
    "/memories/{memory_id}/links",
    response_model=schemas.MemoryLinkResponse,
    status_code=201,
)
def create_memory_link(
    memory_id: str,
    data: schemas.MemoryLinkCreate,
    db: Session = Depends(get_db),
):
    return crud.create_memory_link(db, memory_id, data)


@router.get(
    "/memories/{memory_id}/links",
    response_model=list[schemas.MemoryLinkResponse],
)
def list_memory_links(memory_id: str, db: Session = Depends(get_db)):
    return crud.get_memory_links(db, memory_id)


@router.delete("/memory-links/{link_id}", status_code=204)
def delete_memory_link(link_id: str, db: Session = Depends(get_db)):
    crud.delete_memory_link(db, link_id)
