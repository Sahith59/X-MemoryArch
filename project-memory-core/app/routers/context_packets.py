"""Sub-phase 1.33 — ContextPackets router."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DBSession

from app import crud, schemas
from app.database import get_db

router = APIRouter(tags=["context_packets"])


@router.post(
    "/projects/{project_id}/context-packets",
    response_model=schemas.ContextPacketResponse,
    status_code=201,
)
def create_context_packet(
    project_id: str,
    data: schemas.ContextPacketCreate,
    db: DBSession = Depends(get_db),
):
    return crud.create_context_packet(db, project_id, data)


@router.get(
    "/projects/{project_id}/context-packets",
    response_model=list[schemas.ContextPacketResponse],
)
def list_context_packets(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: DBSession = Depends(get_db),
):
    return crud.get_context_packets(db, project_id, limit=limit, offset=offset)


@router.get(
    "/context-packets/{packet_id}",
    response_model=schemas.ContextPacketResponse,
)
def get_context_packet(
    packet_id: str,
    db: DBSession = Depends(get_db),
):
    return crud.get_context_packet(db, packet_id)


@router.delete(
    "/context-packets/{packet_id}",
    status_code=204,
)
def delete_context_packet(
    packet_id: str,
    db: DBSession = Depends(get_db),
):
    crud.delete_context_packet(db, packet_id)
