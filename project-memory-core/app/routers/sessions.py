from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(tags=["sessions"])


def _session_response(session, db: Session) -> dict:
    """Build SessionResponse dict with live message_count."""
    count = crud.get_message_count_for_session(db, session.id)
    base = schemas.SessionResponse.model_validate(session).model_dump()
    base["message_count"] = count
    return base


@router.post(
    "/projects/{project_id}/sessions",
    response_model=schemas.SessionResponse,
    status_code=201,
)
def create_session(
    project_id: str, data: schemas.SessionCreate, db: Session = Depends(get_db)
):
    session = crud.create_session(db, project_id, data)
    return _session_response(session, db)


@router.get(
    "/projects/{project_id}/sessions",
    response_model=list[schemas.SessionResponse],
)
def list_sessions(
    project_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    sessions = crud.get_sessions(db, project_id, skip=skip, limit=limit)
    return [_session_response(s, db) for s in sessions]


@router.get("/sessions/{session_id}", response_model=schemas.SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = crud.get_session(db, session_id)
    return _session_response(session, db)


@router.put("/sessions/{session_id}", response_model=schemas.SessionResponse)
def update_session(
    session_id: str, data: schemas.SessionUpdate, db: Session = Depends(get_db)
):
    session = crud.update_session(db, session_id, data)
    return _session_response(session, db)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    crud.delete_session(db, session_id)


@router.post(
    "/sessions/{session_id}/extract-memories",
    response_model=schemas.ExtractionResult,
)
def extract_memories(session_id: str, db: Session = Depends(get_db)):
    from app.services.memory_service import extract_memories_from_session
    return extract_memories_from_session(session_id, db)


# ---------------------------------------------------------------------------
# Message endpoints (Sub-phase 1.31)
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[schemas.MessageResponse],
)
def list_messages(session_id: str, db: Session = Depends(get_db)):
    return crud.get_messages_for_session(db, session_id)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=schemas.MessageResponse,
    status_code=201,
)
def add_message(
    session_id: str, data: schemas.MessageCreate, db: Session = Depends(get_db)
):
    return crud.add_message_to_session(db, session_id, data)


@router.get("/messages/{message_id}", response_model=schemas.MessageResponse)
def get_message(message_id: str, db: Session = Depends(get_db)):
    return crud.get_message(db, message_id)


@router.delete("/messages/{message_id}", status_code=204)
def delete_message(message_id: str, db: Session = Depends(get_db)):
    crud.delete_message(db, message_id)
