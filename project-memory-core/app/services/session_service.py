# Session-level helper logic.
# Currently a thin pass-through; extracted here so Phase 2 can add
# pre/post processing (e.g. chunking long sessions before LLM extraction).
from sqlalchemy.orm import Session as DBSession
from app import crud, schemas


def get_session_with_project(session_id: str, db: DBSession):
    session = crud.get_session(db, session_id)
    project = crud.get_project(db, session.project_id)
    return session, project
