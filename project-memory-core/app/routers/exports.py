from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(tags=["exports"])


@router.get(
    "/projects/{project_id}/export/context.md",
    response_class=PlainTextResponse,
)
def export_context(
    project_id: str,
    max_privacy_level: Optional[str] = Query(
        default="internal",
        description="Maximum privacy level to include: public | internal | sensitive | secret",
    ),
    db: Session = Depends(get_db),
):
    from app.services.context_export_service import generate_context_markdown
    return generate_context_markdown(project_id, db, max_privacy_level=max_privacy_level)


@router.get(
    "/projects/{project_id}/export/memories/{memory_type}.md",
    response_class=PlainTextResponse,
)
def export_type_markdown(
    project_id: str,
    memory_type: str,
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: active | resolved | stale | archived | superseded",
    ),
    min_importance: int = Query(
        default=1, ge=1, le=5,
        description="Only include memories with importance >= this value (1–5)",
    ),
    max_privacy_level: str = Query(
        default="internal",
        description="Maximum privacy level: public | internal | sensitive | secret",
    ),
    db: Session = Depends(get_db),
):
    from app.services.per_type_export_service import generate_type_markdown
    return generate_type_markdown(
        project_id,
        memory_type,
        db,
        status_filter=status,
        min_importance=min_importance,
        max_privacy_level=max_privacy_level,
    )


@router.get(
    "/projects/{project_id}/export/memory.yaml",
    response_class=PlainTextResponse,
    responses={200: {"content": {"application/yaml": {}}}},
)
def export_memory_yaml(
    project_id: str,
    status: Optional[str] = Query(
        default=None,
        description="Filter memories by status: active | resolved | stale | archived | superseded",
    ),
    min_importance: int = Query(
        default=1,
        ge=1,
        le=5,
        description="Only include memories with importance >= this value (1–5)",
    ),
    max_privacy_level: str = Query(
        default="internal",
        description="Maximum privacy level to include: public | internal | sensitive | secret",
    ),
    db: Session = Depends(get_db),
):
    from app.services.yaml_export_service import generate_yaml_export
    return generate_yaml_export(
        project_id,
        db,
        status_filter=status,
        min_importance=min_importance,
        max_privacy_level=max_privacy_level,
    )
