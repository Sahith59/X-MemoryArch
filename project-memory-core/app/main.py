from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from sqlalchemy import inspect, text

from app.database import Base, engine
from app.routers import projects, sessions, memories, exports, memory_links, conflicts, suggestions, context_packets, handoff_events
from app.search import setup_fts


def _assert_db_initialized() -> None:
    """Fail fast with a clear message if the DB has not been migrated yet.

    create_all is intentionally NOT called here — Alembic is the sole source
    of schema truth. Run `alembic upgrade head` before starting the server.
    """
    insp = inspect(engine)
    if "memories" not in insp.get_table_names():
        raise RuntimeError(
            "\n\nDatabase not initialized. Run:\n\n"
            "    alembic upgrade head\n\n"
            "then start the server again."
        )


_assert_db_initialized()
setup_fts(engine)

app = FastAPI(
    title="X-MemoryArch",
    description="Local-first memory core for developer AI context",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(sessions.router)
app.include_router(memories.router)
app.include_router(exports.router)
app.include_router(memory_links.router)
app.include_router(conflicts.router)
app.include_router(suggestions.router)
app.include_router(context_packets.router)
app.include_router(handoff_events.router)

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/test", include_in_schema=False)
def test_ui():
    path = os.path.join(_static_dir, "test_ui.html")
    return FileResponse(path, media_type="text/html")
