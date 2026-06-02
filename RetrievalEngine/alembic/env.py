from __future__ import annotations
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Add Phase 1 to sys.path so we can import its Base (for autogenerate)
# ---------------------------------------------------------------------------
_phase1_root = Path(__file__).resolve().parents[2] / "project-memory-core"
if str(_phase1_root) not in sys.path:
    sys.path.insert(0, str(_phase1_root))

try:
    from app.database import Base
    target_metadata = Base.metadata
except ImportError:
    target_metadata = None

# ---------------------------------------------------------------------------
# Load .env for DATABASE_URL
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parents[1] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        # Fall back to Phase 1's .env
        _p1_env = _phase1_root / ".env"
        if _p1_env.exists():
            load_dotenv(_p1_env)
except ImportError:
    pass

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from environment if set
_db_url = os.environ.get("DATABASE_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Only generate diffs for Phase 2 tables/columns
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
