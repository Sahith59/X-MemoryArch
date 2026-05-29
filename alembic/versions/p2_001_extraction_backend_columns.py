"""Phase 2 Pre-Phase: extraction backend columns on projects + memories.

Adds:
  projects.extraction_backend  TEXT NOT NULL DEFAULT 'rule_based'
  projects.extraction_model    TEXT NULL
  memories.extraction_backend  TEXT NOT NULL DEFAULT 'rule_based'
  memories.canonical_type      TEXT NULL   -- bounded stable family for retrieval/analytics
  memories.type_label          TEXT NULL   -- open LLM-assigned label for display/nuance
  memories.llm_reasoning       TEXT NULL   -- LLM short reasoning for type/importance
  memories.contextual_prefix   TEXT NULL   -- LLM 50-100 token context prefix (Phase 2.7)

Revision ID: p2_001
Revises: (Phase 1 final migration)
Create Date: 2026-05-28
"""
from __future__ import annotations
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "p2_001"
down_revision: Union[str, None] = None   # standalone — Phase 2 migration chain starts here
branch_labels: Union[str, Sequence[str], None] = ("phase2",)
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    """Check if a column already exists (idempotent guard)."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = [c["name"] for c in inspector.get_columns(table)]
    return column in cols


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # projects table
    # ------------------------------------------------------------------ #
    if not _column_exists("projects", "extraction_backend"):
        op.add_column(
            "projects",
            sa.Column(
                "extraction_backend",
                sa.Text(),
                nullable=False,
                server_default="rule_based",
            ),
        )

    if not _column_exists("projects", "extraction_model"):
        op.add_column(
            "projects",
            sa.Column("extraction_model", sa.Text(), nullable=True),
        )

    # ------------------------------------------------------------------ #
    # memories table
    # ------------------------------------------------------------------ #
    if not _column_exists("memories", "extraction_backend"):
        op.add_column(
            "memories",
            sa.Column(
                "extraction_backend",
                sa.Text(),
                nullable=False,
                server_default="rule_based",
            ),
        )

    if not _column_exists("memories", "canonical_type"):
        op.add_column(
            "memories",
            sa.Column("canonical_type", sa.Text(), nullable=True),
        )

    if not _column_exists("memories", "type_label"):
        op.add_column(
            "memories",
            sa.Column("type_label", sa.Text(), nullable=True),
        )

    if not _column_exists("memories", "llm_reasoning"):
        op.add_column(
            "memories",
            sa.Column("llm_reasoning", sa.Text(), nullable=True),
        )

    if not _column_exists("memories", "contextual_prefix"):
        op.add_column(
            "memories",
            sa.Column("contextual_prefix", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    # SQLite does not support DROP COLUMN natively before 3.35.
    # Emit a warning and skip — downgrade requires manual schema recreation.
    import warnings
    warnings.warn(
        "SQLite downgrade: cannot drop columns automatically. "
        "Phase 2 columns on projects/memories remain in schema.",
        stacklevel=2,
    )
