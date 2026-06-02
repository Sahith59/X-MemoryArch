"""suggestions_1_32

Sub-phase 1.32 — MemorySuggestions staging table.

Revision ID: j0k1l2m3n4o5
Revises: i9j0k1l2m3n4
Create Date: 2026-05-26
"""
from alembic import op
import sqlalchemy as sa

revision = "j0k1l2m3n4o5"
down_revision = "i9j0k1l2m3n4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_suggestions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("source_session_id", sa.String(), nullable=True),
        sa.Column("source_message_ids", sa.Text(), nullable=True),
        sa.Column("suggested_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("created_by", sa.String(), nullable=False, server_default="manual"),
        sa.Column("memory_id", sa.String(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_session_id"], ["ai_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_suggestions_project_id", "memory_suggestions", ["project_id"])
    op.create_index("ix_memory_suggestions_status", "memory_suggestions", ["status"])
    op.create_index("ix_memory_suggestions_session_id", "memory_suggestions", ["source_session_id"])


def downgrade() -> None:
    op.drop_index("ix_memory_suggestions_session_id", table_name="memory_suggestions")
    op.drop_index("ix_memory_suggestions_status", table_name="memory_suggestions")
    op.drop_index("ix_memory_suggestions_project_id", table_name="memory_suggestions")
    op.drop_table("memory_suggestions")
