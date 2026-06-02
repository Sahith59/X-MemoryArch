"""messages_1_31

Sub-phase 1.31 — Messages table (granular message storage per session)
and source_message_ids on Memory for message-level source tracing.

Revision ID: i9j0k1l2m3n4
Revises: h8i9j0k1l2m3
Create Date: 2026-05-26
"""
from alembic import op
import sqlalchemy as sa

revision = "i9j0k1l2m3n4"
down_revision = "h8i9j0k1l2m3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False, server_default="text"),
        sa.Column("message_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_code", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_error", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("language", sa.String(), nullable=True),
        sa.Column("extra_metadata", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["ai_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_index("ix_messages_project_id", "messages", ["project_id"])
    op.create_index("ix_messages_role", "messages", ["role"])

    op.add_column(
        "memories",
        sa.Column("source_message_ids", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("memories", "source_message_ids")
    op.drop_index("ix_messages_role", table_name="messages")
    op.drop_index("ix_messages_project_id", table_name="messages")
    op.drop_index("ix_messages_session_id", table_name="messages")
    op.drop_table("messages")
