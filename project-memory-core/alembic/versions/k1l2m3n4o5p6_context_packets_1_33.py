"""context_packets_1_33

Sub-phase 1.33 — ContextPackets table for stored cross-tool handoff packets.

Revision ID: k1l2m3n4o5p6
Revises: j0k1l2m3n4o5
Create Date: 2026-05-26
"""
from alembic import op
import sqlalchemy as sa

revision = "k1l2m3n4o5p6"
down_revision = "j0k1l2m3n4o5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "context_packets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("target_tool", sa.String(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("included_memory_ids", sa.Text(), nullable=True),
        sa.Column("included_session_ids", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_context_packets_project_id", "context_packets", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_context_packets_project_id", table_name="context_packets")
    op.drop_table("context_packets")
