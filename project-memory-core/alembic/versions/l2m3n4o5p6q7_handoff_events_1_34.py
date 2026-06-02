"""handoff_events_1_34

Sub-phase 1.34 — HandoffEvents table for tracking cross-tool context transfers.

Revision ID: l2m3n4o5p6q7
Revises: k1l2m3n4o5p6
Create Date: 2026-05-26
"""
from alembic import op
import sqlalchemy as sa

revision = "l2m3n4o5p6q7"
down_revision = "k1l2m3n4o5p6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "handoff_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("context_packet_id", sa.String(), nullable=True),
        sa.Column("source_tool", sa.String(), nullable=False),
        sa.Column("target_tool", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("handoff_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["context_packet_id"], ["context_packets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_handoff_events_project_id", "handoff_events", ["project_id"])
    op.create_index("ix_handoff_events_status", "handoff_events", ["status"])
    op.create_index("ix_handoff_events_source_tool", "handoff_events", ["source_tool"])
    op.create_index("ix_handoff_events_target_tool", "handoff_events", ["target_tool"])


def downgrade() -> None:
    op.drop_index("ix_handoff_events_target_tool", table_name="handoff_events")
    op.drop_index("ix_handoff_events_source_tool", table_name="handoff_events")
    op.drop_index("ix_handoff_events_status", table_name="handoff_events")
    op.drop_index("ix_handoff_events_project_id", table_name="handoff_events")
    op.drop_table("handoff_events")
