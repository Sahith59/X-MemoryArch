"""entity extraction — sub-phase 1.22

Revision ID: b2c3d4e5f6a7
Revises: f1a2b3c4d5e6
Create Date: 2026-05-25
"""
from alembic import op
import sqlalchemy as sa

revision = "b2c3d4e5f6a7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_entities",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("entity_text", sa.String(), nullable=False),
        sa.Column("entity_label", sa.String(), nullable=False),
        sa.Column("normalized_text", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_entities_memory_id", "memory_entities", ["memory_id"])
    op.create_index("ix_memory_entities_project_id", "memory_entities", ["project_id"])
    op.create_index("ix_memory_entities_normalized_text", "memory_entities", ["normalized_text"])


def downgrade() -> None:
    op.drop_index("ix_memory_entities_normalized_text", table_name="memory_entities")
    op.drop_index("ix_memory_entities_project_id", table_name="memory_entities")
    op.drop_index("ix_memory_entities_memory_id", table_name="memory_entities")
    op.drop_table("memory_entities")
