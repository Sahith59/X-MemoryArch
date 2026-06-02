"""superseded_at on memory_links — sub-phase 1.45

Add superseded_at column to memory_links table.
Populated only on links with relationship_type='supersedes' to record
the exact timestamp the old memory was superseded.

Revision ID: o5p6q7r8s9t0
Revises: n4o5p6q7r8s9
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa

revision = "o5p6q7r8s9t0"
down_revision = "n4o5p6q7r8s9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("memory_links") as batch_op:
        batch_op.add_column(
            sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("memory_links") as batch_op:
        batch_op.drop_column("superseded_at")
