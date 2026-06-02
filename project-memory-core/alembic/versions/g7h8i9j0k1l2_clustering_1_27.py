"""clustering — sub-phase 1.27

Revision ID: g7h8i9j0k1l2
Revises: f6a7b8c9d0e1
Create Date: 2026-05-25

"""
from alembic import op
import sqlalchemy as sa

revision = "g7h8i9j0k1l2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("cluster_id", sa.Integer(), nullable=True))
    op.add_column("memories", sa.Column("cluster_label", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("memories", "cluster_label")
    op.drop_column("memories", "cluster_id")
