"""decay_score — sub-phase 1.28

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2026-05-25

"""
from alembic import op
import sqlalchemy as sa

revision = "h8i9j0k1l2m3"
down_revision = "g7h8i9j0k1l2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("decay_score", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("memories", "decay_score")
