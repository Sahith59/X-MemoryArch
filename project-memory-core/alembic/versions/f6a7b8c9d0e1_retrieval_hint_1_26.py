"""retrieval_hint — sub-phase 1.26

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-25

"""
from alembic import op
import sqlalchemy as sa

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("retrieval_hint", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("memories", "retrieval_hint")
