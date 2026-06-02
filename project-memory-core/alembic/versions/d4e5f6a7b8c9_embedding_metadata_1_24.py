"""embedding metadata columns — sub-phase 1.24

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-25
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(sa.Column("embedding_model", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("embedding_dim", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_column("embedding_dim")
        batch_op.drop_column("embedding_model")
