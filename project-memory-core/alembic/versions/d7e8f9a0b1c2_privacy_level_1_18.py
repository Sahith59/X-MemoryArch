"""privacy_level_1_18

Revision ID: d7e8f9a0b1c2
Revises: c3f9a2b1d4e5
Create Date: 2026-05-25 01:00:00.000000

Adds privacy_level column to memories table (default: "internal").
Values: public | internal | sensitive | secret
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd7e8f9a0b1c2'
down_revision: Union[str, None] = 'c3f9a2b1d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(
            sa.Column("privacy_level", sa.String(), nullable=False, server_default="internal")
        )


def downgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_column("privacy_level")
