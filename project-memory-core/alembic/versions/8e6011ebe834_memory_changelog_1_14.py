"""memory_changelog_1_14

Revision ID: 8e6011ebe834
Revises: 40865db349ad
Create Date: 2026-05-24 23:29:46.431666

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '8e6011ebe834'
down_revision: Union[str, None] = '40865db349ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'memory_changelog',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('memory_id', sa.String(), sa.ForeignKey('memories.id'), nullable=False),
        sa.Column('field', sa.String(), nullable=False),
        sa.Column('old_value', sa.Text(), nullable=True),
        sa.Column('new_value', sa.Text(), nullable=True),
        sa.Column('changed_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('memory_changelog')
