"""semantic_layer_1_20

Revision ID: f1a2b3c4d5e6
Revises: d7e8f9a0b1c2
Create Date: 2026-05-25 02:00:00.000000

Adds embedding column to memories table (sub-phase 1.20).
Stores float32 bytes from all-MiniLM-L6-v2 (384-dim, 1536 bytes per memory).
Phase-2 retrieval engine loads with: np.frombuffer(embedding, dtype=np.float32)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'd7e8f9a0b1c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(
            sa.Column("embedding", sa.LargeBinary(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_column("embedding")
