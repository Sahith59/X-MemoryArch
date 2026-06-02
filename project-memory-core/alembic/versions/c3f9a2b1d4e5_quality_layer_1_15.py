"""quality_layer_1_15

Revision ID: c3f9a2b1d4e5
Revises: 8e6011ebe834
Create Date: 2026-05-25 00:00:00.000000

Adds 9 columns to the memories table for sub-phase 1.15:
  - quality_score       (Float)   — rule-based quality score 0.0–1.0
  - review_status       (String)  — draft/verified/needs_review/auto_extracted/rejected
  - search_text         (Text)    — flattened composite for Phase-2 vector embedding
  - source_type         (String)  — ai_session/manual/code_scan/import
  - source_quote        (Text)    — verbatim sentence that triggered extraction
  - symbol_name         (String)  — function/class/variable name
  - line_start          (Integer) — source line range start
  - line_end            (Integer) — source line range end
  - branch_name         (String)  — git branch when memory was created
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c3f9a2b1d4e5'
down_revision: Union[str, None] = '8e6011ebe834'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(sa.Column("quality_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("review_status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("search_text", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source_type", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("source_quote", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("symbol_name", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("line_start", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("line_end", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("branch_name", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_column("branch_name")
        batch_op.drop_column("line_end")
        batch_op.drop_column("line_start")
        batch_op.drop_column("symbol_name")
        batch_op.drop_column("source_quote")
        batch_op.drop_column("source_type")
        batch_op.drop_column("search_text")
        batch_op.drop_column("review_status")
        batch_op.drop_column("quality_score")
