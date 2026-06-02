"""Phase 2.1: retrieval_runs telemetry table.

Every retrieval call writes one row. Drives latency tracking, privacy leakage
audit (forbidden_candidate_count must always be 0), and eventual weighted RRF
learning from labeled (query, memory) pairs.

Revision ID: p2_002
Revises: p2_001
Create Date: 2026-05-28
"""
from __future__ import annotations
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "p2_002"
down_revision: Union[str, None] = "p2_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrieval_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(), nullable=True),
        sa.Column("filters_applied", sa.Text(), nullable=True),
        sa.Column("candidate_count_bm25", sa.Integer(), nullable=True),
        sa.Column("candidate_count_dense", sa.Integer(), nullable=True),
        sa.Column("candidate_count_entity", sa.Integer(), nullable=True),
        sa.Column("fused_count", sa.Integer(), nullable=True),
        sa.Column("forbidden_candidate_count", sa.Integer(), nullable=True),
        sa.Column("selected_memory_ids", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("packet_token_budget", sa.Integer(), nullable=True),
        sa.Column("packet_compression_ratio", sa.Float(), nullable=True),
        sa.Column("judge_relevance_score", sa.Float(), nullable=True),
        sa.Column("backend_used", sa.String(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_retrieval_runs_project_id", "retrieval_runs", ["project_id"])
    op.create_index("ix_retrieval_runs_created_at", "retrieval_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_retrieval_runs_created_at", table_name="retrieval_runs")
    op.drop_index("ix_retrieval_runs_project_id", table_name="retrieval_runs")
    op.drop_table("retrieval_runs")
