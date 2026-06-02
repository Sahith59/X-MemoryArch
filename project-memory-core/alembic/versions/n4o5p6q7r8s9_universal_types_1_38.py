"""universal_types_1_38

Sub-phase 1.38 — Rename 4 developer-specific memory types to universal equivalents.

  bug              → problem
  architecture     → structure
  code_context     → reference_context
  setup_instruction → how_to

Revision ID: n4o5p6q7r8s9
Revises: m3n4o5p6q7r8
Create Date: 2026-05-27

"""
from alembic import op
from sqlalchemy import text

revision = 'n4o5p6q7r8s9'
down_revision = 'm3n4o5p6q7r8'
branch_labels = None
depends_on = None

_RENAMES = [
    ("bug",               "problem"),
    ("architecture",      "structure"),
    ("code_context",      "reference_context"),
    ("setup_instruction", "how_to"),
]


def upgrade() -> None:
    conn = op.get_bind()
    for old, new in _RENAMES:
        conn.execute(text(f"UPDATE memories SET type = '{new}' WHERE type = '{old}'"))
        conn.execute(text(f"UPDATE memory_suggestions SET suggested_type = '{new}' WHERE suggested_type = '{old}'"))


def downgrade() -> None:
    conn = op.get_bind()
    for old, new in reversed(_RENAMES):
        conn.execute(text(f"UPDATE memories SET type = '{old}' WHERE type = '{new}'"))
        conn.execute(text(f"UPDATE memory_suggestions SET suggested_type = '{old}' WHERE suggested_type = '{new}'"))
