"""Create audit tables

Revision ID: 001
Revises:
Create Date: 2025-01-28

Initial migration to create audit_events and agent_decisions tables.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create audit_events table
    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("event_type", sa.VARCHAR(50), nullable=False),
        sa.Column("agent_id", sa.VARCHAR(100), nullable=True),
        sa.Column("session_id", sa.VARCHAR(100), nullable=True),
        sa.Column("user_id", sa.VARCHAR(100), nullable=True),
        sa.Column("action", sa.VARCHAR(255), nullable=False),
        sa.Column("input_data", postgresql.JSONB, nullable=True),
        sa.Column("output_data", postgresql.JSONB, nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("checksum", sa.VARCHAR(64), nullable=False),
    )

    # Create indexes for audit_events
    op.create_index("idx_audit_events_timestamp", "audit_events", ["timestamp"])
    op.create_index("idx_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("idx_audit_events_agent_id", "audit_events", ["agent_id"])
    op.create_index("idx_audit_events_session_id", "audit_events", ["session_id"])

    # Create agent_decisions table
    op.create_table(
        "agent_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "audit_event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("audit_events.id"),
            nullable=True,
        ),
        sa.Column("decision_type", sa.VARCHAR(50), nullable=False),
        sa.Column("rationale", sa.TEXT, nullable=True),
        sa.Column(
            "alternatives",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("selected_option", sa.VARCHAR(255), nullable=False),
        sa.Column("confidence", sa.FLOAT, nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_agent_decisions_confidence_range",
        ),
    )

    # Create indexes for agent_decisions
    op.create_index(
        "idx_agent_decisions_audit_event_id",
        "agent_decisions",
        ["audit_event_id"],
    )
    op.create_index(
        "idx_agent_decisions_decision_type",
        "agent_decisions",
        ["decision_type"],
    )


def downgrade() -> None:
    # Drop agent_decisions table and indexes
    op.drop_index("idx_agent_decisions_decision_type", table_name="agent_decisions")
    op.drop_index("idx_agent_decisions_audit_event_id", table_name="agent_decisions")
    op.drop_table("agent_decisions")

    # Drop audit_events table and indexes
    op.drop_index("idx_audit_events_session_id", table_name="audit_events")
    op.drop_index("idx_audit_events_agent_id", table_name="audit_events")
    op.drop_index("idx_audit_events_event_type", table_name="audit_events")
    op.drop_index("idx_audit_events_timestamp", table_name="audit_events")
    op.drop_table("audit_events")
