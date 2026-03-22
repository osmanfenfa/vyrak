"""
zelvryn/db/migrations/versions/002_add_shadow_log.py
======================================================
Migration 002: Add shadow_logs table.

Shadow mode is the recommended first deployment phase for Zelvryn — the system
evaluates every session and logs what decisions WOULD have been made, but does
not enforce any CHALLENGE or BLOCK. This generates real behavioral training data
before enforcement begins.

This migration adds the shadow_logs table that persists these shadow evaluations
for later analysis and export as ML training data. The table was separated from
action_logs because:
  1. Shadow logs contain additional fields (real_decision, heuristic_label,
     verified_label, feature_vector) not present in live action logs.
  2. Shadow logs may be generated at very high volume during initial deployment
     and then archived — keeping them separate allows efficient table management.
  3. The analyst review workflow (setting verified_label) is specific to shadow
     logs and does not apply to live action_logs.

New columns vs action_logs:
  real_decision      — The decision that WOULD have been enforced
  trust_score_shadow — Trust score in the shadow evaluation
  bot_prob_shadow    — P(bot) from the shadow ML run
  heuristic_label    — Auto-assigned training label (0=human, 1=bot)
  verified_label     — Analyst-confirmed label (overrides heuristic)
  anomalies_shadow   — JSONB array of shadow anomaly codes
  feature_vector     — JSONB of 20-dimensional feature vector

Revision  : 002_shadow_log
Revises   : 001_initial
Create date: 2025-01-16 00:00:00 UTC

To apply:
    alembic -c db/migrations/alembic.ini upgrade 002_shadow_log

To roll back:
    alembic -c db/migrations/alembic.ini downgrade 001_initial
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# Alembic revision identifiers — do not edit these.
revision: str  = "002_shadow_log"
down_revision  = "001_initial"
branch_labels  = None
depends_on     = None


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE: add shadow_logs table
# ─────────────────────────────────────────────────────────────────────────────

def upgrade() -> None:
    """
    Add the shadow_logs table and its indexes.

    Also adds a GIN index on the anomalies_shadow JSONB column to enable
    efficient queries like:
        SELECT * FROM shadow_logs
        WHERE anomalies_shadow @> '["ML_BOT_DETECTED"]'::jsonb;
    """
    op.create_table(
        "shadow_logs",

        # Primary key
        sa.Column(
            "id",
            sa.BigInteger,
            primary_key=True,
            autoincrement=True,
            comment="Sequential primary key for efficient pagination.",
        ),

        # Foreign key to sessions (CASCADE delete for GDPR erasure)
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
            comment="Parent session — shadow log is deleted when session is purged.",
        ),

        # Denormalised user_id for efficient per-user queries
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            nullable=False,
            comment="Denormalised from sessions to avoid JOIN for user-level queries.",
        ),

        # Action details
        sa.Column(
            "action_type",
            sa.String(16),
            nullable=False,
            comment="Action type that produced this shadow evaluation.",
        ),
        sa.Column(
            "timestamp_action",
            sa.Float,
            nullable=False,
            comment="Client-reported Unix timestamp of the action.",
        ),

        # Server receipt timestamp
        sa.Column(
            "logged_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
            comment="Server-side timestamp when the shadow evaluation was logged.",
        ),

        # Shadow evaluation result
        sa.Column(
            "real_decision",
            sa.String(16),
            nullable=False,
            comment=(
                "The decision Zelvryn would have enforced under full enforcement. "
                "ALLOW | CHALLENGE | BLOCK. Not returned to the client in shadow mode."
            ),
        ),
        sa.Column(
            "trust_score_shadow",
            sa.Float,
            nullable=False,
            server_default="100.0",
            comment="Trust score at the time of shadow evaluation.",
        ),
        sa.Column(
            "bot_prob_shadow",
            sa.Float,
            nullable=False,
            server_default="0.0",
            comment="P(bot) from the ML classifier during shadow evaluation.",
        ),

        # Training labels
        sa.Column(
            "heuristic_label",
            sa.SmallInteger,
            nullable=False,
            server_default="0",
            comment=(
                "Auto-assigned training label: "
                "0 = human (no anomaly detected), "
                "1 = bot (any anomaly detected). "
                "May be inaccurate — use verified_label when available."
            ),
        ),
        sa.Column(
            "verified_label",
            sa.SmallInteger,
            nullable=True,
            comment=(
                "Analyst-confirmed training label. "
                "NULL = not yet reviewed. "
                "When set, takes precedence over heuristic_label for ML training."
            ),
        ),

        # JSONB fields for structured data
        sa.Column(
            "anomalies_shadow",
            JSONB,
            nullable=True,
            comment="JSON array of anomaly signal codes triggered during shadow evaluation.",
        ),
        sa.Column(
            "feature_vector",
            JSONB,
            nullable=True,
            comment=(
                "20-dimensional behavioral feature vector at evaluation time. "
                "Stored here for direct export to ML training without re-computation."
            ),
        ),
    )

    # Standard B-tree indexes for common query patterns
    op.create_index(
        "ix_shadow_logs_session_id",
        "shadow_logs",
        ["session_id"],
    )
    op.create_index(
        "ix_shadow_logs_user_id",
        "shadow_logs",
        ["user_id"],
    )
    op.create_index(
        "ix_shadow_logs_user_ts",
        "shadow_logs",
        ["user_id", "timestamp_action"],
    )
    op.create_index(
        "ix_shadow_logs_decision",
        "shadow_logs",
        ["real_decision"],
    )
    op.create_index(
        "ix_shadow_logs_unverified",
        "shadow_logs",
        ["heuristic_label", "verified_label"],
    )
    op.create_index(
        "ix_shadow_logs_logged_at",
        "shadow_logs",
        ["logged_at"],
    )

    # GIN index for JSONB anomaly queries (e.g. find all ML_BOT_DETECTED entries)
    op.create_index(
        "ix_shadow_logs_anomalies_gin",
        "shadow_logs",
        ["anomalies_shadow"],
        postgresql_using="gin",
    )

    # Partial index: unverified entries that need analyst review
    # This makes the review queue query extremely fast
    op.execute(
        sa.text(
            "CREATE INDEX ix_shadow_logs_needs_review "
            "ON shadow_logs (id DESC) "
            "WHERE verified_label IS NULL "
            "AND heuristic_label = 1"   # bot predictions needing verification
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOWNGRADE: remove shadow_logs table
# ─────────────────────────────────────────────────────────────────────────────

def downgrade() -> None:
    """
    Remove the shadow_logs table and all its indexes.

    All shadow evaluation data will be permanently lost.
    Only run in development or after confirming shadow data is no longer needed.
    """
    # Drop partial index first (created with op.execute, must be dropped with execute)
    op.execute(sa.text("DROP INDEX IF EXISTS ix_shadow_logs_needs_review"))

    # Drop standard indexes
    op.drop_index("ix_shadow_logs_anomalies_gin", "shadow_logs",
                  postgresql_using="gin")
    op.drop_index("ix_shadow_logs_logged_at",     "shadow_logs")
    op.drop_index("ix_shadow_logs_unverified",    "shadow_logs")
    op.drop_index("ix_shadow_logs_decision",      "shadow_logs")
    op.drop_index("ix_shadow_logs_user_ts",       "shadow_logs")
    op.drop_index("ix_shadow_logs_user_id",       "shadow_logs")
    op.drop_index("ix_shadow_logs_session_id",    "shadow_logs")

    # Drop the table
    op.drop_table("shadow_logs")
