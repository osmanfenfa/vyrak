"""
zelvryn/db/migrations/versions/001_initial_schema.py
=====================================================
Migration 001: Initial database schema for Zelvryn ABSA.

Creates all base tables required by the system:
  - users           — User account registry and consent tracking
  - sessions        — Session lifecycle records
  - action_logs     — Append-only per-action audit trail
  - feature_vectors — Labeled ML training samples
  - model_versions  — ML model lifecycle registry
  - consent_records — GDPR consent audit trail

Revision  : 001_initial
Revises   : (none — this is the first migration)
Create date: 2025-01-15 00:00:00 UTC

To apply:
    alembic -c db/migrations/alembic.ini upgrade 001_initial

To roll back:
    alembic -c db/migrations/alembic.ini downgrade base
"""

from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# Alembic revision identifiers — do not edit these.
revision: str       = "001_initial"
down_revision       = None           # first migration — no parent
branch_labels       = None
depends_on          = None


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE: create all tables
# ─────────────────────────────────────────────────────────────────────────────

def upgrade() -> None:
    """
    Create the initial Zelvryn database schema.

    All tables are created inside this single migration so that a fresh
    database can be brought to the initial state with a single command:
        alembic upgrade head
    """

    # ── Table: users ─────────────────────────────────────────────────────
    op.create_table(
        "users",

        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            comment="Unique user identifier — matches user_id used across the system.",
        ),
        sa.Column(
            "email",
            sa.String(320),
            nullable=True,
            unique=True,
            comment="User email address. Nullable for SSO or non-email auth systems.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
            comment="Account creation timestamp (server clock).",
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of most recent session start.",
        ),
        sa.Column(
            "consent_given",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("FALSE"),
            comment="Whether the user has consented to behavioral monitoring.",
        ),
        sa.Column(
            "consent_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of consent grant.",
        ),
        sa.Column(
            "consent_revoked_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of consent revocation. NULL = still active.",
        ),
        sa.Column(
            "consent_version",
            sa.String(32),
            nullable=True,
            comment="Version of the privacy notice the user accepted.",
        ),
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("TRUE"),
            comment="FALSE = account suspended or pending deletion.",
        ),
    )
    # Indexes on users
    op.create_index("ix_users_email", "users", ["email"])


    # ── Table: sessions ──────────────────────────────────────────────────
    op.create_table(
        "sessions",

        sa.Column("id",       UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("ended_at",        sa.DateTime(timezone=True), nullable=True),
        sa.Column("final_trust",     sa.Float,   nullable=True),
        sa.Column("risk_rating",     sa.String(32), nullable=True),
        sa.Column("action_count",    sa.Integer, nullable=False, server_default="0"),
        sa.Column("allow_count",     sa.Integer, nullable=False, server_default="0"),
        sa.Column("challenge_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("block_count",     sa.Integer, nullable=False, server_default="0"),
        sa.Column("anomaly_count",   sa.Integer, nullable=False, server_default="0"),
        sa.Column("shadow_mode",     sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("ml_trained",      sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_user_started", "sessions", ["user_id", "started_at"])


    # ── Table: action_logs (append-only audit trail) ─────────────────────
    op.create_table(
        "action_logs",

        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            nullable=False,
            comment="Denormalised from sessions for efficient per-user queries.",
        ),
        sa.Column(
            "action_type",
            sa.String(16),
            nullable=False,
            comment="One of: login, read, write, delete, search, export, config, admin.",
        ),
        sa.Column(
            "resource_id",
            sa.String(32),
            nullable=False,
            comment="First 16 hex chars of SHA-256(raw_resource_id).",
        ),
        sa.Column(
            "timestamp_action",
            sa.Float,
            nullable=False,
            comment="Unix timestamp of the action as reported by the client.",
        ),
        sa.Column(
            "logged_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("trust_at_time", sa.Float,   nullable=False),
        sa.Column("bot_prob",      sa.Float,   nullable=False, server_default="0.0"),
        sa.Column("similarity",    sa.Float,   nullable=False, server_default="1.0"),
        sa.Column("entropy",       sa.Float,   nullable=False, server_default="0.0"),
        sa.Column("temporal_z",    sa.Float,   nullable=False, server_default="0.0"),
        sa.Column("decision",      sa.String(16), nullable=False),
        sa.Column(
            "anomalies",
            JSONB,
            nullable=True,
            comment="JSON array of triggered anomaly signal strings.",
        ),
        sa.Column("ramp_active",   sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
    )
    op.create_index("ix_action_logs_session_id", "action_logs", ["session_id"])
    op.create_index("ix_action_logs_user_id",    "action_logs", ["user_id"])
    op.create_index("ix_action_logs_decision",   "action_logs", ["decision"])
    op.create_index("ix_action_logs_user_ts",    "action_logs", ["user_id", "timestamp_action"])
    op.create_index("ix_action_logs_session_ts", "action_logs", ["session_id", "timestamp_action"])


    # ── Table: feature_vectors ───────────────────────────────────────────
    op.create_table(
        "feature_vectors",

        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.String(64),
            nullable=True,
            comment="Source session identifier (stored as string for flexibility).",
        ),
        sa.Column(
            "vector",
            JSONB,
            nullable=False,
            comment="20-dimensional behavioral feature vector as a JSON array of floats.",
        ),
        sa.Column(
            "label",
            sa.SmallInteger,
            nullable=False,
            comment="0 = human (legitimate), 1 = bot (anomalous).",
        ),
        sa.Column(
            "verified",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("FALSE"),
            comment="TRUE when an analyst has confirmed this label.",
        ),
        sa.Column(
            "source",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'live'"),
            comment="Label origin: live | shadow | corpus | analyst.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("ix_fv_user_id",      "feature_vectors", ["user_id"])
    op.create_index("ix_fv_user_verified","feature_vectors", ["user_id", "verified"])
    op.create_index("ix_fv_label_source", "feature_vectors", ["label", "source"])
    op.create_index("ix_fv_created",      "feature_vectors", ["created_at"])
    op.create_index("ix_fv_session_id",   "feature_vectors", ["session_id"])


    # ── Table: model_versions ────────────────────────────────────────────
    op.create_table(
        "model_versions",

        sa.Column("id",      sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("version", sa.String(64),  nullable=False, unique=True),
        sa.Column("model_path", sa.String(512), nullable=False),
        sa.Column(
            "state",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'trained'"),
            comment="trained | canary | active | retired",
        ),
        sa.Column("n_samples",  sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_human",    sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_bot",      sa.Integer, nullable=False, server_default="0"),
        sa.Column("cv_f1",      sa.Float,   nullable=False, server_default="0.0"),
        sa.Column("cv_far",     sa.Float,   nullable=False, server_default="0.0"),
        sa.Column("cv_frr",     sa.Float,   nullable=False, server_default="0.0"),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("promoted_at",sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes",      sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("ix_model_versions_version",    "model_versions", ["version"])
    op.create_index("ix_model_versions_state",      "model_versions", ["state"])
    op.create_index("ix_model_versions_trained_at", "model_versions", ["trained_at"])


    # ── Table: consent_records ───────────────────────────────────────────
    op.create_table(
        "consent_records",

        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "event_type",
            sa.String(16),
            nullable=False,
            comment="grant | revoke | expire | reconfirm",
        ),
        sa.Column("consent_version", sa.String(32), nullable=True),
        sa.Column("ip_address",      sa.String(64), nullable=True),
        sa.Column("purpose",         sa.Text,       nullable=True),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes",      sa.Text, nullable=True),
    )
    op.create_index("ix_consent_user_id",  "consent_records", ["user_id"])
    op.create_index("ix_consent_user_ts",  "consent_records", ["user_id", "granted_at"])
    op.create_index("ix_consent_event",    "consent_records", ["event_type"])


# ─────────────────────────────────────────────────────────────────────────────
# DOWNGRADE: drop all tables (in reverse creation order)
# ─────────────────────────────────────────────────────────────────────────────

def downgrade() -> None:
    """
    Drop all tables created by this migration.

    Tables are dropped in reverse dependency order to respect foreign key
    constraints. Child tables (with FKs) must be dropped before parent tables.

    WARNING: This permanently destroys all data in the database.
    Only run in development or testing environments.
    """
    # Drop indexes explicitly (some databases require this before table drop)
    op.drop_index("ix_consent_event",    "consent_records")
    op.drop_index("ix_consent_user_ts",  "consent_records")
    op.drop_index("ix_consent_user_id",  "consent_records")
    op.drop_table("consent_records")

    op.drop_index("ix_model_versions_trained_at", "model_versions")
    op.drop_index("ix_model_versions_state",      "model_versions")
    op.drop_index("ix_model_versions_version",    "model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_fv_session_id",   "feature_vectors")
    op.drop_index("ix_fv_created",      "feature_vectors")
    op.drop_index("ix_fv_label_source", "feature_vectors")
    op.drop_index("ix_fv_user_verified","feature_vectors")
    op.drop_index("ix_fv_user_id",      "feature_vectors")
    op.drop_table("feature_vectors")

    op.drop_index("ix_action_logs_session_ts", "action_logs")
    op.drop_index("ix_action_logs_user_ts",    "action_logs")
    op.drop_index("ix_action_logs_decision",   "action_logs")
    op.drop_index("ix_action_logs_user_id",    "action_logs")
    op.drop_index("ix_action_logs_session_id", "action_logs")
    op.drop_table("action_logs")

    op.drop_index("ix_sessions_user_started", "sessions")
    op.drop_index("ix_sessions_user_id",      "sessions")
    op.drop_table("sessions")

    op.drop_index("ix_users_email", "users")
    op.drop_table("users")
