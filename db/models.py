"""
zelvryn/db/models.py
=====================
SQLAlchemy ORM table definitions for the Zelvryn ABSA database.

All tables use SQLAlchemy 2.0 declarative style with typed columns
(Mapped[T]) for full IDE and type-checker support.

Table inventory
---------------
  users           — User account registry and consent tracking
  sessions        — Session lifecycle records and final outcome
  action_logs     — Full per-action audit trail (every evaluate_action() call)
  feature_vectors — Labeled ML training samples
  model_versions  — Model registry: all trained versions and their metrics
  shadow_logs     — Shadow mode evaluation results (added in migration 002)
  consent_records — GDPR consent records for DSAR compliance

Schema design decisions
-----------------------
  1. UUIDs for primary keys — prevents enumeration attacks and supports
     distributed generation without coordination.
  2. JSONB for structured data (anomalies, feature vectors, metadata) —
     PostgreSQL JSONB supports indexed queries on JSON fields.
  3. created_at DEFAULT NOW() — always set by the database, never by
     application code, to avoid clock skew issues.
  4. Separate consent_records table — consent is a legal record and must
     be queryable independently of the profile for DSAR responses.
  5. action_logs is an append-only table — no UPDATE or DELETE queries
     are issued against it (enforced by query discipline in queries.py).

Indexes
-------
  All foreign keys are indexed for JOIN performance.
  (user_id, created_at) composite indexes support per-user time-range queries.
  The feature_vectors table is indexed on (user_id, verified) to support
  the analyst label review workflow.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ─────────────────────────────────────────────────────────────────────────────
# DECLARATIVE BASE
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """
    Shared declarative base for all Zelvryn ORM models.

    All table classes inherit from this. The Base.metadata object is used
    by Alembic to detect schema changes and generate migration scripts.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: users
# ─────────────────────────────────────────────────────────────────────────────

class UserRecord(Base):
    """
    User account registry and consent tracking.

    One row per user account. Created when a user first authenticates.
    The consent_given / consent_revoked_at fields track GDPR consent
    status for deployments that require explicit opt-in.

    Columns
    -------
    id              UUID PK — matches the user_id used throughout the system
    email           Unique email address (nullable for non-email auth systems)
    created_at      Account creation timestamp (server-set)
    last_seen_at    Updated on each session start (for activity tracking)
    consent_given   Whether the user has consented to behavioral monitoring
    consent_at      Timestamp of consent grant
    consent_revoked_at  Timestamp of consent revocation (None = still active)
    consent_version Version of privacy notice the user accepted
    is_active       Whether the account is active (False = suspended/deleted)
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique user identifier — matches user_id used across the system.",
    )
    email: Mapped[Optional[str]] = mapped_column(
        String(320),
        unique=True,
        nullable=True,
        index=True,
        comment="User email address. Nullable for SSO/non-email auth systems.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Account creation timestamp (server clock).",
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of most recent session start.",
    )
    consent_given: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether the user has consented to behavioral monitoring.",
    )
    consent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of consent grant.",
    )
    consent_revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of consent revocation. None = consent still active.",
    )
    consent_version: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="Version of the privacy notice the user accepted.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="False = account suspended or pending deletion.",
    )

    # Relationships
    sessions:        List["SessionRecord"] = relationship(
        "SessionRecord", back_populates="user", cascade="all, delete-orphan"
    )
    feature_vectors: List["FeatureVector"] = relationship(
        "FeatureVector", back_populates="user", cascade="all, delete-orphan"
    )
    consent_records: List["ConsentRecord"] = relationship(
        "ConsentRecord", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"UserRecord(id={self.id!s:.8}... email={self.email!r})"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: sessions
# ─────────────────────────────────────────────────────────────────────────────

class SessionRecord(Base):
    """
    Session lifecycle records — one row per authenticated session.

    Created when POST /session/start is called; finalised when
    DELETE /session/{id} is called or on session timeout.

    Columns
    -------
    id              UUID PK — the session_id returned to the client
    user_id         FK → users.id
    started_at      Session creation timestamp
    ended_at        Session close timestamp (NULL = still active)
    final_trust     Trust score at session end
    risk_rating     Human-readable risk label at session end
    action_count    Total actions evaluated in the session
    allow_count     Number of ALLOW decisions
    challenge_count Number of CHALLENGE decisions
    block_count     Number of BLOCK decisions
    anomaly_count   Total anomaly events
    shadow_mode     Whether shadow mode was active
    ml_trained      Whether the ML classifier was trained at evaluation time
    """
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Session identifier — matches the session_id returned by POST /session/start.",
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner of this session.",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="NULL while session is still active.",
    )
    final_trust: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Trust score at session close.",
    )
    risk_rating: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="LOW RISK / MEDIUM RISK / HIGH RISK / CRITICAL RISK",
    )
    action_count:    Mapped[int] = mapped_column(Integer, default=0)
    allow_count:     Mapped[int] = mapped_column(Integer, default=0)
    challenge_count: Mapped[int] = mapped_column(Integer, default=0)
    block_count:     Mapped[int] = mapped_column(Integer, default=0)
    anomaly_count:   Mapped[int] = mapped_column(Integer, default=0)
    shadow_mode:     Mapped[bool] = mapped_column(Boolean, default=False)
    ml_trained:      Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    user:        "UserRecord"   = relationship("UserRecord", back_populates="sessions")
    action_logs: List["ActionLog"] = relationship(
        "ActionLog", back_populates="session", cascade="all, delete-orphan"
    )
    shadow_logs: List["ShadowLog"] = relationship(
        "ShadowLog", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_sessions_user_started", "user_id", "started_at"),
    )

    def __repr__(self) -> str:
        return (
            f"SessionRecord(id={self.id!s:.8}... "
            f"user={self.user_id!s:.8}... "
            f"trust={self.final_trust})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: action_logs
# ─────────────────────────────────────────────────────────────────────────────

class ActionLog(Base):
    """
    Full per-action audit trail — one row per evaluated action.

    This is an append-only table. No UPDATE or DELETE queries are issued
    against it in normal operation (only GDPR erasure deletes cascade from
    the sessions table via ondelete="CASCADE").

    Columns
    -------
    id              BIGSERIAL PK — sequential for efficient time-range queries
    session_id      FK → sessions.id
    user_id         Denormalised for query efficiency (avoids session JOIN)
    action_type     One of the 8 action types
    resource_id     SHA-256 hashed resource identifier
    timestamp_action Unix timestamp of the action (from the client)
    logged_at       Server-side receipt timestamp
    trust_at_time   Trust score immediately after this action
    bot_prob        P(bot) from the ML classifier
    similarity      Cosine similarity value
    entropy         Shannon entropy value
    temporal_z      Temporal Z-score
    decision        ALLOW / CHALLENGE / BLOCK
    anomalies       JSONB array of triggered anomaly signal codes
    ramp_active     Whether cold start ramp-up was in effect
    """
    __tablename__ = "action_logs"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Sequential primary key — use for time-ordered pagination.",
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Denormalised from sessions for efficient per-user queries.",
    )
    action_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="One of: login, read, write, delete, search, export, config, admin.",
    )
    resource_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="First 16 hex chars of SHA-256(raw_resource_id).",
    )
    timestamp_action: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Unix timestamp of the action as reported by the client.",
    )
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Server-side receipt time.",
    )
    trust_at_time:  Mapped[float]           = mapped_column(Float, nullable=False)
    bot_prob:       Mapped[float]           = mapped_column(Float, nullable=False, default=0.0)
    similarity:     Mapped[float]           = mapped_column(Float, nullable=False, default=1.0)
    entropy:        Mapped[float]           = mapped_column(Float, nullable=False, default=0.0)
    temporal_z:     Mapped[float]           = mapped_column(Float, nullable=False, default=0.0)
    decision:       Mapped[str]             = mapped_column(String(16), nullable=False)
    anomalies:      Mapped[Optional[Any]]   = mapped_column(
        JSONB, nullable=True,
        comment="JSON array of triggered anomaly signal strings.",
    )
    ramp_active:    Mapped[bool]            = mapped_column(Boolean, default=False)

    # Relationships
    session: "SessionRecord" = relationship("SessionRecord", back_populates="action_logs")

    __table_args__ = (
        Index("ix_action_logs_user_ts", "user_id", "timestamp_action"),
        Index("ix_action_logs_session_ts", "session_id", "timestamp_action"),
        Index("ix_action_logs_decision", "decision"),
    )

    def __repr__(self) -> str:
        return (
            f"ActionLog(id={self.id} "
            f"action={self.action_type} "
            f"decision={self.decision} "
            f"trust={self.trust_at_time:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: feature_vectors
# ─────────────────────────────────────────────────────────────────────────────

class FeatureVector(Base):
    """
    Labeled ML training samples — one row per action window evaluation.

    This is the primary training data store for the Random Forest classifier.
    Rows accumulate from live session evaluations (heuristic labels) and
    analyst review (verified labels).

    Columns
    -------
    id              BIGSERIAL PK
    user_id         FK → users.id (for GDPR erasure and per-user analysis)
    session_id      Reference session (not FK — sessions may be purged)
    vector          JSONB — the 20-dimensional feature vector
    label           0 = human, 1 = bot
    verified        True if an analyst has confirmed this label
    source          Origin: 'live', 'shadow', 'corpus', 'analyst'
    created_at      Server timestamp
    """
    __tablename__ = "feature_vectors"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Source session identifier (stored as string for flexibility).",
    )
    vector: Mapped[Any] = mapped_column(
        JSONB,
        nullable=False,
        comment="20-dimensional behavioural feature vector as a JSON array of floats.",
    )
    label: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        comment="0 = human (legitimate), 1 = bot (anomalous).",
    )
    verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True when an analyst has confirmed this label.",
    )
    source: Mapped[str] = mapped_column(
        String(16),
        default="live",
        nullable=False,
        comment="Label origin: live | shadow | corpus | analyst.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: "UserRecord" = relationship("UserRecord", back_populates="feature_vectors")

    __table_args__ = (
        Index("ix_fv_user_verified", "user_id", "verified"),
        Index("ix_fv_label_source", "label", "source"),
        Index("ix_fv_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"FeatureVector(id={self.id} "
            f"label={self.label} "
            f"verified={self.verified} "
            f"source={self.source!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: model_versions
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersion(Base):
    """
    Model registry — one row per trained classifier version.

    This table mirrors the JSON file in models/model_registry.json but
    provides SQL-queryable history for reporting and rollback tracking.
    The JSON file is the authoritative source; this table is a durable audit
    copy that survives even if the models/ directory is wiped.

    Columns
    -------
    id              BIGSERIAL PK
    version         Unique version string (e.g., "rf_v3")
    model_path      Filesystem path to the .joblib file
    state           trained | canary | active | retired
    n_samples       Training dataset size
    n_human / n_bot  Class breakdown
    cv_f1/far/frr   Cross-validation metrics
    trained_at      When the model was trained
    promoted_at     When it was promoted to canary/active (NULL = not yet)
    retired_at      When it was retired (NULL = still active/canary)
    notes           Free-text notes (reason for promotion, rollback note, etc.)
    """
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique version string, e.g. rf_v3 or rf_v5_post_adversarial.",
    )
    model_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Filesystem path to the serialised .joblib model file.",
    )
    state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="trained",
        comment="Lifecycle state: trained | canary | active | retired.",
    )
    n_samples:  Mapped[int]   = mapped_column(Integer, default=0)
    n_human:    Mapped[int]   = mapped_column(Integer, default=0)
    n_bot:      Mapped[int]   = mapped_column(Integer, default=0)
    cv_f1:      Mapped[float] = mapped_column(Float, default=0.0)
    cv_far:     Mapped[float] = mapped_column(Float, default=0.0)
    cv_frr:     Mapped[float] = mapped_column(Float, default=0.0)
    trained_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    promoted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_model_versions_state", "state"),
        Index("ix_model_versions_trained_at", "trained_at"),
    )

    def __repr__(self) -> str:
        return (
            f"ModelVersion({self.version!r} "
            f"state={self.state!r} "
            f"F1={self.cv_f1:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: shadow_logs  (added in migration 002)
# ─────────────────────────────────────────────────────────────────────────────

class ShadowLog(Base):
    """
    Shadow mode evaluation results — added by migration 002_add_shadow_log.

    One row per action evaluation during shadow mode. Unlike action_logs
    (which records what actually happened in enforcement mode), shadow_logs
    records what WOULD have happened — allowing the security team to validate
    detection performance before enabling enforcement.

    The real_decision column is the key field: it captures the decision
    Zelvryn would have issued, even though only ALLOW was returned to the client.

    Columns
    -------
    id              BIGSERIAL PK
    session_id      FK → sessions.id
    user_id         Denormalised for query efficiency
    action_type
    timestamp_action
    logged_at
    real_decision   The decision that WOULD have been enforced (ALLOW/CHALLENGE/BLOCK)
    trust_score_shadow  Trust score in the shadow evaluation
    bot_prob_shadow     P(bot) in the shadow evaluation
    heuristic_label     0=human, 1=bot (auto-assigned, may be wrong)
    verified_label      Analyst-corrected label (NULL = not yet reviewed)
    anomalies_shadow    JSONB array of shadow anomaly signal codes
    feature_vector  JSONB — the 20-dimensional feature vector at evaluation time
    """
    __tablename__ = "shadow_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp_action: Mapped[float] = mapped_column(Float, nullable=False)
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    real_decision: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="The decision Zelvryn would have enforced (not returned to client).",
    )
    trust_score_shadow: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    bot_prob_shadow:    Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    heuristic_label: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="Auto-assigned label: 0=human (no anomaly), 1=bot (anomaly detected).",
    )
    verified_label: Mapped[Optional[int]] = mapped_column(
        SmallInteger,
        nullable=True,
        comment="Analyst-verified label overriding the heuristic. NULL = not reviewed.",
    )
    anomalies_shadow: Mapped[Optional[Any]] = mapped_column(
        JSONB,
        nullable=True,
        comment="JSON array of shadow anomaly signal codes.",
    )
    feature_vector: Mapped[Optional[Any]] = mapped_column(
        JSONB,
        nullable=True,
        comment="20-dimensional feature vector at evaluation time.",
    )

    # Relationships
    session: "SessionRecord" = relationship("SessionRecord", back_populates="shadow_logs")

    __table_args__ = (
        Index("ix_shadow_logs_user_ts", "user_id", "timestamp_action"),
        Index("ix_shadow_logs_decision", "real_decision"),
        Index("ix_shadow_logs_unverified", "heuristic_label", "verified_label"),
    )

    def effective_label(self) -> int:
        """Return verified_label if set, else heuristic_label."""
        return self.verified_label if self.verified_label is not None else self.heuristic_label

    def __repr__(self) -> str:
        return (
            f"ShadowLog(id={self.id} "
            f"decision={self.real_decision!r} "
            f"label={self.effective_label()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE: consent_records
# ─────────────────────────────────────────────────────────────────────────────

class ConsentRecord(Base):
    """
    GDPR consent audit trail — one row per consent grant or revocation.

    This table provides a complete, immutable audit trail of consent events
    for compliance with GDPR Article 7 (conditions for consent) and Article 17
    (right to erasure). Unlike the users.consent_given boolean (which reflects
    current state), this table preserves every historical consent change.

    For a DSAR (Data Subject Access Request), query this table to show the
    complete consent history for a user.

    Columns
    -------
    id              BIGSERIAL PK
    user_id         FK → users.id
    event_type      'grant' | 'revoke' | 'expire' | 'reconfirm'
    consent_version Version of the privacy notice
    ip_address      IP address at time of consent event
    purpose         Plain-language monitoring purpose statement
    granted_at      Timestamp of this event
    expires_at      When this consent expires (for grants)
    notes           Free-text notes (e.g., "user-initiated via account settings")
    """
    __tablename__ = "consent_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="grant | revoke | expire | reconfirm",
    )
    consent_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Expiry timestamp for 'grant' events.",
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: "UserRecord" = relationship("UserRecord", back_populates="consent_records")

    __table_args__ = (
        Index("ix_consent_records_user_ts", "user_id", "granted_at"),
        Index("ix_consent_records_event_type", "event_type"),
    )

    def __repr__(self) -> str:
        return (
            f"ConsentRecord(id={self.id} "
            f"user={self.user_id!s:.8}... "
            f"event={self.event_type!r})"
        )
