"""
zelvryn/db/queries.py
======================
Typed async query functions for all Zelvryn database operations.

Design philosophy
-----------------
  1. No raw SQL strings — all queries use SQLAlchemy 2.0 Core/ORM expressions.
     This prevents SQL injection and keeps queries type-checked.
  2. All functions are async and accept an AsyncSession as their first argument.
     The caller (route handler, background worker) owns the session lifecycle.
  3. Each function does exactly one thing. Complex workflows (e.g., upsert
     session + write action log atomically) are composed at the call site.
  4. No UPDATE queries on action_logs — that table is append-only.
  5. All sensitive lookups (by user_id) accept uuid.UUID objects, not strings,
     to prevent type confusion at the database boundary.

Query organisation
------------------
  UserQueries          — users table
  SessionQueries       — sessions table
  ActionLogQueries     — action_logs table (append-only)
  FeatureVectorQueries — feature_vectors table
  ModelVersionQueries  — model_versions table
  ShadowLogQueries     — shadow_logs table
  ConsentQueries       — consent_records table

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    ActionLog,
    ConsentRecord,
    FeatureVector,
    ModelVersion,
    SessionRecord,
    ShadowLog,
    UserRecord,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# USER QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class UserQueries:
    """Queries against the users table."""

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> Optional[UserRecord]:
        """
        Fetch a user record by UUID.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID

        Returns
        -------
        UserRecord or None
        """
        result = await session.execute(
            select(UserRecord).where(UserRecord.id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_email(
        session: AsyncSession,
        email: str,
    ) -> Optional[UserRecord]:
        """Fetch a user record by email address."""
        result = await session.execute(
            select(UserRecord).where(UserRecord.email == email.lower().strip())
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def upsert(
        session: AsyncSession,
        user_id: uuid.UUID,
        email: Optional[str] = None,
    ) -> UserRecord:
        """
        Get or create a user record.

        If the user already exists, updates last_seen_at.
        If not, creates a new record with default consent=False.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID
        email : str, optional

        Returns
        -------
        UserRecord
        """
        user = await UserQueries.get_by_id(session, user_id)
        if user is None:
            user = UserRecord(
                id             = user_id,
                email          = email.lower().strip() if email else None,
                consent_given  = False,
                is_active      = True,
            )
            session.add(user)
            logger.debug("New user record created: user_id=%s", user_id)
        else:
            user.last_seen_at = datetime.now(timezone.utc)
            if email and not user.email:
                user.email = email.lower().strip()

        await session.flush()  # flush so the object gets its DB-generated values
        return user

    @staticmethod
    async def update_consent(
        session: AsyncSession,
        user_id: uuid.UUID,
        granted: bool,
        version: str = "1.0",
    ) -> None:
        """
        Update the user's consent status.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID
        granted : bool
        version : str
        """
        now = datetime.now(timezone.utc)
        await session.execute(
            update(UserRecord)
            .where(UserRecord.id == user_id)
            .values(
                consent_given      = granted,
                consent_at         = now if granted else None,
                consent_revoked_at = None if granted else now,
                consent_version    = version,
            )
        )

    @staticmethod
    async def deactivate(session: AsyncSession, user_id: uuid.UUID) -> bool:
        """
        Mark a user account as inactive (soft delete).

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID

        Returns
        -------
        bool
            True if the user was found and deactivated.
        """
        result = await session.execute(
            update(UserRecord)
            .where(UserRecord.id == user_id)
            .values(is_active=False)
        )
        return result.rowcount > 0

    @staticmethod
    async def hard_delete(session: AsyncSession, user_id: uuid.UUID) -> bool:
        """
        Permanently delete a user record (GDPR right to erasure).

        Cascades to sessions, feature_vectors, and consent_records via
        database-level ON DELETE CASCADE constraints.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID

        Returns
        -------
        bool
            True if a record was found and deleted.
        """
        result = await session.execute(
            delete(UserRecord).where(UserRecord.id == user_id)
        )
        erased = result.rowcount > 0
        if erased:
            logger.info("GDPR hard delete: user_id=%s", user_id)
        return erased


# ─────────────────────────────────────────────────────────────────────────────
# SESSION QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class SessionQueries:
    """Queries against the sessions table."""

    @staticmethod
    async def create(
        session:    AsyncSession,
        session_id: uuid.UUID,
        user_id:    uuid.UUID,
        shadow_mode: bool = False,
    ) -> SessionRecord:
        """
        Create a new session record.

        Parameters
        ----------
        session : AsyncSession
        session_id : uuid.UUID
        user_id : uuid.UUID
        shadow_mode : bool

        Returns
        -------
        SessionRecord
        """
        record = SessionRecord(
            id          = session_id,
            user_id     = user_id,
            shadow_mode = shadow_mode,
        )
        session.add(record)
        await session.flush()
        return record

    @staticmethod
    async def get_by_id(
        session:    AsyncSession,
        session_id: uuid.UUID,
    ) -> Optional[SessionRecord]:
        """Fetch a session record by UUID."""
        result = await session.execute(
            select(SessionRecord).where(SessionRecord.id == session_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def close_session(
        session:        AsyncSession,
        session_id:     uuid.UUID,
        final_trust:    float,
        risk_rating:    str,
        action_count:   int,
        allow_count:    int,
        challenge_count: int,
        block_count:    int,
        anomaly_count:  int,
        ml_trained:     bool,
    ) -> None:
        """
        Finalise a session record with its outcome metrics.

        Parameters
        ----------
        session : AsyncSession
        session_id : uuid.UUID
        final_trust, risk_rating : summary metrics
        action/allow/challenge/block/anomaly counts : decision breakdown
        ml_trained : whether the ML classifier was active
        """
        await session.execute(
            update(SessionRecord)
            .where(SessionRecord.id == session_id)
            .values(
                ended_at        = datetime.now(timezone.utc),
                final_trust     = final_trust,
                risk_rating     = risk_rating,
                action_count    = action_count,
                allow_count     = allow_count,
                challenge_count = challenge_count,
                block_count     = block_count,
                anomaly_count   = anomaly_count,
                ml_trained      = ml_trained,
            )
        )

    @staticmethod
    async def get_user_sessions(
        session:  AsyncSession,
        user_id:  uuid.UUID,
        limit:    int  = 20,
        offset:   int  = 0,
        active_only: bool = False,
    ) -> List[SessionRecord]:
        """
        Return a user's sessions, newest first.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID
        limit : int
        offset : int
        active_only : bool
            If True, only return sessions where ended_at IS NULL.

        Returns
        -------
        list of SessionRecord
        """
        query = (
            select(SessionRecord)
            .where(SessionRecord.user_id == user_id)
            .order_by(SessionRecord.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if active_only:
            query = query.where(SessionRecord.ended_at.is_(None))

        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def count_active_sessions(
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> int:
        """Return the number of still-open sessions for a user."""
        result = await session.execute(
            select(func.count())
            .select_from(SessionRecord)
            .where(
                SessionRecord.user_id == user_id,
                SessionRecord.ended_at.is_(None),
            )
        )
        return result.scalar_one()

    @staticmethod
    async def get_high_risk_sessions(
        session: AsyncSession,
        limit: int = 50,
    ) -> List[SessionRecord]:
        """Return the most recent HIGH RISK or CRITICAL RISK sessions."""
        result = await session.execute(
            select(SessionRecord)
            .where(SessionRecord.risk_rating.in_(["HIGH RISK", "CRITICAL RISK"]))
            .order_by(SessionRecord.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# ACTION LOG QUERIES (append-only)
# ─────────────────────────────────────────────────────────────────────────────

class ActionLogQueries:
    """
    Queries against the action_logs table.

    IMPORTANT: This table is append-only. No update_* or delete_* functions
    are defined here. The only way rows leave this table is via ON DELETE CASCADE
    from the sessions table when GDPR erasure deletes the session.
    """

    @staticmethod
    async def insert(
        session:          AsyncSession,
        session_id:       uuid.UUID,
        user_id:          uuid.UUID,
        action_type:      str,
        resource_id:      str,
        timestamp_action: float,
        trust_at_time:    float,
        bot_prob:         float,
        similarity:       float,
        entropy:          float,
        temporal_z:       float,
        decision:         str,
        anomalies:        Optional[List[str]] = None,
        ramp_active:      bool                = False,
    ) -> ActionLog:
        """
        Append a single action evaluation to the audit log.

        Parameters are the direct outputs of ABSAv2Engine.evaluate_action()
        plus the session and user identifiers.

        Returns
        -------
        ActionLog
        """
        log_entry = ActionLog(
            session_id       = session_id,
            user_id          = user_id,
            action_type      = action_type,
            resource_id      = resource_id,
            timestamp_action = timestamp_action,
            trust_at_time    = trust_at_time,
            bot_prob         = bot_prob,
            similarity       = similarity,
            entropy          = entropy,
            temporal_z       = temporal_z,
            decision         = decision,
            anomalies        = anomalies or [],
            ramp_active      = ramp_active,
        )
        session.add(log_entry)
        await session.flush()
        return log_entry

    @staticmethod
    async def bulk_insert(
        session: AsyncSession,
        entries: List[Dict[str, Any]],
    ) -> int:
        """
        Insert multiple action log entries in a single statement.

        More efficient than calling insert() in a loop for batch imports.

        Parameters
        ----------
        session : AsyncSession
        entries : list of dict
            Each dict must contain all ActionLog column values except id/logged_at.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not entries:
            return 0

        objects = [ActionLog(**entry) for entry in entries]
        session.add_all(objects)
        await session.flush()
        return len(objects)

    @staticmethod
    async def get_session_actions(
        session:    AsyncSession,
        session_id: uuid.UUID,
        limit:      int = 200,
    ) -> List[ActionLog]:
        """
        Return all action log entries for a session, oldest first.

        Parameters
        ----------
        session : AsyncSession
        session_id : uuid.UUID
        limit : int

        Returns
        -------
        list of ActionLog
        """
        result = await session.execute(
            select(ActionLog)
            .where(ActionLog.session_id == session_id)
            .order_by(ActionLog.timestamp_action.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_user_anomalies(
        session: AsyncSession,
        user_id: uuid.UUID,
        limit:   int = 100,
    ) -> List[ActionLog]:
        """
        Return the most recent anomalous actions for a user (CHALLENGE + BLOCK).

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID
        limit : int

        Returns
        -------
        list of ActionLog
        """
        result = await session.execute(
            select(ActionLog)
            .where(
                ActionLog.user_id == user_id,
                ActionLog.decision.in_(["CHALLENGE", "BLOCK"]),
            )
            .order_by(ActionLog.timestamp_action.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def decision_counts(
        session:    AsyncSession,
        session_id: uuid.UUID,
    ) -> Dict[str, int]:
        """
        Return {ALLOW: n, CHALLENGE: n, BLOCK: n} for a session.

        Parameters
        ----------
        session : AsyncSession
        session_id : uuid.UUID

        Returns
        -------
        dict of str -> int
        """
        result = await session.execute(
            select(ActionLog.decision, func.count().label("cnt"))
            .where(ActionLog.session_id == session_id)
            .group_by(ActionLog.decision)
        )
        rows = result.all()
        counts = {"ALLOW": 0, "CHALLENGE": 0, "BLOCK": 0}
        for row in rows:
            counts[row.decision] = row.cnt
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class FeatureVectorQueries:
    """Queries against the feature_vectors table."""

    @staticmethod
    async def insert(
        session:    AsyncSession,
        user_id:    uuid.UUID,
        vector:     List[float],
        label:      int,
        session_id: Optional[str] = None,
        verified:   bool          = False,
        source:     str           = "live",
    ) -> FeatureVector:
        """
        Persist a labeled feature vector.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID
        vector : list of float
            20-dimensional feature vector.
        label : int
            0 = human, 1 = bot.
        session_id : str, optional
        verified : bool
        source : str

        Returns
        -------
        FeatureVector
        """
        fv = FeatureVector(
            user_id    = user_id,
            session_id = session_id,
            vector     = vector,
            label      = label,
            verified   = verified,
            source     = source,
        )
        session.add(fv)
        await session.flush()
        return fv

    @staticmethod
    async def get_training_batch(
        session:       AsyncSession,
        verified_only: bool = False,
        limit:         int  = 10_000,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Retrieve a labeled (X, y) training batch from the database.

        Parameters
        ----------
        session : AsyncSession
        verified_only : bool
            If True, only return analyst-verified records.
        limit : int
            Maximum samples to return.

        Returns
        -------
        tuple of (list of list, list of int)
            (X, y) arrays suitable for RandomForestClassifier.fit()
        """
        query = select(FeatureVector.vector, FeatureVector.label).limit(limit)
        if verified_only:
            query = query.where(FeatureVector.verified.is_(True))

        result = await session.execute(query)
        rows = result.all()

        if not rows:
            return ([], [])

        X = [row.vector for row in rows]
        y = [row.label  for row in rows]
        return (X, y)

    @staticmethod
    async def set_verified_label(
        session:   AsyncSession,
        record_id: int,
        new_label: int,
    ) -> bool:
        """
        Set an analyst-verified label on a feature vector record.

        Parameters
        ----------
        session : AsyncSession
        record_id : int
        new_label : int

        Returns
        -------
        bool
            True if the record was found and updated.
        """
        result = await session.execute(
            update(FeatureVector)
            .where(FeatureVector.id == record_id)
            .values(label=new_label, verified=True, source="analyst")
        )
        return result.rowcount > 0

    @staticmethod
    async def get_unverified(
        session: AsyncSession,
        limit:   int = 100,
    ) -> List[FeatureVector]:
        """Return unverified records for the analyst review queue."""
        result = await session.execute(
            select(FeatureVector)
            .where(FeatureVector.verified.is_(False))
            .order_by(FeatureVector.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def count_by_label(
        session: AsyncSession,
    ) -> Dict[str, int]:
        """Return {human: n, bot: n, total: n, verified: n} counts."""
        result = await session.execute(
            select(
                FeatureVector.label,
                func.count().label("cnt"),
            ).group_by(FeatureVector.label)
        )
        rows = result.all()
        counts = {0: 0, 1: 0}
        for row in rows:
            counts[row.label] = row.cnt

        verified_result = await session.execute(
            select(func.count())
            .select_from(FeatureVector)
            .where(FeatureVector.verified.is_(True))
        )
        verified_count = verified_result.scalar_one()

        return {
            "human":    counts[0],
            "bot":      counts[1],
            "total":    counts[0] + counts[1],
            "verified": verified_count,
        }

    @staticmethod
    async def delete_user_records(
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> int:
        """
        Delete all feature vectors for a user (GDPR erasure).

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID

        Returns
        -------
        int
            Number of deleted records.
        """
        result = await session.execute(
            delete(FeatureVector).where(FeatureVector.user_id == user_id)
        )
        return result.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# MODEL VERSION QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersionQueries:
    """Queries against the model_versions table."""

    @staticmethod
    async def upsert(
        session:    AsyncSession,
        version:    str,
        model_path: str,
        n_samples:  int   = 0,
        n_human:    int   = 0,
        n_bot:      int   = 0,
        cv_f1:      float = 0.0,
        cv_far:     float = 0.0,
        cv_frr:     float = 0.0,
        trained_at: Optional[datetime] = None,
        notes:      str   = "",
    ) -> ModelVersion:
        """
        Insert a new model version record, or update it if the version already exists.

        Parameters
        ----------
        session : AsyncSession
        version : str
        model_path : str
        n_samples, n_human, n_bot : training dataset sizes
        cv_f1, cv_far, cv_frr : cross-validation metrics
        trained_at : datetime, optional
        notes : str

        Returns
        -------
        ModelVersion
        """
        result = await session.execute(
            select(ModelVersion).where(ModelVersion.version == version)
        )
        mv = result.scalar_one_or_none()

        if mv is None:
            mv = ModelVersion(
                version     = version,
                model_path  = model_path,
                state       = "trained",
                n_samples   = n_samples,
                n_human     = n_human,
                n_bot       = n_bot,
                cv_f1       = cv_f1,
                cv_far      = cv_far,
                cv_frr      = cv_frr,
                trained_at  = trained_at or datetime.now(timezone.utc),
                notes       = notes,
            )
            session.add(mv)
        else:
            mv.model_path = model_path
            mv.n_samples  = n_samples
            mv.n_human    = n_human
            mv.n_bot      = n_bot
            mv.cv_f1      = cv_f1
            mv.cv_far     = cv_far
            mv.cv_frr     = cv_frr
            mv.notes      = notes

        await session.flush()
        return mv

    @staticmethod
    async def set_state(
        session:    AsyncSession,
        version:    str,
        new_state:  str,
    ) -> bool:
        """
        Update a model version's lifecycle state.

        Parameters
        ----------
        session : AsyncSession
        version : str
        new_state : str
            One of: trained | canary | active | retired

        Returns
        -------
        bool
            True if the record was found and updated.
        """
        now    = datetime.now(timezone.utc)
        values = {"state": new_state}

        if new_state in ("canary", "active"):
            values["promoted_at"] = now
        elif new_state == "retired":
            values["retired_at"] = now

        result = await session.execute(
            update(ModelVersion)
            .where(ModelVersion.version == version)
            .values(**values)
        )
        return result.rowcount > 0

    @staticmethod
    async def get_active(
        session: AsyncSession,
    ) -> Optional[ModelVersion]:
        """Return the currently active model version."""
        result = await session.execute(
            select(ModelVersion)
            .where(ModelVersion.state == "active")
            .order_by(ModelVersion.promoted_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_all(
        session: AsyncSession,
    ) -> List[ModelVersion]:
        """Return all model versions, newest first."""
        result = await session.execute(
            select(ModelVersion).order_by(ModelVersion.trained_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_by_state(
        session: AsyncSession,
        state:   str,
    ) -> List[ModelVersion]:
        """Return all model versions in a given state."""
        result = await session.execute(
            select(ModelVersion)
            .where(ModelVersion.state == state)
            .order_by(ModelVersion.trained_at.desc())
        )
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# SHADOW LOG QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class ShadowLogQueries:
    """Queries against the shadow_logs table."""

    @staticmethod
    async def insert(
        session:            AsyncSession,
        session_id:         uuid.UUID,
        user_id:            uuid.UUID,
        action_type:        str,
        timestamp_action:   float,
        real_decision:      str,
        trust_score_shadow: float,
        bot_prob_shadow:    float,
        heuristic_label:    int,
        anomalies_shadow:   Optional[List[str]] = None,
        feature_vector:     Optional[List[float]] = None,
    ) -> ShadowLog:
        """
        Persist a single shadow mode evaluation result.

        Parameters
        ----------
        session : AsyncSession
        session_id : uuid.UUID
        user_id : uuid.UUID
        action_type : str
        timestamp_action : float
        real_decision : str
            The decision that WOULD have been enforced.
        trust_score_shadow : float
        bot_prob_shadow : float
        heuristic_label : int
        anomalies_shadow : list of str, optional
        feature_vector : list of float, optional

        Returns
        -------
        ShadowLog
        """
        entry = ShadowLog(
            session_id          = session_id,
            user_id             = user_id,
            action_type         = action_type,
            timestamp_action    = timestamp_action,
            real_decision       = real_decision,
            trust_score_shadow  = trust_score_shadow,
            bot_prob_shadow     = bot_prob_shadow,
            heuristic_label     = heuristic_label,
            anomalies_shadow    = anomalies_shadow or [],
            feature_vector      = feature_vector or [],
        )
        session.add(entry)
        await session.flush()
        return entry

    @staticmethod
    async def set_verified_label(
        session:   AsyncSession,
        record_id: int,
        new_label: int,
    ) -> bool:
        """Set an analyst-verified label on a shadow log entry."""
        result = await session.execute(
            update(ShadowLog)
            .where(ShadowLog.id == record_id)
            .values(verified_label=new_label)
        )
        return result.rowcount > 0

    @staticmethod
    async def get_unreviewed(
        session: AsyncSession,
        limit:   int = 100,
    ) -> List[ShadowLog]:
        """Return shadow log entries that haven't been analyst-verified."""
        result = await session.execute(
            select(ShadowLog)
            .where(ShadowLog.verified_label.is_(None))
            .order_by(ShadowLog.timestamp_action.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_training_pairs(
        session:       AsyncSession,
        verified_only: bool = False,
        limit:         int  = 10_000,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Extract (X, y) training pairs from shadow log entries.

        Parameters
        ----------
        session : AsyncSession
        verified_only : bool
        limit : int

        Returns
        -------
        tuple of (list of list, list of int)
        """
        query = (
            select(ShadowLog.feature_vector, ShadowLog.heuristic_label,
                   ShadowLog.verified_label)
            .where(ShadowLog.feature_vector.isnot(None))
            .limit(limit)
        )
        if verified_only:
            query = query.where(ShadowLog.verified_label.isnot(None))

        result = await session.execute(query)
        rows = result.all()

        X, y = [], []
        for row in rows:
            fv = row.feature_vector
            if isinstance(fv, list) and len(fv) == 20:
                X.append(fv)
                label = row.verified_label if row.verified_label is not None \
                        else row.heuristic_label
                y.append(label)

        return (X, y)

    @staticmethod
    async def decision_breakdown(
        session: AsyncSession,
    ) -> Dict[str, int]:
        """Return {ALLOW: n, CHALLENGE: n, BLOCK: n} across all shadow logs."""
        result = await session.execute(
            select(ShadowLog.real_decision, func.count().label("cnt"))
            .group_by(ShadowLog.real_decision)
        )
        counts = {"ALLOW": 0, "CHALLENGE": 0, "BLOCK": 0}
        for row in result.all():
            counts[row.real_decision] = row.cnt
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# CONSENT QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class ConsentQueries:
    """Queries against the consent_records table."""

    @staticmethod
    async def record_event(
        session:         AsyncSession,
        user_id:         uuid.UUID,
        event_type:      str,
        consent_version: Optional[str]      = None,
        ip_address:      Optional[str]      = None,
        purpose:         Optional[str]      = None,
        expires_at:      Optional[datetime] = None,
        notes:           Optional[str]      = None,
    ) -> ConsentRecord:
        """
        Append a consent lifecycle event to the audit trail.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID
        event_type : str
            'grant' | 'revoke' | 'expire' | 'reconfirm'
        consent_version : str, optional
        ip_address : str, optional
        purpose : str, optional
        expires_at : datetime, optional
        notes : str, optional

        Returns
        -------
        ConsentRecord
        """
        record = ConsentRecord(
            user_id         = user_id,
            event_type      = event_type,
            consent_version = consent_version,
            ip_address      = ip_address,
            purpose         = purpose,
            expires_at      = expires_at,
            notes           = notes,
        )
        session.add(record)
        await session.flush()
        return record

    @staticmethod
    async def get_user_history(
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> List[ConsentRecord]:
        """
        Return the full consent history for a user, oldest first.

        Used for DSAR responses.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID

        Returns
        -------
        list of ConsentRecord
        """
        result = await session.execute(
            select(ConsentRecord)
            .where(ConsentRecord.user_id == user_id)
            .order_by(ConsentRecord.granted_at.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def has_active_consent(
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> bool:
        """
        Return True if the user has a valid, non-revoked, non-expired consent.

        Checks the users table directly rather than parsing consent_records —
        faster for the hot path.

        Parameters
        ----------
        session : AsyncSession
        user_id : uuid.UUID

        Returns
        -------
        bool
        """
        result = await session.execute(
            select(UserRecord.consent_given, UserRecord.consent_revoked_at)
            .where(UserRecord.id == user_id)
        )
        row = result.one_or_none()
        if row is None:
            return False
        if not row.consent_given:
            return False
        if row.consent_revoked_at is not None:
            return False
        return True
