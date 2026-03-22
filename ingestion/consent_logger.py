"""
zelvryn/ingestion/consent_logger.py
=====================================
ConsentGatedLogger — intercepts all action recording and enforces consent
requirements before any behavioural data flows to the Zelvryn evaluation engine.

GDPR compliance context
------------------------
Under GDPR Article 6 and Recital 47 (legitimate interests), continuous
behavioural monitoring for security purposes can be lawful without explicit
consent if it is proportionate to the security risk and users are clearly
informed. However, many deployments — particularly in consumer-facing
applications — require explicit consent before any monitoring occurs.

ConsentGatedLogger is designed for deployments that require explicit consent.
It gates every call to ABSAv2Engine.evaluate_action() behind a consent check.
If no active consent record exists for the user, the action is logged as
skipped (not evaluated) and the application receives a pass-through ALLOW.

Consent lifecycle
-----------------
  grant_consent(user_id)    — called at account creation or when user accepts
                              the privacy notice. Creates a ConsentRecord.
  revoke_consent(user_id)   — called when user opts out. Marks consent as
                              revoked and triggers a profile deletion request.
  has_consent(user_id)      — called before every evaluate_action(). Returns
                              True only if an active, unexpired record exists.
  get_consent_record(user_id) — returns the full ConsentRecord for audit.

Storage
-------
Consent records are stored in-memory in this implementation. In production,
they should be persisted to the `consent_records` table in PostgreSQL via
db/queries.py. The in-memory store is sufficient for unit testing and for
deployments where the process is long-lived (the API server).

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.action import UserAction
from core.engine import ABSAv2Engine, DecisionResult
from core.constants import TRUST_INITIAL

logger = logging.getLogger(__name__)

# Default consent expiry: 12 months in seconds (GDPR best practice is to
# re-confirm consent annually)
DEFAULT_CONSENT_EXPIRY_SECONDS: int = 365 * 24 * 3600


@dataclass
class ConsentRecord:
    """
    A consent record for a single user.

    Attributes
    ----------
    user_id : str
    granted_at : float
        Unix timestamp when consent was granted.
    expires_at : float
        Unix timestamp when consent expires and must be re-confirmed.
    revoked_at : float or None
        Unix timestamp when consent was revoked. None = still active.
    consent_version : str
        Version of the privacy notice the user consented to. Used to detect
        when the notice has changed and re-confirmation is needed.
    ip_at_grant : str
        IP address of the request that granted consent (for audit trail).
    purpose : str
        Statement of purpose shown to the user at consent time.
    """
    user_id:          str
    granted_at:       float
    expires_at:       float
    revoked_at:       Optional[float]  = None
    consent_version:  str              = "1.0"
    ip_at_grant:      str              = ""
    purpose:          str              = "Security monitoring for session authentication"

    def is_active(self) -> bool:
        """Return True if consent is granted and not revoked or expired."""
        if self.revoked_at is not None:
            return False
        return time.time() < self.expires_at

    def is_expired(self) -> bool:
        """Return True if consent has expired (not yet re-confirmed)."""
        return self.revoked_at is None and time.time() >= self.expires_at

    def days_until_expiry(self) -> float:
        """Return days until expiry. Negative means already expired."""
        return (self.expires_at - time.time()) / 86400.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id":         self.user_id,
            "granted_at":      self.granted_at,
            "expires_at":      self.expires_at,
            "revoked_at":      self.revoked_at,
            "consent_version": self.consent_version,
            "ip_at_grant":     self.ip_at_grant,
            "purpose":         self.purpose,
            "is_active":       self.is_active(),
            "days_until_expiry": round(self.days_until_expiry(), 1),
        }

    def __repr__(self) -> str:
        status = "active" if self.is_active() else ("revoked" if self.revoked_at else "expired")
        return (
            f"ConsentRecord(user={self.user_id!r}, "
            f"status={status}, "
            f"version={self.consent_version!r})"
        )


class ConsentGatedLogger:
    """
    Intercepts action evaluation requests and enforces consent requirements.

    Wraps ABSAv2Engine to ensure no user data is processed without
    an active consent record. Provides full consent lifecycle management.

    Parameters
    ----------
    engine : ABSAv2Engine
        The evaluation engine to gate behind consent checks.
    consent_expiry_seconds : int, optional
        How long a consent record remains valid. Default 12 months.
    strict_mode : bool, optional
        If True, raises ConsentError when consent is absent (blocks the
        request entirely). If False (default), returns a pass-through ALLOW
        response and logs a warning. Choose based on your legal requirement.

    Attributes
    ----------
    _consent_store : dict
        {user_id: ConsentRecord}. In-memory store. Replace with DB calls in
        production by overriding _load_consent() and _save_consent().
    skipped_actions : int
        Total number of actions skipped due to absent consent.
    evaluated_actions : int
        Total number of actions successfully passed to the engine.
    """

    def __init__(
        self,
        engine: ABSAv2Engine,
        consent_expiry_seconds: int = DEFAULT_CONSENT_EXPIRY_SECONDS,
        strict_mode: bool = False,
    ) -> None:
        self.engine                   = engine
        self.consent_expiry_seconds   = consent_expiry_seconds
        self.strict_mode              = strict_mode

        self._consent_store: Dict[str, ConsentRecord] = {}
        self.skipped_actions:   int = 0
        self.evaluated_actions: int = 0

        logger.debug(
            "ConsentGatedLogger initialised: user=%s strict=%s expiry=%dd",
            engine.profile.user_id,
            strict_mode,
            consent_expiry_seconds // 86400
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONSENT LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def grant_consent(
        self,
        user_id: str,
        consent_version: str = "1.0",
        ip_at_grant: str = "",
        purpose: str = "Security monitoring for session authentication",
        expiry_override_seconds: Optional[int] = None,
    ) -> ConsentRecord:
        """
        Record that a user has granted consent for behavioural monitoring.

        This should be called:
          - When the user accepts the privacy notice at account creation
          - When the user re-confirms consent after it has expired
          - When the user opts in from the account settings page

        Parameters
        ----------
        user_id : str
        consent_version : str, optional
            Version of the privacy notice accepted. Default "1.0".
        ip_at_grant : str, optional
            IP address of the grant request (for audit log).
        purpose : str, optional
            Plain-language statement of monitoring purpose.
        expiry_override_seconds : int, optional
            Override the default expiry. Useful for test environments.

        Returns
        -------
        ConsentRecord
            The newly created consent record.
        """
        now = time.time()
        expiry = now + (expiry_override_seconds or self.consent_expiry_seconds)

        record = ConsentRecord(
            user_id         = user_id,
            granted_at      = now,
            expires_at      = expiry,
            revoked_at      = None,
            consent_version = consent_version,
            ip_at_grant     = ip_at_grant,
            purpose         = purpose,
        )
        self._save_consent(user_id, record)

        logger.info(
            "Consent granted: user=%s version=%s expires=%s",
            user_id, consent_version,
            time.strftime("%Y-%m-%d", time.localtime(expiry))
        )
        return record

    def revoke_consent(
        self,
        user_id: str,
        trigger_erasure: bool = True,
    ) -> bool:
        """
        Record that a user has revoked consent for behavioural monitoring.

        After revocation:
          - has_consent() returns False immediately
          - All future evaluate_action() calls are skipped
          - If trigger_erasure=True, the user's behavioural profile data
            should be queued for deletion (profile/redis_store.py delete())

        Parameters
        ----------
        user_id : str
        trigger_erasure : bool, optional
            If True, logs an erasure request. The actual deletion is handled
            by the API layer (DELETE /profile/{user_id}). Default True.

        Returns
        -------
        bool
            True if a consent record was found and revoked. False if no
            record existed (idempotent operation).
        """
        record = self._load_consent(user_id)
        if record is None:
            logger.warning(
                "revoke_consent called for user=%s but no consent record found.",
                user_id
            )
            return False

        record.revoked_at = time.time()
        self._save_consent(user_id, record)

        logger.info(
            "Consent revoked: user=%s (erasure_requested=%s)",
            user_id, trigger_erasure
        )

        if trigger_erasure:
            logger.info(
                "GDPR erasure requested for user=%s — "
                "API layer should call DELETE /profile/%s within 72 hours.",
                user_id, user_id
            )

        return True

    def has_consent(self, user_id: str) -> bool:
        """
        Return True if the user has an active, non-expired consent record.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        bool
        """
        record = self._load_consent(user_id)
        if record is None:
            return False
        return record.is_active()

    def get_consent_record(self, user_id: str) -> Optional[ConsentRecord]:
        """
        Return the consent record for a user, or None if none exists.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        ConsentRecord or None
        """
        return self._load_consent(user_id)

    def needs_reconfirmation(
        self,
        user_id: str,
        current_version: str = "1.0",
        warn_days_before: int = 30,
    ) -> Tuple[bool, str]:
        """
        Return whether the user needs to re-confirm consent.

        Re-confirmation is needed when:
          - The consent record is expired
          - The consent was granted for an older privacy notice version
          - The consent expires within warn_days_before days

        Parameters
        ----------
        user_id : str
        current_version : str
            Current version of the privacy notice.
        warn_days_before : int
            Days before expiry at which to start warning. Default 30.

        Returns
        -------
        tuple of (bool, str)
            (needs_reconf, reason). reason is empty string if no reconf needed.
        """
        record = self._load_consent(user_id)

        if record is None:
            return (True, "NO_CONSENT_RECORD")

        if not record.is_active():
            if record.revoked_at:
                return (True, "CONSENT_REVOKED")
            return (True, "CONSENT_EXPIRED")

        if record.consent_version != current_version:
            return (True, f"VERSION_CHANGED:{record.consent_version}->{current_version}")

        if record.days_until_expiry() < warn_days_before:
            return (True, f"EXPIRING_SOON:{record.days_until_expiry():.0f}_days")

        return (False, "")

    # ─────────────────────────────────────────────────────────────────────────
    # GATED EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_with_consent_check(
        self,
        action: UserAction,
    ) -> DecisionResult:
        """
        Evaluate an action only if the user has active consent.

        If consent is absent:
          - strict_mode=False: returns a pass-through ALLOW DecisionResult
            with anomalies=["CONSENT_ABSENT"] and logs a warning
          - strict_mode=True: raises ConsentError

        Parameters
        ----------
        action : UserAction

        Returns
        -------
        DecisionResult
        """
        user_id = self.engine.profile.user_id

        if not self.has_consent(user_id):
            self.skipped_actions += 1
            logger.warning(
                "Action skipped (no consent): user=%s action=%s",
                user_id, action.action_type
            )

            if self.strict_mode:
                raise ConsentError(
                    f"Cannot evaluate action for user={user_id!r}: "
                    f"no active consent record found."
                )

            # Return a benign pass-through result
            return self._passthrough_result(action, "CONSENT_ABSENT")

        self.evaluated_actions += 1
        return self.engine.evaluate_action(action)

    # ─────────────────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return evaluation and consent statistics.

        Returns
        -------
        dict
        """
        total = self.evaluated_actions + self.skipped_actions
        return {
            "user_id":           self.engine.profile.user_id,
            "evaluated_actions": self.evaluated_actions,
            "skipped_actions":   self.skipped_actions,
            "total_actions":     total,
            "consent_rate_pct":  round(
                100 * self.evaluated_actions / max(total, 1), 1
            ),
            "has_active_consent": self.has_consent(self.engine.profile.user_id),
        }

    def all_consent_records(self) -> List[ConsentRecord]:
        """Return all stored consent records (for admin audit)."""
        return list(self._consent_store.values())

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS (override to use PostgreSQL in production)
    # ─────────────────────────────────────────────────────────────────────────

    def _save_consent(self, user_id: str, record: ConsentRecord) -> None:
        """
        Persist a consent record.

        In-memory implementation. Override to save to PostgreSQL:
          await db.execute(
              "INSERT INTO consent_records ... ON CONFLICT DO UPDATE ...",
              record.to_dict()
          )
        """
        self._consent_store[user_id] = record

    def _load_consent(self, user_id: str) -> Optional[ConsentRecord]:
        """
        Load a consent record by user_id.

        In-memory implementation. Override to load from PostgreSQL:
          row = await db.fetchone(
              "SELECT * FROM consent_records WHERE user_id = $1", user_id
          )
          return ConsentRecord(**row) if row else None
        """
        return self._consent_store.get(user_id)

    @staticmethod
    def _passthrough_result(
        action: UserAction,
        anomaly_code: str,
    ) -> "DecisionResult":
        """
        Build a minimal DecisionResult for pass-through (consent absent) scenarios.
        """
        from core.engine import DecisionResult
        return DecisionResult(
            decision    = "ALLOW",
            trust_score = TRUST_INITIAL,
            bot_prob    = 0.0,
            similarity  = 1.0,
            entropy     = 0.0,
            temporal_z  = 0.0,
            anomalies   = [anomaly_code],
            ramp_active = True,
            session_id  = "",
            action_type = action.action_type,
            timestamp   = action.timestamp,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ConsentGatedLogger("
            f"user={self.engine.profile.user_id!r}, "
            f"evaluated={self.evaluated_actions}, "
            f"skipped={self.skipped_actions}, "
            f"strict={self.strict_mode})"
        )


# Type annotation fix for needs_reconfirmation return type
from typing import Tuple  # noqa: E402


class ConsentError(Exception):
    """
    Raised by ConsentGatedLogger in strict_mode when no consent record exists.

    Catch this in the API layer to return a 403 response with an appropriate
    message directing the user to the consent flow.
    """
    pass
