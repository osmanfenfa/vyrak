"""
zelvryn/profiles/drift_detector.py
=====================================
DriftDetector — detects when a user's recent behaviour has diverged significantly
from their historical profile, distinguishing legitimate behaviour drift from attack.

The drift vs attack distinction problem
-----------------------------------------
Zelvryn's anomaly engine flags sessions where behaviour diverges from the stored
profile. But legitimate users change their behaviour too: a junior developer
promoted to senior may start performing admin operations they never touched before.
A researcher who switches projects changes their document access patterns entirely.

Without drift detection, legitimate role changes produce constant CHALLENGE
decisions for weeks until the EWMA profile slowly catches up. With drift detection,
the system identifies the change as gradual and consistent (drift) rather than
sudden and erratic (attack), and accelerates profile adaptation.

Detection method: KL-divergence on action frequency vectors
-------------------------------------------------------------
The KL-divergence (Kullback-Leibler divergence) measures how different one
probability distribution is from another:

  KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

Where:
  P = recent action frequency distribution (last N actions)
  Q = historical profile distribution (full stored profile)

KL-divergence properties:
  KL = 0.0   : identical distributions (no drift)
  KL = small : minor behavioural shift (normal variation)
  KL = large : significant drift (role change or attack)

The detector uses a sliding window of recent actions and compares it to the
stored profile. Drift is declared when KL-divergence exceeds a threshold
across multiple consecutive windows (avoiding false alarms from temporary spikes).

Drift vs attack differentiation
---------------------------------
  Drift    : gradual increase in KL over many sessions, action distribution
             shifts to a new stable state, timing patterns remain similar
  Attack   : sudden KL spike in a single session, timing anomaly co-occurs,
             sensitive action proportion spikes

The DriftDetector signals that drift may be occurring. The ABSAv2Engine makes
the final determination — if drift is accompanied by timing anomalies, it is
more likely an attack and the engine penalises trust accordingly.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from core.profile import BehavioralProfile
from core.action import UserAction
from core.math_utils import coefficient_of_variation

logger = logging.getLogger(__name__)

# Minimum KL-divergence to classify as significant drift
DEFAULT_KL_THRESHOLD: float = 0.30

# Number of consecutive drift signals before a soft-reset is recommended
DEFAULT_CONSECUTIVE_DRIFT_THRESHOLD: int = 5

# Recent window size (number of actions) for drift measurement
DEFAULT_RECENT_WINDOW: int = 30

# Smoothing constant for KL computation to prevent log(0)
_KL_EPSILON: float = 1e-8


@dataclass
class DriftReport:
    """
    Result of a drift detection analysis.

    Attributes
    ----------
    user_id : str
    kl_divergence : float
        KL divergence between recent and historical action distributions.
    is_drifting : bool
        True if KL divergence exceeds the threshold.
    consecutive_drift_count : int
        How many consecutive windows have shown drift.
    recommend_soft_reset : bool
        True when consecutive drift exceeds the reset threshold — the profile
        should be partially reset to accelerate adaptation.
    drift_direction : dict
        {action_type: delta} showing which action types increased/decreased.
    timing_changed : bool
        Whether timing patterns also changed (drift vs attack signal).
    analysed_at : float
    """
    user_id:                   str
    kl_divergence:             float              = 0.0
    is_drifting:               bool               = False
    consecutive_drift_count:   int                = 0
    recommend_soft_reset:      bool               = False
    drift_direction:           Dict[str, float]   = field(default_factory=dict)
    timing_changed:            bool               = False
    analysed_at:               float              = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"DriftReport(user={self.user_id} "
            f"KL={self.kl_divergence:.3f} "
            f"drifting={self.is_drifting} "
            f"consecutive={self.consecutive_drift_count} "
            f"reset_recommended={self.recommend_soft_reset})"
        )


class DriftDetector:
    """
    Detects gradual behavioural drift in user profiles.

    Maintains a per-user history of KL-divergence readings to identify
    sustained drift patterns vs temporary fluctuations.

    Parameters
    ----------
    kl_threshold : float, optional
        KL-divergence above which a single window is flagged as drifting.
        Default 0.30 (roughly 30% distributional shift).
    consecutive_drift_threshold : int, optional
        Number of consecutive drift signals before a soft-reset is recommended.
        Default 5.
    recent_window : int, optional
        Number of recent actions used to compute the "current" distribution.
        Default 30.
    timing_cv_change_threshold : float, optional
        Minimum change in interval coefficient of variation to flag timing drift.
        Default 0.3 (30% change in timing variability).
    """

    def __init__(
        self,
        kl_threshold:                float = DEFAULT_KL_THRESHOLD,
        consecutive_drift_threshold: int   = DEFAULT_CONSECUTIVE_DRIFT_THRESHOLD,
        recent_window:               int   = DEFAULT_RECENT_WINDOW,
        timing_cv_change_threshold:  float = 0.30,
    ) -> None:
        self.kl_threshold               = kl_threshold
        self.consecutive_threshold      = consecutive_drift_threshold
        self.recent_window              = recent_window
        self.timing_cv_change_threshold = timing_cv_change_threshold

        # {user_id: deque of (kl_value, timestamp)}
        self._kl_history: Dict[str, Deque[Tuple[float, float]]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def check_drift(self, profile: BehavioralProfile) -> DriftReport:
        """
        Analyse a profile for behavioural drift.

        Compares the distribution of the last `recent_window` recorded actions
        against the full historical profile distribution.

        Parameters
        ----------
        profile : BehavioralProfile

        Returns
        -------
        DriftReport
        """
        user_id = profile.user_id

        # Need enough data to measure drift
        if profile.vector_count < self.recent_window * 2:
            return DriftReport(
                user_id       = user_id,
                kl_divergence = 0.0,
                is_drifting   = False,
            )

        # ── Compute recent action distribution ────────────────────────────
        recent_actions = list(profile.action_history)[-self.recent_window:]
        recent_dist    = self._action_distribution(recent_actions)

        # ── Get historical distribution from full profile ─────────────────
        historical_dist = profile.get_mean_vector()

        # ── KL-divergence ─────────────────────────────────────────────────
        kl = self._kl_divergence(recent_dist, historical_dist)

        # ── Update per-user history ───────────────────────────────────────
        history = self._kl_history.setdefault(user_id, deque(maxlen=20))
        history.append((kl, time.time()))

        # Count consecutive above-threshold readings
        consecutive = self._count_consecutive_drift(history)

        is_drifting         = kl > self.kl_threshold
        recommend_reset     = consecutive >= self.consecutive_threshold

        # ── Drift direction ───────────────────────────────────────────────
        drift_direction = self._compute_drift_direction(
            recent_dist, historical_dist
        )

        # ── Timing change detection ───────────────────────────────────────
        timing_changed = self._check_timing_change(profile, recent_actions)

        report = DriftReport(
            user_id                 = user_id,
            kl_divergence           = round(kl, 4),
            is_drifting             = is_drifting,
            consecutive_drift_count = consecutive,
            recommend_soft_reset    = recommend_reset,
            drift_direction         = drift_direction,
            timing_changed          = timing_changed,
        )

        if is_drifting:
            logger.info(
                "Drift detected: %s", report.summary()
            )

        return report

    def apply_soft_reset(
        self,
        profile: BehavioralProfile,
        blend_weight: float = 0.3,
    ) -> BehavioralProfile:
        """
        Apply a soft profile reset by blending the recent distribution into
        the historical profile with a higher weight.

        Unlike a full reset (which would discard all history and trigger cold
        start), a soft reset accelerates EWMA adaptation while preserving the
        overall profile structure.

        After a soft reset:
          - The EWMA vector shifts toward the recent distribution
          - The adaptive threshold is recalculated
          - The KL history for this user is cleared

        Parameters
        ----------
        profile : BehavioralProfile
        blend_weight : float, optional
            Weight given to the recent distribution in the blend.
            Default 0.3 (recent behaviour gets 30% weight in the new profile).

        Returns
        -------
        BehavioralProfile
            The modified profile (modified in place).
        """
        user_id = profile.user_id
        recent_actions = list(profile.action_history)[-self.recent_window:]

        if not recent_actions:
            return profile

        recent_dist  = self._action_distribution(recent_actions)
        current_vec  = profile.get_mean_vector()

        # Blend: new_vec = (1 - w) * historical + w * recent
        blended = [
            (1 - blend_weight) * h + blend_weight * r
            for h, r in zip(current_vec, recent_dist)
        ]

        # Normalise
        total = sum(blended)
        if total > 0:
            blended = [v / total for v in blended]

        # Apply to profile EWMA vector
        if profile.ewma_vector is not None:
            profile.ewma_vector = blended

        # Update vector_sum to reflect the blend
        total_count = profile.vector_count
        profile.vector_sum = [v * total_count for v in blended]

        # Clear KL history to restart drift monitoring
        self._kl_history.pop(user_id, None)

        logger.info(
            "Soft reset applied to user=%s (blend_weight=%.2f)",
            user_id, blend_weight
        )
        return profile

    def kl_trend(self, user_id: str, n: int = 10) -> List[float]:
        """
        Return the last N KL-divergence values for a user.

        Used by the dashboard to plot drift trends over time.

        Parameters
        ----------
        user_id : str
        n : int

        Returns
        -------
        list of float
        """
        history = self._kl_history.get(user_id, deque())
        values  = [kl for kl, _ in history]
        return values[-n:]

    def clear_history(self, user_id: str) -> None:
        """Clear the KL history for a user (called on profile deletion)."""
        self._kl_history.pop(user_id, None)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _action_distribution(actions: List[UserAction]) -> List[float]:
        """
        Compute a normalised action type frequency vector from a list of actions.
        """
        n_types = UserAction.type_count()
        counts  = [0.0] * n_types
        for a in actions:
            counts[a.to_index()] += 1.0
        total = sum(counts)
        if total == 0:
            return [1.0 / n_types] * n_types
        return [c / total for c in counts]

    @staticmethod
    def _kl_divergence(p: List[float], q: List[float]) -> float:
        """
        Compute KL-divergence KL(P || Q).

        Both P and Q must have the same length and sum to 1.0 (probability
        distributions). Epsilon smoothing prevents log(0).

        Parameters
        ----------
        p : list of float
            Recent distribution.
        q : list of float
            Historical distribution.

        Returns
        -------
        float
            KL-divergence >= 0.
        """
        kl = 0.0
        for pi, qi in zip(p, q):
            pi_s = pi + _KL_EPSILON
            qi_s = qi + _KL_EPSILON
            kl  += pi_s * math.log(pi_s / qi_s)
        return max(0.0, kl)

    def _count_consecutive_drift(
        self,
        history: Deque[Tuple[float, float]],
    ) -> int:
        """
        Count the number of consecutive most-recent drift signals.

        Counts backwards from the most recent reading until a non-drift
        reading is encountered.
        """
        count = 0
        for kl, _ in reversed(list(history)):
            if kl > self.kl_threshold:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _compute_drift_direction(
        recent: List[float],
        historical: List[float],
    ) -> Dict[str, float]:
        """
        Compute how each action type's frequency changed (recent - historical).

        Positive delta = this action type is used more than historically.
        Negative delta = used less.
        """
        direction = {}
        for i, action_type in enumerate(UserAction.ACTION_TYPES):
            delta = recent[i] - historical[i]
            if abs(delta) > 0.02:   # only report meaningful changes
                direction[action_type] = round(delta, 4)
        return direction

    def _check_timing_change(
        self,
        profile: BehavioralProfile,
        recent_actions: List[UserAction],
    ) -> bool:
        """
        Check whether timing patterns have changed alongside action distribution.

        A true drift (role change) often changes timing too: an admin doing
        batch operations may act faster than a reader browsing documents.
        An attack is more likely to show timing drift without action drift.

        Computes the CV of intervals from recent actions and compares to the
        profile's historical CV.

        Parameters
        ----------
        profile : BehavioralProfile
        recent_actions : list of UserAction

        Returns
        -------
        bool
            True if timing has changed significantly.
        """
        if len(recent_actions) < 5:
            return False

        # Compute recent intervals
        recent_intervals = [
            abs(recent_actions[i].timestamp - recent_actions[i-1].timestamp)
            for i in range(1, len(recent_actions))
        ]

        if len(recent_intervals) < 3:
            return False

        recent_cv   = coefficient_of_variation(recent_intervals)
        profile_cv  = coefficient_of_variation(profile.time_intervals[-50:]) \
                      if len(profile.time_intervals) >= 5 else 0.5

        cv_change = abs(recent_cv - profile_cv)
        return cv_change > self.timing_cv_change_threshold

    def __repr__(self) -> str:
        return (
            f"DriftDetector("
            f"kl_threshold={self.kl_threshold}, "
            f"consecutive_threshold={self.consecutive_threshold}, "
            f"window={self.recent_window})"
        )
