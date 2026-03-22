"""
DriftDetector — detects when a user's recent behaviour has diverged significantly
from their historical profile, distinguishing legitimate behaviour drift from attack.
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

    # PRIMARY INTERFACE

    def check_drift(self, profile: BehavioralProfile) -> DriftReport:
        """
        Analyse a profile for behavioural drift.
        """
        user_id = profile.user_id

        # Need enough data to measure drift
        if profile.vector_count < self.recent_window * 2:
            return DriftReport(
                user_id       = user_id,
                kl_divergence = 0.0,
                is_drifting   = False,
            )

        # Compute recent action distribution 
        recent_actions = list(profile.action_history)[-self.recent_window:]
        recent_dist    = self._action_distribution(recent_actions)

        # Get historical distribution from full profile
        historical_dist = profile.get_mean_vector()

        # KL-divergence
        kl = self._kl_divergence(recent_dist, historical_dist)

        # Update per-user history
        history = self._kl_history.setdefault(user_id, deque(maxlen=20))
        history.append((kl, time.time()))

        # Count consecutive above-threshold readings
        consecutive = self._count_consecutive_drift(history)

        is_drifting         = kl > self.kl_threshold
        recommend_reset     = consecutive >= self.consecutive_threshold

        # ── Drift direction 
        drift_direction = self._compute_drift_direction(
            recent_dist, historical_dist
        )

        # ── Timing change detection
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

        if profile.ewma_vector is not None:
            profile.ewma_vector = blended

        total_count = profile.vector_count
        profile.vector_sum = [v * total_count for v in blended]
        self._kl_history.pop(user_id, None)

        logger.info(
            "Soft reset applied to user=%s (blend_weight=%.2f)",
            user_id, blend_weight
        )
        return profile

    def kl_trend(self, user_id: str, n: int = 10) -> List[float]:
        """
        Return the last N KL-divergence values for a user.
        """
        history = self._kl_history.get(user_id, deque())
        values  = [kl for kl, _ in history]
        return values[-n:]

    def clear_history(self, user_id: str) -> None:
        """Clear the KL history for a user (called on profile deletion)."""
        self._kl_history.pop(user_id, None)


    # PRIVATE HELPERS
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