"""
zelvryn/coldstart/ramp_manager.py
===================================
RampUpManager — applies relaxed detection thresholds during the cold start
ramp-up period and manages the transition to full enforcement.

Why a ramp-up period is necessary
-----------------------------------
Even with a population prior, a new user's profile will deviate from the prior
simply because their personal behaviour is different from the population average.
A developer who predominantly uses admin and config actions will look anomalous
against a prior built from a mostly read-heavy user base. Without a ramp-up
period, this legitimate user would face constant CHALLENGE decisions before
their personal profile has accumulated enough data to reflect their real patterns.

The ramp-up period (first COLD_START_RAMP_ACTIONS verified actions) applies:
  - No cosine similarity check (too unstable with sparse history)
  - Raised ML confidence threshold (only the most obvious bots are flagged)
  - Raised temporal Z-score limit (new users haven't established a rhythm yet)
  - Doubled trust recovery rate (early anomalies shouldn't permanently damage score)

Smooth transition
-----------------
The ramp-up does not switch off abruptly at action N. Instead, thresholds
interpolate smoothly from their ramp-up values to their full-enforcement values
across the final 20% of the ramp-up period. This prevents a cliff edge where
a user who just crossed the threshold is immediately subjected to the full
detection regime.

Integration with ABSAv2Engine
------------------------------
The engine calls RampUpManager.is_warming_up(action_count) to decide which
threshold set to use. It never calls RampUpManager directly for penalties —
threshold selection happens inside evaluate_action() in engine.py.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import math
from typing import Tuple

from core.constants import (
    COLD_START_RAMP_ACTIONS,
    COLD_START_ML_THRESH,
    COLD_START_RECOVERY_RATE,
    COLD_START_TEMPORAL_LIMIT,
    ML_BOT_CONFIDENCE_THRESH,
    TEMPORAL_ZSCORE_LIMIT,
    TRUST_RECOVERY_RATE,
    SIMILARITY_THRESHOLD,
    SIMILARITY_ADAPTIVE_MIN,
    SIMILARITY_ADAPTIVE_MAX,
)

logger = logging.getLogger(__name__)

# Fraction of COLD_START_RAMP_ACTIONS over which thresholds interpolate
# from ramp values to full-enforcement values. Default: last 20% of ramp.
_INTERPOLATION_FRACTION: float = 0.20


class RampUpManager:
    """
    Manages cold start ramp-up thresholds for new user profiles.

    This class is stateless — all methods take the current action count as
    input rather than maintaining a per-user counter internally. The engine
    maintains action counts via BehavioralProfile.vector_count.

    Parameters
    ----------
    ramp_actions : int, optional
        Total number of actions in the ramp-up period. Defaults to
        COLD_START_RAMP_ACTIONS from constants.
    interpolation_fraction : float, optional
        Fraction of ramp_actions over which thresholds smoothly transition.
        Default 0.20 (last 20% of ramp period).
    """

    def __init__(
        self,
        ramp_actions: int = COLD_START_RAMP_ACTIONS,
        interpolation_fraction: float = _INTERPOLATION_FRACTION,
    ) -> None:
        self.ramp_actions            = ramp_actions
        self.interpolation_fraction  = interpolation_fraction
        self._interp_start           = int(ramp_actions * (1 - interpolation_fraction))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def is_warming_up(self, action_count: int) -> bool:
        """
        Return True if the user is still in the cold start ramp-up period.

        Parameters
        ----------
        action_count : int
            Number of verified human actions recorded in the user's profile
            (BehavioralProfile.vector_count).

        Returns
        -------
        bool
            True if action_count < ramp_actions.
        """
        return action_count < self.ramp_actions

    def ramp_progress(self, action_count: int) -> float:
        """
        Return the fraction of the ramp-up period completed.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        float
            Value in [0.0, 1.0]. 0.0 = just started. 1.0 = ramp complete.
        """
        return min(1.0, action_count / self.ramp_actions)

    def is_interpolating(self, action_count: int) -> bool:
        """
        Return True if the user is in the smooth-transition phase.

        The interpolation phase begins at _interp_start actions and ends
        at ramp_actions. During this phase, thresholds blend between their
        ramp-up values and full-enforcement values.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        bool
        """
        return self._interp_start <= action_count < self.ramp_actions

    def actions_remaining(self, action_count: int) -> int:
        """
        Return the number of actions remaining in the ramp-up period.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        int
            0 if ramp is complete.
        """
        return max(0, self.ramp_actions - action_count)

    # ─────────────────────────────────────────────────────────────────────────
    # THRESHOLD GETTERS (smoothly interpolated)
    # ─────────────────────────────────────────────────────────────────────────

    def get_ml_threshold(self, action_count: int) -> float:
        """
        Return the effective ML bot confidence threshold for this action count.

        During ramp-up:   COLD_START_ML_THRESH (0.92 default — very strict)
        During interp:    linearly blends toward ML_BOT_CONFIDENCE_THRESH
        After ramp-up:    ML_BOT_CONFIDENCE_THRESH (0.78 default — normal)

        A higher threshold during ramp-up means only the most obviously
        automated sessions are flagged, protecting legitimate new users from
        false positives during profile building.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        float
            P(bot) threshold in [ML_BOT_CONFIDENCE_THRESH, COLD_START_ML_THRESH].
        """
        if action_count >= self.ramp_actions:
            return ML_BOT_CONFIDENCE_THRESH

        if not self.is_interpolating(action_count):
            return COLD_START_ML_THRESH

        t = self._interp_t(action_count)
        return self._lerp(COLD_START_ML_THRESH, ML_BOT_CONFIDENCE_THRESH, t)

    def get_temporal_limit(self, action_count: int) -> float:
        """
        Return the effective temporal Z-score limit for this action count.

        During ramp-up:   COLD_START_TEMPORAL_LIMIT (4.5 default — loose)
        After ramp-up:    TEMPORAL_ZSCORE_LIMIT (2.5 default — normal)

        New users haven't established a stable timing rhythm, so wider
        temporal tolerance prevents false positives during exploration.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        float
        """
        if action_count >= self.ramp_actions:
            return TEMPORAL_ZSCORE_LIMIT

        if not self.is_interpolating(action_count):
            return COLD_START_TEMPORAL_LIMIT

        t = self._interp_t(action_count)
        return self._lerp(COLD_START_TEMPORAL_LIMIT, TEMPORAL_ZSCORE_LIMIT, t)

    def get_trust_recovery_rate(self, action_count: int) -> float:
        """
        Return the effective trust recovery rate for this action count.

        During ramp-up:   COLD_START_RECOVERY_RATE (4.0 default — doubled)
        After ramp-up:    TRUST_RECOVERY_RATE (2.0 default — normal)

        Faster trust recovery during ramp-up prevents early accidental
        anomalies from permanently damaging the user's trust floor.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        float
        """
        if action_count >= self.ramp_actions:
            return TRUST_RECOVERY_RATE

        if not self.is_interpolating(action_count):
            return COLD_START_RECOVERY_RATE

        t = self._interp_t(action_count)
        return self._lerp(COLD_START_RECOVERY_RATE, TRUST_RECOVERY_RATE, t)

    def cosine_check_enabled(self, action_count: int) -> bool:
        """
        Return True if the cosine similarity check should run for this action count.

        The cosine similarity check is disabled during the full ramp-up period
        (not just interpolation) because the profile vector is too sparse and
        volatile to produce meaningful comparison results until sufficient
        actions have been recorded.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        bool
            False during ramp-up. True when ramp is complete.
        """
        return not self.is_warming_up(action_count)

    def get_all_thresholds(self, action_count: int) -> dict:
        """
        Return all effective thresholds for the current action count as a dict.

        Convenience method for logging and debugging.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        dict
            Keys: ml_threshold, temporal_limit, trust_recovery_rate,
                  cosine_enabled, is_warming_up, ramp_progress_pct.
        """
        return {
            "ml_threshold":        round(self.get_ml_threshold(action_count), 4),
            "temporal_limit":      round(self.get_temporal_limit(action_count), 4),
            "trust_recovery_rate": round(self.get_trust_recovery_rate(action_count), 4),
            "cosine_enabled":      self.cosine_check_enabled(action_count),
            "is_warming_up":       self.is_warming_up(action_count),
            "ramp_progress_pct":   round(self.ramp_progress(action_count) * 100, 1),
            "actions_remaining":   self.actions_remaining(action_count),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def describe_state(self, action_count: int) -> str:
        """
        Return a human-readable description of the current ramp state.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        str
        """
        if not self.is_warming_up(action_count):
            return f"FULL ENFORCEMENT (ramp complete at {self.ramp_actions} actions)"
        elif self.is_interpolating(action_count):
            remaining = self.actions_remaining(action_count)
            pct = round(self.ramp_progress(action_count) * 100, 1)
            return (
                f"INTERPOLATING ({pct}% complete, {remaining} actions to full enforcement)"
            )
        else:
            remaining = self.actions_remaining(action_count)
            pct = round(self.ramp_progress(action_count) * 100, 1)
            return (
                f"RAMP-UP ({pct}% complete, {remaining} actions remaining, "
                f"cosine disabled, ML thresh={self.get_ml_threshold(action_count):.3f})"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _interp_t(self, action_count: int) -> float:
        """
        Compute the interpolation parameter t in [0.0, 1.0] within the
        transition zone.

        t = 0.0 at the start of the interpolation phase.
        t = 1.0 at the end (= ramp_actions, full enforcement begins).

        Uses a smoothstep curve (3t² - 2t³) for a visually smooth transition
        rather than a harsh linear ramp.

        Parameters
        ----------
        action_count : int

        Returns
        -------
        float
        """
        interp_range = self.ramp_actions - self._interp_start
        if interp_range <= 0:
            return 1.0
        raw_t = (action_count - self._interp_start) / interp_range
        raw_t = max(0.0, min(1.0, raw_t))
        # Smoothstep: S(t) = 3t² - 2t³
        return 3 * raw_t**2 - 2 * raw_t**3

    @staticmethod
    def _lerp(start: float, end: float, t: float) -> float:
        """
        Linear interpolation between start and end by factor t.

        Parameters
        ----------
        start : float
        end : float
        t : float
            Interpolation factor in [0.0, 1.0].

        Returns
        -------
        float
        """
        return start + (end - start) * t

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RampUpManager("
            f"ramp_actions={self.ramp_actions}, "
            f"interp_start={self._interp_start})"
        )