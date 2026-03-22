"""
BehavioralProfile — the long-term behavioural signature for a single user.
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from core.action import UserAction
from core.constants import (
    PROFILE_HISTORY_SIZE,
    SIMILARITY_THRESHOLD,
    SIMILARITY_ADAPTIVE_MIN,
    SIMILARITY_ADAPTIVE_MAX,
    EWMA_DECAY_FACTOR,
    EWMA_MIN_OBSERVATIONS,
    TEMPORAL_DEFAULT_MEAN,
    TEMPORAL_DEFAULT_STDEV,
    ADAPTIVE_THRESHOLD_RECOMPUTE_INTERVAL,
)
from core.math_utils import (
    coefficient_of_variation,
    ewma_update,
    ewma_vector_update,
    compute_interval_stats,
)


class BehavioralProfile:
    """
    Long-term behavioural signature for a single authenticated user.
    The profile is built passively during verified-human sessions and used as
    the reference baseline for all anomaly detection. It continuously updates
    via EWMA to track legitimate behaviour drift.
    """

    def __init__(self, user_id: str, created_at: Optional[float] = None) -> None:
        self.user_id: str = user_id
        self.created_at: float = created_at if created_at is not None else time.time()
        self.last_updated: float = self.created_at

        # ── Action history (rolling buffer) 
        self.action_history: deque = deque(maxlen=PROFILE_HISTORY_SIZE)

        # ── Cumulative action type counts 
        self.action_counts: Dict[str, int] = defaultdict(int)

        # ── Timing history 
        self.time_intervals: List[float] = []

        # ── Simple mean vector (non-EWMA fallback) 
        self.vector_sum: List[float] = [0.0] * UserAction.type_count()
        self.vector_count: int = 0

        # ── EWMA-smoothed statistics (primary for anomaly detection)
        self.ewma_vector: Optional[List[float]] = None
        self.ewma_mean_interval: Optional[float] = None
        self.ewma_std_interval: Optional[float] = None

        # ── ML training data
        self.ml_training_data: List[Tuple[List[float], int]] = []

        # ── Adaptive threshold ────────────────────────────────────────────
        self.adaptive_threshold: float = SIMILARITY_THRESHOLD

        # ── Internal state ────────────────────────────────────────────────
        self._last_action_time: Optional[float] = None
        self._observations_since_threshold_update: int = 0

    # PRIMARY INTERFACE — recording actions
    def record_action(self, action: UserAction) -> None:
        """
        Add a verified-human action to the long-term profile.
        This method should only be called for actions that have been confirmed
        as legitimate (i.e., the evaluate_action() call returned ALLOW and no
        anomalies were detected). Calling it with bot actions would corrupt the
        profile baseline.
        """
        self.action_history.append(action)
        self.action_counts[action.action_type] += 1
        self.vector_count += 1
        self.last_updated = action.timestamp
        idx = action.to_index()
        self.vector_sum[idx] += 1.0

        # Compute and record inter-action interval
        if self._last_action_time is not None:
            interval = max(0.0, action.timestamp - self._last_action_time)
            self._record_interval(interval)

        self._last_action_time = action.timestamp

        # Update EWMA frequency vector
        self._update_ewma_vector(action)

        # Periodically recompute adaptive threshold
        self._observations_since_threshold_update += 1
        if (self._observations_since_threshold_update
                >= ADAPTIVE_THRESHOLD_RECOMPUTE_INTERVAL
                and self.vector_count >= EWMA_MIN_OBSERVATIONS):
            self._recompute_adaptive_threshold()
            self._observations_since_threshold_update = 0

    # VECTOR ACCESS
    def get_mean_vector(self) -> List[float]:
        """
        Return the normalised mean behaviour vector from the entire profile history.
        """
        # Prefer EWMA vector if initialised and has enough data
        if (self.ewma_vector is not None
                and self.vector_count >= EWMA_MIN_OBSERVATIONS):
            total = sum(self.ewma_vector)
            if total > 0:
                return [v / total for v in self.ewma_vector]

        # Fall back to simple cumulative mean
        if self.vector_count == 0:
            return [0.0] * UserAction.type_count()

        total = sum(self.vector_sum)
        if total == 0:
            return [0.0] * UserAction.type_count()
        return [v / total for v in self.vector_sum]

    def get_ewma_vector(self) -> Optional[List[float]]:
        """
        Return the raw EWMA-smoothed action type count vector (un-normalised).
        """
        if self.vector_count < EWMA_MIN_OBSERVATIONS:
            return None
        return list(self.ewma_vector) if self.ewma_vector else None

    # TIMING STATISTICS
    def get_timing_stats(self) -> Tuple[float, float]:
        """
        Return (mean, stdev) of recorded inter-action intervals.
        Uses EWMA-smoothed values when available (requires >= EWMA_MIN_OBSERVATIONS
        recorded intervals). Falls back to raw statistics from time_intervals list,
        and finally to system defaults if insufficient data exists.
        """
        # EWMA-smoothed values (most drift-aware)
        if (self.ewma_mean_interval is not None
                and self.ewma_std_interval is not None
                and len(self.time_intervals) >= EWMA_MIN_OBSERVATIONS):
            return (self.ewma_mean_interval, max(self.ewma_std_interval, 0.1))

        # Raw statistics from recorded intervals
        return compute_interval_stats(
            self.time_intervals,
            default_mean=TEMPORAL_DEFAULT_MEAN,
            default_stdev=TEMPORAL_DEFAULT_STDEV,
        )

    def get_recent_intervals(self, n: int = 20) -> List[float]:
        """
        Return the N most recent inter-action intervals.

        Parameters
        ----------
        n : int
            Number of recent intervals to retrieve.

        Returns
        -------
        list of float
        """
        return self.time_intervals[-n:] if self.time_intervals else []

    # ─────────────────────────────────────────────────────────────────────────
    # ML TRAINING DATA
    # ─────────────────────────────────────────────────────────────────────────

    def add_training_sample(
        self,
        features: List[float],
        label: int,
        verified: bool = False,
    ) -> None:
        """
        Add a labeled feature vector to the ML training dataset.

        Parameters
        ----------
        features : list of float
            20-dimensional feature vector from FeatureExtractor.extract().
        label : int
            0 = human (legitimate user), 1 = bot / anomalous session.
        verified : bool, optional
            True if this label was verified by a human analyst. Unverified
            labels are heuristic auto-labels from the online learning loop.
            (Stored for reference; used by feature_store to separate queues.)
        """
        if len(features) != 20:
            raise ValueError(
                f"Feature vector must have dimension 20, got {len(features)}"
            )
        if label not in (0, 1):
            raise ValueError(f"Label must be 0 (human) or 1 (bot), got {label}")

        self.ml_training_data.append((list(features), label, verified))

    def get_training_data(
        self,
        verified_only: bool = False,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Return the training dataset as (X, y) arrays.

        Parameters
        ----------
        verified_only : bool, optional
            If True, only returns analyst-verified samples. Default False.

        Returns
        -------
        tuple of (list of list, list of int)
            (X, y) where X[i] is a feature vector and y[i] is its label.
        """
        if not self.ml_training_data:
            return ([], [])

        if verified_only:
            data = [(fv, lbl) for fv, lbl, v in self.ml_training_data if v]
        else:
            data = [(fv, lbl) for fv, lbl, *_ in self.ml_training_data]

        if not data:
            return ([], [])

        X = [fv for fv, _ in data]
        y = [lbl for _, lbl in data]
        return (X, y)

    def training_sample_count(self, label: Optional[int] = None) -> int:
        """
        Return the number of stored training samples.

        Parameters
        ----------
        label : int or None
            If None, returns total. If 0 or 1, returns count for that label.

        Returns
        -------
        int
        """
        if label is None:
            return len(self.ml_training_data)
        return sum(1 for _, lbl, *_ in self.ml_training_data if lbl == label)

    # ─────────────────────────────────────────────────────────────────────────
    # PROFILE STATE QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def is_mature(self, min_actions: int = 50) -> bool:
        """
        Return True if the profile has enough history for full detection.

        A profile is considered mature when it has recorded at least min_actions
        verified human actions. Before maturity, cold start rules apply.

        Parameters
        ----------
        min_actions : int
            Minimum actions required. Default 50.

        Returns
        -------
        bool
        """
        return self.vector_count >= min_actions

    def get_action_proportion(self, action_type: str) -> float:
        """
        Return the proportion of total actions for a given action type.

        Parameters
        ----------
        action_type : str
            One of UserAction.ACTION_TYPES.

        Returns
        -------
        float
            Proportion in [0.0, 1.0].
        """
        if self.vector_count == 0:
            return 0.0
        return self.action_counts.get(action_type, 0) / self.vector_count

    def profile_age_seconds(self) -> float:
        """Return the age of this profile in seconds from creation."""
        return time.time() - self.created_at

    def time_since_last_action(self) -> Optional[float]:
        """
        Return seconds since the last recorded action.

        Returns None if no actions have been recorded.
        """
        if self._last_action_time is None:
            return None
        return time.time() - self._last_action_time

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALISATION
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the profile to a plain dict for Redis storage.

        Note: action_history (deque of UserAction objects) is stored as a list
        of serialised action dicts, with the most recent last.

        Returns
        -------
        dict
        """
        return {
            "user_id":            self.user_id,
            "created_at":         self.created_at,
            "last_updated":       self.last_updated,
            "action_history":     [a.to_dict() for a in self.action_history],
            "action_counts":      dict(self.action_counts),
            "time_intervals":     list(self.time_intervals),
            "vector_sum":         list(self.vector_sum),
            "vector_count":       self.vector_count,
            "ewma_vector":        list(self.ewma_vector) if self.ewma_vector else None,
            "ewma_mean_interval": self.ewma_mean_interval,
            "ewma_std_interval":  self.ewma_std_interval,
            "adaptive_threshold": self.adaptive_threshold,
            "ml_training_count":  len(self.ml_training_data),
            # ml_training_data is large; stored separately in the feature store
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralProfile":
        """
        Reconstruct a BehavioralProfile from a serialised dict.

        Parameters
        ----------
        data : dict
            Dict produced by to_dict().

        Returns
        -------
        BehavioralProfile
        """
        profile = cls(
            user_id=data["user_id"],
            created_at=data.get("created_at"),
        )
        profile.last_updated        = data.get("last_updated", profile.created_at)
        profile.action_counts       = defaultdict(int, data.get("action_counts", {}))
        profile.time_intervals      = data.get("time_intervals", [])
        profile.vector_sum          = data.get("vector_sum", [0.0] * UserAction.type_count())
        profile.vector_count        = data.get("vector_count", 0)
        profile.ewma_vector         = data.get("ewma_vector")
        profile.ewma_mean_interval  = data.get("ewma_mean_interval")
        profile.ewma_std_interval   = data.get("ewma_std_interval")
        profile.adaptive_threshold  = data.get("adaptive_threshold", SIMILARITY_THRESHOLD)

        # Reconstruct action_history deque
        for action_dict in data.get("action_history", []):
            profile.action_history.append(UserAction.from_dict(action_dict))

        # Restore last action time from history
        if profile.action_history:
            profile._last_action_time = profile.action_history[-1].timestamp

        return profile

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _record_interval(self, interval: float) -> None:
        """
        Record an inter-action interval and update EWMA timing statistics.

        Parameters
        ----------
        interval : float
            Time in seconds since the previous action.
        """
        self.time_intervals.append(interval)

        # Keep list bounded (prevents unbounded memory growth for long sessions)
        if len(self.time_intervals) > PROFILE_HISTORY_SIZE:
            self.time_intervals.pop(0)

        # Update EWMA timing statistics
        if self.ewma_mean_interval is None:
            # Bootstrap from first interval
            self.ewma_mean_interval = interval
            self.ewma_std_interval  = TEMPORAL_DEFAULT_STDEV
        else:
            # Incremental EWMA update for mean
            self.ewma_mean_interval = ewma_update(
                self.ewma_mean_interval, interval, EWMA_DECAY_FACTOR
            )
            # Incremental EWMA update for stdev via squared deviation proxy
            deviation = abs(interval - self.ewma_mean_interval)
            self.ewma_std_interval = ewma_update(
                self.ewma_std_interval, deviation, EWMA_DECAY_FACTOR
            )
            self.ewma_std_interval = max(self.ewma_std_interval, 0.1)

    def _update_ewma_vector(self, action: UserAction) -> None:
        """
        Update the EWMA action frequency vector with a new action.

        Constructs a one-hot vector for the new action and applies EWMA to
        the running estimate.

        Parameters
        ----------
        action : UserAction
        """
        one_hot = [0.0] * UserAction.type_count()
        one_hot[action.to_index()] = 1.0

        if self.ewma_vector is None:
            # Bootstrap
            self.ewma_vector = list(one_hot)
        else:
            self.ewma_vector = ewma_vector_update(
                self.ewma_vector, one_hot, EWMA_DECAY_FACTOR
            )

    def _recompute_adaptive_threshold(self) -> None:
        """
        Recompute the per-user adaptive cosine similarity threshold.

        Formula:
          CV = stdev(time_intervals) / mean(time_intervals)
          threshold = clamp(
            SIMILARITY_THRESHOLD - (CV * 0.10),
            ADAPTIVE_MIN,
            ADAPTIVE_MAX
          )

        Interpretation:
          - High CV (variable timing) → looser threshold (prevents false positives)
          - Low CV (consistent timing) → tighter threshold (harder to impersonate)

        The threshold is recomputed every ADAPTIVE_THRESHOLD_RECOMPUTE_INTERVAL
        new actions so it tracks gradual lifestyle changes without overreacting
        to individual session variation.
        """
        if len(self.time_intervals) < 10:
            return   # not enough data for reliable CV

        cv = coefficient_of_variation(self.time_intervals)
        raw_threshold = SIMILARITY_THRESHOLD - (cv * 0.10)

        self.adaptive_threshold = max(
            SIMILARITY_ADAPTIVE_MIN,
            min(SIMILARITY_ADAPTIVE_MAX, raw_threshold)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BehavioralProfile(user={self.user_id!r}, "
            f"actions={self.vector_count}, "
            f"threshold={self.adaptive_threshold:.3f}, "
            f"mature={self.is_mature()})"
        )

    def __str__(self) -> str:
        mean_vec = self.get_mean_vector()
        mean_str = ", ".join(f"{v:.3f}" for v in mean_vec)
        mean_interval, std_interval = self.get_timing_stats()
        return (
            f"Profile [{self.user_id}]\n"
            f"  Actions recorded : {self.vector_count}\n"
            f"  Mean vector      : [{mean_str}]\n"
            f"  Timing           : mean={mean_interval:.2f}s  "
            f"stdev={std_interval:.2f}s\n"
            f"  Threshold        : {self.adaptive_threshold:.4f}\n"
            f"  ML samples       : {len(self.ml_training_data)}"
        )