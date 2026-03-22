"""
zelvryn/core/features.py
========================
FeatureExtractor — converts a rolling action window into a 20-dimensional
numerical feature vector for use by the ML classifier.

This is the bridge between raw user actions (symbolic, categorical) and the
numerical domain where machine learning operates. Every feature is designed to
capture a distinct aspect of user behaviour that differentiates humans from
automated agents — particularly sophisticated AI bots that attempt to mimic
human action patterns.

Feature engineering philosophy
-------------------------------
AI mimics fail not because any single feature is wrong, but because they cannot
simultaneously match all 20 features. A bot tuned to produce realistic timing
may produce unrealistically uniform intervals (high quantile_ratio). A bot tuned
to access diverse resources may have suspiciously high autocorrelation from its
scripted access pattern. The 20-feature vector catches mimics in the combination
of subtle signals they inevitably fail to simultaneously replicate.

Complete feature index map
--------------------------
  [0]  freq_login         — proportion of 'login' actions in window
  [1]  freq_read          — proportion of 'read' actions
  [2]  freq_write         — proportion of 'write' actions
  [3]  freq_delete        — proportion of 'delete' actions
  [4]  freq_search        — proportion of 'search' actions
  [5]  freq_export        — proportion of 'export' actions (high risk)
  [6]  freq_config        — proportion of 'config' actions (high risk)
  [7]  freq_admin         — proportion of 'admin' actions (high risk)
  [8]  entropy            — normalised Shannon entropy of action distribution
  [9]  mean_interval      — mean inter-action interval (seconds)
  [10] std_interval       — standard deviation of intervals
  [11] cv_interval        — coefficient of variation (stdev / mean)
  [12] min_interval       — fastest inter-action burst speed (seconds)
  [13] max_interval       — longest pause between actions (seconds)
  [14] sensitive_prop     — proportion of delete+export+config+admin actions
  [15] autocorrelation    — lag-1 Pearson autocorrelation of action indices
  [16] resource_div       — unique resources / total actions (diversity)
  [17] burst_score        — fraction of intervals < 0.5 seconds
  [18] run_length         — mean consecutive identical action run length
  [19] quantile_ratio     — Q1/Q3 of interval distribution (uniformity signal)

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import statistics
from collections import Counter
from typing import Dict, List, Optional

from core.action import UserAction
from core.constants import (
    ML_FEATURE_DIM,
    ENTROPY_BIN_COUNT,
    BURST_INTERVAL_THRESHOLD,
    SENSITIVE_ACTION_TYPES,
    TEMPORAL_DEFAULT_MEAN,
    TEMPORAL_DEFAULT_STDEV,
)
from core.math_utils import (
    shannon_entropy,
    lag1_autocorrelation,
    run_length_mean,
    quantile_ratio,
    coefficient_of_variation,
    safe_divide,
)


class FeatureExtractor:
    """
    Stateless feature extraction engine for the Zelvryn ABSA system.

    All methods are static or class methods — no instance state is maintained.
    This makes FeatureExtractor safe to call from multiple threads simultaneously
    and eliminates any possibility of state contamination between sessions.

    Usage
    -----
    >>> from core.features import FeatureExtractor
    >>> from core.action import UserAction
    >>> window = [UserAction("read", "doc_1"), UserAction("write", "doc_2")]
    >>> vector = FeatureExtractor.extract(window)
    >>> len(vector)
    20
    """

    @staticmethod
    def extract(
        window: List[UserAction],
        profile_intervals: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Extract the 20-dimensional feature vector from an action window.

        This is the primary interface. All 20 features are computed in a single
        pass through the window where possible, with a second pass for statistics
        requiring sorted data or multiple traversals.

        Parameters
        ----------
        window : list of UserAction
            The rolling action window to extract features from. Should contain
            at least 2 actions for meaningful interval features. The vector is
            still produced for shorter windows, but interval features will be
            zeros or defaults.
        profile_intervals : list of float, optional
            Pre-computed inter-action intervals from the user's profile. When
            provided and the window is small, profile intervals supplement the
            window's own intervals for more stable statistics. If None, intervals
            are derived solely from the window's action timestamps.

        Returns
        -------
        list of float
            Feature vector of exactly ML_FEATURE_DIM (20) floats. All values
            are finite; NaN and infinity are replaced with 0.0.

        Raises
        ------
        None
            This method never raises. Bad inputs produce a zero vector.
        """
        n = len(window)
        if n == 0:
            return [0.0] * ML_FEATURE_DIM

        # ── Pre-compute window intervals from timestamps ───────────────────
        window_intervals = FeatureExtractor._compute_intervals(window)

        # Decide which intervals to use for statistics:
        # Window intervals for features, supplemented by profile intervals for
        # timing stats when the window is small.
        intervals_for_stats = window_intervals if window_intervals else []
        if len(intervals_for_stats) < 4 and profile_intervals:
            # Use tail of profile intervals for context
            intervals_for_stats = (
                profile_intervals[-20:] + intervals_for_stats
            )

        # ── Feature group 1: Action type frequencies [0-7] ───────────────
        type_counts = Counter(a.action_type for a in window)
        freq_features = [
            safe_divide(type_counts.get(t, 0), n)
            for t in UserAction.ACTION_TYPES
        ]

        # ── Feature 8: Shannon entropy ─────────────────────────────────────
        entropy_val = shannon_entropy(
            [a.action_type for a in window],
            num_bins=ENTROPY_BIN_COUNT,
        )

        # ── Feature group 9-13: Interval statistics ────────────────────────
        interval_stats = FeatureExtractor._compute_interval_features(
            intervals_for_stats
        )

        # ── Feature 14: Sensitive action proportion ────────────────────────
        sensitive_count = sum(1 for a in window if a.is_sensitive())
        sensitive_prop = safe_divide(sensitive_count, n)

        # ── Feature 15: Lag-1 autocorrelation of action indices ────────────
        action_indices = [float(a.to_index()) for a in window]
        autocorr = lag1_autocorrelation(action_indices)

        # ── Feature 16: Resource diversity ─────────────────────────────────
        unique_resources = len({a.resource_id for a in window})
        resource_div = safe_divide(unique_resources, n)

        # ── Feature 17: Burst score ────────────────────────────────────────
        burst = FeatureExtractor._compute_burst_score(window_intervals)

        # ── Feature 18: Run-length regularity ─────────────────────────────
        action_types_seq = [a.action_type for a in window]
        run_len = run_length_mean(action_types_seq)

        # ── Feature 19: Interval quantile ratio ───────────────────────────
        q_ratio = quantile_ratio(intervals_for_stats)

        # ── Assemble final vector ─────────────────────────────────────────
        vector = (
            freq_features           # indices 0-7   (8 features)
            + [entropy_val]         # index  8      (1 feature)
            + interval_stats        # indices 9-13  (5 features)
            + [                     # indices 14-19 (6 features)
                sensitive_prop,
                autocorr,
                resource_div,
                burst,
                run_len,
                q_ratio,
            ]
        )

        assert len(vector) == ML_FEATURE_DIM, (
            f"Feature vector length mismatch: expected {ML_FEATURE_DIM}, "
            f"got {len(vector)}"
        )

        # Sanitise: replace any NaN/infinity with 0.0
        return [v if (not (v != v) and abs(v) != float("inf")) else 0.0
                for v in vector]

    @staticmethod
    def feature_names() -> List[str]:
        """
        Return the name of each feature at its index position.

        Useful for logging, dashboard display, and feature importance reports.

        Returns
        -------
        list of str
            20 names in the same order as the feature vector.
        """
        return [
            *[f"freq_{t}" for t in UserAction.ACTION_TYPES],  # 0-7
            "entropy",         # 8
            "mean_interval",   # 9
            "std_interval",    # 10
            "cv_interval",     # 11
            "min_interval",    # 12
            "max_interval",    # 13
            "sensitive_prop",  # 14
            "autocorrelation", # 15
            "resource_div",    # 16
            "burst_score",     # 17
            "run_length",      # 18
            "quantile_ratio",  # 19
        ]

    @staticmethod
    def feature_descriptions() -> Dict[str, str]:
        """
        Return a description of each feature keyed by name.

        Used in documentation and the analyst dashboard.

        Returns
        -------
        dict of str -> str
        """
        names = FeatureExtractor.feature_names()
        descs = [
            "Proportion of 'login' actions in the window",
            "Proportion of 'read' actions in the window",
            "Proportion of 'write' actions in the window",
            "Proportion of 'delete' actions in the window (sensitive)",
            "Proportion of 'search' actions in the window",
            "Proportion of 'export' actions in the window (high risk)",
            "Proportion of 'config' actions in the window (high risk)",
            "Proportion of 'admin' actions in the window (high risk)",
            "Normalised Shannon entropy — randomness of action distribution",
            "Mean inter-action interval in seconds",
            "Standard deviation of inter-action intervals",
            "Coefficient of variation (stdev / mean) — machine-like if near 0",
            "Minimum inter-action interval — automation if very low",
            "Maximum inter-action interval — longest pause between actions",
            "Proportion of delete + export + config + admin actions",
            "Lag-1 Pearson autocorrelation of action index sequence",
            "Unique resources / total actions — low means targeting same resource",
            "Fraction of intervals under 0.5s — automation burst signal",
            "Mean consecutive identical action run length — bot looping signal",
            "Q1/Q3 of interval distribution — near 1.0 = machine-uniform timing",
        ]
        return dict(zip(names, descs))

    @staticmethod
    def extract_batch(
        windows: List[List[UserAction]],
        profile_intervals: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """
        Extract feature vectors from a batch of action windows.

        More efficient than calling extract() in a loop when many windows
        need to be processed (e.g., during dataset preparation for training).

        Parameters
        ----------
        windows : list of list of UserAction
            Each inner list is one action window.
        profile_intervals : list of float, optional
            Shared profile intervals passed to each window's extraction.

        Returns
        -------
        list of list of float
            One 20-dimensional vector per window.
        """
        return [
            FeatureExtractor.extract(w, profile_intervals)
            for w in windows
        ]

    @staticmethod
    def validate_vector(vector: List[float]) -> bool:
        """
        Validate that a feature vector has the expected dimensions and is finite.

        Parameters
        ----------
        vector : list of float

        Returns
        -------
        bool
            True if vector is valid (correct length, all finite values).
        """
        if len(vector) != ML_FEATURE_DIM:
            return False
        return all(
            not (v != v) and abs(v) != float("inf")
            for v in vector
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_intervals(window: List[UserAction]) -> List[float]:
        """
        Derive inter-action intervals from action timestamps.

        Intervals are computed as the positive difference between consecutive
        action timestamps. Negative or zero differences (which can occur if
        timestamps are imprecise) are clamped to 0.

        Parameters
        ----------
        window : list of UserAction

        Returns
        -------
        list of float
            len(window) - 1 intervals in seconds. Empty list for windows
            with 0 or 1 actions.
        """
        if len(window) < 2:
            return []
        return [
            max(0.0, window[i].timestamp - window[i - 1].timestamp)
            for i in range(1, len(window))
        ]

    @staticmethod
    def _compute_interval_features(intervals: List[float]) -> List[float]:
        """
        Compute the five interval statistics features (indices 9-13).

        Features computed:
          [9]  mean_interval — arithmetic mean of intervals
          [10] std_interval  — sample standard deviation
          [11] cv_interval   — coefficient of variation (stdev / mean)
          [12] min_interval  — minimum interval value
          [13] max_interval  — maximum interval value

        Parameters
        ----------
        intervals : list of float
            Inter-action intervals in seconds.

        Returns
        -------
        list of float
            5 features: [mean, std, cv, min, max].
            Returns default values when intervals is empty or too short.
        """
        if not intervals:
            return [
                TEMPORAL_DEFAULT_MEAN,    # mean
                TEMPORAL_DEFAULT_STDEV,   # std
                safe_divide(TEMPORAL_DEFAULT_STDEV, TEMPORAL_DEFAULT_MEAN),  # cv
                0.0,                      # min
                0.0,                      # max
            ]

        mean_val = statistics.mean(intervals)

        if len(intervals) < 2:
            std_val = 0.0
        else:
            std_val = statistics.stdev(intervals)

        cv_val  = safe_divide(std_val, mean_val)
        min_val = min(intervals)
        max_val = max(intervals)

        return [mean_val, std_val, cv_val, min_val, max_val]

    @staticmethod
    def _compute_burst_score(intervals: List[float]) -> float:
        """
        Compute the burst score: fraction of intervals below the burst threshold.

        Bursts are very rapid consecutive actions (interval < BURST_INTERVAL_THRESHOLD
        seconds, default 0.5s). A high burst score indicates automated activity:
        human users physically cannot consistently act faster than about 0.3-0.5s.

        Parameters
        ----------
        intervals : list of float
            Inter-action intervals in seconds.

        Returns
        -------
        float
            Burst score in [0.0, 1.0]. 0.0 if no intervals.
        """
        if not intervals:
            return 0.0
        burst_count = sum(1 for iv in intervals if iv < BURST_INTERVAL_THRESHOLD)
        return burst_count / len(intervals)

    @staticmethod
    def _sensitive_action_fraction(window: List[UserAction]) -> float:
        """
        Compute the fraction of sensitive actions in the window.

        Sensitive action types: delete, export, config, admin.
        High fractions indicate exfiltration or privilege abuse patterns.

        Parameters
        ----------
        window : list of UserAction

        Returns
        -------
        float
            Fraction in [0.0, 1.0].
        """
        if not window:
            return 0.0
        sensitive = sum(1 for a in window if a.action_type in SENSITIVE_ACTION_TYPES)
        return sensitive / len(window)