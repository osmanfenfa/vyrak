"""
FeatureExtractor — converts a rolling action window into a 20-dimensional
numerical feature vector for use by the ML classifier.
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
    """

    @staticmethod
    def extract(
        window: List[UserAction],
        profile_intervals: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Extract the 20-dimensional feature vector from an action window.
        """
        n = len(window)
        if n == 0:
            return [0.0] * ML_FEATURE_DIM

        # ── Pre-compute window intervals from timestamps 
        window_intervals = FeatureExtractor._compute_intervals(window)

        # Decide which intervals to use for statistics:
        # Window intervals for features, supplemented by profile intervals for
        # timing stats when the window is small.
        intervals_for_stats = window_intervals if window_intervals else []
        if len(intervals_for_stats) < 4 and profile_intervals:
            intervals_for_stats = (
                profile_intervals[-20:] + intervals_for_stats
            )
            
        type_counts = Counter(a.action_type for a in window)
        freq_features = [
            safe_divide(type_counts.get(t, 0), n)
            for t in UserAction.ACTION_TYPES
        ]

        entropy_val = shannon_entropy(
            [a.action_type for a in window],
            num_bins=ENTROPY_BIN_COUNT,
        )
        
        interval_stats = FeatureExtractor._compute_interval_features(
            intervals_for_stats
        )
        sensitive_count = sum(1 for a in window if a.is_sensitive())
        sensitive_prop = safe_divide(sensitive_count, n)
        action_indices = [float(a.to_index()) for a in window]
        autocorr = lag1_autocorrelation(action_indices)
        unique_resources = len({a.resource_id for a in window})
        resource_div = safe_divide(unique_resources, n)
        burst = FeatureExtractor._compute_burst_score(window_intervals)
        action_types_seq = [a.action_type for a in window]
        run_len = run_length_mean(action_types_seq)
        q_ratio = quantile_ratio(intervals_for_stats)
        vector = (
            freq_features         
            + [entropy_val]        
            + interval_stats      
            + [                    
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
        """
        return [
            FeatureExtractor.extract(w, profile_intervals)
            for w in windows
        ]

    @staticmethod
    def validate_vector(vector: List[float]) -> bool:
        """
        Validate that a feature vector has the expected dimensions and is finite.
        """
        if len(vector) != ML_FEATURE_DIM:
            return False
        return all(
            not (v != v) and abs(v) != float("inf")
            for v in vector
        )

    # PRIVATE HELPERS
    @staticmethod
    def _compute_intervals(window: List[UserAction]) -> List[float]:
        """
        Derive inter-action intervals from action timestamps.
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
        """
        if not intervals:
            return 0.0
        burst_count = sum(1 for iv in intervals if iv < BURST_INTERVAL_THRESHOLD)
        return burst_count / len(intervals)

    @staticmethod
    def _sensitive_action_fraction(window: List[UserAction]) -> float:
        """
        Compute the fraction of sensitive actions in the window.
        """
        if not window:
            return 0.0
        sensitive = sum(1 for a in window if a.action_type in SENSITIVE_ACTION_TYPES)
        return sensitive / len(window)