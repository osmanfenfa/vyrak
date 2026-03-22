"""
OVERRIDING VIA ENVIRONMENT VARIABLES
--------------------------------------
Each constant reads from its matching environment variable if present,
and falls back to the documented default otherwise.  Example:

    ML_BOT_CONFIDENCE_THRESH=0.82 uvicorn api.main:app
"""

import os
from typing import Final

def _float(key: str, default: float) -> float:
    """Read a float from environment, fall back to default."""
    raw = os.environ.get(key, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default

def _int(key: str, default: int) -> int:
    """Read an int from environment, fall back to default."""
    raw = os.environ.get(key, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default

def _bool(key: str, default: bool) -> bool:
    """Read a bool from environment ('true'/'1'/'yes' → True)."""
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes")

# TRUST SCORE DYNAMICS

TRUST_INITIAL: Final[float] = _float("TRUST_INITIAL", 100.0)
TRUST_DECAY_MILD: Final[float] = _float("TRUST_DECAY_MILD", 5.0)
TRUST_DECAY_MODERATE: Final[float] = _float("TRUST_DECAY_MODERATE", 15.0)
TRUST_DECAY_SEVERE: Final[float] = _float("TRUST_DECAY_SEVERE", 35.0)
TRUST_DECAY_AI_BOT: Final[float] = _float("TRUST_DECAY_AI_BOT", 25.0)
TRUST_RECOVERY_RATE: Final[float] = _float("TRUST_RECOVERY_RATE", 2.0)
TRUST_CHALLENGE_THRESHOLD: Final[float] = _float("TRUST_CHALLENGE_THRESHOLD", 50.0)
TRUST_BLOCK_THRESHOLD: Final[float] = _float("TRUST_BLOCK_THRESHOLD", 20.0)

# BEHAVIOURAL WINDOW & RULE ENGINE

BEHAVIOR_WINDOW_SIZE: Final[int] = _int("BEHAVIOR_WINDOW_SIZE", 20)
"""
Number of recent actions kept in the rolling evaluation window.
"""

PROFILE_HISTORY_SIZE: Final[int] = _int("PROFILE_HISTORY_SIZE", 200)
"""
Maximum number of past actions stored in the long-term behavioral profile.
"""

SIMILARITY_THRESHOLD: Final[float] = _float("SIMILARITY_THRESHOLD", 0.75)
"""
Base cosine similarity threshold below which a LOW_SIMILARITY anomaly fires.
"""

TEMPORAL_ZSCORE_LIMIT: Final[float] = _float("TEMPORAL_ZSCORE_LIMIT", 2.5)
"""
Inter-action timing Z-score beyond which TEMPORAL_DRIFT fires.
"""

ENTROPY_HIGH_THRESHOLD: Final[float] = _float("ENTROPY_HIGH_THRESHOLD", 0.92)
"""
Normalised Shannon entropy threshold above which HIGH_ENTROPY fires.
"""

ENTROPY_MIN_WINDOW: Final[int] = _int("ENTROPY_MIN_WINDOW", 10)
"""
Minimum window size before the entropy check is applied.
"""

COSINE_MODERATE_THRESHOLD: Final[float] = _float("COSINE_MODERATE_THRESHOLD", 0.5)
"""
Cosine similarity below this value triggers SEVERE penalty (TRUST_DECAY_SEVERE).
"""

PROFILE_MIN_ACTIONS_FOR_COSINE: Final[int] = _int("PROFILE_MIN_ACTIONS_FOR_COSINE", 20)
"""
Minimum number of recorded profile actions before the cosine similarity
"""

# MACHINE LEARNING CLASSIFIER

ML_FEATURE_DIM: Final[int] = 20
"""
Dimensionality of the feature vector produced by FeatureExtractor.
"""

ML_MIN_TRAIN_SAMPLES: Final[int] = _int("ML_MIN_TRAIN_SAMPLES", 40)
"""
Minimum number of labeled training samples required before the first
model training cycle runs.
"""

ML_NUM_TREES: Final[int] = _int("ML_NUM_TREES", 15)
"""
Number of decision trees in the Random Forest ensemble.
"""

ML_MAX_DEPTH: Final[int] = _int("ML_MAX_DEPTH", 6)
"""
Maximum depth of each decision tree.
Deeper trees overfit; shallower trees underfit.
"""

ML_MIN_SAMPLES_SPLIT: Final[int] = _int("ML_MIN_SAMPLES_SPLIT", 4)
"""
Minimum number of samples required to split an internal tree node.
"""

ML_BOT_CONFIDENCE_THRESH: Final[float] = _float("ML_BOT_CONFIDENCE_THRESH", 0.78)
"""
P(bot) threshold above which the ML classifier fires an ML_BOT_DETECTED
anomaly.  0.78 means the forest must be at least 78% confident that
the current window was produced by a bot.
"""

ML_RETRAIN_INTERVAL: Final[int] = _int("ML_RETRAIN_INTERVAL", 50)
"""
Number of new labeled samples accumulated before the model retrains.
"""

ML_CONFIDENCE_GATE_HIGH: Final[float] = _float("ML_CONFIDENCE_GATE_HIGH", 0.95)
"""
Auto-labeling confidence gate — upper bound.
"""

ML_CONFIDENCE_GATE_LOW: Final[float] = _float("ML_CONFIDENCE_GATE_LOW", 0.10)
"""
Auto-labeling confidence gate — lower bound.
"""

ML_WARM_START_HALF_WINDOW: Final[int] = _int("ML_WARM_START_HALF_WINDOW", 10)
"""
Minimum window fill before ML inference is activated.
"""

# ROFILE & COLD START MANAGEMENT

EWMA_DECAY_FACTOR: Final[float] = _float("EWMA_DECAY_FACTOR", 0.95)
"""
Exponentially Weighted Moving Average decay factor (lambda).
Applied to profile statistics to handle legitimate behaviour drift.
"""

PROFILE_TTL_DAYS: Final[int] = _int("PROFILE_TTL_DAYS", 90)
"""
Number of days after which an inactive profile expires in Redis.
"""

COLD_START_RAMP_ACTIONS: Final[int] = _int("COLD_START_RAMP_ACTIONS", 50)
"""
Number of actions in the cold start ramp-up period.
"""

COLD_START_LOOSE_THRESH: Final[float] = _float("COLD_START_LOOSE_THRESH", 0.92)
"""
ML bot confidence threshold used during the cold start ramp-up period.
this period.
"""

ADAPTIVE_THRESHOLD_MIN: Final[float] = _float("ADAPTIVE_THRESHOLD_MIN", 0.65)
"""
Lower bound for the per-user adaptive similarity threshold.
"""

ADAPTIVE_THRESHOLD_MAX: Final[float] = _float("ADAPTIVE_THRESHOLD_MAX", 0.85)
"""
Upper bound for the per-user adaptive similarity threshold.
"""

ADAPTIVE_THRESHOLD_RECOMPUTE_INTERVAL: Final[int] = _int(
    "ADAPTIVE_THRESHOLD_RECOMPUTE_INTERVAL", 50
)
"""
Number of recorded profile actions between adaptive threshold recomputations.
"""

SHARED_DEVICE_SILHOUETTE_MIN: Final[float] = _float("SHARED_DEVICE_SILHOUETTE_MIN", 0.45)
"""
Minimum silhouette score for the k=2 cluster solution before a profile
is flagged as a shared-device account.
"""

# FEATURE EXTRACTION INTERNALS

BURST_INTERVAL_THRESHOLD: Final[float] = _float("BURST_INTERVAL_THRESHOLD", 0.5)
"""
Inter-action interval (seconds) below which an action is counted as
part of a burst.  Used when computing burst_score (feature [17]).
"""

SENSITIVE_ACTION_TYPES: Final[frozenset] = frozenset(
    {"delete", "export", "admin", "config"}
)
"""
The set of action types considered 'sensitive' for feature [14]
(sensitive_prop). 
"""

AUTOCORR_MIN_WINDOW: Final[int] = _int("AUTOCORR_MIN_WINDOW", 3)
"""
Minimum window size before lag-1 autocorrelation is computed.
"""

QUANTILE_MIN_INTERVALS: Final[int] = _int("QUANTILE_MIN_INTERVALS", 4)
"""
Minimum number of intervals before the Q1/Q3 quantile ratio is computed.
"""

DEFAULT_TIMING_MEAN: Final[float] = _float("DEFAULT_TIMING_MEAN", 5.0)
"""
Default mean inter-action interval (seconds) used when a profile has
fewer than 2 recorded intervals — i.e., not enough data for statistics.
"""

DEFAULT_TIMING_STDEV: Final[float] = _float("DEFAULT_TIMING_STDEV", 3.0)
"""
Default standard deviation of inter-action intervals used when a profile
has fewer than 2 recorded intervals.
"""

# DECISION LABELS  (string constants used in response dicts)

DECISION_ALLOW:     Final[str] = "ALLOW"
DECISION_CHALLENGE: Final[str] = "CHALLENGE"
DECISION_BLOCK:     Final[str] = "BLOCK"

ANOMALY_LOW_SIMILARITY:       Final[str] = "LOW_SIMILARITY"
ANOMALY_HIGH_ENTROPY:         Final[str] = "HIGH_ENTROPY"
ANOMALY_TEMPORAL_DRIFT:       Final[str] = "TEMPORAL_DRIFT"
ANOMALY_ML_BOT_DETECTED:      Final[str] = "ML_BOT_DETECTED"
ANOMALY_DUAL_SIGNAL_ESCALATION: Final[str] = "DUAL_SIGNAL_ESCALATION"

RISK_LOW:      Final[str] = "LOW RISK"
RISK_MEDIUM:   Final[str] = "MEDIUM RISK"
RISK_HIGH:     Final[str] = "HIGH RISK"
RISK_CRITICAL: Final[str] = "CRITICAL RISK"