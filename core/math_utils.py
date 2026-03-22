"""
Pure mathematical functions used throughout the core package.
"""
import math
import statistics
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

# SHANNON ENTROPY

def shannon_entropy(
    action_types: Sequence[str],
    num_bins: int = 8,
) -> float:
    """
    Compute normalised Shannon entropy of an action type sequence.
    Shannon entropy measures how unpredictable / random the distribution
    of action types is within a window.
    1.0
    """
    if not action_types:
        return 0.0

    counts: defaultdict = defaultdict(int)
    for a in action_types:
        counts[a] += 1

    total = len(action_types)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalise: max entropy = log2(number of distinct types actually present)
    n_unique = len(counts)
    if n_unique <= 1:
        return 0.0

    max_entropy = math.log2(min(n_unique, num_bins))
    if max_entropy <= 0:
        return 0.0

    return min(1.0, entropy / max_entropy)

# 2. COSINE SIMILARITY
def cosine_similarity(
    vec_a: Sequence[float],
    vec_b: Sequence[float],
) -> float:
    """
    Compute the cosine similarity between two vectors.
    Cosine similarity measures the cosine of the angle between two
    vectors, regardless of their magnitudes:

      cos(θ) = (A · B) / (|A| × |B|)

    In Zelvryn this is used to compare the live behaviour vector
    (derived from the rolling action window) against the stored
    long-term profile vector.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    if not vec_a:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0

    # Clamp to [0, 1] to avoid floating-point overshoot
    return max(0.0, min(1.0, dot_product / (mag_a * mag_b)))

# 3. TEMPORAL Z-SCORE
def temporal_zscore(
    interval: float,
    profile_mean: float,
    profile_stdev: float,
) -> float:
    """
    Compute the absolute Z-score of a single inter-action interval.

    The Z-score measures how many standard deviations the observed
    interval is from the user's typical rhythm:

      Z = |observed - mean| / stdev
    """
    if profile_stdev <= 0:
        return 0.0
    if interval < 0:
        return 0.0
    return abs(interval - profile_mean) / profile_stdev

# 4. BUILD LIVE VECTOR

def build_live_vector(
    actions: Sequence,
    num_types: int = 8,
) -> List[float]:
    """
    Convert a window of UserAction objects into a normalised frequency vector.
    This vector is used in the cosine similarity check against the
    stored profile mean vector.
    """
    vec = [0.0] * num_types
    if not actions:
        return vec

    for action in actions:
        idx = action.to_index()
        if 0 <= idx < num_types:
            vec[idx] += 1.0

    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]

    return vec

# EXPONENTIALLY WEIGHTED MOVING AVERAGE
def ewma_update(
    current_value: float,
    new_observation: float,
    decay_factor: float = 0.95,
    *,
    decay: Optional[float] = None,
) -> float:
    """
    Compute one step of an Exponentially Weighted Moving Average (EWMA).
    EWMA weights recent observations more heavily than older ones:

      new_ewma = λ × current + (1 − λ) × new_observation

    In Zelvryn, EWMA is applied to profile statistics (vector_sum
    components, timing statistics) to allow the profile to naturally
    follow legitimate behaviour drift over time.
    5.0
    """
    if decay is not None:
        decay_factor = decay

    if not 0.0 <= decay_factor <= 1.0:
        raise ValueError(
            f"decay_factor must be in [0, 1], got {decay_factor}"
        )
    return decay_factor * current_value + (1.0 - decay_factor) * new_observation

def ewma_update_vector(
    current_vector: List[float],
    new_vector: List[float],
    decay_factor: float = 0.95,
) -> List[float]:
    """
    Apply EWMA update element-wise across two vectors.
    Convenience wrapper used to update the entire profile vector_sum
    in a single call.
    """
    if len(current_vector) != len(new_vector):
        raise ValueError(
            f"Vector length mismatch: {len(current_vector)} vs {len(new_vector)}"
        )
    return [
        ewma_update(c, n, decay_factor)
        for c, n in zip(current_vector, new_vector)
    ]

def ewma_vector_update(
    current_vector: List[float],
    new_vector: List[float],
    decay_factor: float = 0.95,
) -> List[float]:
    """
    Backward-compatible alias for ewma_update_vector.
    """
    return ewma_update_vector(
        current_vector,
        new_vector,
        decay_factor,
    )

# 6. GINI IMPURITY
def gini_impurity(labels: Sequence[int]) -> float:
    """
    Compute the Gini impurity of a sequence of binary class labels.
    Gini impurity measures how mixed the class labels are in a node
    of a decision tree:
      Gini(node) = 1 − Σ p_c²
    """
    n = len(labels)
    if n == 0:
        return 0.0

    n_ones = sum(labels)
    p1 = n_ones / n
    p0 = 1.0 - p1

    return 1.0 - (p0 * p0 + p1 * p1)

# 7. PROFILE STATISTICS HELPERS
def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Divide safely and return default when denominator is zero-ish.
    """
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator

def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    """
    Compute the arithmetic mean, returning `default` for empty sequences.
    """
    if not values:
        return default
    return statistics.mean(values)


def safe_stdev(values: Sequence[float], default: float = 1.0) -> float:
    """
    Compute sample standard deviation, returning `default` for sequences
    with fewer than 2 elements.
    """
    if len(values) < 2:
        return default
    return max(statistics.stdev(values), 1e-9)


def compute_interval_stats(
    intervals: Sequence[float],
    default_mean: float = 5.0,
    default_stdev: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute (mean, stdev) for interval sequences with safe defaults.
    """
    if not intervals:
        return (default_mean, default_stdev)

    mean_val = safe_mean(intervals, default=default_mean)
    if len(intervals) < 2:
        return (mean_val, default_stdev)

    stdev_val = safe_stdev(intervals, default=default_stdev)
    return (mean_val, max(stdev_val, 0.1))


def coefficient_of_variation(
    values: Sequence[float],
    default: float = 0.0,
) -> float:
    """
    Compute the coefficient of variation (CV = stdev / mean).
    CV is a dimensionless measure of relative variability.  
    """
    if len(values) < 2:
        return default
    mean_val = safe_mean(values)
    if abs(mean_val) < 1e-10:
        return default
    stdev_val = safe_stdev(values)
    return stdev_val / mean_val


def lag1_autocorrelation(values: Sequence[float]) -> float:
    """
    Compute the lag-1 Pearson autocorrelation of a sequence.

    Used as feature [15] in the
    20-dimensional feature vector.

      r = Σ[(x_i - mean)(x_{i-1} - mean)] / Σ[(x_i - mean)²]
    """
    n = len(values)
    if n < 3:
        return 0.0

    mean_val = statistics.mean(values)
    numerator   = sum((values[i] - mean_val) * (values[i - 1] - mean_val)
                      for i in range(1, n))
    denominator = sum((x - mean_val) ** 2 for x in values)

    if abs(denominator) < 1e-12:
        return 0.0

    return max(-1.0, min(1.0, numerator / denominator))


def quantile_ratio(
    values: Sequence[float],
    min_count: int = 4,
) -> float:
    """
    Compute Q1/Q3 of a sorted value sequence.
    """
    if len(values) < min_count:
        return 0.0

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]

    if q3 < 1e-10:
        return 0.0

    return max(0.0, min(1.0, q1 / q3))


def run_length_mean(sequence: Sequence) -> float:
    """
    Compute the mean run length of a sequence.
    A "run" is a maximal sub-sequence of consecutive identical elements.
    """
    if len(sequence) <= 1:
        return 1.0

    runs: List[int] = []
    current_run = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    return statistics.mean(runs) if runs else 1.0

# 8. VECTOR NORMALISATION
def l1_normalise(vec: List[float]) -> List[float]:
    """
    L1-normalise a vector so its values sum to 1.0.
    """
    total = sum(abs(v) for v in vec)
    if total < 1e-10:
        return vec[:]
    return [v / total for v in vec]


def l2_normalise(vec: List[float]) -> List[float]:
    """
    L2-normalise a vector to unit length.
    """
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude < 1e-10:
        return vec[:]
    return [v / magnitude for v in vec]


def compute_risk_label(trust_score: float) -> str:
    """
    Convert trust score to the user-facing risk bucket label.
    """
    if trust_score >= 75.0:
        return "LOW RISK"
    if trust_score >= 50.0:
        return "MEDIUM RISK"
    if trust_score >= 20.0:
        return "HIGH RISK"
    return "CRITICAL RISK"
