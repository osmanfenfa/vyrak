"""tests/unit/test_math_utils.py — math_utils unit tests."""

import math
import pytest
from core.math_utils import (
    shannon_entropy, cosine_similarity, build_live_vector,
    temporal_zscore, ewma_update, ewma_vector_update,
    lag1_autocorrelation, run_length_mean, quantile_ratio,
    coefficient_of_variation, safe_divide, compute_risk_label,
)


# ─── Shannon entropy ──────────────────────────────────────────────────────────

def test_entropy_single_value_is_zero():
    assert shannon_entropy(["read"] * 10) == 0.0


def test_entropy_uniform_is_one():
    vals = list("abcdefgh")
    assert shannon_entropy(vals) == pytest.approx(1.0, abs=0.01)


def test_entropy_empty_is_zero():
    assert shannon_entropy([]) == 0.0


def test_entropy_range():
    v = shannon_entropy(["read","write","search","login"], num_bins=8)
    assert 0.0 <= v <= 1.0


# ─── Cosine similarity ────────────────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_zero_vector():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_cosine_range():
    a = [0.4, 0.3, 0.2, 0.1]
    b = [0.1, 0.2, 0.3, 0.4]
    s = cosine_similarity(a, b)
    assert -1.0 <= s <= 1.0


# ─── Live vector ─────────────────────────────────────────────────────────────

def test_live_vector_length():
    assert len(build_live_vector([0, 1, 2, 3], num_types=8)) == 8


def test_live_vector_sums_to_one():
    v = build_live_vector([1, 1, 1, 2], num_types=8)
    assert abs(sum(v) - 1.0) < 1e-6


def test_live_vector_empty():
    assert build_live_vector([], 8) == [0.0] * 8


# ─── Temporal Z-score ────────────────────────────────────────────────────────

def test_temporal_zscore_normal():
    z = temporal_zscore(4.2, 4.0, 1.0)
    assert z == pytest.approx(0.2, abs=1e-6)


def test_temporal_zscore_zero_stdev():
    assert temporal_zscore(4.0, 4.0, 0.0) == 0.0


def test_temporal_zscore_fast_action():
    z = temporal_zscore(0.05, 4.0, 1.5)
    assert z > 2.0


# ─── EWMA ─────────────────────────────────────────────────────────────────────

def test_ewma_converges():
    v = 10.0
    for _ in range(200):
        v = ewma_update(v, 0.0, decay=0.9)
    assert v < 0.1


def test_ewma_unchanged_if_equal():
    assert ewma_update(5.0, 5.0, 0.95) == pytest.approx(5.0)


def test_ewma_vector_length():
    c = [1.0, 2.0, 3.0]
    n = [0.0, 0.0, 0.0]
    assert len(ewma_vector_update(c, n, 0.9)) == 3


def test_ewma_vector_mismatch_raises():
    with pytest.raises(ValueError):
        ewma_vector_update([1.0, 2.0], [1.0], 0.9)


# ─── Autocorrelation ─────────────────────────────────────────────────────────

def test_autocorr_alternating_is_negative():
    seq = [float(i % 2) for i in range(20)]
    assert lag1_autocorrelation(seq) < -0.5


def test_autocorr_constant_is_zero():
    assert lag1_autocorrelation([3.0] * 10) == 0.0


def test_autocorr_short_sequence():
    assert lag1_autocorrelation([1.0]) == 0.0


# ─── Run-length ──────────────────────────────────────────────────────────────

def test_run_length_all_same():
    assert run_length_mean(["a"] * 5) == 5.0


def test_run_length_alternating():
    assert run_length_mean(["a","b","a","b"]) == 1.0


def test_run_length_empty():
    assert run_length_mean([]) == 1.0


# ─── Quantile ratio ──────────────────────────────────────────────────────────

def test_quantile_ratio_uniform():
    vals = [1.8, 1.82, 1.79, 1.81, 1.80, 1.81, 1.80, 1.79]
    assert quantile_ratio(vals) > 0.8


def test_quantile_ratio_spread():
    vals = [0.5, 2.3, 8.1, 0.8, 15.2, 1.1, 3.4, 7.2]
    assert quantile_ratio(vals) < 0.5


def test_quantile_ratio_short():
    assert quantile_ratio([1.0, 2.0]) == 0.0


# ─── CV ───────────────────────────────────────────────────────────────────────

def test_cv_zero_mean():
    assert coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0


def test_cv_positive():
    assert coefficient_of_variation([1.0, 2.0, 3.0, 4.0]) > 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def test_safe_divide_zero():
    assert safe_divide(5.0, 0.0) == 0.0
    assert safe_divide(5.0, 0.0, default=99.0) == 99.0


def test_safe_divide_normal():
    assert safe_divide(6.0, 2.0) == 3.0


def test_risk_label():
    assert compute_risk_label(90.0)  == "LOW RISK"
    assert compute_risk_label(60.0)  == "MEDIUM RISK"
    assert compute_risk_label(30.0)  == "HIGH RISK"
    assert compute_risk_label(10.0)  == "CRITICAL RISK"
