"""tests/unit/test_features.py — FeatureExtractor unit tests."""

import time
import pytest
from core.action import UserAction
from core.features import FeatureExtractor
from core.constants import ML_FEATURE_DIM


def make_window(types, interval=3.0):
    t, actions = time.time(), []
    for at in types:
        actions.append(UserAction(at, "res_1", timestamp=t))
        t += interval
    return actions


def test_empty_window_returns_zeros():
    vec = FeatureExtractor.extract([])
    assert len(vec) == ML_FEATURE_DIM
    assert all(v == 0.0 for v in vec)


def test_vector_length():
    w = make_window(["read"] * 10)
    assert len(FeatureExtractor.extract(w)) == ML_FEATURE_DIM


def test_all_finite():
    w = make_window(["read","write","search"] * 5)
    vec = FeatureExtractor.extract(w)
    assert all(not (v != v) and abs(v) != float("inf") for v in vec)


def test_freq_vector_sums_to_one():
    w = make_window(["read","write","search","read","login"])
    vec = FeatureExtractor.extract(w)
    assert abs(sum(vec[:8]) - 1.0) < 0.01


def test_mono_action_entropy_is_zero():
    w = make_window(["read"] * 15)
    vec = FeatureExtractor.extract(w)
    entropy = vec[8]
    assert entropy < 0.05


def test_uniform_action_entropy_is_high():
    types = ["login","read","write","delete","search","export","config","admin"]
    w = make_window(types * 3)
    vec = FeatureExtractor.extract(w)
    assert vec[8] > 0.85


def test_burst_score_high_for_fast_actions():
    t, actions = time.time(), []
    for _ in range(15):
        actions.append(UserAction("read", "r", timestamp=t))
        t += 0.05           # 50ms interval — bot-speed bursts
    vec = FeatureExtractor.extract(actions)
    assert vec[17] > 0.5    # burst_score index 17


def test_burst_score_low_for_slow_actions():
    w = make_window(["read"] * 15, interval=5.0)
    vec = FeatureExtractor.extract(w)
    assert vec[17] < 0.05


def test_sensitive_prop_elevated():
    w = make_window(["export","delete","admin","export","export"])
    vec = FeatureExtractor.extract(w)
    assert vec[14] > 0.4    # sensitive_prop index 14


def test_resource_diversity_full():
    t, actions = time.time(), []
    for i in range(10):
        actions.append(UserAction("read", f"unique_res_{i}", timestamp=t))
        t += 3.0
    vec = FeatureExtractor.extract(actions)
    assert vec[16] > 0.8    # resource_div index 16


def test_feature_names_length():
    assert len(FeatureExtractor.feature_names()) == ML_FEATURE_DIM


def test_validate_vector_pass():
    w   = make_window(["read"] * 5)
    vec = FeatureExtractor.extract(w)
    assert FeatureExtractor.validate_vector(vec)


def test_validate_vector_fail_wrong_dim():
    assert not FeatureExtractor.validate_vector([0.0] * 10)


def test_batch_extract():
    windows = [make_window(["read"] * 5), make_window(["write"] * 5)]
    result  = FeatureExtractor.extract_batch(windows)
    assert len(result) == 2
    assert all(len(v) == ML_FEATURE_DIM for v in result)
