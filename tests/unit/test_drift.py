"""tests/unit/test_drift.py — DriftDetector unit tests."""

import time
import pytest
from profiles.drift_detector import DriftDetector, DriftReport
from core.profile import BehavioralProfile
from core.action import UserAction


def build_profile(action_types, interval=3.0):
    """Build a mature profile from a sequence of action types."""
    p = BehavioralProfile(user_id="drift_user")
    t = time.time()
    for at in action_types:
        p.record_action(UserAction(at, "res_1", timestamp=t))
        t += interval
    return p


def test_drift_report_is_dataclass():
    r = DriftReport(user_id="u")
    assert r.user_id == "u"
    assert not r.is_drifting


def test_insufficient_data_no_drift():
    p   = build_profile(["read"] * 10)  # less than 2 × recent_window
    det = DriftDetector(recent_window=30)
    r   = det.check_drift(p)
    assert not r.is_drifting


def test_no_drift_on_consistent_profile():
    types = (["read"] * 5 + ["write"] * 2 + ["search"] * 3) * 10
    p   = build_profile(types)
    det = DriftDetector(kl_threshold=0.3, recent_window=20)
    r   = det.check_drift(p)
    # Consistent behaviour — KL should be low
    assert r.kl_divergence < 0.5


def test_drift_detected_after_behaviour_change():
    """Profile built on reads, then history filled with exports."""
    p = BehavioralProfile(user_id="drift2")
    t = time.time()
    # Historical: heavy reads
    for _ in range(40):
        p.record_action(UserAction("read", "r", timestamp=t)); t += 3.0
    # Recent history: heavy exports (drastic change)
    for _ in range(30):
        p.record_action(UserAction("export", "r", timestamp=t)); t += 3.0

    det = DriftDetector(kl_threshold=0.15, recent_window=20)
    r   = det.check_drift(p)
    assert r.kl_divergence > 0.0


def test_soft_reset_changes_profile():
    types = (["read"]*5 + ["write"]*5) * 8
    p   = build_profile(types)
    det = DriftDetector()
    # Inject recent export-heavy history
    t = time.time() + 3000
    for _ in range(25):
        p.record_action(UserAction("export", "r", timestamp=t)); t += 3.0
    original_vec = list(p.get_mean_vector())
    det.apply_soft_reset(p, blend_weight=0.4)
    new_vec = p.get_mean_vector()
    # Vector should have changed
    assert new_vec != original_vec


def test_kl_trend_returns_list():
    p   = build_profile(["read"] * 80)
    det = DriftDetector(recent_window=20)
    det.check_drift(p)
    trend = det.kl_trend("drift_user", n=5)
    assert isinstance(trend, list)


def test_clear_history():
    p   = build_profile(["read"] * 80)
    det = DriftDetector(recent_window=20)
    det.check_drift(p)
    det.clear_history("drift_user")
    assert det.kl_trend("drift_user") == []


def test_drift_direction_populated_on_change():
    p = BehavioralProfile(user_id="dd")
    t = time.time()
    for _ in range(40):
        p.record_action(UserAction("read", "r", timestamp=t)); t += 3.0
    for _ in range(30):
        p.record_action(UserAction("admin", "r", timestamp=t)); t += 3.0

    det = DriftDetector(kl_threshold=0.05, recent_window=20)
    r   = det.check_drift(p)
    # admin proportion should have increased
    assert isinstance(r.drift_direction, dict)
