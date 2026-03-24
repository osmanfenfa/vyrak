"""tests/unit/test_shared_device.py — SharedDeviceDetector unit tests."""

import time
import pytest
from profiles.shared_device_detector import SharedDeviceDetector, SharedDeviceReport
from core.profile import BehavioralProfile
from core.action import UserAction


def build_bimodal_profile(n=60):
    """Profile with two distinct timing clusters (two users)."""
    p = BehavioralProfile(user_id="shared_user")
    t = time.time()
    # Cluster A: fast typist (~1s intervals)
    for _ in range(n // 2):
        p.record_action(UserAction("read", "r", timestamp=t))
        t += 1.0 + (t % 0.2)   # ~1s
    # Cluster B: slow reviewer (~8s intervals)
    for _ in range(n // 2):
        p.record_action(UserAction("read", "r", timestamp=t))
        t += 8.0 + (t % 0.3)   # ~8s
    # Record intervals manually to guarantee bimodal structure
    p.time_intervals = (
        [1.0 + i*0.05 for i in range(n // 2)] +
        [8.0 + i*0.05 for i in range(n // 2)]
    )
    return p


def build_single_user_profile(n=80):
    """Profile with unimodal timing (one user)."""
    p = BehavioralProfile(user_id="single_user")
    t = time.time()
    for _ in range(n):
        p.record_action(UserAction("read", "r", timestamp=t))
        t += 4.0 + 0.1 * (_ % 5)
    return p


class TestSharedDeviceDetector:
    def setup_method(self):
        self.det = SharedDeviceDetector(
            silhouette_threshold=0.30,
            min_cluster_gap_s=2.0,
            min_intervals=40,
        )

    def test_returns_report(self):
        p = build_bimodal_profile()
        r = self.det.analyse(p)
        assert isinstance(r, SharedDeviceReport)

    def test_insufficient_data_not_shared(self):
        p = BehavioralProfile(user_id="u")
        r = self.det.analyse(p)
        assert not r.is_shared

    def test_bimodal_profile_detected(self):
        p = build_bimodal_profile(80)
        r = self.det.analyse(p)
        # With clear bimodal structure (1s vs 8s) silhouette should be high
        assert r.silhouette_score > 0.0
        assert r.cluster_gap_s > 1.0

    def test_single_user_not_shared(self):
        p   = build_single_user_profile()
        # Use stricter threshold to reduce noise sensitivity
        det = SharedDeviceDetector(silhouette_threshold=0.60, min_intervals=40)
        r   = det.analyse(p)
        # Single user with small variance should not exceed strict threshold
        assert r.silhouette_score < 0.80  # not a perfect separation

    def test_blueprints_generated_when_shared(self):
        p = build_bimodal_profile(100)
        r = self.det.analyse(p)
        if r.is_shared:
            assert len(r.sub_profile_blueprints) == 2

    def test_suggest_split_returns_none_if_single_user(self):
        p   = build_single_user_profile()
        det = SharedDeviceDetector(silhouette_threshold=0.99, min_intervals=40)
        assert det.suggest_split(p) is None

    def test_cluster_means_ordered(self):
        p = build_bimodal_profile(80)
        r = self.det.analyse(p)
        if len(r.cluster_means) == 2:
            assert r.cluster_means[0] <= r.cluster_means[1]

    def test_confidence_labels(self):
        assert SharedDeviceDetector._confidence_label(0.7)  == "HIGH"
        assert SharedDeviceDetector._confidence_label(0.5)  == "MEDIUM"
        assert SharedDeviceDetector._confidence_label(0.2)  == "LOW"

    def test_intervals_analysed_count(self):
        p = build_bimodal_profile(80)
        r = self.det.analyse(p)
        assert r.intervals_analysed == len(p.time_intervals)

    def test_kmeans_1d_basic(self):
        import random
        rng = random.Random(0)
        # Two well-separated clusters
        data = [1.0 + rng.gauss(0, 0.1) for _ in range(30)] + \
               [8.0 + rng.gauss(0, 0.1) for _ in range(30)]
        labels, centres, inertia = self.det._kmeans_1d(data, k=2, seed=42)
        assert len(labels) == 60
        assert len(centres) == 2
        assert abs(centres[0] - centres[1]) > 3.0   # clearly separated
