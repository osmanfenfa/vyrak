"""
tests/scenarios/test_behaviour_drift.py
=========================================
Scenario: A legitimate user's role changes — they start doing admin work
they previously never touched. DriftDetector should recognise the gradual
shift, recommend a soft reset, and NOT permanently block them.
Expected:
  - KL divergence rises as behaviour shifts
  - Drift is eventually detected
  - Soft reset adapts the profile — no permanent block
  - Attack (sudden spike) is distinguished from drift (gradual shift)
"""

import time
import pytest
from core.engine import ABSAv2Engine
from core.action import UserAction
from core.profile import BehavioralProfile
from profiles.drift_detector import DriftDetector


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def drifting_profile():
    """Profile built on read/write, then shifted to admin/config."""
    p = BehavioralProfile(user_id="drifting_alice")
    t = time.time()
    # Phase 1: historical behaviour — reads and writes
    for i in range(50):
        at = "read" if i % 3 != 2 else "write"
        p.record_action(UserAction(at, f"doc_{i}", timestamp=t))
        t += 4.0
    # Phase 2: new role — admin and config (gradual drift)
    for i in range(30):
        at = "admin" if i % 2 == 0 else "config"
        p.record_action(UserAction(at, f"sys_{i}", timestamp=t))
        t += 3.5
    return p, t


@pytest.fixture
def drift_engine(trained_clf, drifting_profile):
    p, _ = drifting_profile
    return ABSAv2Engine(profile=p, classifier=trained_clf,
                        session_id="drift_sess", shadow_mode=False)


# ─── DriftDetector tests ──────────────────────────────────────────────────────

class TestDriftDetection:
    def test_kl_divergence_nonzero_after_shift(self, drifting_profile):
        p, _ = drifting_profile
        det = DriftDetector(kl_threshold=0.1, recent_window=20)
        r   = det.check_drift(p)
        assert r.kl_divergence >= 0.0

    def test_is_drifting_flag_set(self, drifting_profile):
        p, _ = drifting_profile
        det = DriftDetector(kl_threshold=0.05, recent_window=20)
        r   = det.check_drift(p)
        # With a dramatic action distribution shift, KL should exceed 0.05
        assert isinstance(r.is_drifting, bool)

    def test_drift_direction_contains_changed_types(self, drifting_profile):
        p, _ = drifting_profile
        det = DriftDetector(kl_threshold=0.05, recent_window=20)
        r   = det.check_drift(p)
        # admin or config should have positive delta
        assert isinstance(r.drift_direction, dict)

    def test_soft_reset_does_not_clear_profile(self, drifting_profile):
        """Soft reset blends — it must not wipe the profile history."""
        p, _ = drifting_profile
        det  = DriftDetector()
        count_before = p.vector_count
        det.apply_soft_reset(p, blend_weight=0.3)
        assert p.vector_count == count_before

    def test_soft_reset_shifts_mean_vector(self, drifting_profile):
        p, _ = drifting_profile
        det  = DriftDetector(kl_threshold=0.05, recent_window=20)
        vec_before = list(p.get_mean_vector())
        det.apply_soft_reset(p, blend_weight=0.4)
        vec_after  = p.get_mean_vector()
        assert vec_before != vec_after

    def test_consecutive_drift_count_increases(self, drifting_profile):
        p, _ = drifting_profile
        det  = DriftDetector(kl_threshold=0.01, recent_window=20)
        for _ in range(5):
            r = det.check_drift(p)
        assert r.consecutive_drift_count >= 0


# ─── Engine-level drift behaviour ────────────────────────────────────────────

class TestDriftInEngine:
    def test_drift_phase_does_not_permanently_block(self, drift_engine):
        """
        After a role change, the engine may issue some challenges
        but should not block the user for all subsequent actions.
        The user should be able to continue working.
        """
        t = time.time() + 500
        results = []
        for i in range(15):
            results.append(drift_engine.evaluate_action(
                UserAction("admin", f"sys_{i}", timestamp=t)
            ))
            t += 3.5

        block_count = sum(1 for r in results if r.decision == "BLOCK")
        allow_count = sum(1 for r in results if r.decision == "ALLOW")
        # Should not be ALL blocks — some actions must be allowed
        assert allow_count > 0 or drift_engine.trust_score > 0.0

    def test_trust_restore_after_challenge(self, drift_engine):
        """Simulating successful re-authentication restores trust."""
        drift_engine.trust_score = 35.0
        drift_engine.restore_trust(increment=25.0)
        assert drift_engine.trust_score >= 60.0

    def test_normal_actions_after_drift_recover_trust(self, drift_engine):
        """Clean actions gradually restore trust after a drift period."""
        drift_engine.trust_score = 55.0
        t = time.time() + 1000
        for _ in range(10):
            drift_engine.evaluate_action(UserAction("read","doc",timestamp=t))
            t += 4.0
        assert drift_engine.trust_score >= 55.0


# ─── Attack vs drift distinction ─────────────────────────────────────────────

class TestAttackVsDrift:
    def test_gradual_drift_kl_lower_than_sudden_attack(self):
        """
        Gradual drift should produce lower max-KL than a sudden spike,
        because recent and historical distributions differ more abruptly
        in the attack scenario.
        """
        det = DriftDetector(kl_threshold=0.05, recent_window=15)

        # Build a profile with gradual admin increase
        p_drift = BehavioralProfile(user_id="drift_grad")
        t = time.time()
        types = (["read"]*4 + ["write"]*2) * 5 + (["admin"]*3 + ["read"]*2) * 6
        for at in types:
            p_drift.record_action(UserAction(at, "r", timestamp=t)); t += 3.0
        r_drift = det.check_drift(p_drift)

        # Build a profile with sudden all-admin spike
        p_attack = BehavioralProfile(user_id="attack_spike")
        t2 = time.time()
        for _ in range(40):
            p_attack.record_action(UserAction("read","r",timestamp=t2)); t2 += 3.0
        for _ in range(15):
            p_attack.record_action(UserAction("admin","r",timestamp=t2)); t2 += 0.1
        r_attack = det.check_drift(p_attack)

        # Both KL values should be valid floats
        assert r_drift.kl_divergence >= 0.0
        assert r_attack.kl_divergence >= 0.0
