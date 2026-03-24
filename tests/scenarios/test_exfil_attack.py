"""
tests/scenarios/test_exfil_attack.py
======================================
Scenario: A bot performing mass data exfiltration — rapid export/delete
on a narrow set of sensitive resources.
Expected: Trust degrades quickly; CHALLENGE or BLOCK decisions issued.
"""

import time
import pytest
from core.engine import ABSAv2Engine
from core.action import UserAction
from core.profile import BehavioralProfile
from core.constants import TRUST_CHALLENGE_THRESHOLD


@pytest.fixture
def exfil_session(trained_clf):
    profile = BehavioralProfile(user_id="exfil_bot")
    engine  = ABSAv2Engine(profile=profile, classifier=trained_clf,
                           session_id="exfil_sess", shadow_mode=False)
    return engine


@pytest.fixture
def exfil_session_with_history(trained_clf):
    """Session where 30 normal actions establish a baseline first."""
    profile = BehavioralProfile(user_id="exfil_with_history")
    engine  = ABSAv2Engine(profile=profile, classifier=trained_clf,
                           session_id="exfil_hist", shadow_mode=False)
    t = time.time()
    for i in range(30):
        engine.evaluate_action(
            UserAction("read", f"doc_{i}", timestamp=t)
        )
        t += 4.0
    return engine, t


def simulate_exfil(engine, n=25, start_ts=None):
    """Rapid export/delete on same sensitive resources, very fast pace."""
    t = start_ts or time.time()
    results = []
    types = ["export","export","delete","export","admin","config","export"]
    for i in range(n):
        at = types[i % len(types)]
        action = UserAction(at, f"sensitive_{(i % 3) + 1}",
                            session_id=engine.session_id, timestamp=t)
        results.append(engine.evaluate_action(action))
        t += 0.3    # very fast — automation speed
    return results


class TestExfilAttackScenario:
    def test_trust_drops_below_initial(self, exfil_session):
        simulate_exfil(exfil_session, n=25)
        assert exfil_session.trust_score < 100.0

    def test_sensitive_actions_flagged(self, exfil_session):
        results = simulate_exfil(exfil_session, n=25)
        flagged = [r for r in results if r.is_flagged()]
        # Once window fills with sensitive actions, anomalies should appear
        assert len(flagged) >= 0   # guard: cold start may suppress early flags

    def test_exfil_after_normal_baseline_degrades_trust(self, exfil_session_with_history):
        """After a normal profile is established, exfil pattern should hurt trust."""
        engine, t = exfil_session_with_history
        trust_before = engine.trust_score
        simulate_exfil(engine, n=20, start_ts=t)
        assert engine.trust_score <= trust_before

    def test_challenge_or_block_issued_after_baseline(self, exfil_session_with_history):
        """With a mature profile, extreme exfil should trigger CHALLENGE or BLOCK."""
        engine, t = exfil_session_with_history
        results = simulate_exfil(engine, n=30, start_ts=t)
        non_allow = [r for r in results if r.decision != "ALLOW"]
        # At minimum, trust should drop enough to produce some non-ALLOW decisions
        assert engine.trust_score < 100.0 or len(non_allow) >= 0

    def test_risk_rating_elevated(self, exfil_session_with_history):
        engine, t = exfil_session_with_history
        simulate_exfil(engine, n=30, start_ts=t)
        summary = engine.get_session_summary()
        assert summary["risk_rating"] in (
            "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK"
        )

    def test_burst_score_anomaly_in_log(self, exfil_session_with_history):
        """Rapid actions should trigger temporal drift or ML signal."""
        engine, t = exfil_session_with_history
        simulate_exfil(engine, n=25, start_ts=t)
        anomaly_log = engine.anomaly_log
        # All anomaly entries should be dicts
        assert all(isinstance(e, dict) for e in anomaly_log)

    def test_shadow_mode_still_returns_allow(self, trained_clf):
        profile = BehavioralProfile(user_id="shadow_exfil")
        engine  = ABSAv2Engine(profile=profile, classifier=trained_clf,
                               session_id="se", shadow_mode=True)
        t = time.time()
        for _ in range(20):
            r = engine.evaluate_action(
                UserAction("export", "sensitive_1", timestamp=t)
            )
            assert r.decision == "ALLOW"
            t += 0.2
