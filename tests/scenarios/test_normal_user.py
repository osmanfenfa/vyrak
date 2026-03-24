"""
tests/scenarios/test_normal_user.py
=====================================
Scenario: A legitimate user performs a normal work session.
Expected: All actions ALLOW, trust stays high, no anomalies flagged.
"""

import time
import pytest
from core.engine import ABSAv2Engine
from core.action import UserAction
from core.profile import BehavioralProfile


@pytest.fixture
def normal_session(trained_clf):
    profile = BehavioralProfile(user_id="normal_alice")
    engine  = ABSAv2Engine(profile=profile, classifier=trained_clf,
                           session_id="normal_sess", shadow_mode=False)
    return engine


def simulate_normal_session(engine, n=20):
    """Simulate a realistic read-heavy work session."""
    patterns = [
        ("login","portal"),("read","inbox"),("read","doc_1"),
        ("search","query_1"),("read","doc_2"),("write","note_1"),
        ("read","doc_3"),("read","doc_4"),("search","query_2"),
        ("read","doc_5"),("write","note_2"),("read","report_1"),
        ("search","query_3"),("read","doc_6"),("read","doc_7"),
        ("write","note_3"),("read","doc_8"),("search","query_4"),
        ("read","doc_9"),("read","doc_10"),
    ]
    t = time.time()
    results = []
    for i in range(n):
        at, rid = patterns[i % len(patterns)]
        action = UserAction(at, rid, session_id=engine.session_id, timestamp=t)
        results.append(engine.evaluate_action(action))
        t += 3.5 + (i % 3) * 1.5   # 3.5–8.0 second intervals
    return results


class TestNormalUserScenario:
    def test_all_actions_allowed(self, normal_session):
        results = simulate_normal_session(normal_session, n=20)
        blocked = [r for r in results if r.decision == "BLOCK"]
        assert len(blocked) == 0

    def test_trust_stays_above_50(self, normal_session):
        results = simulate_normal_session(normal_session, n=20)
        min_trust = min(r.trust_score for r in results)
        assert min_trust >= 50.0

    def test_final_trust_high(self, normal_session):
        simulate_normal_session(normal_session, n=20)
        assert normal_session.trust_score >= 75.0

    def test_risk_rating_low(self, normal_session):
        simulate_normal_session(normal_session, n=20)
        summary = normal_session.get_session_summary()
        assert summary["risk_rating"] in ("LOW RISK", "MEDIUM RISK")

    def test_session_summary_correct_counts(self, normal_session):
        n = 20
        simulate_normal_session(normal_session, n=n)
        summary = normal_session.get_session_summary()
        assert summary["total_actions"] == n

    def test_profile_grows_on_clean_actions(self, normal_session):
        before = normal_session.profile.vector_count
        simulate_normal_session(normal_session, n=10)
        assert normal_session.profile.vector_count > before
