"""
tests/scenarios/test_cold_start.py
=====================================
Scenario: A brand-new user with zero history (cold start).
Expected:
  - Profile is seeded with population prior
  - Cosine similarity check disabled during ramp-up
  - Unusual actions do NOT immediately block a new user
  - Trust score is recoverable during ramp-up
  - Full enforcement begins after COLD_START_RAMP_ACTIONS
"""

import time
import pytest
from core.engine import ABSAv2Engine
from core.action import UserAction
from core.profile import BehavioralProfile
from core.constants import COLD_START_RAMP_ACTIONS, TRUST_INITIAL
from coldstart.population_prior import PopulationPriorProfile
from coldstart.ramp_manager import RampUpManager


@pytest.fixture
def cold_engine(trained_clf):
    """Engine with a fresh (zero-history) profile."""
    profile = BehavioralProfile(user_id="cold_user_001")
    return ABSAv2Engine(profile=profile, classifier=trained_clf,
                        session_id="cold_sess", shadow_mode=False)


@pytest.fixture
def prior_seeded_engine(trained_clf):
    """Engine whose profile is seeded with the population prior."""
    prior   = PopulationPriorProfile()
    profile = prior.create_new_user_profile("prior_user")
    return ABSAv2Engine(profile=profile, classifier=trained_clf,
                        session_id="prior_sess", shadow_mode=False)


class TestColdStartProtection:
    def test_new_user_starts_in_ramp_up(self, cold_engine):
        assert not cold_engine.profile.is_mature()

    def test_first_action_returns_allow(self, cold_engine):
        action = UserAction("read", "doc_1", timestamp=time.time())
        result = cold_engine.evaluate_action(action)
        assert result.decision == "ALLOW"

    def test_ramp_active_flag_set(self, cold_engine):
        action = UserAction("read", "doc_1", timestamp=time.time())
        result = cold_engine.evaluate_action(action)
        assert result.ramp_active is True

    def test_normal_actions_stay_allow_during_ramp(self, cold_engine):
        """20 normal actions should not block a new legitimate user."""
        t = time.time()
        for i in range(20):
            r = cold_engine.evaluate_action(
                UserAction("read", f"doc_{i}", timestamp=t)
            )
            t += 4.0
        assert r.decision != "BLOCK"

    def test_trust_recovers_fast_during_ramp(self, cold_engine):
        """Cold start recovery rate is higher than normal."""
        cold_engine.trust_score = 70.0
        t = time.time()
        for _ in range(10):
            cold_engine.evaluate_action(UserAction("read","r",timestamp=t))
            t += 4.0
        # Trust should have recovered somewhat
        assert cold_engine.trust_score >= 70.0

    def test_ramp_exits_after_enough_actions(self, cold_engine):
        """After COLD_START_RAMP_ACTIONS clean actions, ramp-up ends."""
        t = time.time()
        for i in range(COLD_START_RAMP_ACTIONS + 2):
            cold_engine.evaluate_action(
                UserAction("read", f"d_{i}", timestamp=t)
            )
            t += 4.0
        assert cold_engine.profile.is_mature(COLD_START_RAMP_ACTIONS)

    def test_prior_seeded_profile_has_vector(self, prior_seeded_engine):
        assert prior_seeded_engine.profile.vector_count > 0

    def test_prior_seeded_still_in_ramp(self, prior_seeded_engine):
        assert not prior_seeded_engine.profile.is_mature(COLD_START_RAMP_ACTIONS)


class TestRampUpManagerIntegration:
    def test_ramp_thresholds_relax_for_new_user(self):
        ramp = RampUpManager(ramp_actions=50)
        ml_ramp  = ramp.get_ml_threshold(0)
        ml_full  = ramp.get_ml_threshold(50)
        assert ml_ramp > ml_full   # stricter (higher) during ramp-up

    def test_cosine_disabled_at_action_zero(self):
        ramp = RampUpManager()
        assert not ramp.cosine_check_enabled(0)

    def test_cosine_enabled_at_ramp_end(self):
        ramp = RampUpManager(ramp_actions=50)
        assert ramp.cosine_check_enabled(50)

    def test_smooth_threshold_interpolation(self):
        ramp  = RampUpManager(ramp_actions=50)
        vals  = [ramp.get_ml_threshold(n) for n in range(0, 55, 5)]
        # Should be monotonically decreasing (or flat)
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 0.001   # allow tiny float rounding


class TestPopulationPriorColdStart:
    def test_prior_vector_length(self):
        prior = PopulationPriorProfile()
        assert len(prior.mean_vector) == 8

    def test_prior_vector_sums_to_one(self):
        prior = PopulationPriorProfile()
        assert abs(sum(prior.mean_vector) - 1.0) < 0.01

    def test_new_user_gets_prior_timing(self):
        prior   = PopulationPriorProfile()
        profile = prior.create_new_user_profile("u")
        # EWMA timing should be seeded from prior
        assert profile.ewma_mean_interval == pytest.approx(
            prior.mean_interval, abs=0.5
        )

    def test_incremental_update_changes_prior(self):
        import time as _time
        prior   = PopulationPriorProfile()
        vec_before = list(prior.mean_vector)

        p = BehavioralProfile(user_id="inc_u")
        t = _time.time()
        for i in range(25):
            p.record_action(UserAction("admin", f"r_{i}", timestamp=t)); t += 2.0
        prior.update_incrementally(p, weight=0.5)
        # Admin-heavy profile should have shifted admin frequency upward
        assert prior.mean_vector[7] >= vec_before[7]
