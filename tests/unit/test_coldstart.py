"""tests/unit/test_coldstart.py — Cold start unit tests."""

import pytest
from coldstart.ramp_manager import RampUpManager
from coldstart.population_prior import PopulationPriorProfile
from coldstart.device_signals import DeviceSignalBridge
from core.profile import BehavioralProfile
from core.constants import (
    COLD_START_RAMP_ACTIONS, ML_BOT_CONFIDENCE_THRESH,
    COLD_START_ML_THRESH, TRUST_RECOVERY_RATE, COLD_START_RECOVERY_RATE,
)


# ─── RampUpManager ────────────────────────────────────────────────────────────

class TestRampUpManager:
    def setup_method(self):
        self.ramp = RampUpManager(ramp_actions=50)

    def test_warming_up_below_threshold(self):
        assert self.ramp.is_warming_up(0)
        assert self.ramp.is_warming_up(49)

    def test_not_warming_up_at_threshold(self):
        assert not self.ramp.is_warming_up(50)
        assert not self.ramp.is_warming_up(100)

    def test_ml_threshold_full_ramp(self):
        assert self.ramp.get_ml_threshold(0) == COLD_START_ML_THRESH

    def test_ml_threshold_post_ramp(self):
        assert self.ramp.get_ml_threshold(50) == ML_BOT_CONFIDENCE_THRESH

    def test_ml_threshold_interpolates(self):
        t40 = self.ramp.get_ml_threshold(40)
        t50 = self.ramp.get_ml_threshold(50)
        assert COLD_START_ML_THRESH >= t40 >= ML_BOT_CONFIDENCE_THRESH
        assert t50 == ML_BOT_CONFIDENCE_THRESH

    def test_cosine_disabled_during_ramp(self):
        assert not self.ramp.cosine_check_enabled(0)
        assert not self.ramp.cosine_check_enabled(49)

    def test_cosine_enabled_after_ramp(self):
        assert self.ramp.cosine_check_enabled(50)

    def test_recovery_rate_higher_during_ramp(self):
        assert self.ramp.get_trust_recovery_rate(0) > TRUST_RECOVERY_RATE

    def test_recovery_rate_normal_after_ramp(self):
        assert self.ramp.get_trust_recovery_rate(50) == TRUST_RECOVERY_RATE

    def test_ramp_progress(self):
        assert self.ramp.ramp_progress(0)  == pytest.approx(0.0)
        assert self.ramp.ramp_progress(25) == pytest.approx(0.5)
        assert self.ramp.ramp_progress(50) == pytest.approx(1.0)

    def test_actions_remaining(self):
        assert self.ramp.actions_remaining(10) == 40
        assert self.ramp.actions_remaining(60) == 0

    def test_get_all_thresholds_keys(self):
        d = self.ramp.get_all_thresholds(10)
        for k in ("ml_threshold","temporal_limit","trust_recovery_rate","cosine_enabled"):
            assert k in d


# ─── PopulationPriorProfile ───────────────────────────────────────────────────

class TestPopulationPrior:
    def test_default_prior_creates(self):
        p = PopulationPriorProfile()
        assert len(p.mean_vector) == 8
        assert abs(sum(p.mean_vector) - 1.0) < 0.01

    def test_create_new_user_profile_returns_behavioral_profile(self):
        prior   = PopulationPriorProfile()
        profile = prior.create_new_user_profile("new_user_001")
        assert isinstance(profile, BehavioralProfile)
        assert profile.user_id == "new_user_001"

    def test_new_user_profile_in_cold_start(self):
        prior   = PopulationPriorProfile()
        profile = prior.create_new_user_profile("u")
        assert not profile.is_mature()

    def test_build_from_profiles(self):
        import time
        profiles = []
        for i in range(15):
            p = BehavioralProfile(user_id=f"user_{i}")
            t = time.time()
            for j in range(25):
                from core.action import UserAction
                p.record_action(UserAction("read", f"r_{j}", timestamp=t))
                t += 3.0
            profiles.append(p)
        prior = PopulationPriorProfile.build_from_profiles(profiles)
        assert prior.source_profile_count == 15
        assert abs(sum(prior.mean_vector) - 1.0) < 0.01

    def test_is_stale_when_built_at_zero(self):
        p = PopulationPriorProfile()
        assert p.is_stale()

    def test_save_load_round_trip(self, tmp_path):
        prior = PopulationPriorProfile()
        path  = str(tmp_path / "prior.json")
        prior.save(path)
        loaded = PopulationPriorProfile.load(path)
        assert loaded.mean_vector == prior.mean_vector

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = PopulationPriorProfile.load(str(tmp_path / "nope.json"))
        assert isinstance(loaded, PopulationPriorProfile)


# ─── DeviceSignalBridge ────────────────────────────────────────────────────────

class TestDeviceSignals:
    def setup_method(self):
        self.bridge = DeviceSignalBridge()

    def test_assess_returns_risk_score(self):
        r = self.bridge.assess("u1", ip="1.2.3.4", user_agent="Mozilla")
        assert 0.0 <= r.composite_score <= 1.0

    def test_blocklisted_ip_scores_high(self):
        self.bridge.add_to_blocklist("10.0.0.1")
        r = self.bridge.assess("u1", ip="10.0.0.1")
        assert r.ip_risk == 1.0
        assert "IP_BLOCKLISTED" in " ".join(r.triggered_signals)

    def test_allowlisted_ip_scores_zero(self):
        self.bridge.add_to_allowlist("192.168.1.1")
        r = self.bridge.assess("u1", ip="192.168.1.1")
        assert r.ip_risk == 0.0

    def test_unknown_device_flagged(self):
        # First visit registers the device
        self.bridge.assess("u2", user_agent="Chrome/100", screen_resolution="1920x1080")
        # Different user-agent = unknown device
        r = self.bridge.assess("u2", user_agent="curl/7.0")
        assert r.fingerprint_risk > 0.0

    def test_clear_user_history(self):
        self.bridge.assess("u3", user_agent="agent")
        self.bridge.clear_user_history("u3")
        assert self.bridge.known_device_count("u3") == 0

    def test_geo_velocity_impossible_speed(self):
        import time
        t = time.time()
        # First login: London
        self.bridge._record_location("u4", 51.5, -0.1, t)
        # Second login 60 seconds later: Sydney (16000 km away)
        r = self.bridge.assess("u4", latitude=-33.8, longitude=151.2,
                               timestamp=t + 60)
        assert r.geo_velocity_risk > 0.0
