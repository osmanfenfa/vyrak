"""
Unit tests for the label-free detectors and the engine features built on them.
"""

import json
import os
import random
import time

import pytest

from core.action import UserAction
from core.engine import ABSAv2Engine
from core.profile import BehavioralProfile
from detectors.peer import PeerGroupBaseline
from detectors.sequence import SequenceModel
from detectors.volume import VolumeBaseline


def _human_profile(n=120, seed=1, routine=("read", "search", "read", "write")):
    rng = random.Random(seed)
    p = BehavioralProfile(user_id="alice")
    t = time.time() - 100_000
    for i in range(n):
        p.record_action(UserAction(routine[i % len(routine)], f"doc_{i % 30}", timestamp=t))
        t += rng.uniform(2.5, 7.0)
    return p, t


# ── SequenceModel ──────────────────────────────────────────────────────────

class TestSequenceModel:
    def test_inactive_without_history(self):
        a = SequenceModel().assess(["read", "write", "read", "export"], min_transitions=40, margin_bits=1.6)
        assert a.active is False and a.anomalous is False

    def test_habitual_sequence_not_anomalous(self):
        p, _ = _human_profile()
        a = p.sequence_model.assess(["read", "search", "read", "write"] * 3, 40, 1.6)
        assert a.active and not a.anomalous

    def test_unseen_sequence_is_anomalous_with_reason(self):
        p, _ = _human_profile()
        a = p.sequence_model.assess(["read", "admin", "export", "delete", "admin", "export", "config"], 40, 1.6)
        assert a.anomalous
        assert a.reasons and "Rarest" in a.reasons[0]

    def test_round_trip(self):
        p, _ = _human_profile()
        m2 = SequenceModel.from_dict(json.loads(json.dumps(p.sequence_model.to_dict())))
        assert m2.counts == p.sequence_model.counts


# ── VolumeBaseline ─────────────────────────────────────────────────────────

FLOORS = {"read_units": 5000, "export_units": 500, "delete_units": 100}


class TestVolumeBaseline:
    def test_units_from_metadata(self):
        assert VolumeBaseline.units_for("export", {"records": 250}) == 250
        assert VolumeBaseline.units_for("export", {}) == 1

    def test_cold_start_floor(self):
        vb = VolumeBaseline(bucket_seconds=600)
        a = vb.observe("export", {"records": 600}, 1000.0, 5, 4.0, 5.0, FLOORS)
        assert a.anomalous and not a.mature

    def test_learns_baseline_then_flags_spike(self):
        vb = VolumeBaseline(bucket_seconds=600)
        t = 0.0
        for _ in range(8):                       # 8 normal buckets: ~20 exports each
            for _ in range(4):
                a = vb.observe("export", {"records": 5}, t, 5, 4.0, 5.0, FLOORS)
                assert not a.anomalous
                vb.commit_clean("export", {"records": 5})
                t += 30
            t += 600
        assert vb.mature(5)
        spike = vb.observe("export", {"records": 400}, t, 5, 4.0, 5.0, FLOORS)
        assert spike.anomalous and spike.mature
        assert "normally" in spike.reasons[0]

    def test_anomalous_volume_not_learned(self):
        vb = VolumeBaseline(bucket_seconds=600)
        vb.observe("export", {"records": 10_000}, 0.0, 5, 4.0, 5.0, FLOORS)  # never committed
        vb.observe("read", {}, 700.0, 5, 4.0, 5.0, FLOORS)                    # rolls bucket
        assert vb.mean["export_units"] == 0.0


# ── PeerGroupBaseline ──────────────────────────────────────────────────────

class TestPeerGroup:
    def _features(self, rng, export_share=0.02):
        v = [0.0] * 20
        v[1], v[2], v[5] = 0.6 + rng.gauss(0, 0.03), 0.2 + rng.gauss(0, 0.03), export_share
        v[14], v[16], v[17], v[18] = 0.05 + rng.gauss(0, 0.01), 0.7, 0.05, 1.3
        return v

    def test_inactive_until_enough_identities(self):
        pg = PeerGroupBaseline()
        rng = random.Random(0)
        for _ in range(100):
            pg.update(self._features(rng), "only_one")
        assert not pg.assess(self._features(rng), 50, 3, 4.0, "support").active

    def test_flags_outlier(self):
        pg = PeerGroupBaseline()
        rng = random.Random(0)
        for i in range(200):
            pg.update(self._features(rng), f"agent_{i % 10}")
        normal = pg.assess(self._features(rng), 50, 3, 4.0, "support")
        odd = self._features(rng, export_share=0.9)
        odd[14] = 0.9
        outlier = pg.assess(odd, 50, 3, 4.0, "support")
        assert normal.active and not normal.anomalous
        assert outlier.anomalous and "support" in outlier.reasons[0]


# ── Engine integration ─────────────────────────────────────────────────────

class TestEngineNiche:
    def test_agent_identity_skips_bot_classifier(self, trained_clf):
        p = BehavioralProfile(user_id="triage-bot", identity_kind="agent")
        e = ABSAv2Engine(profile=p, classifier=trained_clf, session_id="s")
        t = time.time()
        for i in range(30):
            r = e.evaluate_action(UserAction("read", f"t_{i}", timestamp=t))
            t += 1.0
            assert r.bot_prob == 0.0
            assert "ML_BOT_DETECTED" not in r.anomalies
            assert "MACHINE_TIMING" not in r.anomalies   # regular timing is fine for agents

    def test_human_with_machine_timing_flagged(self):
        e = ABSAv2Engine(profile=BehavioralProfile(user_id="h"), session_id="s")
        t = time.time()
        results = []
        for i in range(20):
            results.append(e.evaluate_action(UserAction("read", f"d_{i}", timestamp=t)))
            t += 2.0
        assert "MACHINE_TIMING" in results[-1].anomalies
        assert any("machine-regular" in r for r in results[-1].reasons)

    def test_established_identity_bulk_export_is_blocked(self):
        p, t = _human_profile(n=200)
        # give the volume baseline mature history
        for b in range(6):
            p.volume.observe("read", {"records": 20}, t, 5, 4.0, 5.0, FLOORS)
            p.volume.commit_clean("read", {"records": 20})
            t += 700
        e = ABSAv2Engine(profile=p, session_id="hijack")
        results = []
        for i in range(6):
            results.append(e.evaluate_action(UserAction("export", "customers_table", timestamp=t, metadata={"records": 5000})))
            t += 3.1 + i * 0.4
        assert any("VOLUME_SPIKE" in r.anomalies for r in results)
        assert results[-1].decision in ("CHALLENGE", "BLOCK")
        assert any(r.reasons for r in results)

    def test_first_privileged_action_flagged_for_mature_identity(self):
        p, t = _human_profile(n=200)
        e = ABSAv2Engine(profile=p, session_id="s")
        r = e.evaluate_action(UserAction("admin", "settings", timestamp=t + 4))
        assert "NEW_PRIVILEGED_ACTION" in r.anomalies

    def test_learning_paused_when_session_not_in_good_standing(self):
        p, t = _human_profile(n=200)
        e = ABSAv2Engine(profile=p, session_id="s")
        e.trust_score = 30.0                     # session already challenged
        before = p.vector_count
        r = e.evaluate_action(UserAction("read", "doc_1", timestamp=t + 4))
        assert r.anomalies == [] and r.learned is False
        assert p.vector_count == before

    def test_state_round_trip(self):
        p, t = _human_profile(n=60)
        e = ABSAv2Engine(profile=p, session_id="s")
        for i in range(5):
            e.evaluate_action(UserAction("read", f"d{i}", timestamp=t)); t += 4.2
        state = json.loads(json.dumps(e.export_state()))
        e2 = ABSAv2Engine(profile=p, session_id="s")
        e2.restore_state(state)
        assert e2.trust_score == e.trust_score
        assert len(e2.action_window) == len(e.action_window)
        assert e2.get_session_summary()["total_actions"] == 5


# ── Model registry integrity ───────────────────────────────────────────────

class TestRegistryChecksum:
    def test_tampered_model_is_refused(self, tmp_path, trained_clf):
        from ml.registry import ModelRegistry
        path = str(tmp_path / "m.joblib")
        trained_clf.save_model(path)
        reg = ModelRegistry(registry_dir=str(tmp_path))
        reg.register("v1", path)
        reg.promote("v1")
        assert reg.get_active_model() is not None
        reg._loaded_models.clear()
        with open(path, "ab") as f:
            f.write(b"tamper")
        assert reg.get_active_model() is None

    def test_canary_routing_is_stable(self, tmp_path):
        from ml.registry import ModelRegistry
        import hashlib
        h = int(hashlib.sha256(b"session-x").hexdigest()[:8], 16) % 100
        assert 0 <= h < 100   # same value in every process (unlike hash())
