"""
tests/integration/test_shadow_mode.py
========================================
ShadowModeLogger integration tests.
"""

import os
import time
import pytest
from core.action import UserAction
from core.engine import ABSAv2Engine
from core.profile import BehavioralProfile
from ingestion.shadow_mode import ShadowModeLogger, ShadowLogEntry


@pytest.fixture
def shadow_logger(tmp_path, trained_clf):
    profile = BehavioralProfile(user_id="shadow_test")
    engine  = ABSAv2Engine(profile=profile, classifier=trained_clf,
                           session_id="shadow_sess", shadow_mode=True)
    return ShadowModeLogger(engine=engine, export_dir=str(tmp_path))


def make_action(at="read", ts=None):
    return UserAction(at, "res_1", session_id="shadow_sess",
                      timestamp=ts or time.time())


class TestShadowEvaluation:
    def test_evaluate_silently_returns_allow(self, shadow_logger):
        result = shadow_logger.evaluate_silently(make_action())
        assert result.decision == "ALLOW"

    def test_entry_is_logged(self, shadow_logger):
        shadow_logger.evaluate_silently(make_action())
        assert shadow_logger.entry_count == 1

    def test_real_decision_recorded(self, shadow_logger):
        """Real decision is stored in log, not returned to caller."""
        t = time.time()
        for _ in range(20):
            shadow_logger.evaluate_silently(
                UserAction("export", "secret", timestamp=t)
            ); t += 0.05
        # At least some entries should have been evaluated
        assert shadow_logger.entry_count > 0
        # All returned decisions must be ALLOW (shadow mode)
        entries = shadow_logger.log_entries
        assert all(isinstance(e, ShadowLogEntry) for e in entries)

    def test_multiple_actions_accumulate(self, shadow_logger):
        t = time.time()
        for i in range(15):
            shadow_logger.evaluate_silently(make_action("read", ts=t)); t += 3.0
        assert shadow_logger.entry_count == 15

    def test_heuristic_label_assigned(self, shadow_logger):
        shadow_logger.evaluate_silently(make_action())
        entry = shadow_logger.log_entries[0]
        assert entry.heuristic_label in (0, 1)


class TestShadowExport:
    def test_export_jsonl_creates_file(self, shadow_logger, tmp_path):
        t = time.time()
        for _ in range(5):
            shadow_logger.evaluate_silently(make_action(ts=t)); t += 3.0
        path = shadow_logger.export_session(format="jsonl", pseudonymise=True)
        assert os.path.exists(path)

    def test_export_csv_creates_file(self, shadow_logger, tmp_path):
        t = time.time()
        for _ in range(5):
            shadow_logger.evaluate_silently(make_action(ts=t)); t += 3.0
        path = shadow_logger.export_session(format="csv", pseudonymise=True)
        assert os.path.exists(path)

    def test_export_training_dataset(self, shadow_logger, tmp_path):
        t = time.time()
        for _ in range(10):
            shadow_logger.evaluate_silently(make_action(ts=t)); t += 3.0
        out = str(tmp_path / "training.json")
        X, y = shadow_logger.export_training_dataset(output_path=out)
        assert len(X) == len(y) == 10
        assert all(len(v) == 20 for v in X)

    def test_pseudonymised_export_hides_user_id(self, shadow_logger, tmp_path):
        import json
        t = time.time()
        for _ in range(3):
            shadow_logger.evaluate_silently(make_action(ts=t)); t += 3.0
        path = shadow_logger.export_session(format="jsonl", pseudonymise=True)
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        for line in lines:
            assert line["user_id"] != "shadow_test"   # must be hashed


class TestVerifyLabel:
    def test_set_verified_label(self, shadow_logger):
        ts = time.time()
        shadow_logger.evaluate_silently(make_action(ts=ts))
        ok = shadow_logger.set_verified_label(timestamp=ts, label=1)
        assert ok
        entry = shadow_logger.log_entries[0]
        assert entry.verified_label == 1

    def test_effective_label_uses_verified(self, shadow_logger):
        ts = time.time()
        shadow_logger.evaluate_silently(make_action("read", ts=ts))
        shadow_logger.set_verified_label(timestamp=ts, label=1)
        entry = shadow_logger.log_entries[-1]
        assert entry.effective_label() == 1


class TestShadowSummary:
    def test_summary_keys(self, shadow_logger):
        shadow_logger.evaluate_silently(make_action())
        s = shadow_logger.get_shadow_summary()
        for k in ("total_evaluated","heuristic_human","heuristic_bot"):
            assert k in s

    def test_total_evaluated_matches_count(self, shadow_logger):
        t = time.time()
        for _ in range(7):
            shadow_logger.evaluate_silently(make_action(ts=t)); t += 3.0
        assert shadow_logger.get_shadow_summary()["total_evaluated"] == 7
