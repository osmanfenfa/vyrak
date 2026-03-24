"""
tests/conftest.py
==================
Shared pytest fixtures used across unit, integration, and scenario tests.
"""

import sys
import time
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.action import UserAction
from core.profile import BehavioralProfile
from core.engine import ABSAv2Engine
from core.classifier import RandomForestClassifier
from core.constants import BEHAVIOR_WINDOW_SIZE


# ─── Action helpers ────────────────────────────────────────────────────────────

def make_action(action_type="read", resource="doc_1", ts=None, session="sess"):
    return UserAction(action_type=action_type, resource_id=resource,
                      session_id=session, timestamp=ts or time.time())


def action_sequence(types, base_ts=None, interval=3.0, session="sess"):
    """Build a list of UserActions with regular timing."""
    t = base_ts or time.time()
    actions = []
    for at in types:
        actions.append(make_action(at, "res_1", ts=t, session=session))
        t += interval
    return actions


# ─── Synthetic training data ───────────────────────────────────────────────────

def synthetic_xy(n_each=60, seed=42):
    """Return (X, y) with n_each human and n_each bot samples."""
    import random
    rng = random.Random(seed)

    def _vec(human: bool):
        if human:
            return [
                0.05, 0.48, 0.19, 0.02, 0.17, 0.03, 0.03, 0.02,  # freq 0-7
                0.62,           # entropy
                0.12, 0.09, 0.75, 0.01, 0.28,   # interval stats
                0.08, 0.20, 0.62, 0.03, 0.05, 0.22,   # remaining
            ]
        else:
            return [
                0.04, 0.55, 0.12, 0.02, 0.12, 0.03, 0.02, 0.01,
                0.58,
                0.04, 0.003, 0.08, 0.0, 0.10,
                0.04, 0.82, 0.25, 0.55, 0.35, 0.88,
            ]

    X, y = [], []
    for _ in range(n_each):
        noise = [rng.gauss(0, 0.02) for _ in range(20)]
        hv = [max(0.0, min(1.0, v + noise[i])) for i, v in enumerate(_vec(True))]
        bv = [max(0.0, min(1.0, v + noise[i])) for i, v in enumerate(_vec(False))]
        X.append(hv); y.append(0)
        X.append(bv); y.append(1)
    return X, y


# ─── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def profile():
    return BehavioralProfile(user_id="test_user_001")


@pytest.fixture
def mature_profile():
    """A profile with 60 recorded actions and stable statistics."""
    p = BehavioralProfile(user_id="mature_user")
    t = time.time()
    for i in range(60):
        at = ["read","write","search","read","read"][i % 5]
        p.record_action(make_action(at, f"doc_{i%20}", ts=t))
        t += 3.5 + (i % 3) * 0.5
    return p


@pytest.fixture
def trained_clf():
    """A RandomForestClassifier trained on synthetic data."""
    clf = RandomForestClassifier(n_trees=5, max_depth=4, random_state=0)
    X, y = synthetic_xy(n_each=40)
    clf.fit(X, y)
    return clf


@pytest.fixture
def engine(mature_profile, trained_clf):
    return ABSAv2Engine(
        profile=mature_profile,
        classifier=trained_clf,
        session_id="test_session",
        shadow_mode=False,
    )


@pytest.fixture
def shadow_engine(mature_profile, trained_clf):
    return ABSAv2Engine(
        profile=mature_profile,
        classifier=trained_clf,
        session_id="shadow_session",
        shadow_mode=True,
    )


@pytest.fixture
def fresh_engine():
    """Engine with a brand-new (cold-start) profile — no history."""
    profile = BehavioralProfile(user_id="fresh_user")
    return ABSAv2Engine(profile=profile, session_id="fresh_sess")
