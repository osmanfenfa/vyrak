"""
tests/integration/test_redis_store.py
========================================
RedisProfileStore integration tests using fakeredis (in-memory Redis).
"""

import time
import pytest
from core.profile import BehavioralProfile
from core.action import UserAction
from profiles.redis_store import RedisProfileStore, ProfileStorageError

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")


@pytest.fixture
def store():
    rc = fakeredis.FakeRedis(decode_responses=True)
    return RedisProfileStore(redis_client=rc, ttl_seconds=3600)


@pytest.fixture
def sample_profile():
    p = BehavioralProfile(user_id="redis_test_user")
    t = time.time()
    for i in range(25):
        p.record_action(UserAction("read", f"doc_{i}", timestamp=t))
        t += 3.0
    return p


class TestSaveAndGet:
    def test_save_and_get_round_trip(self, store, sample_profile):
        store.save("redis_test_user", sample_profile)
        loaded = store.get("redis_test_user")
        assert loaded is not None
        assert loaded.user_id == "redis_test_user"
        assert loaded.vector_count == sample_profile.vector_count

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent_user_xyz") is None

    def test_exists_true_after_save(self, store, sample_profile):
        store.save("redis_test_user", sample_profile)
        assert store.exists("redis_test_user")

    def test_exists_false_before_save(self, store):
        assert not store.exists("never_saved_xyz")

    def test_action_history_preserved(self, store, sample_profile):
        store.save("redis_test_user", sample_profile)
        loaded = store.get("redis_test_user")
        assert len(loaded.action_history) == len(sample_profile.action_history)

    def test_ewma_stats_preserved(self, store, sample_profile):
        store.save("redis_test_user", sample_profile)
        loaded = store.get("redis_test_user")
        assert loaded.ewma_mean_interval == pytest.approx(
            sample_profile.ewma_mean_interval, abs=0.01
        )

    def test_adaptive_threshold_preserved(self, store, sample_profile):
        store.save("redis_test_user", sample_profile)
        loaded = store.get("redis_test_user")
        assert loaded.adaptive_threshold == pytest.approx(
            sample_profile.adaptive_threshold, abs=0.0001
        )


class TestDelete:
    def test_delete_returns_true_when_found(self, store, sample_profile):
        store.save("del_user", sample_profile)
        assert store.delete("del_user") is True

    def test_delete_removes_key(self, store, sample_profile):
        store.save("del_user2", sample_profile)
        store.delete("del_user2")
        assert not store.exists("del_user2")

    def test_delete_missing_returns_false(self, store):
        assert store.delete("ghost_xyz") is False


class TestGetOrCreate:
    def test_creates_new_profile_if_missing(self, store):
        p = store.get_or_create("brand_new_user")
        assert p.user_id == "brand_new_user"
        assert p.vector_count == 0

    def test_returns_existing_profile(self, store, sample_profile):
        store.save("existing_user", sample_profile)
        p = store.get_or_create("existing_user")
        assert p.vector_count == sample_profile.vector_count

    def test_uses_fallback_profile(self, store):
        fallback = BehavioralProfile(user_id="fb_user")
        fallback.vector_count = 99
        p = store.get_or_create("fb_user", fallback_profile=fallback)
        assert p.vector_count == 99


class TestTTL:
    def test_ttl_extend_returns_true_for_existing(self, store, sample_profile):
        store.save("ttl_user", sample_profile)
        assert store.ttl_extend("ttl_user") is True

    def test_ttl_extend_false_for_missing(self, store):
        assert store.ttl_extend("missing_ttl_user") is False


class TestListUsers:
    def test_list_active_users(self, store):
        for i in range(5):
            p = BehavioralProfile(user_id=f"list_user_{i}")
            store.save(f"list_user_{i}", p)
        users = store.list_active_users()
        assert any("list_user_" in u for u in users)
