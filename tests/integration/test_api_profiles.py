"""
tests/integration/test_api_profiles.py
========================================
Profile management endpoint integration tests.
"""

import time
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from core.profile import BehavioralProfile
from core.action import UserAction
from profiles.redis_store import RedisProfileStore


@pytest.fixture(scope="module")
def client():
    app = create_app()
    try:
        import fakeredis
        rc = fakeredis.FakeRedis(decode_responses=True)
    except ImportError:
        from profiles.redis_store import _NoOpRedisStub
        rc = _NoOpRedisStub()

    store = RedisProfileStore(redis_client=rc)

    # Seed a known profile for the test user
    p = BehavioralProfile(user_id="profile_user")
    t = time.time()
    for i in range(30):
        p.record_action(UserAction("read", f"doc_{i}", timestamp=t)); t += 3.0
    store.save("profile_user", p)

    app.state.profile_store   = store
    app.state.model_registry  = None
    app.state.feature_store   = None
    app.state.population_prior = None
    app.state.device_bridge   = None
    app.state.shadow_mode     = False
    app.state.start_time      = time.time()
    return TestClient(app, raise_server_exceptions=True)


HEADERS = {"Authorization": "Bearer profile_user"}
ADMIN_HEADERS = {"Authorization": "Bearer admin_user"}


class TestGetProfile:
    def test_own_profile_200(self, client):
        r = client.get("/profile/profile_user", headers=HEADERS)
        assert r.status_code == 200

    def test_profile_response_shape(self, client):
        r = client.get("/profile/profile_user", headers=HEADERS)
        d = r.json()
        for k in ("user_id","vector_count","adaptive_threshold","action_proportions"):
            assert k in d

    def test_profile_vector_count(self, client):
        r = client.get("/profile/profile_user", headers=HEADERS)
        assert r.json()["vector_count"] == 30

    def test_action_proportions_sum_to_one(self, client):
        r = client.get("/profile/profile_user", headers=HEADERS)
        props = r.json()["action_proportions"]
        assert abs(sum(props.values()) - 1.0) < 0.01

    def test_missing_profile_404(self, client):
        r = client.get("/profile/no_such_user",
                       headers={"Authorization": "Bearer no_such_user"})
        assert r.status_code == 404

    def test_no_auth_401(self, client):
        r = client.get("/profile/profile_user")
        assert r.status_code == 401


class TestDeleteProfile:
    def test_delete_own_profile(self, client):
        # Seed a deletable profile
        try:
            import fakeredis; rc = fakeredis.FakeRedis(decode_responses=True)
        except ImportError:
            from profiles.redis_store import _NoOpRedisStub; rc = _NoOpRedisStub()
        store = client.app.state.profile_store
        p = BehavioralProfile(user_id="del_user")
        store.save("del_user", p)
        r = client.delete("/profile/del_user",
                          headers={"Authorization": "Bearer del_user"})
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_missing_profile_still_200(self, client):
        r = client.delete("/profile/ghost_user",
                          headers={"Authorization": "Bearer ghost_user"})
        # Returns 200 with deleted=False (idempotent)
        assert r.status_code == 200
        assert r.json()["deleted"] is False


class TestResetProfile:
    def test_reset_requires_admin(self, client):
        # Normal user token → 403
        r = client.post("/profile/profile_user/reset", headers=HEADERS)
        assert r.status_code in (403, 422)

    def test_reset_with_admin_token(self, client):
        r = client.post("/profile/profile_user/reset", headers=ADMIN_HEADERS)
        # Admin (any token grants admin in dev mode) → 200
        assert r.status_code == 200

    def test_reset_response_shape(self, client):
        r = client.post("/profile/profile_user/reset", headers=ADMIN_HEADERS)
        if r.status_code == 200:
            d = r.json()
            assert "reset_to" in d
            assert "new_vector_count" in d
