"""
tests/integration/test_api_sessions.py
========================================
FastAPI session endpoint integration tests using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from core.profile import BehavioralProfile
from profiles.redis_store import RedisProfileStore


@pytest.fixture(scope="module")
def app():
    application = create_app()
    # Inject in-memory fakes so no Redis/DB required
    try:
        import fakeredis
        redis_client = fakeredis.FakeRedis(decode_responses=True)
    except ImportError:
        from profiles.redis_store import _NoOpRedisStub
        redis_client = _NoOpRedisStub()

    application.state.profile_store   = RedisProfileStore(redis_client=redis_client)
    application.state.model_registry  = None
    application.state.feature_store   = None
    application.state.population_prior = None
    application.state.device_bridge   = None
    application.state.shadow_mode     = False
    application.state.start_time      = __import__("time").time()
    return application


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app, raise_server_exceptions=True)


TOKEN = "test_user_001"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class TestSessionStart:
    def test_start_returns_201(self, client):
        r = client.post("/session/start", json={"user_id": TOKEN}, headers=HEADERS)
        assert r.status_code == 201

    def test_start_response_shape(self, client):
        r = client.post("/session/start", json={"user_id": TOKEN}, headers=HEADERS)
        d = r.json()
        for k in ("session_id","user_id","trust_score","ramp_active"):
            assert k in d

    def test_start_trust_is_100(self, client):
        r = client.post("/session/start", json={"user_id": TOKEN}, headers=HEADERS)
        assert r.json()["trust_score"] == 100.0

    def test_start_without_auth_returns_401(self, client):
        r = client.post("/session/start", json={"user_id": "x"})
        assert r.status_code == 401


class TestActionEvaluation:
    @pytest.fixture
    def session_id(self, client):
        r = client.post("/session/start", json={"user_id": TOKEN}, headers=HEADERS)
        return r.json()["session_id"]

    def test_evaluate_action_200(self, client, session_id):
        r = client.post(
            f"/session/{session_id}/action",
            json={"action_type": "read", "resource_id": "doc_1"},
            headers=HEADERS,
        )
        assert r.status_code == 200

    def test_evaluate_response_shape(self, client, session_id):
        r = client.post(
            f"/session/{session_id}/action",
            json={"action_type": "read", "resource_id": "doc_1"},
            headers=HEADERS,
        )
        d = r.json()
        for k in ("decision","trust_score","bot_prob","anomalies"):
            assert k in d

    def test_decision_is_valid(self, client, session_id):
        r = client.post(
            f"/session/{session_id}/action",
            json={"action_type": "write", "resource_id": "doc_2"},
            headers=HEADERS,
        )
        assert r.json()["decision"] in ("ALLOW","CHALLENGE","BLOCK")

    def test_invalid_action_type_422(self, client, session_id):
        r = client.post(
            f"/session/{session_id}/action",
            json={"action_type": "fly", "resource_id": "doc"},
            headers=HEADERS,
        )
        assert r.status_code == 422

    def test_missing_resource_id_422(self, client, session_id):
        r = client.post(
            f"/session/{session_id}/action",
            json={"action_type": "read"},
            headers=HEADERS,
        )
        assert r.status_code == 422

    def test_unknown_session_404(self, client):
        r = client.post(
            "/session/no_such_session/action",
            json={"action_type": "read", "resource_id": "r"},
            headers=HEADERS,
        )
        assert r.status_code == 404


class TestSessionSummary:
    @pytest.fixture
    def active_session(self, client):
        r = client.post("/session/start", json={"user_id": TOKEN}, headers=HEADERS)
        sid = r.json()["session_id"]
        client.post(f"/session/{sid}/action",
                    json={"action_type":"read","resource_id":"r"}, headers=HEADERS)
        return sid

    def test_summary_200(self, client, active_session):
        r = client.get(f"/session/{active_session}/summary", headers=HEADERS)
        assert r.status_code == 200

    def test_summary_has_risk_rating(self, client, active_session):
        r = client.get(f"/session/{active_session}/summary", headers=HEADERS)
        assert "risk_rating" in r.json()


class TestSessionDelete:
    def test_delete_200(self, client):
        r  = client.post("/session/start", json={"user_id": TOKEN}, headers=HEADERS)
        sid = r.json()["session_id"]
        rd = client.delete(f"/session/{sid}", headers=HEADERS)
        assert rd.status_code == 200
        assert rd.json()["session_id"] == sid
