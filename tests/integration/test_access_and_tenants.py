"""
Access control, tenant isolation, API keys, alerts, batch events and
cross-worker session state.
"""

import time

import pytest
from fastapi.testclient import TestClient

from api.auth import issue_jwt
from api.main import configure_state, create_app
from profiles.memory_redis import MemoryRedis


def H(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def shared_redis():
    return MemoryRedis()


@pytest.fixture
def app(shared_redis):
    a = create_app()
    configure_state(a, shared_redis)
    a.state.model_registry = None
    a.state.population_prior = None
    a.state.device_bridge = None
    a.state.shadow_mode = False
    return a


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=True)


# ── Roles ──────────────────────────────────────────────────────────────────

class TestRoles:
    def test_plain_user_cannot_use_platform_admin_routes(self, client):
        assert client.get("/admin-gate", headers=H("alice")).status_code == 403
        assert client.get("/admin-gate", headers=H("admin:acme_admin")).status_code == 403

    def test_platform_admin_can(self, client):
        assert client.get("/admin-gate", headers=H("platform_admin:ops")).status_code == 200

    def test_user_cannot_read_alerts(self, client):
        assert client.get("/alerts", headers=H("alice")).status_code == 403

    def test_viewer_can_read_alerts_but_not_label(self, client):
        assert client.get("/alerts", headers=H("viewer:v")).status_code == 200
        r = client.patch("/alerts/nope", json={"status": "false_positive"}, headers=H("viewer:v"))
        assert r.status_code == 403

    def test_user_cannot_start_session_for_someone_else(self, client):
        r = client.post("/session/start", json={"user_id": "bob"}, headers=H("alice"))
        assert r.status_code == 403

    def test_user_cannot_touch_other_users_session(self, client):
        sid = client.post("/session/start", json={"user_id": "alice"}, headers=H("alice")).json()["session_id"]
        r = client.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": "x"}, headers=H("bob"))
        assert r.status_code == 403

    def test_analyst_can_read_but_not_act_on_sessions(self, client):
        sid = client.post("/session/start", json={"user_id": "alice"}, headers=H("alice")).json()["session_id"]
        assert client.get(f"/session/{sid}/summary", headers=H("analyst:an")).status_code == 200
        r = client.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": "x"}, headers=H("analyst:an"))
        assert r.status_code == 403

    def test_plaintext_tokens_rejected_when_not_opted_in(self, client, monkeypatch):
        monkeypatch.setenv("AUTH_ALLOW_PLAINTEXT_TOKENS", "false")
        assert client.get("/auth/me", headers=H("alice")).status_code == 401

    def test_plaintext_tokens_never_allowed_in_production(self, client, monkeypatch):
        monkeypatch.setenv("ZELVRYN_ENV", "production")
        monkeypatch.setenv("JWT_SECRET_KEY", "x" * 40)
        assert client.get("/auth/me", headers=H("platform_admin:ops")).status_code == 401


# ── JWT ────────────────────────────────────────────────────────────────────

class TestJWT:
    def test_jwt_carries_tenant_and_roles(self, client):
        tok = issue_jwt("carol", "acme", ["analyst"])
        me = client.get("/auth/me", headers=H(tok)).json()
        assert me["tenant_id"] == "acme" and me["roles"] == ["analyst"]

    def test_expired_jwt_rejected(self, client):
        tok = issue_jwt("carol", "acme", ["analyst"], ttl_seconds=-10)
        assert client.get("/auth/me", headers=H(tok)).status_code == 401

    def test_forged_jwt_rejected(self, client):
        import jwt as pyjwt
        tok = pyjwt.encode({"sub": "x", "exp": time.time() + 60, "roles": ["platform_admin"]}, "wrong-secret-" * 4, algorithm="HS256")
        assert client.get("/auth/me", headers=H(tok)).status_code == 401

    def test_demo_sign_in_removed(self, client):
        r = client.post("/auth/dev-login", json={"subject": "dee", "tenant_id": "acme", "roles": ["viewer"]})
        assert r.status_code in (404, 405)


# ── API keys ───────────────────────────────────────────────────────────────

class TestApiKeys:
    def _admin(self):
        return H(issue_jwt("acme-admin", "acme", ["admin"]))

    def test_create_use_revoke(self, client):
        r = client.post("/auth/api-keys", json={"name": "backend"}, headers=self._admin())
        assert r.status_code == 201
        key, key_id = r.json()["api_key"], r.json()["key_id"]
        me = client.get("/auth/me", headers=H(key)).json()
        assert me["tenant_id"] == "acme" and me["roles"] == ["integration"]
        listed = client.get("/auth/api-keys", headers=self._admin()).json()["items"]
        assert any(k["key_id"] == key_id for k in listed)
        assert all("secret_hash" not in k and "api_key" not in k for k in listed)
        assert client.delete(f"/auth/api-keys/{key_id}", headers=self._admin()).status_code == 200
        assert client.get("/auth/me", headers=H(key)).status_code == 401

    def test_wrong_secret_rejected(self, client):
        key = client.post("/auth/api-keys", json={"name": "b"}, headers=self._admin()).json()["api_key"]
        assert client.get("/auth/me", headers=H(key[:-3] + "abc")).status_code == 401

    def test_cannot_mint_platform_admin_key(self, client):
        r = client.post("/auth/api-keys", json={"name": "x", "roles": ["platform_admin"]}, headers=self._admin())
        assert r.status_code == 403

    def test_integration_key_acts_for_any_identity_in_tenant(self, client):
        key = client.post("/auth/api-keys", json={"name": "b"}, headers=self._admin()).json()["api_key"]
        r = client.post("/session/start", json={"user_id": "agent-7", "identity_kind": "agent"}, headers=H(key))
        assert r.status_code == 201 and r.json()["identity_kind"] == "agent"
        sid = r.json()["session_id"]
        r = client.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": "t1"}, headers=H(key))
        assert r.status_code == 200


# ── Tenant isolation ───────────────────────────────────────────────────────

class TestTenantIsolation:
    def test_sessions_invisible_across_tenants(self, client):
        a = H(issue_jwt("svc", "acme", ["integration"]))
        b = H(issue_jwt("svc", "globex", ["integration", "admin"]))
        sid = client.post("/session/start", json={"user_id": "u1"}, headers=a).json()["session_id"]
        assert client.get(f"/session/{sid}/summary", headers=b).status_code == 404
        assert client.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": "x"}, headers=b).status_code == 404

    def test_profiles_isolated_across_tenants(self, client):
        a = H(issue_jwt("svc", "acme", ["integration"]))
        sid = client.post("/session/start", json={"user_id": "same-name"}, headers=a).json()["session_id"]
        client.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": "x"}, headers=a)
        acme_admin = H(issue_jwt("ad", "acme", ["admin"]))
        globex_admin = H(issue_jwt("ad", "globex", ["admin"]))
        assert client.get("/profile/same-name", headers=acme_admin).status_code == 200
        assert client.get("/profile/same-name", headers=globex_admin).status_code == 404


# ── Sessions survive across workers ────────────────────────────────────────

class TestSharedState:
    def test_session_continues_on_another_worker(self, shared_redis):
        apps = []
        for _ in range(2):
            a = create_app()
            configure_state(a, shared_redis)
            a.state.model_registry = None
            a.state.population_prior = None
            a.state.device_bridge = None
            a.state.shadow_mode = False
            apps.append(TestClient(a))
        w1, w2 = apps
        sid = w1.post("/session/start", json={"user_id": "alice"}, headers=H("alice")).json()["session_id"]
        t = time.time() - 100
        for i in range(3):
            w1.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": f"d{i}", "timestamp": t}, headers=H("alice"))
            t += 4.3 + i
        r = w2.post(f"/session/{sid}/action", json={"action_type": "read", "resource_id": "d9", "timestamp": t}, headers=H("alice"))
        assert r.status_code == 200
        assert w2.get(f"/session/{sid}/summary", headers=H("alice")).json()["metrics"]["total_actions"] == 4


# ── Alerts & events ────────────────────────────────────────────────────────

class TestAlertsAndEvents:
    def test_exfil_batch_raises_alert_and_label_exports_training(self, client):
        integ = H(issue_jwt("svc", "acme", ["integration"]))
        analyst = H(issue_jwt("ana", "acme", ["analyst"]))
        t = time.time() - 50
        events = [
            {"identity_id": "agent-9", "identity_kind": "agent", "session_id": "run-1",
             "action_type": "export", "resource_id": "customers", "records": 400, "timestamp": t + i}
            for i in range(5)
        ]
        r = client.post("/events", json={"events": events}, headers=integ)
        assert r.status_code == 200
        body = r.json()
        assert body["processed"] == 5 and body["alerts"]
        assert any("VOLUME_SPIKE" in x["anomalies"] for x in body["results"])

        alerts = client.get("/alerts", headers=analyst).json()
        assert alerts["total"] >= 1
        aid = body["alerts"][0]
        detail = client.get(f"/alerts/{aid}", headers=analyst).json()
        assert detail["identity_id"] == "agent-9" and detail["summary"]

        assert client.patch(f"/alerts/{aid}", json={"status": "confirmed_malicious"}, headers=analyst).status_code == 200
        export = client.get("/alerts/training-export", headers=analyst)
        lines = [l for l in export.text.splitlines() if l]
        assert lines and all('"label":1' in l for l in lines)

    def test_alerts_not_visible_to_other_tenant(self, client):
        integ = H(issue_jwt("svc", "acme", ["integration"]))
        client.post("/events", json={"events": [
            {"identity_id": "x", "session_id": "r", "action_type": "export",
             "resource_id": "c", "records": 5000}]}, headers=integ)
        other = H(issue_jwt("ana", "globex", ["analyst"]))
        assert client.get("/alerts", headers=other).json()["total"] == 0

    def test_tenant_shadow_mode(self, client):
        admin = H(issue_jwt("ad", "acme", ["admin"]))
        integ = H(issue_jwt("svc", "acme", ["integration"]))
        assert client.patch("/tenant/settings", json={"shadow_mode": True}, headers=admin).json()["shadow_mode"] is True
        r = client.post("/events", json={"events": [
            {"identity_id": "x", "session_id": "r2", "action_type": "export",
             "resource_id": "c", "records": 50_000}]}, headers=integ).json()
        assert r["results"][0]["decision"] == "ALLOW"
        assert r["results"][0]["alert_id"]
