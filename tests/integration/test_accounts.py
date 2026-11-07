"""
Accounts: operator bootstrap, customer onboarding, invites, team management,
lockout, suspension, password reset and token revocation.
"""

import logging
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from api.accounts import bootstrap_platform_admin
from api.main import configure_state, create_app
from profiles.memory_redis import MemoryRedis

OP_EMAIL = "owner@zelvryn.example"
OP_PASSWORD = "operator-pass-123"


def H(token):
    return {"Authorization": f"Bearer {token}"}


def token_from(link):
    return parse_qs(urlparse(link.replace("/#/", "/")).query)["token"][0]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_ADMIN_EMAIL", OP_EMAIL)
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", OP_PASSWORD)
    monkeypatch.setenv("DASHBOARD_URL", "https://console.example")
    app = create_app()
    configure_state(app, MemoryRedis())
    app.state.model_registry = None
    app.state.population_prior = None
    app.state.device_bridge = None
    bootstrap_platform_admin(app.state.account_store, app.state.tenant_store, logging.getLogger("t"))
    return TestClient(app)


def login(client, email, password):
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def onboard(client, op, name="Acme Corp", admin="lead@acme.example"):
    r = client.post("/platform/customers", json={"name": name, "admin_email": admin}, headers=H(op))
    assert r.status_code == 201, r.text
    return r.json()


def accept(client, link, name="Lead", password="lead-password-1"):
    r = client.post("/auth/accept-invite", json={"token": token_from(link), "name": name, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


class TestOperator:
    def test_bootstrap_operator_can_sign_in(self, client):
        tok = login(client, OP_EMAIL, OP_PASSWORD)
        assert client.get("/auth/me", headers=H(tok)).json()["roles"] == ["platform_admin"]

    def test_email_is_case_insensitive(self, client):
        login(client, OP_EMAIL.upper(), OP_PASSWORD)

    def test_wrong_password_is_generic(self, client):
        r1 = client.post("/auth/login", json={"email": OP_EMAIL, "password": "nope-nope-nope"})
        r2 = client.post("/auth/login", json={"email": "nobody@x.example", "password": "nope-nope-nope"})
        assert r1.status_code == r2.status_code == 401
        assert r1.json()["detail"] == r2.json()["detail"]

    def test_lockout_after_repeated_failures(self, client):
        for _ in range(5):
            client.post("/auth/login", json={"email": OP_EMAIL, "password": "wrong-password"})
        r = client.post("/auth/login", json={"email": OP_EMAIL, "password": OP_PASSWORD})
        assert r.status_code == 429

    def test_non_operator_cannot_onboard(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        lead = accept(client, onboard(client, op)["invite_link"])
        r = client.post("/platform/customers", json={"name": "X Co", "admin_email": "a@x.example"}, headers=H(lead))
        assert r.status_code == 403


class TestOnboarding:
    def test_create_customer_and_first_admin_joins(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        data = onboard(client, op)
        assert data["customer"]["tenant_id"] == "acme-corp"
        assert data["invite_link"].startswith("https://console.example/#/accept-invite?token=")

        info = client.get(f"/auth/invites/{token_from(data['invite_link'])}").json()
        assert info["organisation"] == "Acme Corp" and info["roles"] == ["admin"]

        lead = accept(client, data["invite_link"])
        me = client.get("/auth/me", headers=H(lead)).json()
        assert me["tenant_id"] == "acme-corp" and me["roles"] == ["admin"]
        # new customers start in shadow mode
        assert client.get("/tenant/settings", headers=H(lead)).json()["shadow_mode"] is True

    def test_invite_link_works_once(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        link = onboard(client, op)["invite_link"]
        accept(client, link)
        r = client.post("/auth/accept-invite", json={"token": token_from(link), "name": "x", "password": "another-pass-1"})
        assert r.status_code == 404

    def test_duplicate_org_id_rejected(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        onboard(client, op)
        r = client.post("/platform/customers", json={"name": "Acme Corp", "admin_email": "b@acme.example"}, headers=H(op))
        assert r.status_code == 409

    def test_short_password_rejected(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        link = onboard(client, op)["invite_link"]
        r = client.post("/auth/accept-invite", json={"token": token_from(link), "name": "L", "password": "short"})
        assert r.status_code == 422

    def test_customer_list(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        onboard(client, op)
        items = client.get("/platform/customers", headers=H(op)).json()["items"]
        assert [c["tenant_id"] for c in items] == ["acme-corp"]
        assert items[0]["pending_invites"] == 1


class TestTeam:
    def _lead(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        return op, accept(client, onboard(client, op)["invite_link"])

    def test_admin_invites_analyst(self, client):
        _, lead = self._lead(client)
        r = client.post("/team/invites", json={"email": "ana@acme.example", "role": "analyst"}, headers=H(lead))
        assert r.status_code == 201
        ana = accept(client, r.json()["invite_link"], "Ana", "analyst-pass-1")
        assert client.get("/alerts", headers=H(ana)).status_code == 200
        assert client.get("/team/users", headers=H(ana)).status_code == 403   # analysts can't manage team

    def test_cannot_invite_platform_admin(self, client):
        _, lead = self._lead(client)
        r = client.post("/team/invites", json={"email": "x@acme.example", "role": "platform_admin"}, headers=H(lead))
        assert r.status_code == 422

    def test_role_change_revokes_existing_tokens(self, client):
        _, lead = self._lead(client)
        link = client.post("/team/invites", json={"email": "v@acme.example", "role": "analyst"}, headers=H(lead)).json()["invite_link"]
        tok = accept(client, link, "V", "viewer-pass-12")
        client.patch("/team/users/v@acme.example", json={"role": "viewer"}, headers=H(lead))
        assert client.get("/auth/me", headers=H(tok)).status_code == 401
        new = login(client, "v@acme.example", "viewer-pass-12")
        assert client.get("/auth/me", headers=H(new)).json()["roles"] == ["viewer"]

    def test_disable_blocks_sign_in(self, client):
        _, lead = self._lead(client)
        link = client.post("/team/invites", json={"email": "d@acme.example", "role": "viewer"}, headers=H(lead)).json()["invite_link"]
        accept(client, link, "D", "disable-pass-1")
        client.patch("/team/users/d@acme.example", json={"status": "disabled"}, headers=H(lead))
        r = client.post("/auth/login", json={"email": "d@acme.example", "password": "disable-pass-1"})
        assert r.status_code == 403

    def test_cannot_demote_self(self, client):
        _, lead = self._lead(client)
        r = client.patch("/team/users/lead@acme.example", json={"role": "viewer"}, headers=H(lead))
        assert r.status_code == 400

    def test_team_isolated_between_customers(self, client):
        op, lead = self._lead(client)
        other = accept(client, onboard(client, op, "Globex", "boss@globex.example")["invite_link"], "B", "globex-pass-1")
        assert client.patch("/team/users/lead@acme.example", json={"status": "disabled"}, headers=H(other)).status_code == 404
        emails = [u["email"] for u in client.get("/team/users", headers=H(other)).json()["items"]]
        assert emails == ["boss@globex.example"]

    def test_reset_link_flow(self, client):
        _, lead = self._lead(client)
        link = client.post("/team/invites", json={"email": "r@acme.example", "role": "analyst"}, headers=H(lead)).json()["invite_link"]
        old = accept(client, link, "R", "first-password-1")
        reset = client.post("/team/users/r@acme.example/reset-link", headers=H(lead)).json()["reset_link"]
        r = client.post("/auth/reset-password", json={"token": token_from(reset), "password": "second-password-2"})
        assert r.status_code == 200
        assert client.get("/auth/me", headers=H(old)).status_code == 401            # signed out everywhere
        login(client, "r@acme.example", "second-password-2")

    def test_change_password(self, client):
        _, lead = self._lead(client)
        r = client.post("/auth/change-password", json={"current_password": "lead-password-1", "new_password": "new-lead-pass-9"}, headers=H(lead))
        assert r.status_code == 200
        login(client, "lead@acme.example", "new-lead-pass-9")


class TestSuspension:
    def test_suspended_customer_locked_out_including_api_keys(self, client):
        op = login(client, OP_EMAIL, OP_PASSWORD)
        lead = accept(client, onboard(client, op)["invite_link"])
        key = client.post("/auth/api-keys", json={"name": "b"}, headers=H(lead)).json()["api_key"]
        client.patch("/platform/customers/acme-corp", json={"status": "suspended"}, headers=H(op))
        assert client.get("/auth/me", headers=H(lead)).status_code == 403
        assert client.get("/auth/me", headers=H(key)).status_code == 403
        assert client.post("/auth/login", json={"email": "lead@acme.example", "password": "lead-password-1"}).status_code == 403
        client.patch("/platform/customers/acme-corp", json={"status": "active"}, headers=H(op))
        assert client.get("/auth/me", headers=H(lead)).status_code == 200
