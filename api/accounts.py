"""
Accounts, organisations (tenants) and one-time links.

Model
-----
* The platform operator (you) has a ``platform_admin`` account, created at
  startup from BOOTSTRAP_ADMIN_EMAIL / BOOTSTRAP_ADMIN_PASSWORD.
* The operator creates each customer organisation together with its first
  admin, who receives a one-time invite link.
* A tenant admin invites analysts, viewers and other admins the same way.
* Everyone signs in with email + password. There is no public sign-up.

Security
--------
* Passwords: scrypt (stdlib), per-user random salt, constant-time compare.
* Lockout: 5 failed attempts → 15 minutes.
* Invite and reset links are random 32-byte tokens; only their SHA-256 is
  stored, they expire, and they work once.
* Every account carries a ``token_version``; disabling an account, changing
  its role or password bumps it, which invalidates every token issued before.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional

PLATFORM_TENANT = "zelvryn"
CONSOLE_ROLES = ("admin", "analyst", "viewer")      # roles a person can be invited as
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 200
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 15 * 60
INVITE_TTL_SECONDS = 7 * 24 * 3600
RESET_TTL_SECONDS = 2 * 3600

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# scrypt parameters (~16 MB, tens of ms per hash)
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email)) and len(email) <= 254


def password_problem(password: str) -> Optional[str]:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"Password must be at most {MAX_PASSWORD_LENGTH} characters."
    if password.strip() != password or not password.strip():
        return "Password must not start or end with spaces."
    return None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    if not stored:
        # Burn comparable time so unknown emails aren't distinguishable by timing
        hashlib.scrypt(b"x", salt=b"0" * 16, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
        return False
    try:
        _, n, r, p, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                                n=int(n), r=int(r), p=int(p), dklen=len(digest_hex) // 2)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def slugify_tenant(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:40] or "org"


def dashboard_url() -> str:
    return os.environ.get("DASHBOARD_URL", "http://localhost:5173").rstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# Stores
# ─────────────────────────────────────────────────────────────────────────────

class TenantStore:
    _KEY = "zelvryn:tenant:"
    _INDEX = "zelvryn:tenants"

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def get(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._KEY + tenant_id)
        return json.loads(raw) if raw else None

    def exists(self, tenant_id: str) -> bool:
        return bool(self._redis.exists(self._KEY + tenant_id))

    def create(self, tenant_id: str, name: str, created_by: str) -> Dict[str, Any]:
        record = {
            "tenant_id": tenant_id, "name": name, "status": "active",
            "created_at": time.time(), "created_by": created_by,
        }
        if not self._redis.set(self._KEY + tenant_id, json.dumps(record), nx=True):
            raise ValueError(f"Organisation id '{tenant_id}' is already taken.")
        index = self._index()
        index.append(tenant_id)
        self._redis.set(self._INDEX, json.dumps(index))
        return record

    def update(self, tenant_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        record = self.get(tenant_id)
        if record is None:
            return None
        record.update(patch)
        record["updated_at"] = time.time()
        self._redis.set(self._KEY + tenant_id, json.dumps(record))
        return record

    def list(self) -> List[Dict[str, Any]]:
        return [t for t in (self.get(tid) for tid in self._index()) if t]

    def delete(self, tenant_id: str) -> None:
        self._redis.delete(self._KEY + tenant_id)
        self._redis.set(self._INDEX, json.dumps([t for t in self._index() if t != tenant_id]))

    def is_suspended(self, tenant_id: str) -> bool:
        record = self.get(tenant_id)
        return bool(record and record.get("status") == "suspended")

    def _index(self) -> List[str]:
        raw = self._redis.get(self._INDEX)
        return json.loads(raw) if raw else []


class AccountStore:
    """One account per email address, belonging to exactly one organisation."""

    _KEY = "zelvryn:account:"
    _TENANT_INDEX = "zelvryn:accounts:tenant:"

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def get(self, email: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._KEY + normalize_email(email))
        return json.loads(raw) if raw else None

    def save(self, account: Dict[str, Any]) -> None:
        self._redis.set(self._KEY + account["email"], json.dumps(account))

    def create_invited(self, email: str, tenant_id: str, roles: List[str],
                       invited_by: str, name: str = "") -> Dict[str, Any]:
        email = normalize_email(email)
        existing = self.get(email)
        if existing and existing["status"] != "invited":
            raise ValueError("An account with this email already exists.")
        if existing and existing["tenant_id"] != tenant_id:
            raise ValueError("This email has a pending invite to another organisation.")
        account = existing or {
            "email": email, "tenant_id": tenant_id, "created_at": time.time(),
            "password_hash": None, "failed_logins": 0, "locked_until": 0,
            "last_login_at": None, "token_version": 1,
        }
        account.update({
            "name": name or account.get("name") or "", "roles": list(roles),
            "status": "invited", "invited_by": invited_by, "invited_at": time.time(),
        })
        self.save(account)
        if not existing:
            index = self._index(tenant_id)
            index.append(email)
            self._redis.set(self._TENANT_INDEX + tenant_id, json.dumps(index))
        return account

    def list(self, tenant_id: str) -> List[Dict[str, Any]]:
        return [a for a in (self.get(e) for e in self._index(tenant_id)) if a]

    def count(self, tenant_id: str) -> int:
        return len(self._index(tenant_id))

    def _index(self, tenant_id: str) -> List[str]:
        raw = self._redis.get(self._TENANT_INDEX + tenant_id)
        return json.loads(raw) if raw else []

    @staticmethod
    def public(account: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in account.items() if k not in ("password_hash",)}


class OneTimeLinkStore:
    """Invite and password-reset tokens; only hashes are stored."""

    _KEY = "zelvryn:link:"

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    @staticmethod
    def _h(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue(self, email: str, purpose: str, ttl: int) -> str:
        token = secrets.token_urlsafe(32)
        record = {"email": normalize_email(email), "purpose": purpose,
                  "expires_at": time.time() + ttl}
        self._redis.set(self._KEY + self._h(token), json.dumps(record), ex=ttl)
        return token

    def peek(self, token: str, purpose: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._KEY + self._h(token or ""))
        if not raw:
            return None
        record = json.loads(raw)
        if record["purpose"] != purpose or record["expires_at"] < time.time():
            return None
        return record

    def consume(self, token: str, purpose: str) -> Optional[Dict[str, Any]]:
        record = self.peek(token, purpose)
        if record:
            self._redis.delete(self._KEY + self._h(token))
        return record


def invite_link(token: str) -> str:
    return f"{dashboard_url()}/#/accept-invite?token={token}"


def reset_link(token: str) -> str:
    return f"{dashboard_url()}/#/reset-password?token={token}"


def bootstrap_platform_admin(accounts: AccountStore, tenants: TenantStore, logger) -> None:
    """Create the operator account from environment variables if it doesn't exist."""
    email = normalize_email(os.environ.get("BOOTSTRAP_ADMIN_EMAIL", ""))
    password = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "")
    if not email:
        if not any(True for _ in tenants.list()):
            logger.warning(
                "No platform operator configured. Set BOOTSTRAP_ADMIN_EMAIL and "
                "BOOTSTRAP_ADMIN_PASSWORD to create your admin account."
            )
        return
    if accounts.get(email):
        return
    problem = password_problem(password)
    if not valid_email(email) or problem:
        logger.error("Bootstrap admin not created: %s", problem or "invalid email")
        return
    if not tenants.exists(PLATFORM_TENANT):
        tenants.create(PLATFORM_TENANT, "Zelvryn (operator)", "bootstrap")
    account = accounts.create_invited(email, PLATFORM_TENANT, ["platform_admin"], "bootstrap",
                                      name=os.environ.get("BOOTSTRAP_ADMIN_NAME", "Platform admin"))
    account.update({"status": "active", "password_hash": hash_password(password),
                    "activated_at": time.time()})
    accounts.save(account)
    logger.info("Platform operator account created for %s.", email)
