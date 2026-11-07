"""
Authentication, roles and tenant scoping for the Zelvryn API.

Every request resolves to a ``Principal``: who is calling, which tenant
(customer organisation) they belong to, and which roles they hold.

Three credential types are accepted as ``Authorization: Bearer <token>``:

1. **JWT** issued by Zelvryn's own email/password login (account tokens,
   re-checked against the account on every request) or by a customer's
   identity provider. Required claims: ``sub``, ``exp``. Optional claims:
   ``tenant`` (default ``"default"``), ``roles`` (list).
2. **API key** (``zk_<key_id>_<secret>``) created by a tenant admin and used
   by the customer's backend to send events. Keys are stored hashed.
3. **Plaintext dev token** — only when ``ZELVRYN_ENV`` is not ``production``
   AND ``AUTH_ALLOW_PLAINTEXT_TOKENS=true``. Format: ``<role>:<subject>`` or
   just ``<subject>`` (role ``user``). Never available in production.

Roles
-----
platform_admin  Operates the Zelvryn deployment: global config, model registry.
admin           Tenant administrator: API keys, profile resets, shadow runs,
                everything an analyst can do.
analyst         Reviews alerts, labels them, reads sessions/profiles/audit.
viewer          Read-only access to alerts, summaries and profile statistics.
integration     Machine credential for the customer's backend: starts
                sessions and submits actions for any identity in the tenant.
user            An end user evaluating their own session (direct mode) and
                exercising GDPR rights over their own profile.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated, Dict, FrozenSet, Iterable, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "default"

ROLE_PLATFORM_ADMIN = "platform_admin"
ROLE_ADMIN = "admin"
ROLE_ANALYST = "analyst"
ROLE_VIEWER = "viewer"
ROLE_INTEGRATION = "integration"
ROLE_USER = "user"

ALL_ROLES: FrozenSet[str] = frozenset({
    ROLE_PLATFORM_ADMIN, ROLE_ADMIN, ROLE_ANALYST,
    ROLE_VIEWER, ROLE_INTEGRATION, ROLE_USER,
})

# Permission → roles that hold it. platform_admin implicitly holds everything.
PERMISSIONS: Dict[str, FrozenSet[str]] = {
    # Evaluate sessions on behalf of any identity in the tenant
    "sessions:act_any":   frozenset({ROLE_INTEGRATION, ROLE_ADMIN}),
    # Read any session summary in the tenant
    "sessions:read_any":  frozenset({ROLE_INTEGRATION, ROLE_ADMIN, ROLE_ANALYST, ROLE_VIEWER}),
    "profiles:read_any":  frozenset({ROLE_ADMIN, ROLE_ANALYST, ROLE_VIEWER}),
    "profiles:delete_any": frozenset({ROLE_ADMIN}),
    "profiles:reset":     frozenset({ROLE_ADMIN}),
    "alerts:read":        frozenset({ROLE_ADMIN, ROLE_ANALYST, ROLE_VIEWER}),
    "alerts:label":       frozenset({ROLE_ADMIN, ROLE_ANALYST}),
    "events:ingest":      frozenset({ROLE_INTEGRATION, ROLE_ADMIN}),
    "shadow:run":         frozenset({ROLE_ADMIN}),
    "audit:read":         frozenset({ROLE_ADMIN, ROLE_ANALYST}),
    "apikeys:manage":     frozenset({ROLE_ADMIN}),
    "team:manage":        frozenset({ROLE_ADMIN}),
    "platform:manage":    frozenset(),   # platform_admin only
}


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    subject: str
    tenant_id: str = DEFAULT_TENANT
    roles: FrozenSet[str] = field(default_factory=lambda: frozenset({ROLE_USER}))
    credential: str = "jwt"            # jwt | api_key | dev
    key_id: Optional[str] = None

    def has_role(self, *roles: str) -> bool:
        return ROLE_PLATFORM_ADMIN in self.roles or any(r in self.roles for r in roles)

    def can(self, permission: str) -> bool:
        if ROLE_PLATFORM_ADMIN in self.roles:
            return True
        allowed = PERMISSIONS.get(permission)
        if allowed is None:
            raise KeyError(f"Unknown permission: {permission}")
        return bool(self.roles & allowed)

    def to_dict(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "tenant_id": self.tenant_id,
            "roles": sorted(self.roles),
            "credential": self.credential,
            "key_id": self.key_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

_INSECURE_SECRETS = {
    "", "zelvryn-dev-secret", "zelvryn-dev-secret-CHANGE-IN-PROD", "changeme",
}


def zelvryn_env() -> str:
    return os.environ.get("ZELVRYN_ENV", "development").strip().lower()


def is_production() -> bool:
    return zelvryn_env() == "production"


def plaintext_tokens_allowed() -> bool:
    """Plaintext dev tokens: opt-in, and never in production."""
    if is_production():
        return False
    return os.environ.get("AUTH_ALLOW_PLAINTEXT_TOKENS", "false").strip().lower() == "true"


def jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY", "")
    if is_production() and (secret in _INSECURE_SECRETS or len(secret) < 32):
        raise RuntimeError(
            "JWT_SECRET_KEY must be set to a random value of at least 32 "
            "characters in production."
        )
    return secret or "zelvryn-insecure-development-secret-not-for-production"


def jwt_algorithm() -> str:
    return os.environ.get("JWT_ALGORITHM", "HS256")


def validate_security_config() -> None:
    """Fail fast at startup on unsafe production configuration."""
    if is_production():
        jwt_secret()


def normalize_roles(raw: Iterable[str]) -> FrozenSet[str]:
    roles = frozenset(str(r).strip().lower() for r in raw if str(r).strip())
    unknown = roles - ALL_ROLES
    if unknown:
        raise ValueError(f"Unknown role(s): {sorted(unknown)}")
    return roles


def valid_tenant_id(tenant_id: str) -> bool:
    return (
        0 < len(tenant_id) <= 64
        and all(c.isalnum() or c in "-_" for c in tenant_id)
    )


# ─────────────────────────────────────────────────────────────────────────────
# JWT
# ─────────────────────────────────────────────────────────────────────────────

def issue_jwt(
    subject: str,
    tenant_id: str = DEFAULT_TENANT,
    roles: Iterable[str] = (ROLE_USER,),
    ttl_seconds: int = 3600,
    extra_claims: Optional[Dict[str, object]] = None,
) -> str:
    import jwt

    now = int(time.time())
    payload = {
        "sub": subject,
        "tenant": tenant_id,
        "roles": sorted(normalize_roles(roles)),
        "iat": now,
        "exp": now + int(ttl_seconds),
        **(extra_claims or {}),
    }
    issuer = os.environ.get("JWT_ISSUER")
    audience = os.environ.get("JWT_AUDIENCE")
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience
    return jwt.encode(payload, jwt_secret(), algorithm=jwt_algorithm())


def _decode_jwt(token: str) -> Tuple[Principal, Dict[str, object]]:
    import jwt

    issuer = os.environ.get("JWT_ISSUER") or None
    audience = os.environ.get("JWT_AUDIENCE") or None
    options = {"require": ["sub", "exp"]}
    payload = jwt.decode(
        token,
        jwt_secret(),
        algorithms=[jwt_algorithm()],
        options=options,
        issuer=issuer,
        audience=audience,
    )
    tenant = str(payload.get("tenant") or DEFAULT_TENANT)
    if not valid_tenant_id(tenant):
        raise ValueError("invalid tenant claim")
    raw_roles = payload.get("roles") or [ROLE_USER]
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    principal = Principal(
        subject=str(payload["sub"]),
        tenant_id=tenant,
        roles=normalize_roles(raw_roles),
        credential="jwt",
    )
    return principal, payload


# ─────────────────────────────────────────────────────────────────────────────
# API keys
# ─────────────────────────────────────────────────────────────────────────────

API_KEY_PREFIX = "zk_"


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class ApiKeyStore:
    """
    Stores API keys hashed in Redis (or the in-memory fallback).

    Key format handed to the customer: ``zk_<key_id>_<secret>``. Only the
    SHA-256 of the secret is stored, so a database leak does not leak keys.
    """

    _PREFIX = "zelvryn:apikey:"
    _TENANT_INDEX = "zelvryn:apikeys:tenant:"

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def create(
        self,
        tenant_id: str,
        name: str,
        roles: Iterable[str] = (ROLE_INTEGRATION,),
        created_by: str = "",
    ) -> Dict[str, object]:
        role_set = normalize_roles(roles)
        if ROLE_PLATFORM_ADMIN in role_set:
            raise ValueError("API keys cannot hold the platform_admin role.")
        key_id = secrets.token_hex(6)
        secret = secrets.token_urlsafe(32)
        record = {
            "key_id": key_id,
            "tenant_id": tenant_id,
            "name": name,
            "roles": sorted(role_set),
            "secret_hash": _hash_secret(secret),
            "created_by": created_by,
            "created_at": time.time(),
            "revoked": False,
            "last_used_at": None,
        }
        self._redis.set(self._PREFIX + key_id, json.dumps(record))
        index = self._load_index(tenant_id)
        index.append(key_id)
        self._redis.set(self._TENANT_INDEX + tenant_id, json.dumps(index))
        public = {k: v for k, v in record.items() if k != "secret_hash"}
        public["api_key"] = f"{API_KEY_PREFIX}{key_id}_{secret}"
        return public

    def list(self, tenant_id: str) -> List[Dict[str, object]]:
        out = []
        for key_id in self._load_index(tenant_id):
            rec = self._get(key_id)
            if rec:
                out.append({k: v for k, v in rec.items() if k != "secret_hash"})
        return out

    def revoke(self, tenant_id: str, key_id: str) -> bool:
        rec = self._get(key_id)
        if not rec or rec.get("tenant_id") != tenant_id:
            return False
        rec["revoked"] = True
        rec["revoked_at"] = time.time()
        self._redis.set(self._PREFIX + key_id, json.dumps(rec))
        return True

    def authenticate(self, token: str) -> Optional[Principal]:
        if not token.startswith(API_KEY_PREFIX):
            return None
        body = token[len(API_KEY_PREFIX):]
        key_id, sep, secret = body.partition("_")
        if not sep or not key_id or not secret:
            return None
        rec = self._get(key_id)
        if not rec or rec.get("revoked"):
            return None
        if not hmac.compare_digest(rec["secret_hash"], _hash_secret(secret)):
            return None
        try:
            rec["last_used_at"] = time.time()
            self._redis.set(self._PREFIX + key_id, json.dumps(rec))
        except Exception:
            pass
        return Principal(
            subject=f"apikey:{key_id}",
            tenant_id=rec["tenant_id"],
            roles=frozenset(rec["roles"]),
            credential="api_key",
            key_id=key_id,
        )

    def _get(self, key_id: str) -> Optional[Dict[str, object]]:
        raw = self._redis.get(self._PREFIX + key_id)
        return json.loads(raw) if raw else None

    def _load_index(self, tenant_id: str) -> List[str]:
        raw = self._redis.get(self._TENANT_INDEX + tenant_id)
        return json.loads(raw) if raw else []


# ─────────────────────────────────────────────────────────────────────────────
# Plaintext dev tokens
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dev_token(token: str) -> Principal:
    """``role:subject`` (optionally ``role:subject@tenant``) or ``subject``."""
    role = ROLE_USER
    rest = token
    if ":" in token:
        maybe_role, _, remainder = token.partition(":")
        if maybe_role.lower() in ALL_ROLES:
            role, rest = maybe_role.lower(), remainder
    tenant = DEFAULT_TENANT
    if "@" in rest:
        subject_part, _, maybe_tenant = rest.rpartition("@")
        # Only treat the suffix as a tenant if it looks like one (emails have dots)
        if valid_tenant_id(maybe_tenant) and "." not in maybe_tenant:
            rest, tenant = subject_part, maybe_tenant
    return Principal(subject=rest, tenant_id=tenant, roles=frozenset({role}), credential="dev")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependencies
# ─────────────────────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_principal(
    request: Request,
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(_bearer_scheme)
    ] = None,
) -> Principal:
    if credentials is None:
        raise _unauthorized("Bearer token required.")
    token = (credentials.credentials or "").strip()
    if len(token) < 3:
        raise _unauthorized("Invalid token.")

    principal: Optional[Principal] = None

    if token.startswith(API_KEY_PREFIX):
        store = getattr(request.app.state, "api_key_store", None)
        principal = store.authenticate(token) if store else None
        if principal is None:
            raise _unauthorized("Invalid or revoked API key.")
    elif token.count(".") == 2:
        try:
            principal, claims = _decode_jwt(token)
        except ImportError:
            logger.error("PyJWT is not installed; JWT authentication unavailable.")
            raise _unauthorized("Token validation unavailable.")
        except Exception as e:
            logger.info("JWT validation failed: %s", e)
            raise _unauthorized("Invalid or expired token.")
        if claims.get("acct"):
            principal = _account_principal(request, principal, claims)
    elif plaintext_tokens_allowed():
        principal = _parse_dev_token(token)
    else:
        raise _unauthorized("Invalid token.")

    if ROLE_PLATFORM_ADMIN not in principal.roles:
        tenants = getattr(request.app.state, "tenant_store", None)
        if tenants is not None and tenants.is_suspended(principal.tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This organisation's access is suspended. Contact your Zelvryn administrator.",
            )

    request.state.principal = principal
    request.state.user_id = principal.subject
    request.state.tenant_id = principal.tenant_id
    request.state.roles = sorted(principal.roles)
    return principal


def _account_principal(request: Request, principal: Principal, claims: Dict[str, object]) -> Principal:
    """
    Tokens from Zelvryn's own login are checked against the live account, so
    disabling a user, changing their role or password revokes old tokens.
    """
    accounts = getattr(request.app.state, "account_store", None)
    account = accounts.get(principal.subject) if accounts else None
    if (
        account is None
        or account.get("status") != "active"
        or int(account.get("token_version", 1)) != int(claims.get("ver", 0))
    ):
        raise _unauthorized("Your session has ended. Please sign in again.")
    return Principal(
        subject=account["email"],
        tenant_id=account["tenant_id"],
        roles=frozenset(account["roles"]),
        credential="account",
    )


def require_permission(permission: str):
    """Dependency factory: caller must hold ``permission``."""
    if permission not in PERMISSIONS:
        raise KeyError(f"Unknown permission: {permission}")

    def _dep(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if not principal.can(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission '{permission}'.",
            )
        return principal

    return _dep


def require_platform_admin(
    principal: Annotated[Principal, Depends(get_principal)],
) -> Principal:
    if ROLE_PLATFORM_ADMIN not in principal.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="platform_admin role required.",
        )
    return principal
