"""
FastAPI dependencies: authentication shortcuts, shared-service accessors and
rate limiting.

Session state no longer lives in this process — see api.session_service and
api.stores — so any worker can serve any request.
"""

import hashlib
import logging
import os
import uuid
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from api.auth import (
    Principal,
    get_principal,
    require_permission,
    require_platform_admin,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION SHORTCUTS (backwards-compatible names)
# ─────────────────────────────────────────────────────────────────────────────

def get_current_user(principal: Annotated[Principal, Depends(get_principal)]) -> str:
    """Return the authenticated subject (user id, or ``apikey:<id>``)."""
    return principal.subject


def require_admin(
    principal: Annotated[Principal, Depends(require_platform_admin)],
) -> str:
    """Platform operations (global config, model registry). Returns subject."""
    return principal.subject


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SERVICES
# ─────────────────────────────────────────────────────────────────────────────

def _service(request: Request, name: str, label: str):
    svc = getattr(request.app.state, name, None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{label} not initialised.",
        )
    return svc


def get_profile_store(request: Request):
    return _service(request, "profile_store", "Profile store")


def get_model_registry(request: Request):
    return _service(request, "model_registry", "Model registry")


def get_feature_store(request: Request):
    return _service(request, "feature_store", "Feature store")


def get_alert_store(request: Request):
    return _service(request, "alert_store", "Alert store")


def get_population_prior(request: Request):
    return getattr(request.app.state, "population_prior", None)


def get_device_bridge(request: Request):
    return getattr(request.app.state, "device_bridge", None)


def get_active_session_count(request: Optional[Request] = None) -> int:
    """Number of live sessions (all tenants), read from the shared store."""
    if request is None:
        return 0
    store = getattr(request.app.state, "session_store", None)
    try:
        return store.count() if store else 0
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING (shared across workers via Redis)
# ─────────────────────────────────────────────────────────────────────────────

_RATE_LIMIT_WINDOW_S = 60


def rate_limit_check(
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """
    Fixed one-minute window per credential (tenant + subject), so one noisy
    integration can't starve others and the limit holds across workers.
    """
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    bucket = hashlib.sha256(
        f"{principal.tenant_id}|{principal.subject}".encode("utf-8")
    ).hexdigest()[:24]
    try:
        count = limiter.hit(bucket)
    except Exception as e:   # never fail a request because the limiter is down
        logger.warning("Rate limiter unavailable: %s", e)
        return
    if count > limiter.limit:
        logger.warning(
            "Rate limit exceeded: tenant=%s subject=%s count=%d",
            principal.tenant_id, principal.subject, count,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limiter.limit} requests per minute.",
            headers={
                "Retry-After": str(_RATE_LIMIT_WINDOW_S),
                "X-RateLimit-Limit": str(limiter.limit),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_request_id(
    x_request_id: Annotated[Optional[str], Header(alias="X-Request-ID")] = None,
) -> str:
    return x_request_id or str(uuid.uuid4())


def trust_proxy_headers() -> bool:
    """Only honour X-Forwarded-For behind a proxy you control (it is spoofable)."""
    return os.environ.get("TRUST_PROXY_HEADERS", "false").strip().lower() == "true"


def client_ip(request: Request) -> str:
    if trust_proxy_headers():
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_get_client_ip = client_ip


def extract_metadata_from_request(request: Request) -> dict:
    return {
        "ip": client_ip(request),
        "user_agent": request.headers.get("User-Agent", ""),
    }


__all__ = [
    "Principal",
    "get_principal",
    "require_permission",
    "require_platform_admin",
    "get_current_user",
    "require_admin",
    "get_profile_store",
    "get_model_registry",
    "get_feature_store",
    "get_alert_store",
    "get_population_prior",
    "get_device_bridge",
    "get_active_session_count",
    "rate_limit_check",
    "get_request_id",
    "client_ip",
    "extract_metadata_from_request",
]
