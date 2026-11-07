"""
Identity & access endpoints: who am I, API keys and tenant settings.
Email/password login, invites and team management live in routers/accounts.py.
"""

import time
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.auth import (
    ALL_ROLES,
    ROLE_INTEGRATION,
    ROLE_PLATFORM_ADMIN,
    Principal,
    get_principal,
    normalize_roles,
    require_permission,
)

router = APIRouter(tags=["Access"])


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    roles: List[str] = Field(default_factory=lambda: [ROLE_INTEGRATION])


class TenantSettingsPatch(BaseModel):
    shadow_mode: Optional[bool] = None


@router.get("/auth/me", summary="Describe the current credential")
def whoami(principal: Annotated[Principal, Depends(get_principal)]) -> Dict[str, Any]:
    return principal.to_dict()


def _key_store(request: Request):
    store = getattr(request.app.state, "api_key_store", None)
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API key store not initialised.")
    return store


@router.get("/auth/api-keys", summary="List this tenant's API keys")
def list_api_keys(
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("apikeys:manage"))],
) -> Dict[str, Any]:
    return {"tenant_id": principal.tenant_id, "items": _key_store(request).list(principal.tenant_id)}


@router.post(
    "/auth/api-keys",
    status_code=status.HTTP_201_CREATED,
    summary="Create an API key (the secret is shown once)",
)
def create_api_key(
    body: ApiKeyCreateRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("apikeys:manage"))],
) -> Dict[str, Any]:
    try:
        roles = normalize_roles(body.roles)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    if ROLE_PLATFORM_ADMIN in roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "API keys cannot hold platform_admin.")
    # A tenant admin cannot mint a key more powerful than allowed for tenants
    created = _key_store(request).create(principal.tenant_id, body.name, roles, principal.subject)
    request.state.tenant_id = principal.tenant_id
    request.state.audit_event = {
        "event_type": "api_key_created",
        "actor_id": principal.subject,
        "key_id": created["key_id"],
        "roles": created["roles"],
        "at": time.time(),
    }
    return created


@router.delete("/auth/api-keys/{key_id}", summary="Revoke an API key")
def revoke_api_key(
    key_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("apikeys:manage"))],
) -> Dict[str, Any]:
    if not _key_store(request).revoke(principal.tenant_id, key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found.")
    request.state.tenant_id = principal.tenant_id
    request.state.audit_event = {
        "event_type": "api_key_revoked",
        "actor_id": principal.subject,
        "key_id": key_id,
        "at": time.time(),
    }
    return {"key_id": key_id, "revoked": True}


@router.get("/tenant/settings", summary="This tenant's settings")
def get_tenant_settings(
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("alerts:read"))],
) -> Dict[str, Any]:
    from api.session_service import tenant_shadow_mode
    store = getattr(request.app.state, "tenant_settings", None)
    settings = store.get(principal.tenant_id) if store else {}
    settings["shadow_mode"] = tenant_shadow_mode(request, principal.tenant_id)
    return {"tenant_id": principal.tenant_id, **settings}


@router.patch(
    "/tenant/settings",
    summary="Update tenant settings (e.g. switch from shadow to enforcement)",
)
def patch_tenant_settings(
    body: TenantSettingsPatch,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("apikeys:manage"))],
) -> Dict[str, Any]:
    store = getattr(request.app.state, "tenant_settings", None)
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Tenant settings not initialised.")
    patch = body.model_dump(exclude_none=True)
    updated = store.update(principal.tenant_id, patch)
    request.state.tenant_id = principal.tenant_id
    request.state.audit_event = {
        "event_type": "tenant_settings_changed",
        "actor_id": principal.subject,
        "after_state": patch,
        "at": time.time(),
    }
    return {"tenant_id": principal.tenant_id, **updated}


__all__ = ["router", "ALL_ROLES"]
