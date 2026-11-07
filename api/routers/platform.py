"""
Customer onboarding for the platform operator.

Creating a customer creates its organisation (tenant) and invites its first
admin in one step. The operator can suspend an organisation (all of its
people and API keys stop working immediately) and reactivate it.
"""

import re
import time
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.accounts import PLATFORM_TENANT, AccountStore, slugify_tenant
from api.auth import Principal, require_platform_admin, valid_tenant_id
from api.routers.accounts import make_invite

router = APIRouter(prefix="/platform", tags=["Platform"])

_RESERVED = {PLATFORM_TENANT, "default", "admin", "platform", "api"}


class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    tenant_id: Optional[str] = Field(None, max_length=40, description="Short id; generated from the name if omitted.")
    admin_email: str = Field(..., max_length=254)
    admin_name: str = Field("", max_length=120)
    shadow_mode: bool = Field(True, description="Start observing without enforcing (recommended).")


class AdminInvite(BaseModel):
    email: str = Field(..., max_length=254)
    name: str = Field("", max_length=120)


class CustomerPatch(BaseModel):
    status: Optional[str] = Field(None, pattern="^(active|suspended)$")
    name: Optional[str] = Field(None, min_length=2, max_length=120)


def _stores(request: Request):
    return request.app.state.tenant_store, request.app.state.account_store


def _summary(request: Request, tenant: Dict[str, Any]) -> Dict[str, Any]:
    _, accounts = _stores(request)
    people = accounts.list(tenant["tenant_id"])
    alerts = getattr(request.app.state, "alert_store", None)
    open_alerts = alerts.list(tenant["tenant_id"], status="open", limit=1)["total"] if alerts else 0
    return {
        **tenant,
        "people": len(people),
        "active_people": sum(1 for a in people if a["status"] == "active"),
        "pending_invites": sum(1 for a in people if a["status"] == "invited"),
        "open_alerts": open_alerts,
    }


@router.get("/customers", summary="All customer organisations")
def list_customers(request: Request,
                   principal: Annotated[Principal, Depends(require_platform_admin)]) -> Dict[str, Any]:
    tenants, _ = _stores(request)
    items = [_summary(request, t) for t in tenants.list() if t["tenant_id"] != PLATFORM_TENANT]
    items.sort(key=lambda t: t.get("created_at", 0), reverse=True)
    return {"items": items}


@router.post("/customers", status_code=status.HTTP_201_CREATED,
             summary="Onboard a customer: create the organisation and invite its first admin")
def create_customer(body: CustomerCreate, request: Request,
                    principal: Annotated[Principal, Depends(require_platform_admin)]) -> Dict[str, Any]:
    tenants, _ = _stores(request)
    tenant_id = (body.tenant_id or slugify_tenant(body.name)).strip().lower()
    if not valid_tenant_id(tenant_id) or not re.match(r"^[a-z0-9][a-z0-9-_]*$", tenant_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Organisation id: lowercase letters, numbers, - and _ only.")
    if tenant_id in _RESERVED:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"'{tenant_id}' is reserved.")
    try:
        tenant = tenants.create(tenant_id, body.name.strip(), principal.subject)
    except ValueError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))

    try:
        invite = make_invite(request, tenant_id, body.admin_email, ["admin"],
                             principal.subject, body.admin_name.strip())
    except HTTPException:
        # Roll back the organisation so a typo'd email can simply be retried
        tenants.delete(tenant_id)
        raise

    settings = getattr(request.app.state, "tenant_settings", None)
    if settings is not None:
        settings.update(tenant_id, {"shadow_mode": bool(body.shadow_mode)})

    request.state.tenant_id = tenant_id
    request.state.audit_event = {
        "event_type": "customer_created", "actor_id": principal.subject,
        "after_state": {"tenant_id": tenant_id, "name": tenant["name"],
                        "admin_email": invite["account"]["email"], "shadow_mode": body.shadow_mode},
        "at": time.time(),
    }
    return {"customer": _summary(request, tenant), **invite}


@router.get("/customers/{tenant_id}", summary="One customer with its people")
def get_customer(tenant_id: str, request: Request,
                 principal: Annotated[Principal, Depends(require_platform_admin)]) -> Dict[str, Any]:
    tenants, accounts = _stores(request)
    tenant = tenants.get(tenant_id)
    if tenant is None or tenant_id == PLATFORM_TENANT:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found.")
    return {**_summary(request, tenant),
            "accounts": [AccountStore.public(a) for a in accounts.list(tenant_id)]}


@router.patch("/customers/{tenant_id}", summary="Rename, suspend or reactivate a customer")
def update_customer(tenant_id: str, body: CustomerPatch, request: Request,
                    principal: Annotated[Principal, Depends(require_platform_admin)]) -> Dict[str, Any]:
    tenants, _ = _stores(request)
    if tenant_id == PLATFORM_TENANT or tenants.get(tenant_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found.")
    patch = body.model_dump(exclude_none=True)
    tenant = tenants.update(tenant_id, patch)
    request.state.tenant_id = tenant_id
    request.state.audit_event = {
        "event_type": "customer_updated", "actor_id": principal.subject,
        "after_state": patch, "at": time.time(),
    }
    return _summary(request, tenant)


@router.post("/customers/{tenant_id}/admins", status_code=status.HTTP_201_CREATED,
             summary="Invite another admin to a customer (e.g. if their admin left)")
def invite_customer_admin(tenant_id: str, request: Request, body: AdminInvite,
                          principal: Annotated[Principal, Depends(require_platform_admin)]) -> Dict[str, Any]:
    tenants, _ = _stores(request)
    if tenant_id == PLATFORM_TENANT or tenants.get(tenant_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found.")
    return make_invite(request, tenant_id, body.email, ["admin"], principal.subject, body.name)
