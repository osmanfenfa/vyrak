"""
Email/password sign-in, invites, password changes and team management.

There is no public sign-up: organisations are created by the platform
operator (routers/platform.py) and people join through one-time invite links.
No email service is wired in yet, so invite and reset links are returned to
the admin who created them, to copy and send.
"""

import os
import time
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.accounts import (
    CONSOLE_ROLES,
    INVITE_TTL_SECONDS,
    LOCKOUT_SECONDS,
    MAX_FAILED_LOGINS,
    RESET_TTL_SECONDS,
    AccountStore,
    hash_password,
    invite_link,
    normalize_email,
    password_problem,
    reset_link,
    valid_email,
    verify_password,
)
from api.auth import Principal, get_principal, issue_jwt, require_permission
from api.dependencies import client_ip

router = APIRouter(tags=["Accounts"])

LOGIN_RATE_PER_MINUTE = 20


def _session_seconds() -> int:
    return int(float(os.environ.get("ACCOUNT_SESSION_HOURS", "8")) * 3600)


def _stores(request: Request):
    st = request.app.state
    if getattr(st, "account_store", None) is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Account store not initialised.")
    return st.account_store, st.tenant_store, st.link_store


def _issue_session(account: Dict[str, Any]) -> Dict[str, Any]:
    ttl = _session_seconds()
    token = issue_jwt(
        account["email"], account["tenant_id"], account["roles"], ttl,
        extra_claims={"acct": 1, "ver": int(account.get("token_version", 1))},
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ttl,
        "account": AccountStore.public(account),
    }


def _audit(request: Request, tenant_id: str, event: Dict[str, Any]) -> None:
    request.state.tenant_id = tenant_id
    request.state.audit_event = {**event, "at": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# Sign-in
# ─────────────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=200)


@router.post("/auth/login", summary="Sign in with email and password")
def login(body: LoginRequest, request: Request) -> Dict[str, Any]:
    accounts, tenants, _ = _stores(request)

    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None and limiter.hit(f"login:{client_ip(request)}") > LOGIN_RATE_PER_MINUTE:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many sign-in attempts. Try again in a minute.")

    email = normalize_email(body.email)
    account = accounts.get(email)
    generic = HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password.")

    if account and account.get("locked_until", 0) > time.time():
        mins = int((account["locked_until"] - time.time()) // 60) + 1
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Too many failed attempts. Try again in {mins} minute(s).")

    ok = verify_password(body.password, account.get("password_hash") if account else None)
    if not account or not ok or account.get("status") != "active":
        if account and not ok:
            account["failed_logins"] = int(account.get("failed_logins", 0)) + 1
            if account["failed_logins"] >= MAX_FAILED_LOGINS:
                account["locked_until"] = time.time() + LOCKOUT_SECONDS
                account["failed_logins"] = 0
            accounts.save(account)
        if account and ok and account.get("status") == "disabled":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "This account has been disabled.")
        raise generic

    if account["roles"] != ["platform_admin"] and tenants.is_suspended(account["tenant_id"]):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "This organisation's access is suspended. Contact your Zelvryn administrator.")

    account.update({"failed_logins": 0, "locked_until": 0, "last_login_at": time.time()})
    accounts.save(account)
    _audit(request, account["tenant_id"], {"event_type": "login", "actor_id": email})
    return _issue_session(account)


# ─────────────────────────────────────────────────────────────────────────────
# Invites & password reset (public, token-gated)
# ─────────────────────────────────────────────────────────────────────────────

class AcceptInviteRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=200)
    name: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., max_length=200)


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=200)
    password: str = Field(..., max_length=200)


@router.get("/auth/invites/{token}", summary="Describe an invite before accepting it")
def describe_invite(token: str, request: Request) -> Dict[str, Any]:
    accounts, tenants, links = _stores(request)
    link = links.peek(token, "invite")
    account = accounts.get(link["email"]) if link else None
    if not link or not account or account["status"] != "invited":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This invite link is invalid, used or expired.")
    tenant = tenants.get(account["tenant_id"]) or {}
    return {
        "email": account["email"],
        "name": account.get("name", ""),
        "organisation": tenant.get("name", account["tenant_id"]),
        "roles": account["roles"],
        "expires_at": link["expires_at"],
    }


@router.post("/auth/accept-invite", summary="Set a password and activate an invited account")
def accept_invite(body: AcceptInviteRequest, request: Request) -> Dict[str, Any]:
    accounts, _, links = _stores(request)
    problem = password_problem(body.password)
    if problem:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, problem)
    link = links.consume(body.token, "invite")
    account = accounts.get(link["email"]) if link else None
    if not link or not account or account["status"] != "invited":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This invite link is invalid, used or expired.")
    account.update({
        "name": body.name.strip(), "status": "active",
        "password_hash": hash_password(body.password), "activated_at": time.time(),
        "last_login_at": time.time(),
    })
    accounts.save(account)
    _audit(request, account["tenant_id"], {"event_type": "invite_accepted", "actor_id": account["email"]})
    return _issue_session(account)


@router.post("/auth/reset-password", summary="Choose a new password using a reset link")
def reset_password(body: ResetPasswordRequest, request: Request) -> Dict[str, Any]:
    accounts, _, links = _stores(request)
    problem = password_problem(body.password)
    if problem:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, problem)
    link = links.consume(body.token, "reset")
    account = accounts.get(link["email"]) if link else None
    if not link or not account or account["status"] != "active":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This reset link is invalid, used or expired.")
    account.update({
        "password_hash": hash_password(body.password),
        "token_version": int(account.get("token_version", 1)) + 1,   # sign out everywhere
        "failed_logins": 0, "locked_until": 0,
    })
    accounts.save(account)
    _audit(request, account["tenant_id"], {"event_type": "password_reset", "actor_id": account["email"]})
    return _issue_session(account)


# ─────────────────────────────────────────────────────────────────────────────
# Own account
# ─────────────────────────────────────────────────────────────────────────────

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., max_length=200)
    new_password: str = Field(..., max_length=200)


@router.get("/auth/account", summary="The signed-in person's account")
def my_account(request: Request, principal: Annotated[Principal, Depends(get_principal)]) -> Dict[str, Any]:
    accounts, tenants, _ = _stores(request)
    account = accounts.get(principal.subject) if principal.credential == "account" else None
    tenant = tenants.get(principal.tenant_id) or {}
    return {
        "principal": principal.to_dict(),
        "account": AccountStore.public(account) if account else None,
        "organisation": tenant.get("name", principal.tenant_id),
    }


@router.post("/auth/change-password", summary="Change your password (signs out other sessions)")
def change_password(body: ChangePasswordRequest, request: Request,
                    principal: Annotated[Principal, Depends(get_principal)]) -> Dict[str, Any]:
    accounts, _, _ = _stores(request)
    account = accounts.get(principal.subject) if principal.credential == "account" else None
    if account is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only email/password accounts have a password.")
    if not verify_password(body.current_password, account.get("password_hash")):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Current password is incorrect.")
    problem = password_problem(body.new_password)
    if problem:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, problem)
    account.update({
        "password_hash": hash_password(body.new_password),
        "token_version": int(account.get("token_version", 1)) + 1,
    })
    accounts.save(account)
    _audit(request, account["tenant_id"], {"event_type": "password_changed", "actor_id": account["email"]})
    return _issue_session(account)


# ─────────────────────────────────────────────────────────────────────────────
# Team management (tenant admin)
# ─────────────────────────────────────────────────────────────────────────────

class InviteRequest(BaseModel):
    email: str = Field(..., max_length=254)
    role: str = Field("analyst")
    name: str = Field("", max_length=120)


class AccountPatch(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(active|disabled)$")


def make_invite(request: Request, tenant_id: str, email: str, roles: List[str],
                invited_by: str, name: str = "") -> Dict[str, Any]:
    """Shared by team invites and platform onboarding."""
    accounts, _, links = _stores(request)
    email = normalize_email(email)
    if not valid_email(email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Enter a valid email address.")
    try:
        account = accounts.create_invited(email, tenant_id, roles, invited_by, name=name)
    except ValueError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    token = links.issue(email, "invite", INVITE_TTL_SECONDS)
    return {
        "account": AccountStore.public(account),
        "invite_link": invite_link(token),
        "expires_in_days": INVITE_TTL_SECONDS // 86400,
    }


def _team_account(request: Request, principal: Principal, email: str) -> Dict[str, Any]:
    accounts, _, _ = _stores(request)
    account = accounts.get(email)
    if account is None or account["tenant_id"] != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such person in your organisation.")
    return account


@router.get("/team/users", summary="People in your organisation")
def list_team(request: Request,
              principal: Annotated[Principal, Depends(require_permission("team:manage"))]) -> Dict[str, Any]:
    accounts, _, _ = _stores(request)
    return {"tenant_id": principal.tenant_id,
            "items": [AccountStore.public(a) for a in accounts.list(principal.tenant_id)]}


@router.post("/team/invites", status_code=status.HTTP_201_CREATED, summary="Invite a person")
def invite_member(body: InviteRequest, request: Request,
                  principal: Annotated[Principal, Depends(require_permission("team:manage"))]) -> Dict[str, Any]:
    if body.role not in CONSOLE_ROLES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {list(CONSOLE_ROLES)}")
    result = make_invite(request, principal.tenant_id, body.email, [body.role], principal.subject, body.name)
    _audit(request, principal.tenant_id, {
        "event_type": "user_invited", "actor_id": principal.subject,
        "user_id": result["account"]["email"], "roles": [body.role],
    })
    return result


@router.patch("/team/users/{email}", summary="Change a person's role or disable/enable them")
def update_member(email: str, body: AccountPatch, request: Request,
                  principal: Annotated[Principal, Depends(require_permission("team:manage"))]) -> Dict[str, Any]:
    accounts, _, _ = _stores(request)
    account = _team_account(request, principal, normalize_email(email))
    if account["email"] == principal.subject:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You can't change your own role or status.")
    changes: Dict[str, Any] = {}
    if body.role is not None:
        if body.role not in CONSOLE_ROLES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {list(CONSOLE_ROLES)}")
        changes["roles"] = [body.role]
    if body.status is not None:
        if account["status"] == "invited":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Pending invites can't be enabled or disabled.")
        changes["status"] = body.status
    if not changes:
        return AccountStore.public(account)
    account.update(changes)
    account["token_version"] = int(account.get("token_version", 1)) + 1   # revoke old sessions
    accounts.save(account)
    _audit(request, principal.tenant_id, {
        "event_type": "user_updated", "actor_id": principal.subject,
        "user_id": account["email"], "after_state": changes,
    })
    return AccountStore.public(account)


@router.post("/team/users/{email}/invite-link", summary="New invite link for a pending invite")
def reissue_invite(email: str, request: Request,
                   principal: Annotated[Principal, Depends(require_permission("team:manage"))]) -> Dict[str, Any]:
    account = _team_account(request, principal, normalize_email(email))
    if account["status"] != "invited":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This person has already joined.")
    _, _, links = _stores(request)
    token = links.issue(account["email"], "invite", INVITE_TTL_SECONDS)
    return {"invite_link": invite_link(token), "expires_in_days": INVITE_TTL_SECONDS // 86400}


@router.post("/team/users/{email}/reset-link", summary="Password reset link for a team member")
def reset_link_for_member(email: str, request: Request,
                          principal: Annotated[Principal, Depends(require_permission("team:manage"))]) -> Dict[str, Any]:
    account = _team_account(request, principal, normalize_email(email))
    if account["status"] != "active":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only active accounts can reset their password.")
    _, _, links = _stores(request)
    token = links.issue(account["email"], "reset", RESET_TTL_SECONDS)
    _audit(request, principal.tenant_id, {
        "event_type": "reset_link_issued", "actor_id": principal.subject, "user_id": account["email"],
    })
    return {"reset_link": reset_link(token), "expires_in_hours": RESET_TTL_SECONDS // 3600}
