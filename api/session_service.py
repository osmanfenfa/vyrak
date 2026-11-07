"""
Session evaluation service — the request-independent core of the session
and event endpoints.

Every call is stateless with respect to the worker process:

    load session state (Redis) → lock identity → load profile + peer baseline
    → rebuild engine → evaluate → save profile, session, peer baseline
    → open/update alert

so any number of API workers can serve the same session, and a restart
loses nothing.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status

from api.auth import Principal
from core.action import UserAction
from core.constants import IDENTITY_KIND_HUMAN, IDENTITY_KINDS, TRUST_INITIAL
from core.engine import ABSAv2Engine, DecisionResult
from core.profile import BehavioralProfile
from profiles.redis_store import ProfileLockTimeout

logger = logging.getLogger(__name__)

# Initial-trust penalty applied from device/network risk at session start
DEVICE_RISK_MAX_PENALTY = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state(request: Request):
    return request.app.state


def _runtime(request: Request) -> Dict[str, Any]:
    return getattr(request.app.state, "runtime_config", {}) or {}


def tenant_shadow_mode(request: Request, tenant_id: str) -> bool:
    st = _state(request)
    settings_store = getattr(st, "tenant_settings", None)
    global_shadow = bool(getattr(st, "shadow_mode", False))
    if settings_store is None:
        return global_shadow
    return bool(settings_store.get(tenant_id, fallback_shadow=global_shadow)["shadow_mode"])


def _profile_store(request: Request, tenant_id: str):
    store = getattr(_state(request), "profile_store", None)
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Profile store not initialised.")
    return store.for_tenant(tenant_id) if hasattr(store, "for_tenant") else store


def _session_store(request: Request):
    store = getattr(_state(request), "session_store", None)
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session store not initialised.")
    return store


def resolve_identity(principal: Principal, requested: Optional[str]) -> str:
    """
    Integration keys and tenant admins act on behalf of any identity in their
    tenant; everyone else may only act as themselves.
    """
    if principal.can("sessions:act_any"):
        if not requested:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "user_id is required.")
        return requested
    if requested and requested != principal.subject:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "You can only start sessions for yourself.",
        )
    return principal.subject


def _new_profile(request: Request, identity_id: str, identity_kind: str) -> BehavioralProfile:
    prior = getattr(_state(request), "population_prior", None)
    if prior is not None and identity_kind == IDENTITY_KIND_HUMAN:
        profile = prior.create_new_user_profile(identity_id)
    else:
        profile = BehavioralProfile(user_id=identity_id)
    profile.identity_kind = identity_kind
    return profile


def _classifier(request: Request, session_id: str):
    registry = getattr(_state(request), "model_registry", None)
    if registry is None:
        return None
    try:
        return registry.get_serving_model(session_id)
    except Exception as e:
        logger.warning("Failed to load classifier: %s", e)
        return None


def _build_engine(
    request: Request,
    tenant_id: str,
    session_id: str,
    profile: BehavioralProfile,
    identity_kind: str,
    peer_group: str,
    shadow: bool,
) -> ABSAv2Engine:
    peer_store = getattr(_state(request), "peer_store", None)
    peer = peer_store.load(tenant_id, peer_group) if peer_store else None
    engine = ABSAv2Engine(
        profile=profile,
        classifier=_classifier(request, session_id),
        session_id=session_id,
        shadow_mode=shadow,
        identity_kind=identity_kind,
        peer_baseline=peer,
        peer_group=peer_group,
    )
    engine.set_ramp_actions(int(_runtime(request).get("cold_start_ramp_actions", engine._ramp.ramp_actions)))
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start_session(
    request: Request,
    principal: Principal,
    identity_id: str,
    identity_kind: str = IDENTITY_KIND_HUMAN,
    peer_group: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    external_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    if identity_kind not in IDENTITY_KINDS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"identity_kind must be one of {sorted(IDENTITY_KINDS)}")
    tenant_id = principal.tenant_id
    session_id = session_id or str(uuid.uuid4())
    store = _profile_store(request, tenant_id)
    shadow = tenant_shadow_mode(request, tenant_id)

    with _locked(store, identity_id):
        profile = store.get(identity_id)
        if profile is None:
            profile = _new_profile(request, identity_id, identity_kind)
        elif profile.identity_kind != identity_kind:
            # An identity's kind is fixed by its history; don't let a caller
            # flip a human to "agent" to dodge automation checks.
            identity_kind = profile.identity_kind
        store.save(identity_id, profile)

    engine = _build_engine(request, tenant_id, session_id, profile, identity_kind,
                           peer_group or identity_kind, shadow)

    trust_initial = min(100.0, max(0.0, float(_runtime(request).get("trust_initial", TRUST_INITIAL))))
    device_risk = _assess_device(request, tenant_id, identity_id, metadata)
    if device_risk is not None:
        trust_initial = max(0.0, trust_initial - DEVICE_RISK_MAX_PENALTY * float(device_risk["composite_score"]))
    engine.trust_score = trust_initial

    state = engine.export_state()
    state.update({
        "tenant_id": tenant_id,
        "identity_id": identity_id,
        "external_session_id": external_session_id,
        "alert_id": None,
        "device_risk": device_risk,
        "created_by": principal.subject,
    })
    _session_store(request).save(tenant_id, session_id, state)

    return {
        "session_id": session_id,
        "user_id": identity_id,
        "identity_kind": identity_kind,
        "tenant_id": tenant_id,
        "trust_score": round(engine.trust_score, 3),
        "ramp_active": not profile.is_mature(min_actions=engine._ramp.ramp_actions),
        "shadow_mode": shadow,
        "started_at": time.time(),
        "device_risk": device_risk,
    }


def load_session(request: Request, principal: Principal, session_id: str, write: bool) -> Dict[str, Any]:
    state = _session_store(request).load(principal.tenant_id, session_id)
    if state is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Session '{session_id}' not found. Start a session first via POST /session/start.",
        )
    own = state["identity_id"] == principal.subject
    permission = "sessions:act_any" if write else "sessions:read_any"
    if not own and not principal.can(permission):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Session does not belong to the authenticated user.")
    return state


def evaluate(
    request: Request,
    principal: Principal,
    session_id: str,
    action: UserAction,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[DecisionResult, Dict[str, Any], Dict[str, Any]]:
    """Returns (result, session_state_after, context_for_audit)."""
    tenant_id = principal.tenant_id
    state = state or load_session(request, principal, session_id, write=True)
    identity_id = state["identity_id"]
    store = _profile_store(request, tenant_id)

    with _locked(store, identity_id):
        # Re-read under the lock so concurrent requests can't lose updates
        state = _session_store(request).load(tenant_id, session_id) or state
        profile = store.get(identity_id) or _new_profile(request, identity_id, state.get("identity_kind", IDENTITY_KIND_HUMAN))
        engine = _build_engine(
            request, tenant_id, session_id, profile,
            state.get("identity_kind", IDENTITY_KIND_HUMAN),
            state.get("peer_group") or state.get("identity_kind", IDENTITY_KIND_HUMAN),
            tenant_shadow_mode(request, tenant_id),
        )
        engine.restore_state(state)
        engine.shadow_mode = tenant_shadow_mode(request, tenant_id)
        trust_before = float(engine.trust_score)
        decision_before = engine.current_decision()

        result = engine.evaluate_action(action)
        would_be = engine.current_decision()

        store.save(identity_id, profile)
        if engine.peer_dirty and engine.peer_baseline is not None:
            peer_store = getattr(_state(request), "peer_store", None)
            if peer_store:
                peer_store.save(tenant_id, engine.peer_group, engine.peer_baseline)

        new_state = engine.export_state()
        for k in ("tenant_id", "identity_id", "external_session_id", "alert_id", "device_risk", "created_by"):
            new_state[k] = state.get(k)

        if result.anomalies:
            alerts = getattr(_state(request), "alert_store", None)
            if alerts is not None:
                new_state["alert_id"] = alerts.record(
                    tenant_id, state.get("alert_id"),
                    session_id=session_id,
                    identity_id=identity_id,
                    identity_kind=engine.identity_kind,
                    decision=would_be,
                    enforced=not engine.shadow_mode,
                    trust_score=result.trust_score,
                    anomalies=result.anomalies,
                    reasons=result.reasons,
                    action_type=result.action_type,
                    timestamp=result.timestamp,
                    features=result.features,
                )
        _session_store(request).save(tenant_id, session_id, new_state)

    ctx = {
        "trust_before": trust_before,
        "decision_before": decision_before,
        "would_be_decision": would_be,
        "identity_kind": engine.identity_kind,
        "alert_id": new_state.get("alert_id"),
        "shadow_mode": engine.shadow_mode,
    }
    return result, new_state, ctx


def restore_trust(request: Request, principal: Principal, session_id: str, increment: float) -> Dict[str, Any]:
    state = load_session(request, principal, session_id, write=True)
    store = _profile_store(request, principal.tenant_id)
    with _locked(store, state["identity_id"]):
        state = _session_store(request).load(principal.tenant_id, session_id) or state
        state["trust_score"] = min(100.0, float(state["trust_score"]) + max(0.0, increment))
        _session_store(request).save(principal.tenant_id, session_id, state)
    return state


def summarize(request: Request, principal: Principal, session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = load_session(request, principal, session_id, write=False)
    engine = _engine_from_state(request, principal.tenant_id, session_id, state)
    return engine.get_session_summary(), state


def close(request: Request, principal: Principal, session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = load_session(request, principal, session_id, write=True)
    engine = _engine_from_state(request, principal.tenant_id, session_id, state)
    summary = engine.get_session_summary()
    store = _profile_store(request, principal.tenant_id)
    try:
        store.ttl_extend(state["identity_id"])
    except Exception:
        pass
    _session_store(request).delete(principal.tenant_id, session_id)
    return summary, state


def _engine_from_state(request: Request, tenant_id: str, session_id: str, state: Dict[str, Any]) -> ABSAv2Engine:
    store = _profile_store(request, tenant_id)
    profile = store.get(state["identity_id"]) or BehavioralProfile(user_id=state["identity_id"])
    kind = state.get("identity_kind", IDENTITY_KIND_HUMAN)
    engine = ABSAv2Engine(
        profile=profile,
        classifier=_classifier(request, session_id),
        session_id=session_id,
        shadow_mode=bool(state.get("shadow_mode")),
        identity_kind=kind,
        peer_group=state.get("peer_group") or kind,
    )
    engine.restore_state(state)
    return engine


def _assess_device(request: Request, tenant_id: str, identity_id: str, metadata: Optional[Dict[str, Any]]):
    bridge = getattr(_state(request), "device_bridge", None)
    if bridge is None or not metadata:
        return None
    try:
        risk = bridge.assess(
            user_id=f"{tenant_id}/{identity_id}",
            ip=metadata.get("ip"),
            user_agent=metadata.get("user_agent"),
            latitude=metadata.get("latitude"),
            longitude=metadata.get("longitude"),
        )
        return risk.to_dict()
    except Exception as e:
        logger.warning("Device risk assessment failed: %s", e)
        return None


class _locked:
    """Identity lock that maps lock timeouts to HTTP 409."""

    def __init__(self, store, identity_id: str) -> None:
        self._cm = store.lock(identity_id) if hasattr(store, "lock") else None

    def __enter__(self):
        if self._cm is None:
            return None
        try:
            return self._cm.__enter__()
        except ProfileLockTimeout:
            raise HTTPException(status.HTTP_409_CONFLICT, "Identity is busy; retry shortly.")

    def __exit__(self, *exc):
        if self._cm is not None:
            return self._cm.__exit__(*exc)
        return False
