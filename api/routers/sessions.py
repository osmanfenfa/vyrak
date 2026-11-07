"""
Session endpoints — the real-time evaluation surface of the Zelvryn API.

Two calling patterns are supported:

* **Integration mode** (recommended): the customer's backend calls with an
  API key (role ``integration``) and names the identity in ``user_id``.
  Works for humans, AI agents and service accounts.
* **Direct mode**: an end user's own token (role ``user``); they can only
  evaluate their own sessions.

Endpoints are synchronous on purpose: they do blocking Redis I/O and take a
per-identity lock, so FastAPI runs them in its thread pool.
"""

import logging
import time
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

import core.engine as engine_mod   # thresholds are runtime-patchable module globals
from api import session_service as svc
from api.auth import Principal, get_principal
from api.dependencies import rate_limit_check
from api.schemas import (
    ActionRequest,
    AnomalyEntry,
    DecisionResponse,
    FeatureImportanceItem,
    RestoreTrustRequest,
    RiskLabelEnum,
    SessionDeleteResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionSummaryMetrics,
    SessionSummaryResponse,
)
from core.action import UserAction

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/session",
    tags=["Sessions"],
    responses={
        401: {"description": "Unauthorised — missing or invalid credentials"},
        403: {"description": "Forbidden — session belongs to another identity"},
        429: {"description": "Too many requests — rate limit exceeded"},
    },
)


def _audit_state(request: Request, principal: Principal, session_id: str, identity_id: str) -> None:
    request.state.user_id = identity_id
    request.state.actor_id = principal.subject
    request.state.tenant_id = principal.tenant_id
    request.state.session_id = session_id


@router.post(
    "/start",
    response_model=SessionStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Begin a new evaluated session",
    description=(
        "Call right after the identity authenticates in your application. "
        "Integration keys name the identity in `user_id`; end users may only "
        "start sessions for themselves."
    ),
)
def start_session(
    body: SessionStartRequest,
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
    _rate: Annotated[None, Depends(rate_limit_check)] = None,
) -> SessionStartResponse:
    identity_id = svc.resolve_identity(principal, body.user_id)
    data = svc.start_session(
        request,
        principal,
        identity_id=identity_id,
        identity_kind=body.identity_kind.value,
        peer_group=body.peer_group,
        metadata=body.metadata.model_dump() if body.metadata else None,
    )
    _audit_state(request, principal, data["session_id"], identity_id)
    logger.info(
        "Session started: tenant=%s session=%s identity=%s kind=%s shadow=%s",
        principal.tenant_id, data["session_id"], identity_id,
        data["identity_kind"], data["shadow_mode"],
    )
    return SessionStartResponse(**data)


@router.post(
    "/{session_id}/action",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate an action",
    description=(
        "Call for every protected operation. Returns ALLOW / CHALLENGE / BLOCK "
        "with plain-language reasons. In shadow mode `decision` is always ALLOW "
        "and `would_be_decision` shows what enforcement would have done."
    ),
)
def evaluate_action(
    session_id: str,
    body: ActionRequest,
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
    _rate: Annotated[None, Depends(rate_limit_check)] = None,
) -> DecisionResponse:
    state = svc.load_session(request, principal, session_id, write=True)
    try:
        action = UserAction(
            action_type=body.action_type.value,
            resource_id=body.resource_id,
            session_id=session_id,
            timestamp=body.timestamp or time.time(),
            metadata=body.metadata.model_dump(exclude_none=True) if body.metadata else {},
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    result, _, ctx = svc.evaluate(request, principal, session_id, action, state=state)

    _audit_state(request, principal, session_id, state["identity_id"])
    request.state.trust_score_before = ctx["trust_before"]
    request.state.decision_before = ctx["decision_before"]
    request.state.zelvryn_decision = result.decision
    request.state.would_be_decision = ctx["would_be_decision"]
    request.state.trust_score = result.trust_score
    request.state.bot_prob = result.bot_prob
    request.state.anomalies = result.anomalies
    request.state.reasons = result.reasons

    if ctx["would_be_decision"] in ("CHALLENGE", "BLOCK"):
        logger.info(
            "%s%s: tenant=%s session=%s identity=%s trust=%.1f anomalies=%s",
            ctx["would_be_decision"], " (shadow)" if ctx["shadow_mode"] else "",
            principal.tenant_id, session_id, state["identity_id"],
            result.trust_score, result.anomalies,
        )

    return DecisionResponse(
        decision=result.decision,
        trust_score=result.trust_score,
        bot_prob=result.bot_prob,
        similarity=result.similarity,
        entropy=result.entropy,
        temporal_z=result.temporal_z,
        anomalies=result.anomalies,
        ramp_active=result.ramp_active,
        session_id=session_id,
        action_type=result.action_type,
        timestamp=result.timestamp,
        reasons=result.reasons,
        signals=result.signals,
        would_be_decision=ctx["would_be_decision"],
        alert_id=ctx["alert_id"],
    )


@router.post(
    "/{session_id}/restore-trust",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Restore trust after successful re-authentication",
    description=(
        "Call after the identity passes an out-of-band challenge (OTP, push, "
        "manager approval). Partially restores trust without a full reset."
    ),
)
def restore_trust(
    session_id: str,
    body: RestoreTrustRequest,
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
) -> DecisionResponse:
    state = svc.restore_trust(request, principal, session_id, body.increment)
    trust = float(state["trust_score"])
    _audit_state(request, principal, session_id, state["identity_id"])
    request.state.audit_event = {
        "event_type": "trust_restored",
        "actor_id": principal.subject,
        "user_id": state["identity_id"],
        "session_id": session_id,
        "reason": body.reason,
        "after_state": {"trust_score": trust},
        "at": time.time(),
    }
    decision = (
        "BLOCK" if trust <= engine_mod.TRUST_BLOCK_THRESHOLD
        else "CHALLENGE" if trust <= engine_mod.TRUST_CHALLENGE_THRESHOLD
        else "ALLOW"
    )
    return DecisionResponse(
        decision=decision,
        trust_score=trust,
        bot_prob=0.0,
        similarity=1.0,
        entropy=0.0,
        temporal_z=0.0,
        anomalies=[f"TRUST_RESTORED:{body.reason}"],
        ramp_active=False,
        session_id=session_id,
        action_type="login",
        timestamp=time.time(),
        reasons=[f"Trust restored by {body.increment:.0f} after '{body.reason}'."],
        would_be_decision=decision,
        alert_id=state.get("alert_id"),
    )


_RISK_MAP = {
    "LOW RISK": RiskLabelEnum.LOW,
    "MEDIUM RISK": RiskLabelEnum.MEDIUM,
    "HIGH RISK": RiskLabelEnum.HIGH,
    "CRITICAL RISK": RiskLabelEnum.CRITICAL,
}


def _summary_response(session_id: str, summary: Dict[str, Any], state: Dict[str, Any]) -> SessionSummaryResponse:
    results = state.get("result_history") or []
    avg_trust = (
        sum(float(r["trust_score"]) for r in results) / len(results)
        if results else float(state.get("trust_score", 100.0))
    )
    avg_bot = sum(float(r.get("bot_prob", 0.0)) for r in results) / len(results) if results else 0.0
    metrics = SessionSummaryMetrics(
        total_actions=summary["total_actions"],
        allowed=summary["allowed"],
        challenged=summary["challenged"],
        blocked=summary["blocked"],
        anomaly_count=summary["anomaly_count"],
        avg_trust_score=round(avg_trust, 2),
        avg_bot_prob=round(avg_bot, 4),
    )
    anomaly_entries = []
    for entry in summary.get("anomaly_log", []):
        anomaly_entries.append(AnomalyEntry(
            action_type=entry.get("action_type", "unknown"),
            timestamp=entry.get("timestamp", 0.0),
            trust_score=entry.get("trust_score", 0.0),
            anomalies=entry.get("anomalies", []),
            bot_prob=entry.get("bot_prob", 0.0),
            decision=entry.get("decision", "ALLOW"),
            reasons=entry.get("reasons", []),
        ))
    return SessionSummaryResponse(
        user_id=state["identity_id"],
        session_id=session_id,
        final_trust=summary["final_trust"],
        risk_rating=_RISK_MAP.get(summary.get("risk_rating", "LOW RISK"), RiskLabelEnum.LOW),
        metrics=metrics,
        ml_trained=summary.get("ml_trained", False),
        ml_train_samples=summary.get("ml_train_samples", 0),
        top_features=[
            FeatureImportanceItem(name=n, importance=i)
            for n, i in (summary.get("top_features") or [])
        ],
        session_duration_s=summary.get("session_duration_s", 0.0),
        shadow_mode=summary.get("shadow_mode", False),
        identity_kind=summary.get("identity_kind", "human"),
        alert_id=state.get("alert_id"),
        anomaly_log=anomaly_entries,
    )


@router.get(
    "/{session_id}/summary",
    response_model=SessionSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get session evaluation summary",
)
def get_session_summary(
    session_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
) -> SessionSummaryResponse:
    summary, state = svc.summarize(request, principal, session_id)
    return _summary_response(session_id, summary, state)


@router.delete(
    "/{session_id}",
    response_model=SessionDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="End a session",
)
def delete_session(
    session_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
) -> SessionDeleteResponse:
    summary, state = svc.close(request, principal, session_id)
    _audit_state(request, principal, session_id, state["identity_id"])
    request.state.audit_event = {
        "event_type": "session_closed",
        "actor_id": principal.subject,
        "user_id": state["identity_id"],
        "session_id": session_id,
        "final_trust": summary["final_trust"],
        "total_actions": summary["total_actions"],
        "risk_rating": summary.get("risk_rating"),
        "before_state": {"session_open": True, "trust_score": summary["final_trust"]},
        "after_state": {"session_open": False, "risk_rating": summary.get("risk_rating")},
        "at": time.time(),
    }
    return SessionDeleteResponse(
        session_id=session_id,
        archived=True,
        final_trust=summary["final_trust"],
        total_actions=summary["total_actions"],
    )
