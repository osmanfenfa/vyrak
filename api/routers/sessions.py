"""
Session management endpoints — the core evaluation surface of the Zelvryn API.
"""

import logging
import time
import uuid
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, status
from api.dependencies import (
    get_current_user,
    get_device_bridge,
    get_engine_for_session,
    get_or_create_engine,
    get_profile_store,
    close_session_engine,
    rate_limit_check,
)
from api.schemas import (
    ActionRequest,
    DecisionResponse,
    RestoreTrustRequest,
    SessionDeleteResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionSummaryMetrics,
    SessionSummaryResponse,
    FeatureImportanceItem,
    AnomalyEntry,
    RiskLabelEnum,
)
from core.action import UserAction
from core.constants import TRUST_INITIAL

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/session",
    tags=["Sessions"],
    responses={
        401: {"description": "Unauthorised — missing or invalid Bearer token"},
        429: {"description": "Too many requests — rate limit exceeded"},
    },
)


def _runtime_ramp_actions(request: Request) -> int:
    cfg = getattr(request.app.state, "runtime_config", {}) or {}
    return max(1, int(cfg.get("cold_start_ramp_actions", 50)))


def _runtime_trust_initial(request: Request) -> float:
    cfg = getattr(request.app.state, "runtime_config", {}) or {}
    val = float(cfg.get("trust_initial", TRUST_INITIAL))
    return min(100.0, max(0.0, val))


@router.post(
    "/start",
    response_model=SessionStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Begin a new authenticated session",
    description=(
        "Called immediately after successful authentication in the host application. "
        "Creates a new evaluation session and returns a session_id to be used in "
        "all subsequent action evaluation calls."
    ),
)
async def start_session(
    body:       SessionStartRequest,
    request:    Request,
    user_id:    Annotated[str, Depends(get_current_user)],
    _rate:      Annotated[None, Depends(rate_limit_check)] = None,
) -> SessionStartResponse:
    """
    Initialise a Zelvryn evaluation session for an authenticated user.
    """
    session_id = str(uuid.uuid4())

    # Attach device signals to the cold start assessment
    device_bridge = get_device_bridge(request)
    device_risk   = None
    if device_bridge and body.metadata:
        meta = body.metadata
        device_risk = device_bridge.assess(
            user_id    = user_id,
            ip         = meta.ip,
            user_agent = meta.user_agent,
            latitude   = meta.latitude,
            longitude  = meta.longitude,
        )

    engine = get_or_create_engine(session_id, user_id, request)
    request.state.user_id    = user_id
    request.state.session_id = session_id
    shadow_mode  = getattr(request.app.state, "shadow_mode", False)
    ramp_active  = not engine.profile.is_mature(
        min_actions=_runtime_ramp_actions(request)
    )
    trust_initial = _runtime_trust_initial(request)

    logger.info(
        "Session started: session=%s user=%s ramp=%s shadow=%s",
        session_id, user_id, ramp_active, shadow_mode
    )

    return SessionStartResponse(
        session_id  = session_id,
        user_id     = user_id,
        trust_score = trust_initial,
        ramp_active = ramp_active,
        shadow_mode = shadow_mode,
        started_at  = time.time(),
    )


@router.post(
    "/{session_id}/action",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a user action",
    description=(
        "The core evaluation endpoint. Call this for every protected operation "
        "performed by the user. Returns a decision (ALLOW / CHALLENGE / BLOCK) "
        "within 50ms at p95."
    ),
)
async def evaluate_action(
    session_id: str,
    body:       ActionRequest,
    request:    Request,
    user_id:    Annotated[str, Depends(get_current_user)],
    _rate:      Annotated[None, Depends(rate_limit_check)] = None,
) -> DecisionResponse:
    """
    Evaluate a single user action against the session's behavioural profile.
    """
    engine = get_engine_for_session(session_id, request)

    # Verify the session belongs to the authenticated user
    if engine.profile.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session does not belong to the authenticated user.",
        )

    # Build UserAction
    try:
        action = UserAction(
            action_type = body.action_type.value,
            resource_id = body.resource_id,
            session_id  = session_id,
            timestamp   = body.timestamp or time.time(),
            metadata    = body.metadata.model_dump() if body.metadata else {},
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # Evaluate
    result = engine.evaluate_action(action)
    
    profile_store = get_profile_store(request)
    try:
        profile_store.save(user_id, engine.profile)
    except Exception as e:
        logger.warning(
            "Profile save failed after action: user=%s error=%s", user_id, e
        )

    # Set state for AuditMiddleware
    request.state.user_id         = user_id
    request.state.session_id      = session_id
    request.state.zelvryn_decision = result.decision
    request.state.trust_score     = result.trust_score
    request.state.bot_prob        = result.bot_prob
    request.state.anomalies       = result.anomalies

    if result.decision in ("CHALLENGE", "BLOCK"):
        logger.info(
            "%s: session=%s user=%s trust=%.1f P(bot)=%.3f anomalies=%s",
            result.decision, session_id, user_id,
            result.trust_score, result.bot_prob, result.anomalies
        )

    return DecisionResponse(
        decision    = result.decision,
        trust_score = result.trust_score,
        bot_prob    = result.bot_prob,
        similarity  = result.similarity,
        entropy     = result.entropy,
        temporal_z  = result.temporal_z,
        anomalies   = result.anomalies,
        ramp_active = result.ramp_active,
        session_id  = session_id,
        action_type = result.action_type,
        timestamp   = result.timestamp,
    )


@router.post(
    "/{session_id}/restore-trust",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Restore trust after successful re-authentication",
    description=(
        "Called after the user successfully completes an out-of-band "
        "re-authentication challenge (email OTP, push notification, etc.). "
        "Partially restores the trust score without a full reset."
    ),
)
async def restore_trust(
    session_id: str,
    body:       RestoreTrustRequest,
    request:    Request,
    user_id:    Annotated[str, Depends(get_current_user)],
) -> DecisionResponse:
    """
    Restore session trust after a successful challenge completion.
    """
    engine = get_engine_for_session(session_id, request)

    if engine.profile.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session does not belong to the authenticated user.",
        )

    engine.restore_trust(increment=body.increment)

    logger.info(
        "Trust restored: session=%s user=%s reason=%s new_trust=%.1f",
        session_id, user_id, body.reason, engine.trust_score
    )

    # Persist
    profile_store = get_profile_store(request)
    try:
        profile_store.save(user_id, engine.profile)
    except Exception as e:
        logger.warning("Profile save failed after trust restore: %s", e)

    return DecisionResponse(
        decision    = engine._compute_decision(),
        trust_score = engine.trust_score,
        bot_prob    = 0.0,
        similarity  = 1.0,
        entropy     = 0.0,
        temporal_z  = 0.0,
        anomalies   = [f"TRUST_RESTORED:{body.reason}"],
        ramp_active = not engine.profile.is_mature(
            min_actions=_runtime_ramp_actions(request)
        ),
        session_id  = session_id,
        action_type = "login",
        timestamp   = time.time(),
    )


@router.get(
    "/{session_id}/summary",
    response_model=SessionSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get session evaluation summary",
    description=(
        "Returns the full session report: decision counts, average trust score, "
        "top feature importances, and the complete anomaly log."
    ),
)
async def get_session_summary(
    session_id: str,
    request:    Request,
    user_id:    Annotated[str, Depends(get_current_user)],
) -> SessionSummaryResponse:
    """
    Return a comprehensive summary of the session's evaluation results.
    """
    engine = get_engine_for_session(session_id, request)

    if engine.profile.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session does not belong to the authenticated user.",
        )

    summary = engine.get_session_summary()
    decisions = engine.decision_log

    # Compute average trust and bot prob from result history
    results = engine.result_history
    avg_trust = (
        sum(r.trust_score for r in results) / len(results)
        if results else _runtime_trust_initial(request)
    )
    avg_bot_p = (
        sum(r.bot_prob for r in results) / len(results)
        if results else 0.0
    )

    metrics = SessionSummaryMetrics(
        total_actions  = summary["total_actions"],
        allowed        = summary["allowed"],
        challenged     = summary["challenged"],
        blocked        = summary["blocked"],
        anomaly_count  = summary["anomaly_count"],
        avg_trust_score = round(avg_trust, 2),
        avg_bot_prob   = round(avg_bot_p, 4),
    )

    top_features = [
        FeatureImportanceItem(name=name, importance=imp)
        for name, imp in (summary.get("top_features") or [])
    ]

    # Convert anomaly log entries
    anomaly_entries = []
    for entry in summary.get("anomaly_log", []):
        try:
            anomaly_entries.append(AnomalyEntry(
                action_type = entry.get("action_type", "unknown"),
                timestamp   = entry.get("timestamp", 0.0),
                trust_score = entry.get("trust_score", 0.0),
                anomalies   = entry.get("anomalies", []),
                bot_prob    = entry.get("bot_prob", 0.0),
                decision    = entry.get("decision", "ALLOW"),
            ))
        except Exception:
            pass

    # Map risk_rating string to enum
    risk_str = summary.get("risk_rating", "LOW RISK")
    risk_map = {
        "LOW RISK":      RiskLabelEnum.LOW,
        "MEDIUM RISK":   RiskLabelEnum.MEDIUM,
        "HIGH RISK":     RiskLabelEnum.HIGH,
        "CRITICAL RISK": RiskLabelEnum.CRITICAL,
    }
    risk_label = risk_map.get(risk_str, RiskLabelEnum.LOW)

    return SessionSummaryResponse(
        user_id            = user_id,
        session_id         = session_id,
        final_trust        = summary["final_trust"],
        risk_rating        = risk_label,
        metrics            = metrics,
        ml_trained         = summary.get("ml_trained", False),
        ml_train_samples   = summary.get("ml_train_samples", 0),
        top_features       = top_features,
        session_duration_s = summary.get("session_duration_s", 0.0),
        shadow_mode        = summary.get("shadow_mode", False),
        anomaly_log        = anomaly_entries,
    )


@router.delete(
    "/{session_id}",
    response_model=SessionDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="End and archive a session",
    description=(
        "Ends the session, saves the final profile state to Redis, "
    ),
)
async def delete_session(
    session_id: str,
    request:    Request,
    user_id:    Annotated[str, Depends(get_current_user)],
) -> SessionDeleteResponse:
    """
    Close a session, persist the final profile, and clean up memory.
    """
    engine = get_engine_for_session(session_id, request)

    if engine.profile.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session does not belong to the authenticated user.",
        )

    summary      = engine.get_session_summary()
    final_trust  = summary["final_trust"]
    total_actions = summary["total_actions"]

    # Persist updated profile
    profile_store = get_profile_store(request)
    archived = False
    try:
        profile_store.save(user_id, engine.profile)
        profile_store.ttl_extend(user_id)
        archived = True
    except Exception as e:
        logger.warning(
            "Profile persistence failed on session close: session=%s error=%s",
            session_id, e
        )

    close_session_engine(session_id)

    # Audit
    request.state.audit_event = {
        "event_type":     "session_closed",
        "user_id":        user_id,
        "session_id":     session_id,
        "final_trust":    final_trust,
        "total_actions":  total_actions,
        "risk_rating":    summary.get("risk_rating"),
    }

    logger.info(
        "Session closed: session=%s user=%s final_trust=%.1f actions=%d",
        session_id, user_id, final_trust, total_actions
    )

    return SessionDeleteResponse(
        session_id    = session_id,
        archived      = archived,
        final_trust   = final_trust,
        total_actions = total_actions,
    )
