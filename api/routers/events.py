"""
Batch event ingestion for integrations.

The fastest way for a customer to start: forward their application's audit
events (from a log pipeline, webhook, or a periodic job). No per-action
round-trip and no SDK required.

Each event names the identity and the customer's own session id; Zelvryn
maps that to an internal session (creating it on first sight) and evaluates
events in order. Decisions are returned per event. In shadow mode nothing is
enforced, but alerts are still raised for analysts.
"""

import logging
import time
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from api import session_service as svc
from api.auth import Principal, require_permission
from api.dependencies import rate_limit_check
from api.schemas import ActionTypeEnum, IdentityKindEnum
from core.action import UserAction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Events"])

_NS = uuid.UUID("6f1c2a7e-1b9d-4c55-9a77-0c2b1d6f8e10")


class EventIn(BaseModel):
    identity_id: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(..., min_length=1, max_length=256, description="Your own session/run identifier.")
    action_type: ActionTypeEnum
    resource_id: str = Field(..., min_length=1, max_length=512)
    timestamp: Optional[float] = None
    identity_kind: IdentityKindEnum = IdentityKindEnum.HUMAN
    peer_group: Optional[str] = Field(None, max_length=64)
    records: Optional[int] = Field(None, ge=0)
    bytes: Optional[int] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventBatch(BaseModel):
    events: List[EventIn] = Field(..., min_length=1, max_length=1000)


def internal_session_id(tenant_id: str, external_session_id: str) -> str:
    return str(uuid.uuid5(_NS, f"{tenant_id}:{external_session_id}"))


@router.post("", summary="Evaluate a batch of application events")
def ingest_events(
    body: EventBatch,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("events:ingest"))],
    _rate: Annotated[None, Depends(rate_limit_check)] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    session_store = request.app.state.session_store
    results: List[Dict[str, Any]] = []
    counts = {"ALLOW": 0, "CHALLENGE": 0, "BLOCK": 0}
    alerts = set()

    for ev in body.events:
        sid = internal_session_id(principal.tenant_id, ev.session_id)
        if session_store.load(principal.tenant_id, sid) is None:
            svc.start_session(
                request, principal,
                identity_id=ev.identity_id,
                identity_kind=ev.identity_kind.value,
                peer_group=ev.peer_group,
                session_id=sid,
                external_session_id=ev.session_id,
            )
        meta = dict(ev.metadata)
        if ev.records is not None:
            meta["records"] = ev.records
        if ev.bytes is not None:
            meta["bytes"] = ev.bytes
        action = UserAction(
            action_type=ev.action_type.value,
            resource_id=ev.resource_id,
            session_id=sid,
            timestamp=ev.timestamp or time.time(),
            metadata=meta,
        )
        result, state, ctx = svc.evaluate(request, principal, sid, action)
        if state["identity_id"] != ev.identity_id:
            # A customer session id reused for a different identity
            logger.warning("Session %s reused across identities", ev.session_id)
        counts[ctx["would_be_decision"]] += 1
        if ctx["alert_id"]:
            alerts.add(ctx["alert_id"])
        results.append({
            "session_id": ev.session_id,
            "identity_id": state["identity_id"],
            "decision": result.decision,
            "would_be_decision": ctx["would_be_decision"],
            "trust_score": result.trust_score,
            "anomalies": result.anomalies,
            "reasons": result.reasons,
            "alert_id": ctx["alert_id"],
        })

    request.state.tenant_id = principal.tenant_id
    return {
        "tenant_id": principal.tenant_id,
        "processed": len(results),
        "would_be": counts,
        "alerts": sorted(alerts),
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "results": results,
    }
