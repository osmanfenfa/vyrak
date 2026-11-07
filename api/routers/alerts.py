"""
Alert endpoints for security analysts.

An alert is opened on a session's first anomaly and updated as more arrive.
Analysts label alerts; labelled alerts are exported as analyst-verified
training data — the feedback loop that lets a supervised model be trained
on *your* customers' real attacks and false positives instead of synthetic
data.
"""

import json
import time
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from api.auth import Principal, require_permission
from api.dependencies import get_alert_store
from api.stores import ALERT_STATUSES

router = APIRouter(prefix="/alerts", tags=["Alerts"])


class AlertLabelRequest(BaseModel):
    status: str = Field(..., description=f"One of {list(ALERT_STATUSES)}")
    note: str = Field("", max_length=2000)


@router.get("", summary="List alerts for this tenant (newest first)")
def list_alerts(
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("alerts:read"))],
    status_filter: Optional[str] = Query(None, alias="status"),
    severity: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    identity_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    if status_filter and status_filter not in ALERT_STATUSES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"status must be one of {list(ALERT_STATUSES)}")
    data = get_alert_store(request).list(
        principal.tenant_id, status=status_filter, severity=severity,
        identity_id=identity_id, limit=limit, offset=offset,
    )
    return {"tenant_id": principal.tenant_id, "limit": limit, "offset": offset, **data}


@router.get("/training-export", summary="Export analyst-labelled alerts as training data (JSONL)")
def export_training_samples(
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("alerts:label"))],
) -> Response:
    samples = get_alert_store(request).training_samples(principal.tenant_id)
    request.state.tenant_id = principal.tenant_id
    request.state.audit_event = {
        "event_type": "alert_training_export",
        "actor_id": principal.subject,
        "count": len(samples),
        "at": time.time(),
    }
    body = "\n".join(json.dumps(s, separators=(",", ":")) for s in samples)
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="labelled_alerts_{int(time.time())}.jsonl"'},
    )


@router.get("/{alert_id}", summary="Get one alert with its event timeline")
def get_alert(
    alert_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("alerts:read"))],
) -> Dict[str, Any]:
    alert = get_alert_store(request).get(principal.tenant_id, alert_id)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.")
    for e in alert["events"]:
        e.pop("features", None)
    return alert


@router.patch("/{alert_id}", summary="Label an alert (confirm, dismiss, or mark expected)")
def label_alert(
    alert_id: str,
    body: AlertLabelRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_permission("alerts:label"))],
) -> Dict[str, Any]:
    try:
        alert = get_alert_store(request).label(
            principal.tenant_id, alert_id, body.status, principal.subject, body.note
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.")
    request.state.tenant_id = principal.tenant_id
    request.state.audit_event = {
        "event_type": "alert_labelled",
        "actor_id": principal.subject,
        "alert_id": alert_id,
        "after_state": {"status": body.status},
        "at": time.time(),
    }
    for e in alert["events"]:
        e.pop("features", None)
    return alert
