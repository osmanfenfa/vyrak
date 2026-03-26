import csv
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field, model_validator

from api.admin_runtime import apply_runtime_patch, get_runtime_config
from api.dependencies import get_active_session_count, require_admin
from api.middleware import AUDIT_LOG_FILE

router = APIRouter(
    prefix="/admin-gate",
    tags=["Admin Gate"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden - admin role required"},
    },
)

_NEXT_RETRAIN_CYCLE_KEYS = {"ml_retrain_interval"}
_IMMEDIATE_EFFECT_KEYS = {
    "trust_initial",
    "trust_decay_mild",
    "trust_decay_moderate",
    "trust_decay_severe",
    "trust_decay_ai_bot",
    "trust_recovery_rate",
    "trust_challenge_threshold",
    "trust_block_threshold",
    "ml_bot_confidence_thresh",
    "cold_start_ramp_actions",
    "cold_start_ml_thresh",
    "cold_start_temporal_limit",
    "cold_start_recovery_rate",
    "ewma_decay_factor",
    "canary_traffic_percent",
    "shadow_mode",
}


class RuntimeConfigPatch(BaseModel):
    trust_initial: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_decay_mild: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_decay_moderate: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_decay_severe: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_decay_ai_bot: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_recovery_rate: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_challenge_threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trust_block_threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    ml_bot_confidence_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ml_retrain_interval: Optional[int] = Field(default=None, ge=1, le=1_000_000)
    cold_start_ramp_actions: Optional[int] = Field(default=None, ge=1, le=100_000)
    cold_start_ml_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cold_start_temporal_limit: Optional[float] = Field(default=None, ge=0.0, le=1000.0)
    cold_start_recovery_rate: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    ewma_decay_factor: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    canary_traffic_percent: Optional[int] = Field(default=None, ge=0, le=100)


class ShadowModePatch(BaseModel):
    enabled: bool


class ModelCanaryPatch(BaseModel):
    version: str = Field(..., min_length=1, max_length=128)
    traffic_percent: Optional[int] = Field(default=None, ge=0, le=100)


class ModelPromotePatch(BaseModel):
    version: str = Field(..., min_length=1, max_length=128)
    notes: str = Field(default="", max_length=500)


class ModelRollbackPatch(BaseModel):
    target_version: Optional[str] = Field(default=None, min_length=1, max_length=128)
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def normalize_target_version(self) -> "ModelRollbackPatch":
        if self.target_version is not None and not self.target_version.strip():
            self.target_version = None
        return self


@router.get("", status_code=status.HTTP_200_OK)
async def admin_gate_home(
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    cfg = get_runtime_config(request.app)
    model_registry = getattr(request.app.state, "model_registry", None)
    model_summary = model_registry.summary() if model_registry else None
    return {
        "route": "/admin-gate",
        "actor": admin_user,
        "shadow_mode": bool(getattr(request.app.state, "shadow_mode", False)),
        "active_sessions": int(get_active_session_count()),
        "runtime_config": cfg,
        "config_effects": _runtime_effect_catalog(),
        "model_registry": model_summary,
        "audit_log_file": AUDIT_LOG_FILE,
        "audit_log_exists": os.path.exists(AUDIT_LOG_FILE),
        "timestamp": time.time(),
    }


@router.get("/config", status_code=status.HTTP_200_OK)
async def get_admin_config(
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    cfg = dict(get_runtime_config(request.app))
    cfg["shadow_mode"] = bool(getattr(request.app.state, "shadow_mode", False))
    return {
        "actor": admin_user,
        "config": cfg,
        "config_effects": _runtime_effect_catalog(),
        "timestamp": time.time(),
    }


@router.patch("/config", status_code=status.HTTP_200_OK)
async def patch_admin_config(
    body: RuntimeConfigPatch,
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    patch = body.model_dump(exclude_none=True)
    if not patch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No configuration fields provided.",
        )

    before = dict(get_runtime_config(request.app))
    merged = dict(before)
    merged.update(patch)
    _validate_thresholds(merged)

    after = apply_runtime_patch(request.app, patch)
    effects = _classify_runtime_effects(patch.keys())

    request.state.audit_event = {
        "event_type": "admin_config_changed",
        "actor_id": admin_user,
        "changed_keys": sorted(patch.keys()),
        "effective_immediately": effects["effective_immediately"],
        "effective_next_retrain_cycle": effects["effective_next_retrain_cycle"],
        "before_state": {k: before.get(k) for k in patch.keys()},
        "after_state": {k: after.get(k) for k in patch.keys()},
        "at": time.time(),
    }

    return {
        "updated": True,
        "changed_keys": sorted(patch.keys()),
        "effective_immediately": effects["effective_immediately"],
        "effective_next_retrain_cycle": effects["effective_next_retrain_cycle"],
        "config_effects": _runtime_effect_catalog(),
        "config": after,
        "timestamp": time.time(),
    }


@router.patch("/shadow-mode", status_code=status.HTTP_200_OK)
async def patch_shadow_mode(
    body: ShadowModePatch,
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    before = bool(getattr(request.app.state, "shadow_mode", False))
    after_cfg = apply_runtime_patch(request.app, {"shadow_mode": bool(body.enabled)})
    after = bool(getattr(request.app.state, "shadow_mode", False))

    request.state.audit_event = {
        "event_type": "admin_shadow_mode_changed",
        "actor_id": admin_user,
        "before_state": {"shadow_mode": before},
        "after_state": {"shadow_mode": after},
        "at": time.time(),
    }

    return {
        "updated": True,
        "shadow_mode": after,
        "runtime_config": after_cfg,
        "timestamp": time.time(),
    }


@router.get("/models", status_code=status.HTTP_200_OK)
async def list_models(
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    model_registry = getattr(request.app.state, "model_registry", None)
    if model_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialised.",
        )

    versions = [v.to_dict() for v in model_registry.list_versions()]
    summary = model_registry.summary()
    return {
        "actor": admin_user,
        "summary": summary,
        "versions": versions,
        "timestamp": time.time(),
    }


@router.post("/models/canary", status_code=status.HTTP_200_OK)
async def set_model_canary(
    body: ModelCanaryPatch,
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    model_registry = getattr(request.app.state, "model_registry", None)
    if model_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialised.",
        )

    before = model_registry.summary()
    if body.traffic_percent is not None:
        apply_runtime_patch(request.app, {"canary_traffic_percent": body.traffic_percent})
    try:
        promoted = model_registry.promote_to_canary(body.version)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    after = model_registry.summary()
    request.state.audit_event = {
        "event_type": "admin_model_canary_promoted",
        "actor_id": admin_user,
        "model_version": body.version,
        "before_state": before,
        "after_state": after,
        "at": time.time(),
    }

    return {
        "updated": True,
        "model": promoted.to_dict(),
        "summary": after,
        "timestamp": time.time(),
    }


@router.post("/models/promote", status_code=status.HTTP_200_OK)
async def promote_model(
    body: ModelPromotePatch,
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    model_registry = getattr(request.app.state, "model_registry", None)
    if model_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialised.",
        )

    before = model_registry.summary()
    try:
        promoted = model_registry.promote(body.version, notes=body.notes)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    after = model_registry.summary()
    request.state.audit_event = {
        "event_type": "admin_model_promoted",
        "actor_id": admin_user,
        "model_version": body.version,
        "notes": body.notes,
        "before_state": before,
        "after_state": after,
        "at": time.time(),
    }

    return {
        "updated": True,
        "model": promoted.to_dict(),
        "summary": after,
        "timestamp": time.time(),
    }


@router.post("/models/rollback", status_code=status.HTTP_200_OK)
async def rollback_model(
    body: ModelRollbackPatch,
    request: Request,
    admin_user: str = Depends(require_admin),
) -> Dict[str, Any]:
    model_registry = getattr(request.app.state, "model_registry", None)
    if model_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialised.",
        )

    before = model_registry.summary()
    try:
        rolled = model_registry.rollback(body.target_version)
        if body.notes:
            rolled.notes = body.notes
            if hasattr(model_registry, "_save_registry"):
                model_registry._save_registry()
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    after = model_registry.summary()
    request.state.audit_event = {
        "event_type": "admin_model_rollback",
        "actor_id": admin_user,
        "target_version": rolled.version,
        "notes": body.notes,
        "before_state": before,
        "after_state": after,
        "at": time.time(),
    }

    return {
        "updated": True,
        "model": rolled.to_dict(),
        "summary": after,
        "timestamp": time.time(),
    }


@router.get("/audit", status_code=status.HTTP_200_OK)
async def query_audit_logs(
    request: Request,
    admin_user: str = Depends(require_admin),
    limit: int = Query(default=200, ge=1, le=5_000),
    offset: int = Query(default=0, ge=0, le=200_000),
    event_type: Optional[str] = Query(default=None),
    actor_id: Optional[str] = Query(default=None),
    since_ts: Optional[float] = Query(default=None),
    until_ts: Optional[float] = Query(default=None),
) -> Dict[str, Any]:
    entries = _load_audit_entries()
    filtered = _filter_audit_entries(
        entries=entries,
        event_type=event_type,
        actor_id=actor_id,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    filtered.sort(key=_entry_ts, reverse=True)
    page = filtered[offset: offset + limit]

    return {
        "actor": admin_user,
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
        "items": page,
        "timestamp": time.time(),
    }


@router.get("/audit/export")
async def export_audit_logs(
    request: Request,
    admin_user: str = Depends(require_admin),
    format: str = Query(default="jsonl", pattern="^(jsonl|json|csv)$"),
    event_type: Optional[str] = Query(default=None),
    actor_id: Optional[str] = Query(default=None),
    since_ts: Optional[float] = Query(default=None),
    until_ts: Optional[float] = Query(default=None),
    limit: int = Query(default=20_000, ge=1, le=200_000),
) -> Response:
    entries = _load_audit_entries()
    filtered = _filter_audit_entries(
        entries=entries,
        event_type=event_type,
        actor_id=actor_id,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    filtered.sort(key=_entry_ts, reverse=True)
    payload = filtered[:limit]

    request.state.audit_event = {
        "event_type": "admin_audit_export",
        "actor_id": admin_user,
        "format": format,
        "count": len(payload),
        "filters": {
            "event_type": event_type,
            "actor_id": actor_id,
            "since_ts": since_ts,
            "until_ts": until_ts,
            "limit": limit,
        },
        "at": time.time(),
    }

    timestamp = int(time.time())
    if format == "jsonl":
        content = "\n".join(json.dumps(item, separators=(",", ":")) for item in payload)
        return Response(
            content=content,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="audit_export_{timestamp}.jsonl"'
            },
        )

    if format == "json":
        content = json.dumps(payload, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="audit_export_{timestamp}.json"'
            },
        )

    rows = payload
    all_keys = sorted({k for row in rows for k in row.keys()})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _to_csv_cell(row.get(k)) for k in all_keys})
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="audit_export_{timestamp}.csv"'
        },
    )


def _validate_thresholds(cfg: Dict[str, Any]) -> None:
    challenge = float(cfg["trust_challenge_threshold"])
    block = float(cfg["trust_block_threshold"])
    if block > challenge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "trust_block_threshold must be less than or equal to "
                "trust_challenge_threshold."
            ),
        )


def _load_audit_entries() -> List[Dict[str, Any]]:
    if not os.path.exists(AUDIT_LOG_FILE):
        return []

    records: List[Dict[str, Any]] = []
    with open(AUDIT_LOG_FILE, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _filter_audit_entries(
    entries: List[Dict[str, Any]],
    event_type: Optional[str],
    actor_id: Optional[str],
    since_ts: Optional[float],
    until_ts: Optional[float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in entries:
        if event_type and str(item.get("event_type")) != event_type:
            continue

        item_actor = _audit_actor(item)
        if actor_id and item_actor != actor_id:
            continue

        ts = _entry_ts(item)
        if since_ts is not None and ts < since_ts:
            continue
        if until_ts is not None and ts > until_ts:
            continue
        out.append(item)
    return out


def _audit_actor(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("actor_id", "admin_user", "user_id", "reset_by", "deleted_by"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _entry_ts(entry: Dict[str, Any]) -> float:
    for key in ("logged_at", "timestamp", "at"):
        val = entry.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _to_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _classify_runtime_effects(changed_keys) -> Dict[str, List[str]]:
    immediate: List[str] = []
    next_cycle: List[str] = []

    for key in sorted(set(str(k) for k in changed_keys)):
        if key in _NEXT_RETRAIN_CYCLE_KEYS:
            next_cycle.append(key)
        elif key in _IMMEDIATE_EFFECT_KEYS:
            immediate.append(key)
        else:
            immediate.append(key)

    return {
        "effective_immediately": immediate,
        "effective_next_retrain_cycle": next_cycle,
    }


def _runtime_effect_catalog() -> Dict[str, List[str]]:
    return {
        "effective_immediately": sorted(_IMMEDIATE_EFFECT_KEYS),
        "effective_next_retrain_cycle": sorted(_NEXT_RETRAIN_CYCLE_KEYS),
    }
