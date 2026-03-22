"""
zelvryn/api/routers/shadow.py
===============================
Shadow mode ingestion and training data export endpoints.

Endpoints
---------
  POST /shadow/ingest   — Process a batch of historical action events in
                          shadow mode (evaluate, log, no enforcement)
  GET  /shadow/export   — Export shadow session logs as training datasets
  GET  /shadow/stats    — Shadow session statistics summary
  POST /shadow/verify   — Set an analyst-verified label on a log entry

These endpoints are used during the initial deployment phase (shadow mode
weeks 1-4) and by the security analyst team for ongoing model maintenance.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import os
import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from api.dependencies import (
    get_current_user,
    get_feature_store,
    get_profile_store,
    require_admin,
    rate_limit_check,
)
from api.schemas import (
    ShadowExportRequest,
    ShadowExportResponse,
    ShadowIngestRequest,
    ShadowIngestResponse,
)
from core.action import UserAction
from core.engine import ABSAv2Engine
from core.profile import BehavioralProfile
from core.constants import SHADOW_MODE_ENABLED
from ingestion.shadow_mode import ShadowLogEntry, ShadowModeLogger

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/shadow",
    tags=["Shadow Mode"],
    responses={
        401: {"description": "Unauthorised"},
        403: {"description": "Forbidden — admin role required"},
    },
)

# In-memory shadow session storage: {session_id: ShadowModeLogger}
_shadow_loggers: dict = {}

# Default export directory
_EXPORT_DIR = os.environ.get("SHADOW_EXPORT_DIR", "data/shadow_exports")


# ─────────────────────────────────────────────────────────────────────────────
# POST /shadow/ingest
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=ShadowIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a batch of events in shadow mode",
    description=(
        "Processes a batch of historical action events through the full Zelvryn "
        "evaluation pipeline without enforcing any decisions. All evaluation "
        "results are logged for later export as training data. "
        "Admin role required."
    ),
)
async def shadow_ingest(
    body:       ShadowIngestRequest,
    request:    Request,
    admin_user: Annotated[str, Depends(require_admin)],
    _rate:      Annotated[None, Depends(rate_limit_check)] = None,
) -> ShadowIngestResponse:
    """
    Process a batch of action events through the shadow evaluation stack.
    """
    t_start = time.time()

    profile_store = get_profile_store(request)
    processed = errors = blocks = challenges = 0

    for event in body.events:
        try:
            user_id    = str(event.get("user_id", "unknown"))
            session_id = str(event.get("session_id", f"{user_id}_shadow"))
            action_str = str(event.get("action_type", "read"))
            resource   = str(event.get("resource_id", "unknown"))
            timestamp  = float(event.get("timestamp") or time.time())

            # Validate and normalise action type
            if action_str not in UserAction.ACTION_TYPES:
                action_str = "read"

            action = UserAction(
                action_type = action_str,
                resource_id = resource,
                session_id  = session_id,
                timestamp   = timestamp,
                metadata    = event.get("metadata", {}),
            )

            # Load or create profile
            try:
                profile = profile_store.get_or_create(user_id)
            except Exception:
                profile = BehavioralProfile(user_id=user_id)

            # Get or create shadow logger for this session
            if session_id not in _shadow_loggers:
                engine = ABSAv2Engine(
                    profile     = profile,
                    session_id  = session_id,
                    shadow_mode = True,
                )
                shadow_logger = ShadowModeLogger(
                    engine     = engine,
                    export_dir = _EXPORT_DIR,
                )
                _shadow_loggers[session_id] = shadow_logger
            else:
                shadow_logger = _shadow_loggers[session_id]

            # Evaluate silently
            result = shadow_logger.evaluate_silently(action)

            # Track real decisions (even though not enforced)
            real_decision = result.anomalies[0] if result.anomalies else "ALLOW"
            # The real decision is stored in the shadow log entry
            if shadow_logger._log:
                last_entry = shadow_logger._log[-1]
                if last_entry.real_decision == "BLOCK":
                    blocks += 1
                elif last_entry.real_decision == "CHALLENGE":
                    challenges += 1

            # Save updated profile
            try:
                profile_store.save(user_id, shadow_logger.engine.profile)
            except Exception:
                pass

            processed += 1

        except Exception as e:
            errors += 1
            logger.debug("Shadow ingest event error: %s", e)

    # Export if requested
    export_path = None
    if body.export_results:
        try:
            # Export all active shadow loggers
            for session_id, shadow_logger in list(_shadow_loggers.items()):
                if shadow_logger.entry_count > 0:
                    export_path = shadow_logger.export_session(
                        format       = "jsonl",
                        pseudonymise = True,
                    )
        except Exception as e:
            logger.warning("Shadow export failed: %s", e)

    duration_ms = (time.time() - t_start) * 1000

    logger.info(
        "Shadow ingest complete: processed=%d errors=%d blocks=%d "
        "challenges=%d duration=%.1fms",
        processed, errors, blocks, challenges, duration_ms
    )

    return ShadowIngestResponse(
        processed   = processed,
        errors      = errors,
        blocks      = blocks,
        challenges  = challenges,
        export_path = export_path,
        duration_ms = round(duration_ms, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /shadow/export
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/export",
    response_model=ShadowExportResponse,
    status_code=status.HTTP_200_OK,
    summary="Export shadow session logs as training data",
    description=(
        "Exports all accumulated shadow session logs as a labeled training dataset. "
        "The output can be used directly with ModelTrainer.train_from_files(). "
        "Admin role required."
    ),
)
async def export_shadow_data(
    request:        Request,
    admin_user:     Annotated[str, Depends(require_admin)],
    session_id:     Optional[str]  = Query(None, description="Export a specific session only."),
    format:         str            = Query("jsonl", pattern="^(jsonl|csv)$"),
    verified_only:  bool           = Query(False),
    pseudonymise:   bool           = Query(True),
) -> ShadowExportResponse:
    """
    Export shadow session data as a training dataset.
    """
    if not _shadow_loggers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No shadow sessions found. "
                "Ingest data first via POST /shadow/ingest or run with "
                "SHADOW_MODE_ENABLED=true."
            ),
        )

    # Select loggers to export
    if session_id:
        if session_id not in _shadow_loggers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shadow session '{session_id}' not found.",
            )
        loggers_to_export = {session_id: _shadow_loggers[session_id]}
    else:
        loggers_to_export = dict(_shadow_loggers)

    # Aggregate all entries
    all_entries: list = []
    for sid, sl in loggers_to_export.items():
        entries = sl.log_entries
        if verified_only:
            entries = [e for e in entries if e.verified_label is not None]
        all_entries.extend(entries)

    if not all_entries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No entries found matching the export criteria.",
        )

    n_human = sum(1 for e in all_entries if e.effective_label() == 0)
    n_bot   = sum(1 for e in all_entries if e.effective_label() == 1)

    # Write export file using the first logger's export method
    first_logger = next(iter(loggers_to_export.values()))
    try:
        export_path = first_logger.export_session(
            format       = format,
            pseudonymise = pseudonymise,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {e}",
        )

    logger.info(
        "Shadow export: %d entries (%d human, %d bot) → %s",
        len(all_entries), n_human, n_bot, export_path
    )

    return ShadowExportResponse(
        export_path    = export_path,
        n_entries      = len(all_entries),
        n_human        = n_human,
        n_bot          = n_bot,
        verified_only  = verified_only,
        format         = format,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /shadow/stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Shadow session statistics",
    description="Returns summary statistics across all active shadow sessions.",
)
async def shadow_stats(
    request:    Request,
    admin_user: Annotated[str, Depends(require_admin)],
) -> dict:
    """
    Return aggregate statistics across all shadow sessions.
    """
    if not _shadow_loggers:
        return {
            "active_sessions":  0,
            "total_entries":    0,
            "total_human":      0,
            "total_bot":        0,
            "total_verified":   0,
        }

    total_entries  = 0
    total_human    = 0
    total_bot      = 0
    total_verified = 0
    session_summaries = []

    for session_id, shadow_logger in _shadow_loggers.items():
        summary = shadow_logger.get_shadow_summary()
        session_summaries.append({
            "session_id": session_id,
            **summary,
        })
        total_entries  += summary.get("total_evaluated", 0)
        total_human    += summary.get("heuristic_human", 0)
        total_bot      += summary.get("heuristic_bot", 0)
        total_verified += summary.get("verified_labels", 0)

    return {
        "active_sessions": len(_shadow_loggers),
        "total_entries":   total_entries,
        "total_human":     total_human,
        "total_bot":       total_bot,
        "total_verified":  total_verified,
        "sessions":        session_summaries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /shadow/verify
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/verify",
    status_code=status.HTTP_200_OK,
    summary="Set analyst-verified label on a shadow entry",
    description=(
        "Allows a security analyst to override the heuristic label on a "
        "shadow log entry with a verified ground-truth label. "
        "Verified labels take priority in ML training. Admin role required."
    ),
)
async def verify_shadow_entry(
    request:    Request,
    admin_user: Annotated[str, Depends(require_admin)],
    session_id: str   = Query(..., description="Session ID of the entry to verify."),
    timestamp:  float = Query(..., description="Timestamp of the action entry."),
    label:      int   = Query(..., ge=0, le=1, description="Verified label: 0=human, 1=bot."),
) -> dict:
    """
    Set an analyst-verified label on a shadow log entry.
    """
    if session_id not in _shadow_loggers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shadow session '{session_id}' not found.",
        )

    shadow_logger = _shadow_loggers[session_id]
    success = shadow_logger.set_verified_label(timestamp=timestamp, label=label)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No log entry found near timestamp {timestamp} in session {session_id}.",
        )

    logger.info(
        "Shadow label verified: session=%s timestamp=%.3f label=%d by=%s",
        session_id, timestamp, label, admin_user
    )

    # Also save to feature store if available
    feature_store = get_feature_store(request)
    if feature_store:
        try:
            feature_store.save(
                user_id    = shadow_logger.engine.profile.user_id,
                vector     = [0.0] * 20,   # placeholder — actual vector in shadow log
                label      = label,
                session_id = session_id,
                verified   = True,
                source     = "analyst",
            )
        except Exception as e:
            logger.warning("Feature store save failed for verified label: %s", e)

    return {
        "verified":    True,
        "session_id":  session_id,
        "timestamp":   timestamp,
        "label":       label,
        "verified_by": admin_user,
    }
