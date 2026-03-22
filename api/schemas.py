"""
zelvryn/api/schemas.py
=======================
Pydantic v2 request and response models for every Zelvryn API endpoint.

Design principles
-----------------
  1. Every field has a type annotation, a description, and a sensible default
     or explicit required marker. This drives auto-generated OpenAPI docs.
  2. Request models validate on assignment so bad payloads are rejected at
     the boundary before touching any business logic.
  3. Response models are strict about which fields are exposed — internal
     profile state (EWMA vectors, raw interval lists) is never serialised
     to API responses.
  4. Nested models are used for grouping (e.g. MetricsBlock inside
     SessionSummaryResponse) to keep the top-level models readable.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class DecisionEnum(str, Enum):
    ALLOW     = "ALLOW"
    CHALLENGE = "CHALLENGE"
    BLOCK     = "BLOCK"


class RiskLabelEnum(str, Enum):
    LOW      = "LOW RISK"
    MEDIUM   = "MEDIUM RISK"
    HIGH     = "HIGH RISK"
    CRITICAL = "CRITICAL RISK"


class ActionTypeEnum(str, Enum):
    LOGIN  = "login"
    READ   = "read"
    WRITE  = "write"
    DELETE = "delete"
    SEARCH = "search"
    EXPORT = "export"
    CONFIG = "config"
    ADMIN  = "admin"


class ModelStateEnum(str, Enum):
    TRAINED  = "trained"
    CANARY   = "canary"
    ACTIVE   = "active"
    RETIRED  = "retired"


# ─────────────────────────────────────────────────────────────────────────────
# COMMON BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class MetadataBlock(BaseModel):
    """Optional supplementary context attached to action events."""
    ip:           Optional[str]  = Field(None, description="Client IP address")
    device:       Optional[str]  = Field(None, description="Device fingerprint hash")
    user_agent:   Optional[str]  = Field(None, description="Browser User-Agent string")
    geo_country:  Optional[str]  = Field(None, description="ISO 3166-1 alpha-2 country code")
    geo_city:     Optional[str]  = Field(None, description="City name")
    latitude:     Optional[float] = Field(None, description="Latitude for geo-velocity check")
    longitude:    Optional[float] = Field(None, description="Longitude for geo-velocity check")

    model_config = {"extra": "allow"}   # allow arbitrary metadata keys


class FeatureImportanceItem(BaseModel):
    name:       str   = Field(..., description="Feature name")
    importance: float = Field(..., description="Normalised permutation importance score")


class AnomalyEntry(BaseModel):
    """Single anomaly signal entry in a session log."""
    action_type:  str        = Field(..., description="Action type that triggered the anomaly")
    timestamp:    float      = Field(..., description="Unix timestamp of the action")
    trust_score:  float      = Field(..., description="Trust score at time of anomaly")
    anomalies:    List[str]  = Field(..., description="Triggered anomaly signal codes")
    bot_prob:     float      = Field(..., description="P(bot) at time of anomaly")
    decision:     str        = Field(..., description="Decision issued")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    """
    Request body for POST /session/start.

    Initiates a new authenticated session in the Zelvryn evaluation engine.
    The user_id must correspond to an authenticated user in the host application.
    """
    user_id:    str  = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier of the authenticated user.",
        examples=["user_abc123"],
    )
    metadata:   Optional[MetadataBlock] = Field(
        None,
        description="Optional device and location context for cold start bridging.",
    )

    @field_validator("user_id")
    @classmethod
    def user_id_no_whitespace(cls, v: str) -> str:
        if v != v.strip():
            raise ValueError("user_id must not have leading or trailing whitespace.")
        return v


class SessionStartResponse(BaseModel):
    """Response from POST /session/start."""
    session_id:      str   = Field(..., description="Unique session identifier to use in subsequent action calls.")
    user_id:         str   = Field(..., description="Echo of the requesting user_id.")
    trust_score:     float = Field(..., description="Initial trust score (always 100.0 for new sessions).")
    ramp_active:     bool  = Field(..., description="True if the user is in the cold start ramp-up period.")
    shadow_mode:     bool  = Field(..., description="True if the server is operating in shadow (observation-only) mode.")
    started_at:      float = Field(..., description="Unix timestamp of session creation.")


class ActionRequest(BaseModel):
    """
    Request body for POST /session/{session_id}/action.

    The core evaluation endpoint. Every protected operation performed by
    the user should trigger one call to this endpoint.
    """
    action_type:  ActionTypeEnum = Field(
        ...,
        description="Type of action performed. Must be one of the 8 supported action types.",
        examples=["read"],
    )
    resource_id:  str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Identifier of the resource accessed (document ID, endpoint path, etc.). Hashed server-side.",
        examples=["document_42"],
    )
    timestamp:    Optional[float] = Field(
        None,
        description="Unix timestamp of the action. Defaults to server receipt time if omitted.",
    )
    metadata:     Optional[MetadataBlock] = Field(
        None,
        description="Optional request context (IP, device, etc.).",
    )

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_future(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v > time.time() + 60:
            raise ValueError("timestamp cannot be more than 60 seconds in the future.")
        return v


class DecisionResponse(BaseModel):
    """
    Response from POST /session/{session_id}/action.

    The primary response object. The application uses `decision` to decide
    whether to serve the request (ALLOW), prompt re-authentication (CHALLENGE),
    or terminate the session (BLOCK).
    """
    decision:     DecisionEnum  = Field(..., description="Enforcement decision: ALLOW, CHALLENGE, or BLOCK.")
    trust_score:  float         = Field(..., ge=0.0, le=100.0, description="Current session trust score (0-100).")
    bot_prob:     float         = Field(..., ge=0.0, le=1.0, description="ML classifier P(bot) estimate (0-1).")
    similarity:   float         = Field(..., ge=0.0, le=1.0, description="Cosine similarity between live window and profile.")
    entropy:      float         = Field(..., ge=0.0, le=1.0, description="Normalised Shannon entropy of the action window.")
    temporal_z:   float         = Field(..., ge=0.0, description="Absolute temporal Z-score of the current interval.")
    anomalies:    List[str]     = Field(default_factory=list, description="List of triggered anomaly signal codes.")
    ramp_active:  bool          = Field(..., description="True if cold start ramp-up rules are in effect.")
    session_id:   str           = Field(..., description="Echo of the evaluated session ID.")
    action_type:  str           = Field(..., description="Echo of the evaluated action type.")
    timestamp:    float         = Field(..., description="Unix timestamp of the evaluation.")


class SessionSummaryMetrics(BaseModel):
    """Nested metrics block within SessionSummaryResponse."""
    total_actions:   int   = Field(..., description="Total actions evaluated in this session.")
    allowed:         int   = Field(..., description="Number of ALLOW decisions.")
    challenged:      int   = Field(..., description="Number of CHALLENGE decisions.")
    blocked:         int   = Field(..., description="Number of BLOCK decisions.")
    anomaly_count:   int   = Field(..., description="Total anomaly events detected.")
    avg_trust_score: float = Field(..., description="Average trust score over the session.")
    avg_bot_prob:    float = Field(..., description="Average P(bot) over the session.")


class SessionSummaryResponse(BaseModel):
    """Response from GET /session/{session_id}/summary."""
    user_id:          str                    = Field(..., description="User who owns this session.")
    session_id:       str                    = Field(..., description="Session identifier.")
    final_trust:      float                  = Field(..., description="Trust score at session end.")
    risk_rating:      RiskLabelEnum          = Field(..., description="Human-readable risk label.")
    metrics:          SessionSummaryMetrics  = Field(..., description="Per-decision and signal statistics.")
    ml_trained:       bool                   = Field(..., description="Whether the ML classifier has been trained.")
    ml_train_samples: int                    = Field(..., description="Total labeled samples accumulated for training.")
    top_features:     List[FeatureImportanceItem] = Field(default_factory=list, description="Top-5 most important features.")
    session_duration_s: float                = Field(..., description="Wall-clock session duration in seconds.")
    shadow_mode:      bool                   = Field(..., description="Whether shadow mode was active.")
    anomaly_log:      List[AnomalyEntry]     = Field(default_factory=list, description="Full log of all anomalous action evaluations.")


class SessionDeleteResponse(BaseModel):
    """Response from DELETE /session/{session_id}."""
    session_id:    str   = Field(..., description="Deleted session ID.")
    archived:      bool  = Field(..., description="True if session was archived before deletion.")
    final_trust:   float = Field(..., description="Trust score at time of deletion.")
    total_actions: int   = Field(..., description="Total actions evaluated in the session.")


class RestoreTrustRequest(BaseModel):
    """Request body for POST /session/{session_id}/restore-trust."""
    increment:  float = Field(
        20.0,
        gt=0.0,
        le=50.0,
        description="Trust points to restore after successful re-authentication. Max 50.",
    )
    reason:     str   = Field(
        "re_authentication",
        description="Reason for trust restoration (e.g., 'email_otp', 'push_confirm').",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    """Response from GET /profile/{user_id}."""
    user_id:             str         = Field(..., description="Profile owner.")
    vector_count:        int         = Field(..., description="Total verified human actions recorded.")
    adaptive_threshold:  float       = Field(..., description="Per-user cosine similarity anomaly threshold.")
    is_mature:           bool        = Field(..., description="True if vector_count >= cold start ramp actions.")
    training_samples:    int         = Field(..., description="Labeled ML training samples accumulated.")
    mean_interval_s:     float       = Field(..., description="EWMA-smoothed mean inter-action interval (seconds).")
    std_interval_s:      float       = Field(..., description="EWMA-smoothed interval standard deviation.")
    action_proportions:  Dict[str, float] = Field(..., description="Current action type frequency distribution.")
    last_updated:        float       = Field(..., description="Unix timestamp of last profile update.")
    profile_age_days:    float       = Field(..., description="Age of the profile in days since creation.")


class ProfileDeleteResponse(BaseModel):
    """Response from DELETE /profile/{user_id}."""
    user_id:          str   = Field(..., description="Deleted user ID.")
    deleted:          bool  = Field(..., description="True if the profile was found and deleted.")
    erasure_complete: bool  = Field(..., description="True if all associated data was erased.")
    deleted_at:       float = Field(..., description="Unix timestamp of deletion.")


class ProfileResetResponse(BaseModel):
    """Response from POST /profile/{user_id}/reset."""
    user_id:          str   = Field(..., description="Reset user ID.")
    reset_to:         str   = Field(..., description="What the profile was reset to: 'population_prior' or 'empty'.")
    new_vector_count: int   = Field(..., description="Vector count after reset.")
    reset_at:         float = Field(..., description="Unix timestamp of reset.")


# ─────────────────────────────────────────────────────────────────────────────
# SHADOW MODE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class ShadowIngestRequest(BaseModel):
    """
    Request body for POST /shadow/ingest.

    Ingests a batch of historical action events through the shadow evaluation
    pipeline. All events are evaluated but no enforcement is applied.
    """
    events: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description=(
            "List of action event dicts. Each must have: user_id, session_id, "
            "action_type, resource_id. Optional: timestamp, metadata."
        ),
    )
    export_results: bool = Field(
        False,
        description="If True, export the processed results to data/shadow_exports/.",
    )

    @field_validator("events")
    @classmethod
    def events_have_required_fields(cls, v: List[Dict]) -> List[Dict]:
        required = {"user_id", "action_type", "resource_id"}
        for i, event in enumerate(v[:10]):   # validate first 10 only for performance
            missing = required - set(event.keys())
            if missing:
                raise ValueError(
                    f"Event at index {i} is missing required fields: {missing}"
                )
        return v


class ShadowIngestResponse(BaseModel):
    """Response from POST /shadow/ingest."""
    processed:     int   = Field(..., description="Number of events successfully processed.")
    errors:        int   = Field(..., description="Number of events that failed processing.")
    blocks:        int   = Field(..., description="Number of BLOCK decisions (shadow — not enforced).")
    challenges:    int   = Field(..., description="Number of CHALLENGE decisions (shadow — not enforced).")
    export_path:   Optional[str] = Field(None, description="Path of the export file if export_results=True.")
    duration_ms:   float = Field(..., description="Total processing time in milliseconds.")


class ShadowExportRequest(BaseModel):
    """Query parameters for GET /shadow/export."""
    session_id:     Optional[str]  = Field(None, description="Specific session to export. None = all.")
    format:         str            = Field("jsonl", pattern="^(jsonl|csv)$", description="Export format: jsonl or csv.")
    verified_only:  bool           = Field(False,  description="Export only analyst-verified labels.")
    pseudonymise:   bool           = Field(True,   description="Replace user_id/session_id with hashes.")


class ShadowExportResponse(BaseModel):
    """Response from GET /shadow/export."""
    export_path:   str   = Field(..., description="Path of the written export file.")
    n_entries:     int   = Field(..., description="Number of records exported.")
    n_human:       int   = Field(..., description="Human-labeled records.")
    n_bot:         int   = Field(..., description="Bot-labeled records.")
    verified_only: bool  = Field(..., description="Whether only verified records were included.")
    format:        str   = Field(..., description="Export format used.")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    """Health status for a single system component."""
    status:        str            = Field(..., description="'healthy', 'degraded', or 'down'.")
    latency_ms:    Optional[float] = Field(None, description="Round-trip latency in milliseconds.")
    details:       Optional[str]  = Field(None, description="Human-readable status details.")


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status:        str                     = Field(..., description="Overall system status.")
    version:       str                     = Field(..., description="Zelvryn API version string.")
    uptime_s:      float                   = Field(..., description="API server uptime in seconds.")
    shadow_mode:   bool                    = Field(..., description="Whether shadow mode is enabled.")
    components:    Dict[str, ComponentStatus] = Field(..., description="Per-component health status.")
    checked_at:    float                   = Field(..., description="Unix timestamp of health check.")


class ModelHealthResponse(BaseModel):
    """Response from GET /health/model."""
    active_version:         Optional[str]   = Field(None, description="Active model version string.")
    canary_version:         Optional[str]   = Field(None, description="Canary model version string (if any).")
    canary_traffic_pct:     int             = Field(..., description="Percentage of traffic routed to canary.")
    total_versions:         int             = Field(..., description="Total registered model versions.")
    active_cv_f1:           Optional[float] = Field(None, description="Active model cross-validation F1 score.")
    active_cv_far:          Optional[float] = Field(None, description="Active model cross-validation FAR.")
    active_cv_frr:          Optional[float] = Field(None, description="Active model cross-validation FRR.")
    canary_cv_f1:           Optional[float] = Field(None, description="Canary model CV F1 (if canary active).")
    versions_by_state:      Dict[str, List[str]] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response body for all 4xx and 5xx responses."""
    error:      str            = Field(..., description="Error type identifier.")
    message:    str            = Field(..., description="Human-readable error description.")
    detail:     Optional[Any]  = Field(None, description="Additional structured error detail.")
    request_id: Optional[str]  = Field(None, description="Request ID for log correlation.")
    timestamp:  float          = Field(default_factory=time.time)


class ValidationErrorResponse(BaseModel):
    """Response body for 422 Unprocessable Entity (request validation failures)."""
    error:   str        = "validation_error"
    message: str        = "Request validation failed."
    errors:  List[Dict] = Field(default_factory=list, description="List of field-level validation errors.")
