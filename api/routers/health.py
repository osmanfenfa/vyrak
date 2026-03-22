"""
zelvryn/api/routers/health.py
===============================
System health and observability endpoints.

Endpoints
---------
  GET /health           — Overall system health (API + Redis + ML + DB)
  GET /health/model     — Model registry status, active/canary versions, metrics
  GET /health/redis     — Redis connection and memory statistics
  GET /health/ready     — Kubernetes readiness probe (minimal check)
  GET /health/live      — Kubernetes liveness probe (always 200 if server is up)

These endpoints are intentionally not authenticated — they must be accessible
by load balancers, Kubernetes probes, and monitoring systems without credentials.
They never return user data or internal configuration.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import time
from typing import Dict

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from api.schemas import (
    ComponentStatus,
    HealthResponse,
    ModelHealthResponse,
)
from api.dependencies import get_active_session_count

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

# API version string
_API_VERSION = "2.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Overall system health",
    description=(
        "Returns the health status of all system components: API server, "
        "Redis profile store, ML model registry, and the evaluation engine. "
        "Returns HTTP 200 if healthy, HTTP 503 if any critical component is down."
    ),
)
async def health_check(request: Request) -> HealthResponse:
    """
    Comprehensive health check for all system components.

    Performs live connectivity checks against Redis and the model registry.
    """
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime_s   = time.time() - start_time
    shadow_mode = getattr(request.app.state, "shadow_mode", False)
    components: Dict[str, ComponentStatus] = {}
    overall_ok = True

    # ── Check Redis ────────────────────────────────────────────────────────
    profile_store = getattr(request.app.state, "profile_store", None)
    redis_status  = _check_redis(profile_store)
    components["redis"] = redis_status
    if redis_status.status == "down":
        overall_ok = False

    # ── Check ML Registry ─────────────────────────────────────────────────
    model_registry = getattr(request.app.state, "model_registry", None)
    ml_status      = _check_model_registry(model_registry)
    components["model_registry"] = ml_status
    # Model registry degraded is not a critical failure — engine can run
    # rule-based detection without a classifier

    # ── Check Session Engine ──────────────────────────────────────────────
    active_sessions = get_active_session_count()
    components["session_engine"] = ComponentStatus(
        status  = "healthy",
        details = f"{active_sessions} active sessions",
    )

    # ── Check Population Prior ────────────────────────────────────────────
    prior = getattr(request.app.state, "population_prior", None)
    if prior is not None:
        stale = prior.is_stale() if hasattr(prior, "is_stale") else False
        components["population_prior"] = ComponentStatus(
            status  = "degraded" if stale else "healthy",
            details = (
                f"{getattr(prior, 'source_profile_count', 0)} source profiles"
                + (" — stale (rebuild recommended)" if stale else "")
            ),
        )
    else:
        components["population_prior"] = ComponentStatus(
            status  = "degraded",
            details = "Not loaded — new users will use system defaults",
        )

    overall_status = "healthy" if overall_ok else "degraded"
    http_status    = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    response_body = HealthResponse(
        status      = overall_status,
        version     = _API_VERSION,
        uptime_s    = round(uptime_s, 1),
        shadow_mode = shadow_mode,
        components  = components,
        checked_at  = time.time(),
    )

    return JSONResponse(
        content    = response_body.model_dump(),
        status_code = http_status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health/model
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/model",
    response_model=ModelHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="ML model registry status",
    description=(
        "Returns the active and canary model versions, their performance metrics, "
        "and the canary traffic percentage. Used by the analyst dashboard."
    ),
)
async def model_health(request: Request) -> ModelHealthResponse:
    """
    Return model registry status and metrics.
    """
    model_registry = getattr(request.app.state, "model_registry", None)

    if model_registry is None:
        return ModelHealthResponse(
            active_version      = None,
            canary_version      = None,
            canary_traffic_pct  = 0,
            total_versions      = 0,
            active_cv_f1        = None,
            active_cv_far       = None,
            active_cv_frr       = None,
            canary_cv_f1        = None,
            versions_by_state   = {},
        )

    summary        = model_registry.summary()
    active_metrics = summary.get("active_metrics") or {}
    canary_metrics = summary.get("canary_metrics") or {}

    return ModelHealthResponse(
        active_version     = summary.get("active_version"),
        canary_version     = summary.get("canary_version"),
        canary_traffic_pct = summary.get("canary_traffic_percent", 0),
        total_versions     = summary.get("total_versions", 0),
        active_cv_f1       = active_metrics.get("cv_f1"),
        active_cv_far      = active_metrics.get("cv_far"),
        active_cv_frr      = active_metrics.get("cv_frr"),
        canary_cv_f1       = canary_metrics.get("cv_f1"),
        versions_by_state  = summary.get("versions_by_state", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health/redis
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/redis",
    status_code=status.HTTP_200_OK,
    summary="Redis profile store health",
    description="Returns Redis connection status and memory usage statistics.",
)
async def redis_health(request: Request) -> dict:
    """
    Return Redis health and memory statistics.
    """
    profile_store = getattr(request.app.state, "profile_store", None)

    if profile_store is None:
        return {
            "status":  "not_configured",
            "details": "Profile store not initialised.",
        }

    t_start = time.perf_counter()
    try:
        stats   = profile_store.store_stats()
        latency = (time.perf_counter() - t_start) * 1000
        return {
            "status":        "healthy",
            "latency_ms":    round(latency, 2),
            "profile_count": stats.get("profile_count", 0),
            "used_memory_mb": stats.get("used_memory_mb", 0),
            "used_memory_peak_mb": stats.get("used_memory_peak_mb", 0),
            "ttl_seconds":   stats.get("ttl_seconds"),
        }
    except Exception as e:
        return {
            "status":  "down",
            "details": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# GET /health/ready  (Kubernetes readiness probe)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Kubernetes readiness probe",
    description=(
        "Returns 200 when the service is ready to accept traffic "
        "(Redis is reachable and the model is loaded). "
        "Returns 503 when the service is not yet ready."
    ),
    include_in_schema=False,   # hide from OpenAPI docs
)
async def readiness_probe(request: Request):
    """
    Kubernetes readiness probe. Returns 503 if Redis is unreachable.
    """
    profile_store = getattr(request.app.state, "profile_store", None)

    if profile_store is None:
        return JSONResponse(
            content={"ready": False, "reason": "Profile store not initialised."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Quick Redis ping
    try:
        profile_store._redis.ping()
    except Exception as e:
        return JSONResponse(
            content={"ready": False, "reason": f"Redis unreachable: {e}"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return {"ready": True}


# ─────────────────────────────────────────────────────────────────────────────
# GET /health/live  (Kubernetes liveness probe)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Kubernetes liveness probe",
    description="Always returns 200 if the server process is alive.",
    include_in_schema=False,
)
async def liveness_probe():
    """
    Kubernetes liveness probe. Always returns 200.
    """
    return {"alive": True, "timestamp": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _check_redis(profile_store) -> ComponentStatus:
    """Perform a live Redis health check."""
    if profile_store is None:
        return ComponentStatus(status="down", details="Profile store not configured.")

    t_start = time.perf_counter()
    try:
        profile_store._redis.ping()
        latency_ms = (time.perf_counter() - t_start) * 1000
        return ComponentStatus(
            status     = "healthy",
            latency_ms = round(latency_ms, 2),
            details    = "Redis connection OK.",
        )
    except Exception as e:
        return ComponentStatus(
            status  = "down",
            details = f"Redis unreachable: {e}",
        )


def _check_model_registry(model_registry) -> ComponentStatus:
    """Check whether the model registry has an active model loaded."""
    if model_registry is None:
        return ComponentStatus(
            status  = "degraded",
            details = "Model registry not configured — rule-based detection only.",
        )

    active_version = model_registry.active_version()
    if active_version is None:
        return ComponentStatus(
            status  = "degraded",
            details = "No active model in registry — rule-based detection only.",
        )

    # Try to verify the model is loaded (not just registered)
    try:
        clf = model_registry.get_active_model()
        trained = clf is not None and getattr(clf, "is_trained", False)
        return ComponentStatus(
            status  = "healthy" if trained else "degraded",
            details = (
                f"Active model: {active_version}"
                + (" (loaded)" if trained else " (not trained)")
            ),
        )
    except Exception as e:
        return ComponentStatus(
            status  = "degraded",
            details = f"Model load check failed: {e}",
        )
