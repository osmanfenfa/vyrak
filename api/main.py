"""
zelvryn/api/main.py
====================
FastAPI application factory for the Zelvryn ABSA API.

The entry point for the API server. Uvicorn starts here:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

Or in development with auto-reload:
  uvicorn api.main:app --reload --log-level debug

Application structure
----------------------
  create_app()   — factory function that builds and returns the FastAPI instance
  lifespan()     — async context manager that runs startup and shutdown logic
  app            — the default app instance (used by uvicorn directly)

Startup sequence (inside lifespan)
------------------------------------
  1. Configure logging
  2. Connect to Redis and initialise RedisProfileStore
  3. Load ModelRegistry (from models/model_registry.json)
  4. Load active model into memory
  5. Initialise FeatureStore
  6. Load PopulationPriorProfile (from data/processed/population_prior.json)
  7. Initialise DeviceSignalBridge
  8. Set app.state.shadow_mode from env
  9. Store start_time for uptime calculation

All infrastructure objects are stored on app.state so they are accessible
to dependency injection functions in dependencies.py without module-level
globals or circular imports.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware import register_middleware
from api.routers import include_routers
from api.schemas import ErrorResponse, ValidationErrorResponse
from core.constants import SHADOW_MODE_ENABLED, LOG_LEVEL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    Configure structured logging for the API server.

    Uses a two-handler setup:
      - console: human-readable format for development
      - audit: JSON format for the audit logger (writes to logs/audit.jsonl)

    In production, replace the console handler with a JSON formatter
    for ingestion by log aggregators (Datadog, ELK, CloudWatch).
    """
    log_config = {
        "version":    1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "format": '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class":     "logging.StreamHandler",
                "formatter": "console",
                "stream":    "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level":    LOG_LEVEL,
        },
        "loggers": {
            "uvicorn":         {"level": "INFO",    "propagate": False, "handlers": ["console"]},
            "uvicorn.access":  {"level": "WARNING", "propagate": False, "handlers": ["console"]},
            "zelvryn.audit":   {"level": "INFO",    "propagate": False, "handlers": ["console"]},
        },
    }
    logging.config.dictConfig(log_config)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN (STARTUP + SHUTDOWN)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown resources.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    # ── Startup ────────────────────────────────────────────────────────────
    _configure_logging()
    logger.info("=" * 60)
    logger.info("Zelvryn ABSA API v2.0 — starting up")
    logger.info("=" * 60)

    app.state.start_time = time.time()

    # Shadow mode flag
    app.state.shadow_mode = (
        os.environ.get("SHADOW_MODE_ENABLED", "false").lower() == "true"
        or SHADOW_MODE_ENABLED
    )
    if app.state.shadow_mode:
        logger.warning(
            "SHADOW MODE ENABLED — evaluations are logged but NOT enforced."
        )

    # ── Redis / Profile Store ──────────────────────────────────────────────
    try:
        from profiles.redis_store import RedisProfileStore

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

        try:
            import redis as redis_lib
            pool   = redis_lib.ConnectionPool.from_url(
                redis_url,
                max_connections = int(os.environ.get("REDIS_MAX_CONNECTIONS", 50)),
                decode_responses = True,
                socket_connect_timeout = 5,
            )
            client = redis_lib.Redis(connection_pool=pool)
            client.ping()
            logger.info("Redis connected: %s", redis_url.split("@")[-1])
        except Exception as e:
            logger.warning(
                "Redis connection failed (%s). Using in-memory stub. "
                "Profiles will not persist across restarts.", e
            )
            client = None   # RedisProfileStore will use _NoOpRedisStub

        app.state.profile_store = RedisProfileStore(redis_client=client)

    except ImportError:
        logger.warning("profiles.redis_store not available. Profiles will not persist.")
        app.state.profile_store = None

    # ── Model Registry ────────────────────────────────────────────────────
    try:
        from ml.registry import ModelRegistry

        model_dir = os.environ.get("MODEL_PATH", "models")
        # Strip filename if a specific .joblib path was given
        if model_dir.endswith(".joblib"):
            model_dir = os.path.dirname(model_dir) or "models"

        registry = ModelRegistry(registry_dir=model_dir)
        app.state.model_registry = registry

        active = registry.active_version()
        if active:
            clf = registry.get_active_model()
            if clf and getattr(clf, "is_trained", False):
                logger.info("Active model loaded: %s", active)
            else:
                logger.warning("Active model %s could not be loaded.", active)
        else:
            logger.warning(
                "No active model in registry. "
                "Run scripts/train_initial_model.py to train the first model."
            )

    except ImportError:
        logger.warning("ml.registry not available. Running rule-based detection only.")
        app.state.model_registry = None

    # ── Feature Store ──────────────────────────────────────────────────────
    try:
        from ml.feature_store import FeatureStore

        db_url = os.environ.get("DATABASE_URL")
        app.state.feature_store = FeatureStore(db_url=db_url)
        logger.info(
            "FeatureStore ready: backend=%s",
            "postgresql" if db_url else "memory"
        )
    except ImportError:
        logger.warning("ml.feature_store not available.")
        app.state.feature_store = None

    # ── Population Prior ───────────────────────────────────────────────────
    try:
        from coldstart.population_prior import PopulationPriorProfile

        prior_path = os.environ.get(
            "POPULATION_PRIOR_PATH",
            "data/processed/population_prior.json"
        )
        prior = PopulationPriorProfile.load(prior_path)
        app.state.population_prior = prior

        if prior.is_stale():
            logger.warning(
                "Population prior is stale (%.1f days old). "
                "Run scripts/build_population_prior.py to rebuild.",
                time.time() / 86400 - prior.built_at / 86400 if prior.built_at else 0
            )

    except ImportError:
        logger.warning("coldstart.population_prior not available.")
        app.state.population_prior = None

    # ── Device Signal Bridge ───────────────────────────────────────────────
    try:
        from coldstart.device_signals import DeviceSignalBridge

        app.state.device_bridge = DeviceSignalBridge(
            geo_velocity_enabled = True,
        )
        logger.info("DeviceSignalBridge ready.")
    except ImportError:
        logger.warning("coldstart.device_signals not available.")
        app.state.device_bridge = None

    # ── Ready ──────────────────────────────────────────────────────────────
    startup_ms = (time.time() - app.state.start_time) * 1000
    logger.info(
        "Zelvryn API ready in %.0f ms — shadow=%s",
        startup_ms, app.state.shadow_mode
    )

    yield   # ← Application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Zelvryn API shutting down...")

    uptime = time.time() - app.state.start_time
    logger.info("Uptime: %.1f seconds", uptime)
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    This is the factory function — call it to create a fresh app instance.
    The module-level `app` variable calls this at import time for uvicorn.

    Returns
    -------
    FastAPI
    """
    app = FastAPI(
        title       = "Zelvryn — Adaptive Behavioral Signature Authentication",
        description = (
            "Continuous session authentication API. Evaluates every user action "
            "against a learned behavioural profile to detect bot activity, "
            "session hijacking, and AI mimics in real time.\n\n"
            "**Developed by Osman Fenfaquee's Vyrak**"
        ),
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        openapi_url = "/openapi.json",
        lifespan    = lifespan,
        contact = {
            "name":  "Vyrak",
            "email": "osmantfenfaquee@gmail.com",
        },
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins_raw = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173"
    )
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = cors_origins,
        allow_credentials = True,
        allow_methods     = ["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers     = [
            "Authorization", "Content-Type",
            "X-Request-ID", "X-Forwarded-For",
        ],
        expose_headers    = [
            "X-Request-ID", "X-Process-Time-Ms", "X-Zelvryn-Version"
        ],
    )

    # ── Custom middleware stack ────────────────────────────────────────────
    register_middleware(app)

    # ── Routers ────────────────────────────────────────────────────────────
    include_routers(app)

    # ── Root redirect ──────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service":     "Zelvryn ABSA API",
            "version":     "2.0.0",
            "docs":        "/docs",
            "health":      "/health",
            "description": "Adaptive Behavioral Signature Authentication",
        }

    # ── Exception handlers ─────────────────────────────────────────────────
    _register_exception_handlers(app)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent error response format."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return a structured 422 response for Pydantic validation errors."""
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            content     = ValidationErrorResponse(
                errors     = exc.errors(),
            ).model_dump() | {"request_id": request_id},
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code = status.HTTP_404_NOT_FOUND,
            content     = ErrorResponse(
                error      = "not_found",
                message    = f"The requested resource was not found: {request.url.path}",
                request_id = request_id,
            ).model_dump(),
        )

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code = status.HTTP_405_METHOD_NOT_ALLOWED,
            content     = ErrorResponse(
                error      = "method_not_allowed",
                message    = f"Method {request.method} is not allowed on {request.url.path}.",
                request_id = request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "Unhandled exception: %s — request_id=%s path=%s",
            exc, request_id, request.url.path,
            exc_info=True,
        )
        return JSONResponse(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            content     = ErrorResponse(
                error      = "internal_server_error",
                message    = "An unexpected error occurred. Please try again.",
                request_id = request_id,
            ).model_dump(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT APP INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

# Module-level app instance for uvicorn:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000
app = create_app()
