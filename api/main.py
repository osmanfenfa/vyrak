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
from api.admin_runtime import apply_runtime_patch, get_runtime_config
from api.accounts import (
    AccountStore,
    OneTimeLinkStore,
    TenantStore,
    bootstrap_platform_admin,
)
from api.auth import ApiKeyStore, is_production, validate_security_config
from api.stores import (
    AlertStore,
    PeerStore,
    RateLimiter,
    SessionStateStore,
    TenantSettingsStore,
)
from core.constants import SHADOW_MODE_ENABLED, LOG_LEVEL
logger = logging.getLogger(__name__)

def _load_local_env_file(path: str = ".env") -> None:
    """
    Lightweight .env loader for local development.
    Existing process environment variables always win.
    """
    if not os.path.exists(path):
        return

    loaded = 0
    try:
        with open(path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key.startswith("export "):
                    key = key[len("export "):].strip()
                if not key or key in os.environ:
                    continue

                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in ("'", '"')
                ):
                    value = value[1:-1]

                os.environ[key] = value
                loaded += 1
    except Exception as e:
        logger.warning("Failed to read %s (%s).", path, e)
        return

    if loaded:
        logger.info("Loaded %d variable(s) from %s.", loaded, path)


# SHARED STATE
def configure_state(app: FastAPI, redis_client=None) -> None:
    """
    Attach every Redis-backed store to app.state.

    With no client (development / tests) an isolated in-memory backend is
    used. Production must pass a real Redis client — see lifespan().
    """
    from profiles.memory_redis import MemoryRedis
    from profiles.redis_store import RedisProfileStore

    client = redis_client if redis_client is not None else MemoryRedis()
    app.state.redis = client
    app.state.profile_store = RedisProfileStore(redis_client=client)
    app.state.session_store = SessionStateStore(client)
    app.state.alert_store = AlertStore(client)
    app.state.peer_store = PeerStore(client)
    app.state.tenant_settings = TenantSettingsStore(client)
    app.state.api_key_store = ApiKeyStore(client)
    app.state.rate_limiter = RateLimiter(
        client, int(os.environ.get("RATE_LIMIT_PER_MINUTE", "1200"))
    )
    app.state.account_store = AccountStore(client)
    app.state.tenant_store = TenantStore(client)
    app.state.link_store = OneTimeLinkStore(client)


# LOGGING CONFIGURATION
def _configure_logging() -> None:
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


# LIFESPAN (STARTUP + SHUTDOWN)
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _load_local_env_file(".env")
    _configure_logging()
    logger.info("=" * 60)
    logger.info("Zelvryn ABSA API v2.0 — starting up")
    logger.info("=" * 60)

    app.state.start_time = time.time()

    app.state.shadow_mode = (
        os.environ.get("SHADOW_MODE_ENABLED", "false").lower() == "true"
        or SHADOW_MODE_ENABLED
    )
    if app.state.shadow_mode:
        logger.warning(
            "SHADOW MODE ENABLED — evaluations are logged but NOT enforced."
        )

    # Runtime-adjustable controls (no restart required for admin updates)
    runtime_cfg = get_runtime_config(app)
    runtime_cfg["shadow_mode"] = bool(app.state.shadow_mode)
    apply_runtime_patch(app, runtime_cfg)

    # Redis — shared state for profiles, sessions, alerts, keys, limits
    validate_security_config()
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    client = None
    try:
        import redis as redis_lib
        pool = redis_lib.ConnectionPool.from_url(
            redis_url,
            max_connections=int(os.environ.get("REDIS_MAX_CONNECTIONS", 50)),
            decode_responses=True,
            socket_connect_timeout=5,
        )
        client = redis_lib.Redis(connection_pool=pool)
        client.ping()
        logger.info("Redis connected: %s", redis_url.split("@")[-1])
    except Exception as e:
        if is_production():
            raise RuntimeError(
                f"Redis is required in production and is unreachable: {e}"
            ) from e
        logger.warning(
            "Redis unavailable (%s). Using in-memory state: development only; "
            "data is lost on restart and not shared between workers.", e
        )
        client = None
    configure_state(app, client)
    bootstrap_platform_admin(app.state.account_store, app.state.tenant_store, logger)

    # Model Registry
    try:
        from ml.registry import ModelRegistry

        model_dir = os.environ.get("MODEL_PATH", "models")
        # Strip filename if a specific .joblib path was given
        if model_dir.endswith(".joblib"):
            model_dir = os.path.dirname(model_dir) or "models"

        registry = ModelRegistry(registry_dir=model_dir)
        registry.canary_traffic_percent = int(
            get_runtime_config(app).get(
                "canary_traffic_percent",
                registry.canary_traffic_percent,
            )
        )
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

    except ImportError as e:
        logger.warning(
            "ml.registry not available (%s). Running rule-based detection only.",
            e,
        )
        app.state.model_registry = None

    try:
        from ml.feature_store import FeatureStore

        db_url = os.environ.get("DATABASE_URL")
        app.state.feature_store = FeatureStore(db_url=db_url)
        logger.info(
            "FeatureStore ready: backend=%s",
            "postgresql" if db_url else "memory"
        )
    except ImportError as e:
        logger.warning("ml.feature_store not available (%s).", e)
        app.state.feature_store = None

    try:
        from coldstart.population_prior import PopulationPriorProfile

        prior_path = os.environ.get(
            "POPULATION_PRIOR_PATH",
            "data/processed/population_prior.json"
        )
        prior = PopulationPriorProfile.load(prior_path)
        app.state.population_prior = prior

        if not getattr(prior, "built_at", 0):
            logger.warning(
                "Population prior has never been built from real profiles. "
                "Run scripts/build_population_prior.py once you have traffic."
            )
        elif prior.is_stale():
            built_at = getattr(prior, "built_at", None)
            age_days = (
                (time.time() - float(built_at)) / 86400.0
                if built_at is not None
                else 0.0
            )
            logger.warning(
                "Population prior is stale (%.1f days old). "
                "Run scripts/build_population_prior.py to rebuild.",
                max(0.0, age_days),
            )

    except ImportError:
        logger.warning("coldstart.population_prior not available.")
        app.state.population_prior = None

    try:
        from coldstart.device_signals import DeviceSignalBridge

        app.state.device_bridge = DeviceSignalBridge(
            geo_velocity_enabled = True,
        )
        logger.info("DeviceSignalBridge ready.")
    except ImportError:
        logger.warning("coldstart.device_signals not available.")
        app.state.device_bridge = None

    startup_ms = (time.time() - app.state.start_time) * 1000
    logger.info(
        "Zelvryn API ready in %.0f ms — shadow=%s",
        startup_ms, app.state.shadow_mode
    )

    yield 
    logger.info("Zelvryn API shutting down...")

    uptime = time.time() - app.state.start_time
    logger.info("Uptime: %.1f seconds", uptime)
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
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

    origins = [
        "https://zelvryn.netlify.app",
        "http://localhost:5173",
    ]
    cors_origins_raw = os.environ.get("CORS_ORIGINS", "")
    extra_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    allow_origins = list(dict.fromkeys(origins + extra_origins))

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = allow_origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
        expose_headers    = [
            "X-Request-ID", "X-Process-Time-Ms", "X-Zelvryn-Version"
        ],
    )

    register_middleware(app)
    include_routers(app)
    # Usable immediately (tests, scripts); lifespan swaps in real Redis
    configure_state(app)
    app.state.shadow_mode = SHADOW_MODE_ENABLED
    app.state.start_time = time.time()
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service":     "Zelvryn ABSA API",
            "version":     "2.0.0",
            "docs":        "/docs",
            "health":      "/health",
            "description": "Adaptive Behavioral Signature Authentication",
        }
        
    _register_exception_handlers(app)

    return app


# EXCEPTION HANDLERS
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
app = create_app()
