import logging
import os
import time
import uuid
from typing import Annotated, Dict, Optional, Tuple
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from core.action import UserAction
from core.engine import ABSAv2Engine
from core.profile import BehavioralProfile
from core.constants import SHADOW_MODE_ENABLED

logger = logging.getLogger(__name__)

# BEARER TOKEN AUTHENTICATION

_bearer_scheme = HTTPBearer(auto_error=False)


def _is_jwt_like(token: str) -> bool:
    return token.count(".") == 2


def _allow_plaintext_tokens(settings) -> bool:
    """
    Allow non-JWT bearer tokens in development unless explicitly disabled.

    Override with AUTH_ALLOW_PLAINTEXT_TOKENS=true|false.
    """
    raw = os.environ.get("AUTH_ALLOW_PLAINTEXT_TOKENS")
    if raw is not None:
        return raw.lower() == "true"

    env = str(getattr(settings, "zelvryn_env", "development")).lower()
    return env != "production"


def get_current_user(
    request: Request,
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(_bearer_scheme)
    ] = None,
) -> str:
    """
    Validate the Bearer JWT token and return the authenticated user_id.

    In production, this function should:
      1. Decode and verify the JWT using the configured secret/public key
      2. Check expiry and issuer claims
      3. Return the 'sub' claim as user_id
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = (credentials.credentials or "").strip()
    if len(token) < 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        import jwt
        from config.settings import settings

        if _allow_plaintext_tokens(settings) and not _is_jwt_like(token):
            logger.debug(
                "Using plaintext dev token as user_id (env=%s).",
                getattr(settings, "zelvryn_env", "unknown"),
            )
            return token

        secret_key = getattr(settings, "jwt_secret_key", "zelvryn-dev-secret")
        algorithm  = getattr(settings, "jwt_algorithm", "HS256")

        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            options={"require": ["sub", "exp"]},
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim.",
            )
        return str(user_id)

    except ImportError:
        # PyJWT not installed — development mode: treat token as user_id directly
        logger.warning(
            "PyJWT not installed. Using token value as user_id (development only). "
            "Install: pip install pyjwt"
        )
        return token

    except Exception as e:
        logger.warning("JWT validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(
    user_id: Annotated[str, Depends(get_current_user)],
    request: Request,
) -> str:
    """
    Extend get_current_user to require the 'admin' role.
    """
    # Check request state for role info (populated by JWT decode in production)
    roles = getattr(request.state, "roles", ["admin"])
    if "admin" not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation.",
        )
    return user_id


def get_profile_store(request: Request):
    """
    Provide the shared RedisProfileStore from application state.
    """
    store = getattr(request.app.state, "profile_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile store not initialised. Check Redis connection.",
        )
    return store


def get_model_registry(request: Request):
    """
    Provide the shared ModelRegistry from application state.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialised.",
        )
    return registry


def get_feature_store(request: Request):
    """
    Provide the shared FeatureStore from application state.
    """
    store = getattr(request.app.state, "feature_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature store not initialised.",
        )
    return store


def get_population_prior(request: Request):
    """
    Provide the PopulationPriorProfile from application state.
    """
    prior = getattr(request.app.state, "population_prior", None)
    return prior  # None is acceptable — cold start uses defaults


def get_device_bridge(request: Request):
    """
    Provide the DeviceSignalBridge from application state.
    """
    bridge = getattr(request.app.state, "device_bridge", None)
    return bridge  # None is acceptable — device signals are optional


# SESSION ENGINE DEPENDENCY
_session_cache: Dict[str, Tuple["ABSAv2Engine", float]] = {}
_SESSION_CACHE_TTL: int = 3600   # 1 hour


def get_or_create_engine(
    session_id: str,
    user_id:    str,
    request:    Request,
) -> ABSAv2Engine:
    """
    Return the ABSAv2Engine for a session, creating one if it does not exist.
    """
    now = time.time()

    # Evict stale engines
    stale = [sid for sid, (_, ts) in _session_cache.items()
             if now - ts > _SESSION_CACHE_TTL]
    for sid in stale:
        _session_cache.pop(sid, None)

    if session_id in _session_cache:
        engine, _ = _session_cache[session_id]
        _session_cache[session_id] = (engine, now)
        return engine

    # Load profile from Redis
    profile_store = getattr(request.app.state, "profile_store", None)
    population_prior = getattr(request.app.state, "population_prior", None)

    if profile_store is not None:
        try:
            profile = profile_store.get_or_create(
                user_id,
                fallback_profile=(
                    population_prior.create_new_user_profile(user_id)
                    if population_prior else BehavioralProfile(user_id=user_id)
                ),
            )
        except Exception as e:
            logger.warning(
                "Profile load failed for user=%s: %s. Using empty profile.",
                user_id, e
            )
            profile = BehavioralProfile(user_id=user_id)
    else:
        profile = BehavioralProfile(user_id=user_id)

    # Get active classifier from registry
    classifier = None
    model_registry = getattr(request.app.state, "model_registry", None)
    if model_registry is not None:
        try:
            classifier = model_registry.get_serving_model(session_id)
        except Exception as e:
            logger.warning("Failed to load classifier: %s", e)

    # Create engine
    shadow_mode = getattr(request.app.state, "shadow_mode", SHADOW_MODE_ENABLED)
    engine = ABSAv2Engine(
        profile     = profile,
        classifier  = classifier,
        session_id  = session_id,
        shadow_mode = shadow_mode,
    )
    _session_cache[session_id] = (engine, now)

    # Apply live runtime controls to newly created engines.
    runtime_cfg = getattr(request.app.state, "runtime_config", {}) or {}
    try:
        if hasattr(engine, "_ramp"):
            ramp_actions = max(
                1,
                int(runtime_cfg.get("cold_start_ramp_actions", engine._ramp.ramp_actions)),
            )
            engine._ramp.ramp_actions = ramp_actions
            engine._ramp._interp_start = int(
                ramp_actions * (1 - engine._ramp.interpolation_fraction)
            )

        trust_initial = float(runtime_cfg.get("trust_initial", engine.trust_score))
        engine.trust_score = min(100.0, max(0.0, trust_initial))
    except Exception as e:
        logger.warning("Failed to apply runtime settings to new engine: %s", e)

    logger.debug(
        "Session engine created: session=%s user=%s shadow=%s",
        session_id, user_id, shadow_mode
    )
    return engine


def get_engine_for_session(
    session_id: str,
    request:    Request,
) -> ABSAv2Engine:
    """
    Return the engine for an existing session (no creation).
    """
    if session_id not in _session_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Session '{session_id}' not found. "
                f"Start a session first via POST /session/start."
            ),
        )
    engine, _ = _session_cache[session_id]
    _session_cache[session_id] = (engine, time.time())  # refresh TTL
    return engine


def close_session_engine(session_id: str) -> Optional[ABSAv2Engine]:
    """
    Remove a session engine from the cache and return it for final processing.
    """
    if session_id in _session_cache:
        engine, _ = _session_cache.pop(session_id)
        return engine
    return None


def get_active_session_count() -> int:
    """Return the number of active session engines in cache."""
    return len(_session_cache)


# RATE LIMITING

# Sliding window rate limit store
_rate_limit_store: Dict[str, list] = {}
_RATE_LIMIT_WINDOW_S: int   = 60    # 1 minute window
_RATE_LIMIT_MAX_REQUESTS: int = 300  # 300 requests per minute per IP


def rate_limit_check(request: Request) -> None:
    """
    Enforce per-IP sliding window rate limiting.
    """
    client_ip = _get_client_ip(request)
    now       = time.time()
    window_start = now - _RATE_LIMIT_WINDOW_S

    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []

    # Remove timestamps outside the current window
    _rate_limit_store[client_ip] = [
        ts for ts in _rate_limit_store[client_ip]
        if ts > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX_REQUESTS:
        logger.warning(
            "Rate limit exceeded: ip=%s requests=%d window=%ds",
            client_ip,
            len(_rate_limit_store[client_ip]),
            _RATE_LIMIT_WINDOW_S,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {_RATE_LIMIT_MAX_REQUESTS} requests "
                f"per {_RATE_LIMIT_WINDOW_S} seconds."
            ),
            headers={
                "Retry-After":        str(_RATE_LIMIT_WINDOW_S),
                "X-RateLimit-Limit":  str(_RATE_LIMIT_MAX_REQUESTS),
                "X-RateLimit-Window": str(_RATE_LIMIT_WINDOW_S),
            },
        )

    _rate_limit_store[client_ip].append(now)


# REQUEST ID
def get_request_id(
    x_request_id: Annotated[Optional[str], Header(alias="X-Request-ID")] = None,
) -> str:
    """
    Return the X-Request-ID header value, or generate a new UUID if absent.
    """
    return x_request_id or str(uuid.uuid4())


# HELPER UTILITIES
def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP from the request, respecting reverse proxy headers.
    """
    # Respect X-Forwarded-For (set by nginx/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # Fall back to direct connection IP
    if request.client:
        return request.client.host

    return "unknown"


def extract_metadata_from_request(request: Request) -> dict:
    """
    Build a metadata dict from request headers for device signal assessment.
    """
    return {
        "ip":         _get_client_ip(request),
        "user_agent": request.headers.get("User-Agent", ""),
    }
