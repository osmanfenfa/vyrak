"""
zelvryn/api/routers/__init__.py
================================
Route registration package for the Zelvryn API.

All four routers are imported here and exposed for registration in main.py
via a single include_routers(app) call. This keeps main.py clean and makes
it easy to add or remove routers without touching the app factory.

Routers
-------
  sessions_router   — /session/* endpoints (core evaluation)
  profiles_router   — /profile/* endpoints (profile management)
  shadow_router     — /shadow/* endpoints (shadow mode data)
  health_router     — /health endpoints (observability)

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

from api.routers.sessions import router as sessions_router
from api.routers.profiles import router as profiles_router
from api.routers.shadow   import router as shadow_router
from api.routers.health   import router as health_router


def include_routers(app) -> None:
    """
    Register all routers on the FastAPI application.

    Parameters
    ----------
    app : FastAPI
    """
    app.include_router(sessions_router)
    app.include_router(profiles_router)
    app.include_router(shadow_router)
    app.include_router(health_router)


__all__ = [
    "sessions_router",
    "profiles_router",
    "shadow_router",
    "health_router",
    "include_routers",
]
