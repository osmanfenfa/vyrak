from api.routers.sessions import router as sessions_router
from api.routers.profiles import router as profiles_router
from api.routers.shadow   import router as shadow_router
from api.routers.health   import router as health_router
from api.routers.admin_gate import router as admin_gate_router

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
    app.include_router(admin_gate_router)

__all__ = [
    "sessions_router",
    "profiles_router",
    "shadow_router",
    "health_router",
    "admin_gate_router",
    "include_routers",
]
