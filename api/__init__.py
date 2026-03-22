"""
zelvryn/api/__init__.py
========================
FastAPI REST API layer for the Zelvryn ABSA system.

This package exposes Zelvryn's evaluation engine, profile management,
shadow mode controls, and system health as a REST API. It is the only
layer that faces external clients — the application developer, the
security analyst dashboard, and the admin tooling.

Architecture position
----------------------
api/ sits at the top of the dependency stack:

  api/  →  profiles/ (RedisProfileStore)
  api/  →  ml/      (ModelRegistry, FeatureStore)
  api/  →  coldstart/ (PopulationPriorProfile, DeviceSignalBridge)
  api/  →  ingestion/ (ConsentGatedLogger, ShadowModeLogger)
  api/  →  core/    (ABSAv2Engine, BehavioralProfile, UserAction)

api/ imports from everything. Nothing imports from api/.

Components
----------
  main.py         — FastAPI app factory, lifespan, CORS, exception handlers
  schemas.py      — Pydantic request/response models for all endpoints
  dependencies.py — FastAPI dependency injection: engine factory, profile store,
                    model registry, rate limiting, authentication
  middleware.py   — Request logging, timing, request-ID injection, audit trail

  routers/
    sessions.py   — POST /session/start, POST /session/{id}/action,
                    GET /session/{id}/summary, DELETE /session/{id}
    profiles.py   — GET/DELETE /profile/{user_id}, POST /profile/{user_id}/reset
    shadow.py     — POST /shadow/ingest, GET /shadow/export
    health.py     — GET /health, GET /health/model, GET /health/redis

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

from api.main import create_app

__all__ = ["create_app"]
__version__ = "2.0.0"
__author__  = "Vyrak"
