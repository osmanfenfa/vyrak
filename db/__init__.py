"""
zelvryn/db/__init__.py
=======================
Database layer for the Zelvryn ABSA system.

This package owns all PostgreSQL interaction: connection pooling, ORM models,
typed query functions, and schema migrations via Alembic.

Architecture position
----------------------
db/ is a peer to api/ and profiles/ — it provides persistence for data that
does not belong in Redis (audit logs, session archives, feature vectors, model
versions, consent records). Redis owns live profile state (sub-millisecond
reads); PostgreSQL owns everything that needs durable storage, complex queries,
or relational integrity.

Dependency rule
---------------
db/ imports from core/ only (for type hints on constants).
No other package imports from db/ — they use it indirectly through the
query functions exposed here, or through async-compatible wrappers in api/.

Components
----------
  connection.py   — SQLAlchemy async engine, session factory, dependency helpers
  models.py       — SQLAlchemy ORM table definitions (mirrors db/migrations/)
  queries.py      — Typed query functions: insert, select, update, delete

  migrations/
    alembic.ini   — Alembic configuration
    env.py        — Migration environment (connects Alembic to the async engine)
    versions/
      001_initial_schema.py  — Creates all base tables
      002_add_shadow_log.py  — Adds shadow_logs table

Usage
-----
    from db import get_async_session, get_engine
    from db.queries import ActionLogQueries, SessionQueries

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

from db.connection import (
    get_engine,
    get_async_session,
    get_session_factory,
    init_db,
    close_db,
    DatabaseNotAvailableError,
)
from db.models import (
    Base,
    UserRecord,
    SessionRecord,
    ActionLog,
    FeatureVector,
    ModelVersion,
    ShadowLog,
    ConsentRecord,
)
from db import queries

__all__ = [
    # Connection helpers
    "get_engine",
    "get_async_session",
    "get_session_factory",
    "init_db",
    "close_db",
    "DatabaseNotAvailableError",
    # ORM models
    "Base",
    "UserRecord",
    "SessionRecord",
    "ActionLog",
    "FeatureVector",
    "ModelVersion",
    "ShadowLog",
    "ConsentRecord",
    # Query module
    "queries",
]

__version__ = "2.0.0"
__author__  = "Vyrak"
