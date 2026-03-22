"""
zelvryn/db/connection.py
=========================
Async PostgreSQL connection management for the Zelvryn ABSA system.

This module provides:
  - An async SQLAlchemy engine backed by asyncpg
  - A session factory (AsyncSessionFactory) for database transactions
  - FastAPI-compatible async dependency (get_async_session)
  - Startup (init_db) and shutdown (close_db) lifecycle functions
  - A synchronous fallback engine for Alembic migrations (which require sync)

Connection strategy
-------------------
SQLAlchemy's async engine wraps asyncpg, the fastest Python PostgreSQL driver.
The engine is created once at application startup (in api/main.py lifespan)
and shared across all request handlers via module-level state. This provides:
  - Connection pooling (NullPool for async, configurable pool size for sync)
  - Prepared statement caching via asyncpg
  - Automatic reconnection on transient failures

Session lifecycle
-----------------
Each database operation gets its own session (unit of work). Sessions are:
  - Created by AsyncSessionFactory() at the start of each operation
  - Committed on success, rolled back on exception
  - Closed automatically via the async context manager

For FastAPI route handlers that need a session, use the `get_async_session`
dependency via Depends(). For background workers and scripts, use
`get_async_session()` directly as an async context manager.

Sync engine for Alembic
-----------------------
Alembic's migration runner is synchronous. A separate sync engine is created
on demand (not pooled) for migration commands only. It is never used by the
running API.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL STATE
# Engine and session factory are singletons — created once, shared everywhere.
# ─────────────────────────────────────────────────────────────────────────────

_engine = None           # AsyncEngine — the primary async engine
_session_factory = None  # async_sessionmaker — creates AsyncSession instances
_sync_engine = None      # Engine — synchronous engine for Alembic only


class DatabaseNotAvailableError(Exception):
    """
    Raised when a database operation is attempted but the engine has not
    been initialised (e.g., DATABASE_URL not set, init_db() not called).
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db(database_url: Optional[str] = None) -> None:
    """
    Initialise the async database engine and session factory.

    Call this once at application startup (inside api/main.py lifespan).
    Subsequent calls are no-ops if the engine is already initialised.

    Parameters
    ----------
    database_url : str, optional
        PostgreSQL DSN in asyncpg format:
          postgresql+asyncpg://user:password@host:5432/zelvryn
        Falls back to DATABASE_URL environment variable if not provided.
        If neither is set, logs a warning and leaves the engine as None
        (the system operates without a database — suitable for testing).

    Side effects
    ------------
    Sets module-level _engine and _session_factory.
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.debug("Database already initialised — skipping.")
        return

    url = database_url or os.environ.get("DATABASE_URL")

    if not url:
        logger.warning(
            "DATABASE_URL not set. Database features (audit logs, feature store, "
            "session archive) are disabled. Set DATABASE_URL to enable."
        )
        return

    # Normalise URL: psycopg2-style URLs need +asyncpg for async driver
    async_url = _normalise_async_url(url)

    try:
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

        pool_size     = int(os.environ.get("DB_POOL_SIZE",     "10"))
        max_overflow  = int(os.environ.get("DB_MAX_OVERFLOW",  "20"))
        pool_timeout  = int(os.environ.get("DB_POOL_TIMEOUT",  "30"))
        pool_recycle  = int(os.environ.get("DB_POOL_RECYCLE",  "1800"))  # 30 min

        _engine = create_async_engine(
            async_url,
            pool_size        = pool_size,
            max_overflow     = max_overflow,
            pool_timeout     = pool_timeout,
            pool_recycle     = pool_recycle,
            pool_pre_ping    = True,   # detect stale connections before use
            echo             = os.environ.get("DB_ECHO", "false").lower() == "true",
            future           = True,
        )

        _session_factory = async_sessionmaker(
            bind           = _engine,
            expire_on_commit = False,   # prevent lazy-load errors after commit
            autoflush      = False,
            autocommit     = False,
        )

        logger.info(
            "Database engine initialised: pool_size=%d max_overflow=%d url=%s",
            pool_size, max_overflow, _redact_url(async_url)
        )

    except ImportError as e:
        logger.error(
            "SQLAlchemy or asyncpg not installed: %s. "
            "Install with: pip install sqlalchemy[asyncio] asyncpg",
            e
        )
    except Exception as e:
        logger.error(
            "Failed to initialise database engine: %s", e, exc_info=True
        )


async def close_db() -> None:
    """
    Dispose the async engine and release all pooled connections.

    Call this at application shutdown (inside api/main.py lifespan).
    """
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine         = None
        _session_factory = None
        logger.info("Database engine disposed.")


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE AND SESSION ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    """
    Return the async engine, or None if not initialised.

    Returns
    -------
    AsyncEngine or None
    """
    return _engine


def get_session_factory():
    """
    Return the async session factory (async_sessionmaker), or None.

    Returns
    -------
    async_sessionmaker or None
    """
    return _session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator:
    """
    Async context manager that yields a database session.

    Commits on success, rolls back on exception, always closes the session.

    Usage (in a background worker or script):
    -----------------------------------------
    async with get_async_session() as session:
        result = await session.execute(select(UserRecord))
        rows = result.scalars().all()

    Usage (as a FastAPI dependency):
    ---------------------------------
    async def my_route(session = Depends(get_async_session_dep)):
        ...

    Yields
    ------
    AsyncSession

    Raises
    ------
    DatabaseNotAvailableError
        If the engine has not been initialised.
    """
    if _session_factory is None:
        raise DatabaseNotAvailableError(
            "Database session requested but engine is not initialised. "
            "Set DATABASE_URL and call init_db() at startup."
        )

    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_async_session_dep():
    """
    FastAPI dependency wrapper for get_async_session.

    Use via Depends(get_async_session_dep) in route handlers that need
    direct database access.

    Yields
    ------
    AsyncSession
    """
    async with get_async_session() as session:
        yield session


# ─────────────────────────────────────────────────────────────────────────────
# SYNCHRONOUS ENGINE FOR ALEMBIC
# ─────────────────────────────────────────────────────────────────────────────

def get_sync_engine(database_url: Optional[str] = None):
    """
    Create and return a synchronous SQLAlchemy engine for Alembic migrations.

    This engine is NOT pooled (NullPool) and should only be used in migration
    scripts, not in the running API. It is created on demand, not cached.

    Parameters
    ----------
    database_url : str, optional
        PostgreSQL DSN in psycopg2 format:
          postgresql+psycopg2://user:password@host:5432/zelvryn
        Falls back to DATABASE_URL env var.

    Returns
    -------
    Engine (synchronous) or None if no URL is available.
    """
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        return None

    sync_url = _normalise_sync_url(url)

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool

        engine = create_engine(
            sync_url,
            poolclass   = NullPool,   # no pooling — used for one-shot migrations
            echo        = False,
            future      = True,
        )
        logger.debug("Sync engine created for Alembic: %s", _redact_url(sync_url))
        return engine
    except ImportError as e:
        logger.error("SQLAlchemy not installed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

async def check_db_health() -> dict:
    """
    Perform a lightweight database connectivity check.

    Executes `SELECT 1` against the database and measures round-trip latency.

    Returns
    -------
    dict
        Keys: status ("healthy" | "down"), latency_ms, details.
    """
    import time

    if _engine is None:
        return {
            "status":  "not_configured",
            "details": "DATABASE_URL not set — database features disabled.",
        }

    t_start = time.perf_counter()
    try:
        from sqlalchemy import text
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "status":     "healthy",
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        return {
            "status":  "down",
            "details": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TABLE CREATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

async def create_all_tables() -> None:
    """
    Create all ORM-defined tables that do not yet exist.

    This is a lightweight alternative to running Alembic migrations —
    suitable for development and testing. In production, always use
    Alembic migrations to get versioned, reversible schema changes.

    Raises
    ------
    DatabaseNotAvailableError
    """
    if _engine is None:
        raise DatabaseNotAvailableError(
            "Cannot create tables: database engine not initialised."
        )

    from db.models import Base

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created (or already existed).")


async def drop_all_tables() -> None:
    """
    Drop all ORM-defined tables.

    WARNING: Destructive — for testing and development only.

    Raises
    ------
    DatabaseNotAvailableError
    """
    if _engine is None:
        raise DatabaseNotAvailableError(
            "Cannot drop tables: database engine not initialised."
        )

    from db.models import Base

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped.")


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE URL NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_async_url(url: str) -> str:
    """
    Convert a database URL to the asyncpg-compatible format.

    Supported conversions:
      postgresql://...         → postgresql+asyncpg://...
      postgresql+psycopg2://... → postgresql+asyncpg://...
      postgres://...           → postgresql+asyncpg://...
    """
    url = url.strip()
    replacements = [
        ("postgresql+psycopg2://", "postgresql+asyncpg://"),
        ("postgresql+psycopg://",  "postgresql+asyncpg://"),
        ("postgres://",            "postgresql+asyncpg://"),
    ]
    for old, new in replacements:
        if url.startswith(old):
            return new + url[len(old):]

    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]

    return url   # already in the right format (or custom driver)


def _normalise_sync_url(url: str) -> str:
    """
    Convert a database URL to psycopg2-compatible synchronous format.

    Supported conversions:
      postgresql+asyncpg://... → postgresql+psycopg2://...
      postgresql://...         → postgresql+psycopg2://...
      postgres://...           → postgresql+psycopg2://...
    """
    url = url.strip()
    replacements = [
        ("postgresql+asyncpg://", "postgresql+psycopg2://"),
        ("postgres://",           "postgresql+psycopg2://"),
    ]
    for old, new in replacements:
        if url.startswith(old):
            return new + url[len(old):]

    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]

    return url


def _redact_url(url: str) -> str:
    """
    Redact the password from a database URL for safe logging.

    postgresql+asyncpg://user:secret@host/db → postgresql+asyncpg://user:***@host/db
    """
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(
                f":{parsed.password}@", ":***@"
            )
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url
