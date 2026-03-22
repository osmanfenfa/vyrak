"""
zelvryn/db/migrations/env.py
==============================
Alembic migration environment for the Zelvryn ABSA database.

This file is automatically invoked by Alembic when you run:
  alembic upgrade head
  alembic revision --autogenerate -m "description"
  alembic downgrade -1

It connects the Alembic migration runner to Zelvryn's SQLAlchemy models
and database connection. Because Zelvryn uses an async engine (asyncpg),
we must use Alembic's run_sync() pattern to run migrations synchronously.

Two running modes
-----------------
  Offline mode (alembic upgrade head --sql)
    Generates the SQL script without connecting to a database.
    Useful for reviewing migrations before applying them or for generating
    SQL to hand off to a DBA.

  Online mode (alembic upgrade head)
    Connects to the database, runs the migration in a transaction, and
    updates the alembic_version table.

Usage
-----
    # Apply all pending migrations
    cd db/migrations && alembic -c alembic.ini upgrade head

    # Generate a new migration from model changes
    alembic -c alembic.ini revision --autogenerate -m "add_new_column"

    # Downgrade one step
    alembic -c alembic.ini downgrade -1

    # Show current migration state
    alembic -c alembic.ini current

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Add project root to sys.path so db.models can be imported.
# ─────────────────────────────────────────────────────────────────────────────

# Resolve project root (four levels up: versions/ → migrations/ → db/ → project/)
_migrations_dir = Path(__file__).resolve().parent
_db_dir         = _migrations_dir.parent
_project_root   = _db_dir.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ─────────────────────────────────────────────────────────────────────────────
# ALEMBIC CONFIG OBJECT
# ─────────────────────────────────────────────────────────────────────────────

# `config` is the Alembic Config object injected by the alembic runner.
# It provides access to alembic.ini settings.
config = context.config

# Inject the database URL from the environment variable into alembic.ini's
# sqlalchemy.url setting. This overrides whatever is in alembic.ini so the
# URL does not need to be hardcoded in the config file.
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Alembic uses synchronous drivers — normalise to psycopg2 format
    sync_url = database_url
    replacements = [
        ("postgresql+asyncpg://", "postgresql+psycopg2://"),
        ("postgres://",           "postgresql+psycopg2://"),
    ]
    for old, new in replacements:
        if sync_url.startswith(old):
            sync_url = new + sync_url[len(old):]
            break

    config.set_main_option("sqlalchemy.url", sync_url)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Interpret the alembic.ini logging configuration if present.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass   # logging config is optional — don't fail migrations over it

# ─────────────────────────────────────────────────────────────────────────────
# METADATA FOR AUTOGENERATE
# ─────────────────────────────────────────────────────────────────────────────

# Import all ORM models so Alembic can detect table additions, column changes,
# index changes, etc. during --autogenerate. Without these imports, autogenerate
# would produce empty (no-op) migration scripts.
from db.models import Base  # noqa: E402 — must be after path setup

target_metadata = Base.metadata

# ─────────────────────────────────────────────────────────────────────────────
# MIGRATION CONTEXT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Tables and schemas to exclude from autogenerate comparison.
# The alembic_version table is managed by Alembic itself and should never
# appear in generated migrations.
EXCLUDE_TABLES = {"alembic_version"}


def include_object(object, name, type_, reflected, compare_to):
    """
    Filter function for --autogenerate: determines which objects to include
    in migration comparisons.

    Excludes the alembic_version table and any external tables that are
    managed outside of Zelvryn's ORM (e.g., PostGIS extension tables).
    """
    if type_ == "table" and name in EXCLUDE_TABLES:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in offline mode.

    Generates SQL statements without connecting to a live database.
    Writes the SQL to stdout or a file depending on alembic command flags.

    This is useful for:
      - Reviewing what a migration will do before applying it
      - Generating SQL to hand to a DBA for manual execution
      - CI pipelines that don't have database access during the build step
    """
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url                    = url,
        target_metadata        = target_metadata,
        literal_binds          = True,
        dialect_opts           = {"paramstyle": "named"},
        include_object         = include_object,
        compare_type           = True,   # detect column type changes
        compare_server_default = True,   # detect server_default changes
        render_as_batch        = False,  # set True for SQLite (not needed for PG)
    )

    with context.begin_transaction():
        context.run_migrations()


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_migrations_online() -> None:
    """
    Run migrations in online mode against a live database.

    Creates a synchronous engine connection, runs migrations inside a
    transaction, and updates the alembic_version table on success.

    The engine is created fresh (NullPool — no connection pooling) since
    migration scripts are one-shot processes, not long-running servers.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix  = "sqlalchemy.",
        poolclass = pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection             = connection,
            target_metadata        = target_metadata,
            include_object         = include_object,
            compare_type           = True,
            compare_server_default = True,
            # Ensure all DDL runs inside a transaction for atomic rollback
            transaction_per_migration = True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
