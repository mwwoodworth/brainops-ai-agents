"""Shared helpers for verifying table existence without DDL.

The agent_worker role (app_agent_role) has NO DDL permissions by design
(P0-LOCK security).  All CREATE TABLE / CREATE INDEX / CREATE EXTENSION /
ALTER TABLE must happen via migrations run as postgres, never at runtime.

These helpers let modules confirm their required tables exist so they can
degrade gracefully when tables are missing.
"""

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def verify_tables_sync(
    table_names: Sequence[str],
    cursor,
    *,
    module_name: str = "unknown",
) -> bool:
    """Check that all *table_names* exist in the public schema (sync/psycopg2).

    Returns True if every table is present, False otherwise.
    Logs an error listing any missing tables.
    """
    try:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (list(table_names),),
        )
        found = {row[0] for row in cursor.fetchall()}
        missing = set(table_names) - found
        if missing:
            logger.error(
                "[%s] Required tables missing (%d/%d present). "
                "Run migrations to create: %s",
                module_name,
                len(found),
                len(table_names),
                ", ".join(sorted(missing)),
            )
            return False
        logger.info("[%s] All %d required tables verified", module_name, len(table_names))
        return True
    except Exception as exc:
        logger.error("[%s] Table verification failed: %s", module_name, exc)
        return False


async def verify_tables_async(
    table_names: Sequence[str],
    pool,
    *,
    module_name: str = "unknown",
) -> bool:
    """Check that all *table_names* exist in the public schema (async/asyncpg).

    Returns True if every table is present, False otherwise.
    Logs an error listing any missing tables.
    """
    try:
        rows = await pool.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY($1)",
            list(table_names),
        )
        found = {r["table_name"] for r in rows}
        missing = set(table_names) - found
        if missing:
            logger.error(
                "[%s] Required tables missing (%d/%d present). "
                "Run migrations to create: %s",
                module_name,
                len(found),
                len(table_names),
                ", ".join(sorted(missing)),
            )
            return False
        logger.info("[%s] All %d required tables verified", module_name, len(table_names))
        return True
    except Exception as exc:
        logger.error("[%s] Table verification failed: %s", module_name, exc)
        return False
