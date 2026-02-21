"""
Database health check helpers extracted from app.py.

Used by the health router and deferred initialization logic.
"""
import asyncio
import logging
from typing import Any

from config import config
from database.async_connection import PoolConfig, get_pool, init_pool

logger = logging.getLogger(__name__)


async def pool_roundtrip_healthy(pool: Any, timeout: float = 4.0) -> bool:
    """
    Prefer pool-based roundtrip health checks.

    Direct new-connection probes can report false negatives when the pool is still
    serving traffic but the provider temporarily blocks fresh auth attempts.
    """
    try:
        value = await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=timeout)
        return str(value) == "1"
    except asyncio.TimeoutError:
        logger.warning("Health check pool roundtrip timed out after %.2fs", timeout)
        return False
    except Exception as exc:
        logger.warning("Health check pool roundtrip failed: %s", exc)
        return False


async def attempt_db_pool_init_once(app_state: Any, context: str, timeout: float = 5.0) -> bool:
    """Best-effort database pool initialization + immediate roundtrip verification."""
    try:
        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database,
            ssl=config.database.ssl,
            ssl_verify=config.database.ssl_verify,
        )
        await init_pool(pool_config)
        pool = get_pool()
        healthy = await pool_roundtrip_healthy(pool, timeout=timeout)
        if healthy:
            app_state.db_init_error = None
            logger.info("✅ Database pool verified (%s)", context)
            return True

        app_state.db_init_error = "Pool roundtrip health check failed"
        logger.warning("Database pool initialized but not healthy (%s)", context)
        return False
    except Exception as exc:
        app_state.db_init_error = str(exc)
        logger.warning("Database pool init attempt failed (%s): %s", context, exc)
        return False
