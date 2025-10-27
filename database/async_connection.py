"""
Async Database Connection Pool - Production Ready
Type-safe, lint-clean, fully tested
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

# Type alias for database records
DbRecord = dict[str, Any]


@dataclass
class PoolConfig:
    """Database pool configuration"""
    host: str
    port: int
    user: str
    password: str
    database: str
    min_size: int = 5
    max_size: int = 20
    command_timeout: int = 60


class AsyncDatabasePool:
    """Async database connection pool manager"""

    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize connection pool"""
        if self._pool is not None:
            logger.warning("Pool already initialized")
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                command_timeout=self.config.command_timeout
            )
            logger.info(
                f"✅ Database pool initialized "
                f"(min={self.config.min_size}, max={self.config.max_size})"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize pool: {e}")
            raise

    async def close(self) -> None:
        """Close connection pool"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("✅ Database pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        """Get pool instance"""
        if self._pool is None:
            raise RuntimeError(
                "Database pool not initialized. Call initialize() first."
            )
        return self._pool

    async def fetch(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> list[asyncpg.Record]:
        """Execute query and return all rows"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """Execute query and return single row"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute query and return single value"""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> str:
        """Execute query without returning data"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def executemany(
        self,
        command: str,
        args: list[Any],
        timeout: Optional[float] = None
    ) -> str:
        """Execute query for multiple parameter sets"""
        async with self.pool.acquire() as conn:
            return await conn.executemany(command, args, timeout=timeout)

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Global pool instance
_pool: Optional[AsyncDatabasePool] = None


async def init_pool(config: PoolConfig) -> AsyncDatabasePool:
    """Initialize global database pool"""
    global _pool
    if _pool is None:
        _pool = AsyncDatabasePool(config)
        await _pool.initialize()
    return _pool


def get_pool() -> AsyncDatabasePool:
    """Get global database pool"""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close global database pool"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
