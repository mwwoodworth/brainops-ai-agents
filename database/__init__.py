"""Database module for async operations"""

from .async_connection import (
    PoolConfig,
    BasePool,
    AsyncDatabasePool,
    InMemoryDatabasePool,
    init_pool,
    get_pool,
    close_pool,
    using_fallback,
)

# Stub for get_tenant_db - provides tenant-specific DB access
async def get_tenant_db(tenant_id: str = None):
    """Get database connection for a specific tenant.

    Falls back to default pool if tenant_id not specified.
    """
    pool = get_pool()
    return pool

__all__ = [
    "PoolConfig",
    "BasePool",
    "AsyncDatabasePool",
    "InMemoryDatabasePool",
    "init_pool",
    "get_pool",
    "close_pool",
    "using_fallback",
    "get_tenant_db",
]
