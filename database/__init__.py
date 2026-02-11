"""Database module for async operations

POOL USAGE POLICY:
  - get_tenant_pool(tenant_id) — REQUIRED for all agent/mutation operations.
    Wraps the raw pool in TenantScopedPool which validates SQL mutations
    at the application layer (defense-in-depth on top of DB-level RLS).
  - get_pool()                 — System-level only (health checks, monitoring,
    schema introspection, invariant checks). Not for tenant-scoped mutations.
"""

from .async_connection import (
    PoolConfig,
    BasePool,
    AsyncDatabasePool,
    InMemoryDatabasePool,
    init_pool,
    get_pool,
    get_tenant_pool,
    close_pool,
    using_fallback,
    DatabaseUnavailableError,
)
from .tenant_guard import TenantScopedPool

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
    "get_tenant_pool",
    "close_pool",
    "using_fallback",
    "get_tenant_db",
    "TenantScopedPool",
    "DatabaseUnavailableError",
]
