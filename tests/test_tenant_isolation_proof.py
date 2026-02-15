import pytest
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, AsyncMock
from database.async_connection import TenantScopedPool


class _MockConn:
    async def execute(self, *args, **kwargs):
        return "UPDATE 1"

    async def fetch(self, *args, **kwargs):
        return []

    async def fetchrow(self, *args, **kwargs):
        return None

    async def fetchval(self, *args, **kwargs):
        return None

    @asynccontextmanager
    async def transaction(self):
        yield


class _MockRawPool:
    @asynccontextmanager
    async def acquire(self, timeout=None):
        yield _MockConn()


class _MockWrapper:
    def __init__(self):
        self.pool = _MockRawPool()
        self._execute_called = False

    async def fetchrow(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        self._execute_called = True
        return "UPDATE 1"

    async def fetch(self, *args, **kwargs):
        return []


@pytest.mark.asyncio
async def test_cross_tenant_mutation_blocked():
    """Prove that an UPDATE without tenant_id fails."""
    mock_pool = _MockWrapper()
    tenant_pool = TenantScopedPool(mock_pool, "tenant-a")

    # malicious query (missing tenant_id)
    unsafe_sql = "UPDATE invoices SET status = 'paid' WHERE id = 'invoice-123'"

    with pytest.raises(ValueError, match="WHERE clause must constrain TENANT_ID"):
        await tenant_pool.execute(unsafe_sql)


@pytest.mark.asyncio
async def test_scoped_mutation_allowed():
    """Prove that a properly scoped UPDATE succeeds."""
    mock_pool = _MockWrapper()
    tenant_pool = TenantScopedPool(mock_pool, "tenant-a")

    # safe query
    safe_sql = "UPDATE invoices SET status = 'paid' WHERE id = 'invoice-123' AND tenant_id = $2"

    await tenant_pool.execute(safe_sql, "invoice-123", "tenant-a")
