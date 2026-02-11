
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from database.async_connection import TenantScopedPool

@pytest.mark.asyncio
async def test_cross_tenant_mutation_blocked():
    """Prove that an UPDATE without tenant_id fails."""
    mock_pool = AsyncMock()
    tenant_pool = TenantScopedPool(mock_pool, "tenant-a")
    
    # malicious query (missing tenant_id)
    unsafe_sql = "UPDATE invoices SET status = 'paid' WHERE id = 'invoice-123'"
    
    with pytest.raises(ValueError, match="WHERE clause must constrain TENANT_ID"):
        await tenant_pool.execute(unsafe_sql)

@pytest.mark.asyncio
async def test_scoped_mutation_allowed():
    """Prove that a properly scoped UPDATE succeeds."""
    mock_pool = AsyncMock()
    tenant_pool = TenantScopedPool(mock_pool, "tenant-a")
    
    # safe query
    safe_sql = "UPDATE invoices SET status = 'paid' WHERE id = 'invoice-123' AND tenant_id = $2"
    
    await tenant_pool.execute(safe_sql, "invoice-123", "tenant-a")
    mock_pool.execute.assert_called_once()
