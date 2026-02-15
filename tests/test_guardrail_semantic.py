import pytest
from contextlib import asynccontextmanager
from database.tenant_guard import TenantScopedPool
from unittest.mock import AsyncMock


class MockConnection:
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


class MockRawPool:
    @asynccontextmanager
    async def acquire(self, timeout=None):
        yield MockConnection()


class MockPool:
    def __init__(self):
        self.pool = MockRawPool()

    async def fetchrow(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        return "UPDATE 1"

    async def fetch(self, *args, **kwargs):
        return []


@pytest.fixture
def tenant_pool():
    return TenantScopedPool(MockPool(), "tenant-123")


@pytest.mark.asyncio
async def test_semantic_advanced_bypasses(tenant_pool):
    """
    Semantic attacks that might fool a regex but not a parser.
    """

    # 1. Tautology with Precedence trick
    # Regex sees "tenant_id", but logic is "tenant_id = X OR 1=1" -> TRUE
    unsafe_or = "UPDATE invoices SET s='paid' WHERE tenant_id = 't1' OR 1=1"
    with pytest.raises(ValueError, match="OR clauses not allowed"):
        await tenant_pool.execute(unsafe_or)

    # 2. CTE Hiding
    # Logic is inside CTE, outer query has tenant_id but does nothing constraining
    unsafe_cte = "WITH rows AS (UPDATE invoices SET s='paid' RETURNING id) SELECT * FROM rows WHERE tenant_id='t1'"
    with pytest.raises(ValueError, match="CTE"):
        await tenant_pool.execute(unsafe_cte)

    # 3. Predicate on WRONG table
    # UPDATE invoices i, tenants t SET s='paid' WHERE t.tenant_id = 't1'
    # The constraint is on 't', not 'i'. 'invoices' is unbounded.
    # This requires knowing which table is being updated.
    # Current Draconian Guard allows this if it sees "tenant_id".
    # A true semantic parser should detect "tenant_id" must apply to the target table or be a general constraint.
    # For now, we accept if "tenant_id" is present in WHERE, assuming straightforward queries.

    # 4. Function call masquerading
    # SELECT my_unsafe_func(tenant_id)
    # If function does mutation, it's unsafe.
    # We should block SELECTs that look like mutations or just block SELECTs in .execute() if possible?
    # No, execute() might run stored procs.

    # 5. Table Aliasing
    # UPDATE invoices i SET s='paid' WHERE i.tenant_id = 't1'
    # Should PASS because "tenant_id" token is present.
    safe_alias = "UPDATE invoices i SET s='paid' WHERE i.tenant_id = 't1'"
    await tenant_pool.execute(safe_alias)

    # 6. Misbound Predicate (The Hardest Case)
    # UPDATE invoices SET s='paid' WHERE EXISTS (SELECT 1 FROM tenants WHERE id = 't1')
    # "tenant_id" is NOT in the query text, but logic might imply it.
    # Our guardrail requires "TENANT_ID" token. So this should FAIL (Good).
    unsafe_exists = (
        "UPDATE invoices SET s='paid' WHERE EXISTS (SELECT 1 FROM tenants WHERE id = 't1')"
    )
    with pytest.raises(ValueError, match="WHERE clause must constrain TENANT_ID"):
        await tenant_pool.execute(unsafe_exists)

    # 7. Misbound Predicate V2
    # UPDATE invoices SET s='paid' WHERE other_col = 'tenant_id'
    # "tenant_id" is present as a string literal value, not a column.
    # Our parser STRIPS string literals. So this should FAIL (Good).
    unsafe_literal = "UPDATE invoices SET s='paid' WHERE other_col = 'tenant_id'"
    with pytest.raises(ValueError, match="WHERE clause must constrain TENANT_ID"):
        await tenant_pool.execute(unsafe_literal)
