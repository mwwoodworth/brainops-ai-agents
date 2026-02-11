
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock
from database.tenant_guard import TenantScopedPool

# Mock the underlying pool
class MockPool:
    async def fetchrow(self, *args, **kwargs): return None
    async def execute(self, *args, **kwargs): return "UPDATE 1"
    async def fetch(self, *args, **kwargs): return []

@pytest.fixture
def tenant_pool():
    return TenantScopedPool(MockPool(), "tenant-123")

@pytest.mark.asyncio
async def test_bypass_attempts(tenant_pool):
    """
    Adversarial test suite to attempt bypassing the tenant guardrail.
    Each test case represents a known SQL injection or obfuscation technique.
    """
    
    # 1. Whitespace Obfuscation
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute("   UPDATE    invoices   SET status='paid' WHERE id='1'")

    # 2. Case Variations
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute("uPdAtE invoices SET status='paid' WHERE id='1'")

    # 3. Comment Injection (Basic)
    # The current regex-like check might be fooled by comments if it strictly looks for "UPDATE"
    # But usually simple string containment "UPDATE" works. 
    # The bypass target is the *absence* of "tenant_id" check.
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute("UPDATE/* comment */invoices SET status='paid' WHERE id='1'")

    # 4. CTE / Subquery Mutation
    # This is tricky: "WITH x AS (UPDATE ...) SELECT ..."
    # The guardrail looks for "UPDATE", so it should catch this.
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute("WITH updated_rows AS (UPDATE invoices SET status='paid' WHERE id='1' RETURNING *) SELECT * FROM updated_rows")

    # 5. Schema Qualification
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute("UPDATE public.invoices SET status='paid' WHERE id='1'")

    # 6. Quoted Identifiers
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute('UPDATE "invoices" SET status=\'paid\' WHERE id=\'1\'')


    # 7. Semicolon Chaining (Multi-statement)
    # Note: asyncpg usually blocks multi-statement by default unless explicitly enabled/configured, 
    # but the guardrail should catch the second statement if it parses the whole string.
    with pytest.raises(ValueError, match="Tenant ID check missing"):
        await tenant_pool.execute("SELECT 1; UPDATE invoices SET status='paid' WHERE id='1'")

@pytest.mark.asyncio
async def test_false_positives(tenant_pool):
    """Ensure safe queries are allowed."""
    # 1. Proper scope
    await tenant_pool.execute("UPDATE invoices SET status='paid' WHERE id='1' AND tenant_id = $1", "tenant-123")
    
    # 2. Case insensitive tenant_id check
    await tenant_pool.execute("UPDATE invoices SET status='paid' WHERE id='1' AND TENANT_ID = $1", "tenant-123")

@pytest.mark.asyncio
async def test_potential_bypass_vectors(tenant_pool):
    """
    These are edge cases that MIGHT bypass a naive string check.
    We assert they FAIL (raise ValueError). If they pass, we have a vulnerability.
    """
    
    # Bypass 1: "tenant_id" in a comment but not in logic
    # e.g. "UPDATE invoices ... -- tenant_id check ignored"
    # Current implementation: looks for "TENANT_ID" string. 
    # Vulnerability: If I write "UPDATE ... -- tenant_id", logic sees "tenant_id" and allows it.
    
    # We EXPECT this to be blocked (require semantic check), but based on current implementation (string check),
    # it might pass. We document the result.
    
    bypass_payload = "UPDATE invoices SET status='paid' WHERE id='1' -- tenant_id check"
    
    try:
        await tenant_pool.execute(bypass_payload)
        pytest.fail("CRITICAL VULNERABILITY: Comment-based bypass succeeded! (String check found 'tenant_id' in comment)")
    except ValueError:
        pass # Good, it failed (or logic is smarter than we think)

    # Bypass 2: "tenant_id" in a string literal
    # e.g. UPDATE invoices SET notes = 'checked tenant_id' WHERE id=1
    bypass_string = "UPDATE invoices SET notes = 'verified tenant_id manually' WHERE id='1'"
    try:
        await tenant_pool.execute(bypass_string)
        pytest.fail("CRITICAL VULNERABILITY: String literal bypass succeeded!")
    except ValueError:
        pass

