
import pytest
import asyncpg
from database.async_connection import get_pool

# This test assumes the env vars are set to the RESTRICTED user.
# If run with superuser env vars, it will FAIL (which is good, it proves we are not hardened yet).

@pytest.mark.asyncio
async def test_agent_cannot_delete_users():
    pool = get_pool()
    try:
        # Attempt forbidden action
        await pool.execute("DELETE FROM users WHERE id = 'some-uuid'")
        pytest.fail("SECURITY FAILURE: Agent was able to DELETE from users table!")
    except asyncpg.InsufficientPrivilegeError:
        pass # SUCCESS: DB blocked the action
    except Exception as e:
        # Fallback for other errors (like syntax/FK), but ideally we want specific permission error
        if "permission denied" in str(e).lower():
            pass
        else:
            raise e

@pytest.mark.asyncio
async def test_agent_cannot_update_tenants():
    pool = get_pool()
    try:
        await pool.execute("UPDATE tenants SET name = 'Hacked' WHERE id = 'some-uuid'")
        pytest.fail("SECURITY FAILURE: Agent was able to UPDATE tenants table!")
    except asyncpg.InsufficientPrivilegeError:
        pass
    except Exception as e:
        if "permission denied" in str(e).lower():
            pass
        else:
            raise e
