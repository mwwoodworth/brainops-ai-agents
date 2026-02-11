
import logging
from typing import Any
from database.simple_sql_parser import SimpleSQLParser

logger = logging.getLogger(__name__)


class TenantScopedPool:
    """
    A wrapper around the asyncpg pool that enforces tenant isolation checks
    at the application layer using a lightweight SQL tokenizer/parser.

    All query methods are wrapped so that mutations (INSERT/UPDATE/DELETE)
    are validated for tenant_id scoping before execution.  Read queries
    are passed through but still parsed for early detection of unsafe patterns.

    This is DEFENSE-IN-DEPTH on top of database-level RLS — not a replacement.
    """

    def __init__(self, pool: Any, tenant_id: str):
        self.pool = pool
        self.tenant_id = tenant_id

    def _validate_query(self, query: str) -> None:
        """Parse query to enforce mutation safety (tenant_id in WHERE/INSERT)."""
        parser = SimpleSQLParser(query)
        parser.validate_mutation_safety("TENANT_ID")

    # ── Core query methods ──────────────────────────────────────────────

    async def fetch(self, query: str, *args: Any) -> list:
        self._validate_query(query)
        return await self.pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any):
        self._validate_query(query)
        return await self.pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any):
        self._validate_query(query)
        return await self.pool.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        self._validate_query(query)
        return await self.pool.execute(query, *args)

    async def executemany(self, query: str, args: list) -> None:
        self._validate_query(query)
        return await self.pool.executemany(query, args)

    # ── Pass-through for connection-level operations ────────────────────

    async def acquire(self):
        """Acquire a raw connection from the underlying pool.

        WARNING: Operations on the raw connection bypass tenant validation.
        Use only for schema introspection or system-level queries.
        """
        logger.warning(
            "TenantScopedPool.acquire() called — bypasses tenant validation. "
            "Prefer using fetch/execute methods for tenant-safe operations."
        )
        return await self.pool.acquire()

    async def release(self, connection: Any) -> None:
        return await self.pool.release(connection)

    # ── Metadata ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return getattr(self.pool, "size", 0)

    def __repr__(self) -> str:
        return f"TenantScopedPool(tenant={self.tenant_id}, pool={self.pool!r})"
