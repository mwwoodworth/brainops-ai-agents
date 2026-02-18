
import asyncio
import logging
from typing import Any, Optional

from database.simple_sql_parser import SimpleSQLParser

logger = logging.getLogger(__name__)

# Sentinel for tenant_id that's missing or clearly invalid
_INVALID_TENANT_IDS = {"", "null", "None", "undefined"}


class TenantScopedPool:
    """
    A wrapper around the asyncpg pool that enforces tenant isolation at BOTH
    the application layer (SQL parser validation) AND the database layer
    (SET LOCAL app.current_tenant_id inside every transaction).

    Every query method:
      1. Validates mutation safety via SimpleSQLParser
      2. Acquires a raw connection from the underlying asyncpg pool
      3. Starts an explicit transaction
      4. Calls SET LOCAL app.current_tenant_id = '<tenant_id>' (transaction-scoped)
      5. Executes the query
      6. Returns the result (transaction auto-commits on exit)

    This ensures that RLS policies using `current_tenant_id()` see the correct
    tenant for every single query — and the setting is automatically reverted
    when the transaction ends, preventing pool connection leakage.
    """

    def __init__(self, pool: Any, tenant_id: str):
        self.pool = pool
        self.tenant_id = tenant_id
        # Fail-closed: reject clearly invalid tenant IDs at construction time
        if not tenant_id or tenant_id in _INVALID_TENANT_IDS:
            raise ValueError(
                f"TenantScopedPool requires a valid tenant_id, got: {tenant_id!r}"
            )

    def _validate_query(self, query: str) -> None:
        """Parse query to enforce mutation safety (tenant_id in WHERE/INSERT)."""
        if ";" in query:
            raise ValueError("Tenant guardrail blocks multi-statement SQL (semicolon detected).")

        parser = SimpleSQLParser(query)
        parser.validate_mutation_safety("TENANT_ID")

    def _get_raw_pool(self):
        """Get the underlying asyncpg.Pool for direct connection acquisition."""
        # AsyncDatabasePool exposes .pool -> asyncpg.Pool
        raw = getattr(self.pool, "pool", None) or getattr(self.pool, "_pool", None)
        if raw is None:
            raise RuntimeError("Cannot access underlying asyncpg pool from TenantScopedPool.pool")
        return raw

    async def _execute_scoped(
        self,
        operation: str,
        query: str,
        *args: Any,
        column: int = 0,
        max_retries: int = 2,
    ) -> Any:
        """
        Execute a query inside a transaction with tenant context set.

        Retries on transient connection errors (same pattern as AsyncDatabasePool).
        """
        import asyncpg as _asyncpg

        raw_pool = self._get_raw_pool()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                async with raw_pool.acquire(timeout=10.0) as conn:
                    async with conn.transaction():
                        # Set tenant context (transaction-local → auto-reverts on commit/rollback)
                        await conn.execute(
                            "SELECT set_config('app.current_tenant_id', $1, true)",
                            self.tenant_id,
                        )
                        if operation == "fetch":
                            return await conn.fetch(query, *args)
                        elif operation == "fetchrow":
                            return await conn.fetchrow(query, *args)
                        elif operation == "fetchval":
                            return await conn.fetchval(query, *args, column=column)
                        elif operation == "execute":
                            return await conn.execute(query, *args)
                        else:
                            raise ValueError(f"Unknown operation: {operation}")

            except asyncio.TimeoutError as e:
                last_error = e
                logger.error(
                    "TenantScopedPool acquire timed out on %s (attempt %d/%d, tenant=%s)",
                    operation, attempt + 1, max_retries + 1, self.tenant_id[:8],
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                raise

            except (
                _asyncpg.ConnectionDoesNotExistError,
                _asyncpg.InterfaceError,
                _asyncpg.InternalClientError,
                _asyncpg.PostgresConnectionError,
            ) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Connection error on %s (attempt %d/%d, tenant=%s): %s",
                        operation, attempt + 1, max_retries + 1, self.tenant_id[:8], e,
                    )
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    logger.error(
                        "Connection error on %s after %d attempts (tenant=%s): %s",
                        operation, max_retries + 1, self.tenant_id[:8], e,
                    )

            except asyncio.CancelledError:
                raise

            except Exception:
                raise

        if last_error:
            raise last_error
        raise RuntimeError(f"Unexpected state: {operation} failed without error")

    # ── Core query methods ──────────────────────────────────────────────

    async def fetch(self, query: str, *args: Any) -> list:
        self._validate_query(query)
        return await self._execute_scoped("fetch", query, *args)

    async def fetchrow(self, query: str, *args: Any):
        self._validate_query(query)
        return await self._execute_scoped("fetchrow", query, *args)

    async def fetchval(self, query: str, *args: Any):
        self._validate_query(query)
        return await self._execute_scoped("fetchval", query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        self._validate_query(query)
        return await self._execute_scoped("execute", query, *args)

    async def executemany(self, query: str, args: list) -> None:
        self._validate_query(query)
        raw_pool = self._get_raw_pool()
        async with raw_pool.acquire(timeout=10.0) as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_tenant_id', $1, true)",
                    self.tenant_id,
                )
                return await conn.executemany(query, args)

    # ── Pass-through for connection-level operations ────────────────────

    async def acquire(self):
        """Acquire a raw connection with tenant context pre-set.

        The connection has app.current_tenant_id set at the session level.
        Callers MUST use transactions if they want isolation guarantees.
        """
        logger.warning(
            "TenantScopedPool.acquire() called — setting session-level tenant context. "
            "Prefer using fetch/execute methods for full transaction-scoped isolation."
        )
        raw_pool = self._get_raw_pool()
        conn = await raw_pool.acquire(timeout=10.0)
        try:
            # Session-level (false) so it persists for the connection lifetime
            await conn.execute(
                "SELECT set_config('app.current_tenant_id', $1, false)",
                self.tenant_id,
            )
        except Exception:
            await raw_pool.release(conn)
            raise
        return conn

    async def release(self, connection: Any) -> None:
        """Release a connection and clear tenant context."""
        try:
            # Clear tenant context to prevent leakage
            await connection.execute("RESET app.current_tenant_id")
        except Exception:
            pass  # Connection may already be closed
        raw_pool = self._get_raw_pool()
        return await raw_pool.release(connection)

    # ── Metadata ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return getattr(self.pool, "size", 0)

    def __repr__(self) -> str:
        return f"TenantScopedPool(tenant={self.tenant_id[:8]}..., pool={self.pool!r})"
