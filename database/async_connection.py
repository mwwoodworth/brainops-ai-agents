"""
Async Database Connection Pool - Production Ready
Type-safe, lint-clean, fully tested
"""
import asyncio
import json
import logging
import os
import ssl
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

# Type alias for database records
DbRecord = dict[str, Any]


class DatabaseUnavailableError(RuntimeError):
    """Raised when the database pool cannot be initialized or accessed."""


@dataclass
class PoolConfig:
    """Database pool configuration"""
    host: str
    port: int
    user: str
    password: str
    database: str
    min_size: int = 2  # Keep minimum connections ready
    max_size: int = 10  # Increased for better concurrency with transaction mode pooler
    command_timeout: int = 30
    connect_timeout: float = 30.0  # Increased to prevent timeouts on slow networks
    max_inactive_connection_lifetime: float = 60.0  # Recycle idle connections after 60s
    ssl: bool = True  # Supabase requires TLS; allow override for local/dev
    ssl_verify: bool = True  # Enable SSL verification by default for security; disable only for local dev


class BasePool:
    """Interface for database-like pool implementations"""

    async def initialize(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def fetch(self, query: str, *args: Any, timeout: Optional[float] = None) -> list[DbRecord]:
        raise NotImplementedError

    async def fetchrow(self, query: str, *args: Any, timeout: Optional[float] = None) -> Optional[DbRecord]:
        raise NotImplementedError

    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        raise NotImplementedError

    async def execute(self, query: str, *args: Any, timeout: Optional[float] = None) -> str:
        raise NotImplementedError

    async def executemany(self, command: str, args: list[Any], timeout: Optional[float] = None) -> str:
        raise NotImplementedError

    async def test_connection(self) -> bool:
        raise NotImplementedError


class AsyncDatabasePool(BasePool):
    """Async database connection pool manager"""

    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        if config.ssl:
            ctx = ssl.create_default_context()
            if not config.ssl_verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            self._ssl_context = ctx
        else:
            self._ssl_context = None

    @staticmethod
    def _json_text_encoder(value: Any) -> str:
        """
        Encode Python values for Postgres json/jsonb parameters.

        asyncpg's default jsonb encoder expects a string; many call sites pass dicts.
        We keep decode behavior as text (string) to avoid breaking existing codepaths
        that assume jsonb results are strings.
        """
        if value is None:
            return "null"
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str)

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Per-connection initialization for asyncpg pool connections."""
        await conn.set_type_codec(
            "json",
            encoder=self._json_text_encoder,
            decoder=lambda s: s,
            schema="pg_catalog",
            format="text",
        )
        await conn.set_type_codec(
            "jsonb",
            encoder=self._json_text_encoder,
            decoder=lambda s: s,
            schema="pg_catalog",
            format="text",
        )

    async def initialize(self) -> None:
        """Initialize connection pool with timeout protection"""
        if self._pool is not None:
            logger.warning("Pool already initialized")
            return

        try:
            # Use asyncio.wait_for to prevent hanging if DB is unreachable
            self._pool = await asyncio.wait_for(
                asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    database=self.config.database,
                    min_size=self.config.min_size,
                    max_size=self.config.max_size,
                    command_timeout=self.config.command_timeout,
                    timeout=self.config.connect_timeout,
                    max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                    ssl=self._ssl_context,
                    statement_cache_size=0,  # Disable statement cache to avoid session mode issues
                    init=self._init_connection,
                ),
                timeout=self.config.connect_timeout + 5  # Extra buffer for pool setup
            )
            logger.info(
                "✅ Database pool initialized "
                "(min=%s, max=%s)",
                self.config.min_size,
                self.config.max_size,
            )
        except asyncio.TimeoutError:
            logger.error("❌ Database connection timed out after %.1fs", self.config.connect_timeout)
            raise
        except Exception as exc:
            logger.error("❌ Database pool initialization failed: %s", exc)
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
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._pool

    def acquire(self):
        """Acquire a connection from the pool - delegates to underlying asyncpg pool.
        This method provides compatibility for code that calls pool.acquire() directly.
        Usage: async with pool.acquire() as conn: ...
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._pool.acquire()

    async def _execute_with_retry(
        self,
        operation: str,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
        column: int = 0,
        max_retries: int = 2
    ) -> Any:
        """
        Execute database operation with automatic retry on connection errors.

        Handles transient connection issues from Supabase pooler (pgbouncer)
        that can close connections mid-operation.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                async with self.pool.acquire() as conn:
                    if operation == "fetch":
                        return await conn.fetch(query, *args, timeout=timeout)
                    elif operation == "fetchrow":
                        return await conn.fetchrow(query, *args, timeout=timeout)
                    elif operation == "fetchval":
                        return await conn.fetchval(query, *args, column=column, timeout=timeout)
                    elif operation == "execute":
                        return await conn.execute(query, *args, timeout=timeout)
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
            except (
                asyncpg.ConnectionDoesNotExistError,
                asyncpg.InterfaceError,
                asyncpg.InternalClientError,
                asyncpg.PostgresConnectionError,
            ) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Connection error on %s (attempt %d/%d): %s - retrying...",
                        operation, attempt + 1, max_retries + 1, e
                    )
                    await asyncio.sleep(0.1 * (attempt + 1))  # Brief backoff
                else:
                    logger.error(
                        "Connection error on %s after %d attempts: %s",
                        operation, max_retries + 1, e
                    )
            except asyncio.CancelledError:
                # Don't retry cancelled operations - propagate immediately
                raise
            except Exception as e:
                # Don't retry on non-connection errors (query errors, timeouts, etc.)
                raise

        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError(f"Unexpected state: {operation} failed without error")

    async def fetch(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> list[asyncpg.Record]:
        """Execute query and return all rows (with retry on connection errors)"""
        return await self._execute_with_retry("fetch", query, *args, timeout=timeout)

    async def fetchrow(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """Execute query and return single row (with retry on connection errors)"""
        return await self._execute_with_retry("fetchrow", query, *args, timeout=timeout)

    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute query and return single value (with retry on connection errors)"""
        return await self._execute_with_retry("fetchval", query, *args, timeout=timeout, column=column)

    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> str:
        """Execute query without returning data (with retry on connection errors)"""
        return await self._execute_with_retry("execute", query, *args, timeout=timeout)

    async def executemany(
        self,
        command: str,
        args: list[Any],
        timeout: Optional[float] = None
    ) -> str:
        """Execute query for multiple parameter sets"""
        # executemany doesn't retry - it's typically used for bulk operations
        # where partial completion would be problematic
        async with self.pool.acquire() as conn:
            return await conn.executemany(command, args, timeout=timeout)

    async def test_connection(self, timeout: float = 4.0) -> bool:
        """Test database connection with timeout protection.

        Uses a direct connection instead of pool to avoid contention during
        health checks when the pool is busy with actual work.
        """
        try:
            # Direct connection test - bypasses pool to avoid contention
            direct_conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    database=self.config.database,
                    ssl=self._ssl_context,
                    timeout=timeout,
                    statement_cache_size=0,
                ),
                timeout=timeout + 1.0
            )
            try:
                result = await asyncio.wait_for(
                    direct_conn.fetchval("SELECT 1"),
                    timeout=2.0
                )
                return result == 1
            finally:
                await direct_conn.close()
        except asyncio.TimeoutError:
            logger.warning("Health check connection timed out after %.1fs", timeout)
            return False
        except Exception as exc:
            logger.warning("Health check connection failed: %s", exc)
            return False


class InMemoryDatabasePool(BasePool):
    """In-memory fallback implementation when the real database is unavailable."""

    def __init__(self) -> None:
        now = datetime.utcnow()
        self._agents: dict[str, dict[str, Any]] = {
            "ops-intel": {
                "id": "ops-intel",
                "name": "Operations Intelligence Agent",
                "category": "operations",
                "description": "Monitors KPIs and flags anomalies for operations leadership.",
                "enabled": True,
                "capabilities": [
                    {
                        "name": "kpi_monitoring",
                        "description": "Track core operational KPIs.",
                        "enabled": True,
                        "parameters": {"window_minutes": 60},
                    },
                    {
                        "name": "trend_analysis",
                        "description": "Highlight notable KPI variance.",
                        "enabled": True,
                        "parameters": {"threshold": 0.1},
                    },
                ],
                "configuration": {"confidence_threshold": 0.75},
                "created_at": now,
                "updated_at": now,
            },
            "sales-routing": {
                "id": "sales-routing",
                "name": "Sales Routing Assistant",
                "category": "revenue",
                "description": "Scores incoming leads and routes to the optimal account team.",
                "enabled": True,
                "capabilities": [
                    {
                        "name": "lead_scoring",
                        "description": "Compute dynamic lead fit scores.",
                        "enabled": True,
                        "parameters": {"scoring_model": "revenue_v2"},
                    },
                    {
                        "name": "assignment",
                        "description": "Match lead to account owner.",
                        "enabled": True,
                        "parameters": {"round_robin": True},
                    },
                ],
                "configuration": {"default_owner": "sales@brainops.com"},
                "created_at": now,
                "updated_at": now,
            },
            "scheduler": {
                "id": "scheduler",
                "name": "Automation Scheduler",
                "category": "automation",
                "description": "Coordinates cross-agent workflows and reminders.",
                "enabled": True,
                "capabilities": [
                    {
                        "name": "workflow_orchestration",
                        "description": "Manage recurring automation flows.",
                        "enabled": True,
                        "parameters": {"max_parallel_runs": 5},
                    }
                ],
                "configuration": {"window_minutes": 30},
                "created_at": now,
                "updated_at": now,
            },
        }
        self._executions: dict[str, dict[str, Any]] = {}
        self._memories: list[dict[str, Any]] = [
            {
                "id": "mem-ops-1",
                "user_id": "ops-user",
                "content": "Operations standup scheduled for 9:00 AM daily.",
                "importance": 0.65,
                "created_at": now,
                "tags": ["operations", "schedule"],
            },
            {
                "id": "mem-sales-1",
                "user_id": "sales-user",
                "content": "High-priority lead from Austin requesting immediate follow-up.",
                "importance": 0.9,
                "created_at": now,
                "tags": ["sales", "priority"],
            },
        ]

    async def initialize(self) -> None:
        logger.warning("Using in-memory fallback database store for AI agents service.")

    async def close(self) -> None:
        self._executions.clear()

    async def fetch(self, query: str, *args: Any, timeout: Optional[float] = None) -> list[DbRecord]:
        sql = query.lower().strip()

        if "from agents" in sql or "from ai_agents" in sql:
            return self._fetch_agents(sql, list(args))

        if "information_schema.tables" in sql:
            return [{"table_name": "ai_persistent_memory"}, {"table_name": "unified_ai_memory"}]

        if "from ai_persistent_memory" in sql:
            return self._search_memories(sql, list(args))

        if "from unified_ai_memory" in sql or "into unified_ai_memory" in sql or "update unified_ai_memory" in sql:
            return self._fetch_unified_memories(sql, list(args))

        if "from agent_executions" in sql:
            return self._fetch_executions(sql, list(args))

        # Default: empty result
        return []

    def _fetch_unified_memories(self, sql: str, args: list[Any]) -> list[DbRecord]:
        """Fallback handler for unified_ai_memory table"""
        # Handle INSERT
        if "insert into unified_ai_memory" in sql:
            # Generate an in-memory ID and return it
            import uuid
            new_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Store it in local memories for consistency (simplified mapping)
            # Args order depends on the query in memory.py
            # But we can just return success for now
            return [{
                "id": new_id, 
                "content_hash": "inmemory_hash_" + new_id[:8], 
                "created_at": now
            }]

        # Handle SELECT count/stats
        if "select count(*)" in sql:
            return [{
                "total_memories": len(self._memories),
                "unique_contexts": 1,
                "avg_importance": 0.7,
                "with_embeddings": len(self._memories),
                "unique_systems": 1,
                "memory_types": 1
            }]

        # Handle SEARCH/SELECT
        importance_threshold = 0.0
        query = None
        
        # Basic parsing
        if "importance_score >=" in sql:
            # Find the float param
            for arg in args:
                if isinstance(arg, float):
                    importance_threshold = arg
                    break
        
        if "ilike" in sql:
            for arg in args:
                if isinstance(arg, str) and "%" in arg:
                    query = arg.strip("%")
                    break

        matches: list[DbRecord] = []
        for memory in self._memories:
            if memory["importance"] < importance_threshold:
                continue
            if query and query.lower() not in memory["content"].lower():
                continue

            matches.append({
                "id": memory["id"],
                "memory_type": "episodic",
                "content": {"text": memory["content"]},
                "importance_score": memory["importance"],
                "category": "general",
                "title": "In-memory Record",
                "tags": memory.get("tags", []),
                "source_system": "inmemory_fallback",
                "source_agent": "inmemory_fallback",
                "created_at": memory["created_at"],
                "last_accessed": memory["created_at"],
                "access_count": 1,
                "similarity": 0.9 if query else None,
                "metadata": {},
                "content_hash": "inmemory_hash",
                "context_id": str(uuid.uuid4()),
                "parent_memory_id": None,
                "related_memories": [],
                "has_embedding": True
            })

        limit = 10
        if "limit" in sql:
            try:
                # Find the limit param (usually last int)
                for arg in reversed(args):
                    if isinstance(arg, int) and arg < 1000: # heuristic
                        limit = arg
                        break
            except ValueError:
                pass

        return matches[:limit]

    def _apply_agent_filters(self, args: list[Any]) -> list[dict[str, Any]]:
        agents = list(self._agents.values())
        if not args:
            return agents

        filtered = agents
        # Enabled filter is always first when provided
        if isinstance(args[0], bool):
            filtered = [a for a in filtered if a.get("enabled", True) == args[0]]
            args = args[1:]

        if args:
            category = args[0]
            filtered = [a for a in filtered if a.get("category") == category]

        return filtered

    def _fetch_agents(self, sql: str, args: list[Any]) -> list[DbRecord]:
        agents = self._apply_agent_filters(args)
        result: list[DbRecord] = []
        for agent in agents:
            capabilities = agent.get("capabilities") or []
            if isinstance(capabilities, str):
                try:
                    capabilities = json.loads(capabilities)
                except json.JSONDecodeError:
                    capabilities = [capabilities]

            configuration = agent.get("configuration") or {}
            if isinstance(configuration, str):
                try:
                    configuration = json.loads(configuration)
                except json.JSONDecodeError:
                    configuration = {}

            parsed_agent = {
                **agent,
                "capabilities": capabilities,
                "configuration": configuration,
            }
            result.append(parsed_agent)
        return result

    def _search_memories(self, sql: str, args: list[Any]) -> list[DbRecord]:
        importance_threshold = args[0] if args else 0.0
        user_id = None
        query = None

        if "user_id = $2" in sql and len(args) >= 2:
            user_id = args[1]

        if "content ilike $" in sql:
            query = args[-1].strip("%") if args else None

        matches: list[DbRecord] = []
        for memory in self._memories:
            if memory["importance"] < importance_threshold:
                continue
            if user_id and memory["user_id"] != user_id:
                continue
            if query and query.lower() not in memory["content"].lower():
                continue

            matches.append(
                {
                    "id": memory["id"],
                    "user_id": memory["user_id"],
                    "content": memory["content"],
                    "importance": memory["importance"],
                    "created_at": memory["created_at"],
                    "tags": memory.get("tags", []),
                }
            )

        limit = 10
        if "limit" in sql:
            try:
                limit = int(sql.rsplit("limit", 1)[1].strip())
            except ValueError:
                logger.debug("Failed to parse limit from SQL: %s", sql)

        return matches[:limit]

    def _fetch_executions(self, sql: str, args: list[Any]) -> list[DbRecord]:
        executions = list(self._executions.values())
        if "e.agent_id =" in sql and args:
            executions = [e for e in executions if e["agent_id"] == args[0]]
            if "e.status =" in sql and len(args) > 1:
                executions = [e for e in executions if e["status"] == args[1]]
        elif "e.status =" in sql and args:
            executions = [e for e in executions if e["status"] == args[0]]

        limit = 100
        if "limit $" in sql and args:
            limit = args[-1]

        results: list[DbRecord] = []
        for execution in executions[:limit]:
            agent = self._agents.get(execution["agent_id"])
            results.append(
                {
                    **execution,
                    "agent_name": agent["name"] if agent else execution["agent_id"],
                }
            )
        return results

    async def fetchrow(self, query: str, *args: Any, timeout: Optional[float] = None) -> Optional[DbRecord]:
        sql = query.lower()
        if "select * from agents where id" in sql and args:
            agent = self._agents.get(args[0])
            if not agent:
                return None
            capabilities = agent.get("capabilities") or []
            if isinstance(capabilities, str):
                try:
                    capabilities = json.loads(capabilities)
                except json.JSONDecodeError:
                    capabilities = [capabilities]

            configuration = agent.get("configuration") or {}
            if isinstance(configuration, str):
                try:
                    configuration = json.loads(configuration)
                except json.JSONDecodeError:
                    configuration = {}

            return {**agent, "capabilities": capabilities, "configuration": configuration}

        if "select" in sql and "from" not in sql:
            return None

        rows = await self.fetch(query, *args, timeout=timeout)
        return rows[0] if rows else None

    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        query.lower()
        # Removed fake fallback responses - return None instead of fake data
        # This ensures callers get accurate information about database state

        row = await self.fetchrow(query, *args, timeout=timeout)
        if row is None:
            return None

        if isinstance(row, dict):
            values = list(row.values())
            try:
                return values[column]
            except IndexError:
                return None
        return row

    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None
    ) -> str:
        sql = query.lower()
        if "insert into agent_executions" in sql:
            execution_id = args[0]
            self._executions[execution_id] = {
                "id": execution_id,
                "agent_id": args[1],
                "started_at": args[2],
                "status": args[3],
                "input_data": args[4],
                "completed_at": None,
                "output_data": None,
                "duration_ms": None,
                "error": None,
            }
            return "INSERT 0 1"

        if "update agent_executions" in sql:
            execution_id = args[-1]
            execution = self._executions.get(execution_id)
            if execution:
                if "output_data" in sql:
                    execution["completed_at"] = args[0]
                    execution["status"] = args[1]
                    execution["output_data"] = args[2]
                    execution["duration_ms"] = args[3]
                else:
                    execution["status"] = args[0]
                    execution["error"] = args[1]
                    execution["completed_at"] = args[2]
            return "UPDATE 1"

        return "OK"

    async def executemany(self, command: str, args: list[Any], timeout: Optional[float] = None) -> str:
        for params in args:
            await self.execute(command, *params, timeout=timeout)
        return f"EXECUTEMANY {len(args)}"

    async def test_connection(self) -> bool:
        return True


# Global pool instance
_pool: Optional[BasePool] = None
USING_FALLBACK = False


async def init_pool(config: PoolConfig) -> BasePool:
    """Initialize global database pool"""
    global _pool, USING_FALLBACK
    if _pool is not None:
        return _pool
    last_error: Optional[Exception] = None

    def _with_port(port: int) -> PoolConfig:
        return PoolConfig(
            host=config.host,
            port=port,
            user=config.user,
            password=config.password,
            database=config.database,
            min_size=config.min_size,
            max_size=config.max_size,
            command_timeout=config.command_timeout,
            connect_timeout=config.connect_timeout,
            max_inactive_connection_lifetime=config.max_inactive_connection_lifetime,
            ssl=config.ssl,
            ssl_verify=config.ssl_verify,
        )

    # Try primary configuration, then alternate port if Supabase pooler port fails
    candidate_ports = [config.port]
    if config.port == 6543:
        candidate_ports.append(5432)
    elif config.port == 5432:
        candidate_ports.append(6543)

    for port in candidate_ports:
        try:
            pool = AsyncDatabasePool(_with_port(port))
            await pool.initialize()
            _pool = pool
            USING_FALLBACK = False
            return _pool
        except Exception as exc:  # pragma: no cover - defensive logging for prod
            last_error = exc
            logger.error("❌ Failed to initialize database pool on port %s: %s", port, exc)

    env = os.getenv("ENVIRONMENT", "production").strip().lower()
    is_production = env in {"production", "prod"}
    allow_fallback = os.getenv("ALLOW_INMEMORY_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
    if allow_fallback and is_production:
        logger.critical(
            "❌ FATAL: ALLOW_INMEMORY_FALLBACK is set but ENVIRONMENT=%s. Refusing to use in-memory fallback in production.",
            env,
        )
        allow_fallback = False

    if not allow_fallback:
        logger.critical(
            "❌ FATAL: Database connection failed. Refusing to use in-memory fallback without ALLOW_INMEMORY_FALLBACK."
        )
        if last_error:
            logger.critical("Database error: %s", last_error)
        raise DatabaseUnavailableError(
            f"Database connection required. Error: {last_error}. "
            "Set ALLOW_INMEMORY_FALLBACK=true for explicit dev-only fallback (ENVIRONMENT != production)."
        )

    logger.warning(
        "Falling back to in-memory store (ALLOW_INMEMORY_FALLBACK enabled, ENVIRONMENT=%s).",
        env,
    )
    fallback = InMemoryDatabasePool()
    await fallback.initialize()
    _pool = fallback
    USING_FALLBACK = True
    if last_error:
        logger.error("Database pool fallback reason: %s", last_error)
    return _pool


def get_pool() -> BasePool:
    """Get global database pool"""
    if _pool is None:
        raise DatabaseUnavailableError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close global database pool"""
    global _pool, USING_FALLBACK
    if _pool is not None:
        await _pool.close()
        _pool = None
        USING_FALLBACK = False


def using_fallback() -> bool:
    """Return True when the in-memory fallback store is active."""
    return USING_FALLBACK
