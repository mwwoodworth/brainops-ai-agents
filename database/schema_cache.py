"""
Schema Cache - Caches database schema information to avoid expensive information_schema queries.

The information_schema queries were taking 72.3 seconds on average with 37 calls.
Schema rarely changes, so caching with a 1-hour TTL dramatically reduces load.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    value: Any
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class SchemaCache:
    """
    Thread-safe cache for database schema queries.

    Dramatically reduces load from information_schema queries which were
    taking 72.3 seconds on average.
    """
    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    default_ttl_seconds: int = 3600  # 1 hour - schema rarely changes

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                logger.debug(f"Schema cache HIT: {key}")
                return entry.value
            if entry:
                # Clean up expired entry
                del self._cache[key]
                logger.debug(f"Schema cache EXPIRED: {key}")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set cached value with TTL"""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        async with self._lock:
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )
            logger.debug(f"Schema cache SET: {key} (TTL: {ttl}s)")

    async def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"Schema cache INVALIDATED: {key}")

    async def invalidate_all(self) -> None:
        """Clear all cached entries"""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Schema cache CLEARED: {count} entries")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            now = time.time()
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "keys": list(self._cache.keys())
            }


# Global schema cache instance
_schema_cache: Optional[SchemaCache] = None


def get_schema_cache() -> SchemaCache:
    """Get or create the global schema cache instance"""
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = SchemaCache()
    return _schema_cache


async def cached_schema_query(
    pool,
    cache_key: str,
    query: str,
    *args,
    ttl_seconds: int = 3600
) -> list[Any]:
    """
    Execute a schema query with caching.

    Use this for expensive information_schema queries that don't change often.

    Args:
        pool: Database pool
        cache_key: Unique key for this query
        query: SQL query to execute
        *args: Query parameters
        ttl_seconds: Cache TTL (default 1 hour)

    Returns:
        Query results (cached or fresh)
    """
    cache = get_schema_cache()

    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached

    # Execute query
    logger.info(f"Schema cache MISS - executing query: {cache_key}")
    start = time.time()
    result = await pool.fetch(query, *args)
    duration = time.time() - start

    # Convert to list of dicts for caching
    result_list = [dict(r) for r in result]

    # Cache the result
    await cache.set(cache_key, result_list, ttl_seconds)

    logger.info(f"Schema query completed in {duration:.2f}s, cached {len(result_list)} rows: {cache_key}")
    return result_list


async def get_cached_foreign_keys(pool, schema: str = "public") -> list[dict]:
    """
    Get foreign key relationships with caching.

    This query was taking 72+ seconds without caching.
    """
    cache_key = f"fk_relationships_{schema}"
    query = """
        SELECT
            tc.table_name as source_table,
            kcu.column_name as source_column,
            ccu.table_name as target_table,
            ccu.column_name as target_column,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = $1
        ORDER BY tc.table_name
    """
    return await cached_schema_query(pool, cache_key, query, schema)


async def get_cached_table_constraints(pool, schema: str, table: str) -> list[dict]:
    """
    Get table constraints with caching.
    """
    cache_key = f"constraints_{schema}_{table}"
    query = """
        SELECT
            constraint_name,
            constraint_type,
            table_name,
            table_schema
        FROM information_schema.table_constraints
        WHERE table_schema = $1 AND table_name = $2
    """
    return await cached_schema_query(pool, cache_key, query, schema, table)


async def get_cached_primary_keys(pool, schema: str, table: str) -> list[str]:
    """
    Get primary key columns with caching.
    """
    cache_key = f"pk_columns_{schema}_{table}"
    query = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = $1
            AND tc.table_name = $2
    """
    result = await cached_schema_query(pool, cache_key, query, schema, table)
    return [row['column_name'] for row in result]


async def get_cached_constraint_count(pool, constraint_type: str = "FOREIGN KEY") -> int:
    """
    Get count of constraints with caching.
    """
    cache_key = f"constraint_count_{constraint_type}"
    query = """
        SELECT COUNT(*) as count FROM information_schema.table_constraints
        WHERE constraint_type = $1
    """
    result = await cached_schema_query(pool, cache_key, query, constraint_type)
    return result[0]['count'] if result else 0


async def get_cached_tables(pool, schema: str = "public") -> list[dict]:
    """
    Get all tables in a schema with caching.
    """
    cache_key = f"tables_{schema}"
    query = """
        SELECT
            table_name,
            table_schema,
            table_type
        FROM information_schema.tables
        WHERE table_schema = $1
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    return await cached_schema_query(pool, cache_key, query, schema)


async def get_cached_columns(pool, schema: str, table: str) -> list[dict]:
    """
    Get columns for a table with caching.
    """
    cache_key = f"columns_{schema}_{table}"
    query = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        ORDER BY ordinal_position
    """
    return await cached_schema_query(pool, cache_key, query, schema, table)
