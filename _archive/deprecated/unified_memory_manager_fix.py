#!/usr/bin/env python3
"""
Fixed version of unified_memory_manager with proper type handling
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class UnifiedMemoryManager:
    """Fixed unified memory system with proper type handling"""

    def __init__(self):
        # Try individual DB vars first, fall back to DATABASE_URL
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME', 'postgres')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_port = os.getenv('DB_PORT', '5432')

        # Fallback to DATABASE_URL if individual vars not set (Render provides DATABASE_URL)
        if not all([db_host, db_user, db_password]):
            database_url = os.getenv('DATABASE_URL', '')
            if database_url:
                parsed = urlparse(database_url)
                db_host = parsed.hostname or db_host
                db_name = parsed.path.lstrip('/') if parsed.path else db_name
                db_user = parsed.username or db_user
                db_password = parsed.password or db_password
                db_port = str(parsed.port) if parsed.port else db_port
                logger.info(f"unified_memory_manager_fix: Parsed DATABASE_URL: host={db_host}, db={db_name}")
            else:
                raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

        self.db_config = {
            "host": db_host,
            "database": db_name,
            "user": db_user,
            "password": db_password,
            "port": int(db_port)
        }
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool - USE SHARED POOL"""
        try:
            # CRITICAL: Use the shared pool from database/async_connection.py
            # instead of creating our own pool to prevent pool exhaustion
            from database.async_connection import get_pool, using_fallback

            try:
                shared_pool = get_pool()
                if not using_fallback():
                    self.pool = shared_pool
                    logger.info("✅ Unified memory using SHARED database pool")
                else:
                    logger.warning("⚠️ Shared pool using fallback, unified memory DB disabled")
                    self.pool = None
            except RuntimeError:
                # Pool not yet initialized - this is OK, app.py will initialize it
                logger.warning("⚠️ Shared pool not initialized yet, unified memory DB disabled")
                self.pool = None

        except Exception as e:
            logger.error(f"❌ Failed to initialize unified memory pool: {e}")
            self.pool = None

    async def close(self):
        """Close connection pool - DO NOT close shared pool"""
        # NOTE: We don't close the pool here since it's a SHARED pool
        # managed by database/async_connection.py
        # Only set our reference to None
        self.pool = None

    async def store(
        self,
        key: str,
        value: Any,
        context_type: str = "system",
        context_id: str = None,
        importance: int = 5,
        expires_in_hours: Optional[int] = None,
        tags: list[str] = None
    ) -> bool:
        """Store memory with proper type conversion"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                # Convert context_id to string to match varchar type
                if context_id is None:
                    context_id = "default"
                else:
                    context_id = str(context_id)[:255]  # Ensure it fits varchar(255)

                # Calculate expiration
                expires_at = None
                if expires_in_hours:
                    expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

                # Store memory with ON CONFLICT handling (using CANONICAL unified_ai_memory)
                await conn.execute("""
                    INSERT INTO unified_ai_memory (
                        memory_type, content, source_system, source_agent,
                        created_by, importance_score, tags, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (content_hash)
                    DO UPDATE SET
                        content = $2,
                        importance_score = $6,
                        tags = $7,
                        expires_at = $8,
                        updated_at = NOW(),
                        access_count = unified_ai_memory.access_count + 1
                """, 'semantic', json.dumps(value), context_type, context_id, key,
                    importance / 10.0, tags or [], expires_at)

                logger.info(f"✅ Stored memory: {key} in {context_type}/{context_id}")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to store memory: {e}")
                return False

    async def recall(
        self,
        query: str = None,
        context_type: str = None,
        context_id: str = None,
        limit: int = 10,
        min_importance: int = 0
    ) -> list[dict]:
        """Recall memories with proper type handling"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                # Build query with proper type casting (using CANONICAL unified_ai_memory)
                sql = """
                    SELECT
                        id::text as id,
                        memory_type,
                        context_id,
                        source_system,
                        content,
                        importance_score,
                        tags,
                        created_at,
                        updated_at,
                        last_accessed,
                        access_count
                    FROM unified_ai_memory
                    WHERE importance_score >= $1
                """
                params = [min_importance / 10.0]  # Convert 0-10 to 0-1 scale
                param_count = 1

                # Add context filters with proper string comparison
                if context_type:
                    param_count += 1
                    sql += f" AND source_system = ${param_count}"
                    params.append(context_type)

                if context_id:
                    param_count += 1
                    # Cast to text for comparison
                    sql += f" AND context_id::text = ${param_count}::text"
                    params.append(str(context_id))

                if query:
                    param_count += 1
                    sql += f" AND (search_text ILIKE ${param_count} OR content::text ILIKE ${param_count})"
                    params.append(f"%{query}%")

                # Add ordering and limit
                sql += f"""
                    ORDER BY importance_score DESC, last_accessed DESC
                    LIMIT ${param_count + 1}
                """
                params.append(limit)

                # Execute query
                rows = await conn.fetch(sql, *params)

                # Update access timestamps
                if rows:
                    memory_ids = [row['id'] for row in rows]
                    await conn.execute("""
                        UPDATE unified_ai_memory
                        SET last_accessed = NOW(),
                            access_count = access_count + 1
                        WHERE id = ANY($1::uuid[])
                    """, memory_ids)

                # Parse and return memories
                memories = []
                for row in rows:
                    memory = dict(row)
                    try:
                        memory['value'] = json.loads(memory['value'])
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.debug("Failed to decode memory value: %s", exc)
                    memories.append(memory)

                logger.info(f"✅ Recalled {len(memories)} memories")
                return memories

            except Exception as e:
                logger.error(f"❌ Failed to recall memories: {e}")
                return []

    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics with proper aggregation (using CANONICAL unified_ai_memory)"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT source_system) as unique_systems,
                        COUNT(DISTINCT context_id) as unique_contexts,
                        AVG(importance_score)::float as avg_importance,
                        MAX(access_count) as max_access_count,
                        COUNT(DISTINCT memory_type) as unique_types
                    FROM unified_ai_memory
                    WHERE expires_at IS NULL OR expires_at > NOW()
                """)

                return dict(stats) if stats else {}

            except Exception as e:
                logger.error(f"❌ Failed to get memory stats: {e}")
                return {}

    async def cleanup_expired(self) -> int:
        """Clean up expired memories (using CANONICAL unified_ai_memory)"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute("""
                    DELETE FROM unified_ai_memory
                    WHERE expires_at IS NOT NULL
                    AND expires_at < NOW()
                """)

                deleted_count = int(result.split()[-1])
                if deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {deleted_count} expired memories")

                return deleted_count

            except Exception as e:
                logger.error(f"❌ Failed to cleanup expired memories: {e}")
                return 0

# Test the fixed version
async def test_fixed_memory():
    """Test the fixed memory manager"""
    manager = UnifiedMemoryManager()

    try:
        await manager.initialize()

        # Test store
        success = await manager.store(
            key="test_fix",
            value={"test": "Fixed version works"},
            context_type="system",
            context_id="hotfix_test",
            importance=10
        )
        print(f"Store test: {'✅' if success else '❌'}")

        # Test recall
        memories = await manager.recall(
            context_type="system",
            context_id="hotfix_test"
        )
        print(f"Recall test: ✅ Found {len(memories)} memories")

        # Test stats
        stats = await manager.get_stats()
        print(f"Stats test: ✅ {stats}")

        await manager.close()

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_fixed_memory())
