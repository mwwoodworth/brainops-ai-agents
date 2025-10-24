#!/usr/bin/env python3
"""
Fixed version of unified_memory_manager with proper type handling
"""

import os
import json
import asyncpg
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

class UnifiedMemoryManager:
    """Fixed unified memory system with proper type handling"""

    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
            "database": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
            "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
            "port": int(os.getenv("DB_PORT", 5432))
        }
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=1,
                max_size=10,
                command_timeout=60,
                max_inactive_connection_lifetime=300
            )
            logger.info("✅ Unified memory system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory pool: {e}")
            raise

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

    async def store(
        self,
        key: str,
        value: Any,
        context_type: str = "system",
        context_id: str = None,
        importance: int = 5,
        expires_in_hours: Optional[int] = None,
        tags: List[str] = None
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

                # Store memory with ON CONFLICT handling
                await conn.execute("""
                    INSERT INTO unified_memory (
                        context_type, context_id, key, value,
                        importance, tags, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (context_type, context_id, key)
                    DO UPDATE SET
                        value = $4,
                        importance = $5,
                        tags = $6,
                        expires_at = $7,
                        updated_at = NOW(),
                        access_count = unified_memory.access_count + 1
                """, context_type, context_id, key, json.dumps(value),
                    importance, tags or [], expires_at)

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
    ) -> List[Dict]:
        """Recall memories with proper type handling"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                # Build query with proper type casting
                sql = """
                    SELECT
                        id::text as id,
                        context_type,
                        context_id,
                        key,
                        value,
                        importance,
                        tags,
                        created_at,
                        updated_at,
                        accessed_at,
                        access_count
                    FROM unified_memory
                    WHERE importance >= $1
                """
                params = [min_importance]
                param_count = 1

                # Add context filters with proper string comparison
                if context_type:
                    param_count += 1
                    sql += f" AND context_type = ${param_count}"
                    params.append(context_type)

                if context_id:
                    param_count += 1
                    # Cast to text for comparison
                    sql += f" AND context_id::text = ${param_count}::text"
                    params.append(str(context_id))

                if query:
                    param_count += 1
                    sql += f" AND (key ILIKE ${param_count} OR value::text ILIKE ${param_count})"
                    params.append(f"%{query}%")

                # Add ordering and limit
                sql += """
                    ORDER BY importance DESC, accessed_at DESC
                    LIMIT ${}
                """.format(param_count + 1)
                params.append(limit)

                # Execute query
                rows = await conn.fetch(sql, *params)

                # Update access timestamps
                if rows:
                    memory_ids = [row['id'] for row in rows]
                    await conn.execute("""
                        UPDATE unified_memory
                        SET accessed_at = NOW(),
                            access_count = access_count + 1
                        WHERE id = ANY($1::uuid[])
                    """, memory_ids)

                # Parse and return memories
                memories = []
                for row in rows:
                    memory = dict(row)
                    try:
                        memory['value'] = json.loads(memory['value'])
                    except:
                        pass  # Keep as string if not JSON
                    memories.append(memory)

                logger.info(f"✅ Recalled {len(memories)} memories")
                return memories

            except Exception as e:
                logger.error(f"❌ Failed to recall memories: {e}")
                return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics with proper aggregation"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT context_type) as unique_systems,
                        COUNT(DISTINCT context_id) as unique_contexts,
                        AVG(importance)::float as avg_importance,
                        MAX(access_count) as max_access_count,
                        COUNT(DISTINCT tags) as unique_tags
                    FROM unified_memory
                    WHERE expires_at IS NULL OR expires_at > NOW()
                """)

                return dict(stats) if stats else {}

            except Exception as e:
                logger.error(f"❌ Failed to get memory stats: {e}")
                return {}

    async def cleanup_expired(self) -> int:
        """Clean up expired memories"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute("""
                    DELETE FROM unified_memory
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