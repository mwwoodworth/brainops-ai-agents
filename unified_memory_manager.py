#!/usr/bin/env python3
"""
Unified Memory Manager - Enterprise Grade Memory System
Consolidates 53 chaotic memory tables into one intelligent system
"""

import asyncio
import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, ContextManager, Optional, Union

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor
from utils.embedding_provider import generate_embedding_sync

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


# Use shared connection pool - CRITICAL for preventing connection exhaustion
try:
    from database.sync_pool import get_sync_pool

    USING_SHARED_POOL = True
except ImportError:
    USING_SHARED_POOL = False
    logger.warning("⚠️ Shared sync_pool not available, falling back to direct connections")


class MemoryType(Enum):
    """Types of memory in the unified system"""

    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Short-term active memory
    META = "meta"  # Memory about memories


@dataclass
class Memory:
    """Unified memory structure"""

    memory_type: MemoryType
    content: dict[str, Any]
    source_system: str
    source_agent: str
    created_by: str
    importance_score: float = 0.5
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    context_id: Optional[str] = None
    parent_memory_id: Optional[str] = None
    related_memories: Optional[list[str]] = None
    expires_at: Optional[datetime] = None
    tenant_id: Optional[str] = None


class UnifiedMemoryManager:
    """Enterprise-grade unified memory management system"""

    def __init__(self, tenant_id: Optional[str] = None):
        self.embedding_cache = {}
        self.consolidation_threshold = 0.85  # Similarity threshold for consolidation
        self.tenant_id = tenant_id or os.getenv("TENANT_ID") or os.getenv("DEFAULT_TENANT_ID")
        self._pool = None
        self._init_pool()

    def _init_pool(self) -> None:
        """Initialize shared connection pool"""
        if USING_SHARED_POOL:
            try:
                self._pool = get_sync_pool()
                logger.info("✅ Unified memory using SHARED connection pool")
            except Exception as exc:
                logger.error("❌ Failed to get shared pool: %s", exc, exc_info=True)
                self._pool = None
        else:
            logger.warning("⚠️ Running without shared pool - connection exhaustion risk")

    def _get_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Get connection from shared pool"""
        if self._pool:
            return self._pool.get_connection()
        return None

    @property
    def conn(self) -> Optional[psycopg2.extensions.connection]:
        """Backward compatibility - get connection from pool for current operation"""
        if hasattr(self, "_current_conn") and self._current_conn:
            return self._current_conn
        return None

    def _get_cursor(self) -> ContextManager[Optional[RealDictCursor]]:
        """Get cursor from shared pool - backward compatible context manager"""
        from contextlib import contextmanager

        @contextmanager
        def cursor_context():
            if not self._pool:
                logger.error("❌ No connection pool available")
                yield None
                return

            with self._pool.get_connection() as conn:
                if not conn:
                    logger.error("❌ Failed to get connection from pool")
                    yield None
                    return
                self._current_conn = conn  # Store for backward compat
                # Disable autocommit so SET LOCAL + query share one transaction.
                # Supavisor transaction-mode pooling assigns backend connections
                # per-transaction, so autocommit (1 stmt = 1 txn) loses the GUC.
                prev_autocommit = conn.autocommit
                conn.autocommit = False
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    if self.tenant_id:
                        cursor.execute(
                            "SELECT set_config('app.current_tenant_id', %s, true)",
                            [self.tenant_id],
                        )
                    # Increase IVFFlat probes from default 1 to 10 for better recall.
                    # With lists=300, probes=10 searches ~3.3% of the index which
                    # balances query speed with result quality.
                    cursor.execute("SET LOCAL ivfflat.probes = 10")
                    yield cursor
                    conn.commit()
                    cursor.close()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.autocommit = prev_autocommit
                    self._current_conn = None

        return cursor_context()

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Union[bool, dict[str, Any], list[dict[str, Any]]]]:
        """Execute query using shared pool - returns results or None"""
        if not self._pool:
            logger.error("❌ No connection pool available")
            return None

        with self._pool.get_connection() as conn:
            if not conn:
                logger.error("❌ Failed to get connection from pool")
                return None
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, params)
                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall()
                else:
                    result = True
                cursor.close()
                return result
            except Exception as exc:
                logger.error("❌ Query execution failed: %s", exc, exc_info=True)
                return None

    def log_to_brain(self, system: str, action: str, data: dict[str, Any]) -> None:
        """Log significant events to the unified brain logs.

        NOTE: The unified_brain_logs table MUST already exist in the database.
        DDL (CREATE TABLE) was removed because the agent_worker role (app_agent_role)
        correctly does not have DDL permissions. The table is created by migrations,
        not at runtime.
        """
        try:
            import uuid

            with self._get_cursor() as cur:
                if not cur:
                    return

                cur.execute(
                    """
                    INSERT INTO unified_brain_logs (id, system, action, data, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                """,
                    (
                        str(uuid.uuid4()),
                        system,
                        action,
                        json.dumps(data, cls=CustomJSONEncoder),
                    ),
                )

        except Exception as exc:
            logger.warning("Failed to log to unified brain: %s", exc, exc_info=True)

    def store(self, memory: Memory) -> str:
        """Store a memory with deduplication and linking"""
        if not memory.tenant_id:
            # Try to use instance tenant_id if memory tenant_id is missing
            if self.tenant_id:
                memory.tenant_id = self.tenant_id
            else:
                raise ValueError("tenant_id is mandatory for memory storage")

        try:
            # Check for duplicates
            existing = self._find_duplicate(memory)
            if existing:
                # Reinforce existing memory instead of creating duplicate
                mem_id = self._reinforce_memory(existing["id"], memory)
                self.log_to_brain(
                    "memory_system",
                    "memory_reinforced",
                    {
                        "memory_id": mem_id,
                        "type": memory.memory_type.value,
                        "source": memory.source_system,
                    },
                )
                return mem_id

            # Generate embedding if we have content
            embedding = self._generate_embedding(memory.content)
            if embedding is None:
                logger.warning(
                    "Memory will be stored without embedding - semantic search unavailable for this memory"
                )

            # Find related memories
            related = self._find_related_memories(memory.content, memory.tenant_id, limit=5)

            # Store the memory using shared pool via backward-compat cursor
            with self._get_cursor() as cur:
                if not cur:
                    logger.error("❌ Failed to get cursor from pool")
                    return None

                query = """
                INSERT INTO unified_ai_memory (
                    memory_type, content, source_system, source_agent,
                    created_by, importance_score, tags, metadata,
                    context_id, parent_memory_id, related_memories,
                    expires_at, tenant_id, embedding, search_text
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s, %s, %s
                ) RETURNING id
                """

                search_text = self._generate_search_text(memory)

                # Serialize with custom encoder to handle datetime and Enum
                content_json = json.dumps(memory.content, cls=CustomJSONEncoder)
                metadata_json = json.dumps(memory.metadata or {}, cls=CustomJSONEncoder)

                cur.execute(
                    query,
                    (
                        memory.memory_type.value,
                        content_json,
                        memory.source_system,
                        memory.source_agent,
                        memory.created_by,
                        memory.importance_score,
                        memory.tags or [],
                        metadata_json,
                        memory.context_id,
                        memory.parent_memory_id,
                        [r["id"] for r in related] if related else None,
                        memory.expires_at,
                        memory.tenant_id,
                        embedding,
                        search_text,
                    ),
                )

                result = cur.fetchone()
                memory_id = result["id"] if result else None

                logger.info(f"✅ Stored memory {memory_id} ({memory.memory_type.value})")

                self.log_to_brain(
                    "memory_system",
                    "memory_stored",
                    {
                        "memory_id": memory_id,
                        "type": memory.memory_type.value,
                        "source": memory.source_system,
                        "tags": memory.tags,
                    },
                )

                return memory_id

        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            return None

    async def store_async(
        self,
        content: str,
        memory_type: str = "operational",
        category: str = None,
        metadata: dict = None,
    ) -> str:
        """Async wrapper for store to match app.py interface"""
        # Map string memory_type to Enum
        try:
            mem_type = MemoryType(memory_type.lower())
        except ValueError:
            mem_type = MemoryType.SEMANTIC  # Default

        # Construct Memory object
        mem = Memory(
            memory_type=mem_type,
            content={"text": content, "category": category}
            if isinstance(content, str)
            else content,
            source_system="api",
            source_agent="user",
            created_by="api_user",
            importance_score=0.5,
            tags=[category] if category else [],
            metadata=metadata or {},
            tenant_id=self.tenant_id,
        )

        # Run sync store in thread pool
        return await asyncio.to_thread(self.store, mem)

    def recall(
        self,
        query: Union[str, dict],
        tenant_id: str = None,
        context: Optional[str] = None,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> list[dict]:
        """Recall relevant memories with semantic search"""
        # Use instance tenant_id if not provided
        tenant_id = tenant_id or self.tenant_id

        if not tenant_id:
            raise ValueError("tenant_id is required for memory recall")

        try:
            # Generate query embedding
            if isinstance(query, str):
                query_content = {"query": query}
            else:
                query_content = query

            query_embedding = self._generate_embedding(query_content)

            # If embedding generation failed, fall back to keyword search
            if query_embedding is None:
                logger.warning("Embedding generation failed - falling back to keyword search")
                return self._keyword_search(query, tenant_id, context, limit, memory_type)

            with self._get_cursor() as cur:
                # Convert embedding list to pgvector string format [0.1,0.2,...]
                # psycopg2 serializes lists as {0.1,0.2,...} (array format) but
                # pgvector ::vector cast expects [0.1,0.2,...] (vector string format)
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                # Build the query
                base_query = """
                SELECT
                    id, memory_type, content, source_system, source_agent,
                    importance_score, access_count, created_at, tags, metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM unified_ai_memory
                WHERE tenant_id = %s
                  AND embedding IS NOT NULL
                """

                params = [embedding_str, tenant_id]

                # Add filters
                filters = []
                if context:
                    filters.append("context_id = %s")
                    params.append(context)

                if memory_type:
                    filters.append("memory_type = %s")
                    params.append(memory_type.value)

                if filters:
                    base_query += " AND " + " AND ".join(filters)

                # Order by relevance (similarity + importance)
                base_query += """
                ORDER BY (1 - (embedding <=> %s::vector)) * importance_score DESC
                LIMIT %s
                """
                params.extend([embedding_str, limit])

                cur.execute(base_query, params)
                memories = cur.fetchall()

                # Update access counts
                if memories:
                    memory_ids = [m["id"] for m in memories]
                    self._update_access_counts(memory_ids)

                    self.log_to_brain(
                        "memory_system",
                        "memory_recalled",
                        {
                            "query": query if isinstance(query, str) else "vector",
                            "count": len(memories),
                            "top_score": float(memories[0].get("similarity") or 0.0),
                        },
                    )

                logger.info(f"📚 Recalled {len(memories)} relevant memories")
                return [dict(m) for m in memories]

        except Exception as e:
            logger.error(f"❌ Failed to recall memories: {e}")
            return []

    async def search(self, query: str, limit: int = 10, memory_type: str = None) -> list[dict]:
        """Async wrapper for recall"""
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type.lower())
            except ValueError as exc:
                logger.debug("Invalid memory_type %s: %s", memory_type, exc)

        return await asyncio.to_thread(
            self.recall, query, self.tenant_id, limit=limit, memory_type=mem_type
        )

    async def recall_async(self, query, tenant_id=None, context=None, limit=10, memory_type=None):
        """Async wrapper for recall - use from async contexts to avoid blocking the event loop"""
        return await asyncio.to_thread(self.recall, query, tenant_id, context, limit, memory_type)

    async def synthesize_async(self, tenant_id=None, time_window=timedelta(hours=24)):
        """Async wrapper for synthesize"""
        return await asyncio.to_thread(self.synthesize, tenant_id, time_window)

    async def consolidate_async(self, aggressive=False):
        """Async wrapper for consolidate"""
        return await asyncio.to_thread(self.consolidate, aggressive)

    async def apply_retention_policy_async(self, tenant_id=None, aggressive=False):
        """Async wrapper for apply_retention_policy"""
        return await asyncio.to_thread(self.apply_retention_policy, tenant_id, aggressive)

    async def auto_garbage_collect_async(self, tenant_id=None, dry_run=False):
        """Async wrapper for auto_garbage_collect"""
        return await asyncio.to_thread(self.auto_garbage_collect, tenant_id, dry_run)

    async def get_stats_async(self, tenant_id=None):
        """Async wrapper for get_stats"""
        return await asyncio.to_thread(self.get_stats, tenant_id)

    async def store_memory_async(self, memory):
        """Async wrapper for store (Memory object) - use from async contexts"""
        return await asyncio.to_thread(self.store, memory)

    def _keyword_search(
        self,
        query: Union[str, dict],
        tenant_id: str,
        context: Optional[str] = None,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> list[dict]:
        """Fallback keyword search when embeddings are unavailable"""
        try:
            search_term = query if isinstance(query, str) else str(query)
            # Basic sanitization
            search_term = search_term.replace("%", "")

            with self._get_cursor() as cur:
                base_query = """
                SELECT
                    id, memory_type, content, source_system, source_agent,
                    importance_score, access_count, created_at, tags, metadata,
                    0.5 as similarity
                FROM unified_ai_memory
                WHERE tenant_id = %s
                AND (search_text ILIKE %s OR content::text ILIKE %s)
                """

                params = [tenant_id, f"%{search_term}%", f"%{search_term}%"]

                if context:
                    base_query += " AND context_id = %s"
                    params.append(context)

                if memory_type:
                    base_query += " AND memory_type = %s"
                    params.append(memory_type.value)

                base_query += " ORDER BY importance_score DESC LIMIT %s"
                params.append(limit)

                cur.execute(base_query, params)
                memories = cur.fetchall()

                if memories:
                    logger.info(
                        f"🔎 Keyword recall found {len(memories)} results for '{search_term}'"
                    )

                return [dict(m) for m in memories]

        except Exception as e:
            logger.error(f"❌ Keyword search failed: {e}")
            return []

    def synthesize(
        self, tenant_id: str = None, time_window: timedelta = timedelta(hours=24)
    ) -> list[dict]:
        """Synthesize insights from recent memories"""
        tenant_id = tenant_id or self.tenant_id
        if not tenant_id:
            raise ValueError("tenant_id is required for synthesis")

        try:
            with self._get_cursor() as cur:
                # Get recent high-importance memories
                query = """
                SELECT
                    memory_type, source_system, content, importance_score, tags
                FROM unified_ai_memory
                WHERE created_at > %s
                    AND importance_score > 0.7
                    AND tenant_id = %s
                ORDER BY importance_score DESC
                LIMIT 100
                """

                cutoff_time = datetime.now() - time_window
                cur.execute(query, (cutoff_time, tenant_id))
                memories = cur.fetchall()

                insights = []

                # Group by type and look for patterns
                patterns = self._identify_patterns(memories)

                for pattern in patterns:
                    insight = {
                        "type": "pattern",
                        "confidence": pattern["confidence"],
                        "description": pattern["description"],
                        "supporting_memories": pattern["memory_count"],
                        "recommended_action": pattern["action"],
                        "impact": pattern["estimated_impact"],
                        "discovered_at": datetime.now().isoformat(),
                    }
                    insights.append(insight)

                    # Store the insight as a meta memory
                    self.store(
                        Memory(
                            memory_type=MemoryType.META,
                            content=insight,
                            source_system="memory_synthesis",
                            source_agent="synthesizer",
                            created_by="unified_memory_manager",
                            importance_score=pattern["confidence"],
                            tags=["insight", "pattern", "synthesis"],
                            tenant_id=tenant_id,
                        )
                    )

                logger.info(
                    f"🧠 Synthesized {len(insights)} insights from {len(memories)} memories"
                )
                return insights

        except Exception as e:
            logger.error(f"❌ Failed to synthesize insights: {e}")
            return []

    def consolidate(self, aggressive: bool = False):
        """Consolidate similar memories to reduce redundancy"""
        try:
            threshold = 0.7 if aggressive else self.consolidation_threshold

            with self._get_cursor() as cur:
                # Find similar memories
                query = """
                WITH similarity_pairs AS (
                    SELECT
                        m1.id as id1, m2.id as id2,
                        m1.content as content1, m2.content as content2,
                        1 - (m1.embedding <=> m2.embedding) as similarity
                    FROM unified_ai_memory m1
                    JOIN unified_ai_memory m2 ON m1.id < m2.id
                    WHERE m1.memory_type = m2.memory_type
                        AND m1.source_system = m2.source_system
                        AND m1.tenant_id = m2.tenant_id
                        AND 1 - (m1.embedding <=> m2.embedding) > %s
                )
                SELECT * FROM similarity_pairs
                ORDER BY similarity DESC
                LIMIT 100
                """

                cur.execute(query, (threshold,))
                similar_pairs = cur.fetchall()

                consolidated_count = 0

                for pair in similar_pairs:
                    # Merge the memories
                    merged_content = self._merge_memories(pair["content1"], pair["content2"])

                    # Update the first memory with merged content
                    update_query = """
                    UPDATE unified_ai_memory
                    SET content = %s,
                        importance_score = importance_score + 0.1,
                        reinforcement_count = reinforcement_count + 1
                    WHERE id = %s
                    """
                    cur.execute(update_query, (Json(merged_content), pair["id1"]))

                    # Mark the second memory as consolidated
                    delete_query = """
                    UPDATE unified_ai_memory
                    SET expires_at = NOW(),
                        metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{consolidated_into}',
                            %s::jsonb
                        )
                    WHERE id = %s
                    """
                    cur.execute(delete_query, (json.dumps(str(pair["id1"])), pair["id2"]))

                    consolidated_count += 1

                self.conn.commit()
                logger.info(f"♻️ Consolidated {consolidated_count} memory pairs")

                if consolidated_count > 0:
                    self.log_to_brain(
                        "memory_system",
                        "memory_consolidated",
                        {"count": consolidated_count, "aggressive": aggressive},
                    )

        except Exception as e:
            logger.error(f"❌ Failed to consolidate memories: {e}")
            # Note: Connection rollback is handled automatically by the pool

    def migrate_from_chaos(self, tenant_id: str = None, limit: int = 1000):
        """Migrate data from the 53 chaotic memory tables"""
        tenant_id = tenant_id or self.tenant_id
        if not tenant_id:
            raise ValueError("tenant_id is required for migration")

        # NOTE: These are LEGACY tables to migrate FROM.
        # All data should migrate TO unified_ai_memory (the CANONICAL table).
        # Do NOT add unified_ai_memory to this list!
        tables_to_migrate = [
            "ai_context_memory",
            "ai_persistent_memory",
            "agent_memory",
            "cross_ai_memory",
            "production_memory",
            "unified_memory",  # Legacy table - migrate data to unified_ai_memory
            "ai_memory",  # Legacy table with 1 row
            "ai_memories",  # Legacy table with 3 rows
            "ai_memory_store",  # Legacy table with 3 rows
            "system_memory",
        ]

        total_migrated = 0

        for table in tables_to_migrate:
            try:
                migrated = self._migrate_table(table, tenant_id, limit)
                total_migrated += migrated
                logger.info(f"📦 Migrated {migrated} memories from {table}")
            except Exception as e:
                logger.warning(f"⚠️ Could not migrate from {table}: {e}")

        logger.info(f"✅ Total migrated: {total_migrated} memories from chaos to order")

    def _migrate_table(self, table_name: str, tenant_id: str, limit: int) -> int:
        """Migrate a single table to unified memory"""
        # Validate table name to prevent SQL injection
        import re

        if not re.match(r"^[a-z_][a-z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        try:
            with self._get_cursor() as cur:
                # First check if table exists
                check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
                """
                cur.execute(check_query, (table_name,))
                if not cur.fetchone()["exists"]:
                    return 0

                # Get recent data from old table
                query = f"""
                SELECT * FROM {table_name}
                WHERE created_at > NOW() - INTERVAL '30 days'
                ORDER BY created_at DESC
                LIMIT %s
                """
                cur.execute(query, (limit,))
                old_memories = cur.fetchall()

                migrated = 0
                for old_mem in old_memories:
                    try:
                        # Convert to unified format
                        memory = Memory(
                            memory_type=MemoryType.SEMANTIC,
                            content=old_mem.get("content")
                            or old_mem.get("memory_data")
                            or dict(old_mem),
                            source_system=table_name,
                            source_agent="migrated",
                            created_by="migration",
                            importance_score=old_mem.get("importance", 0.5),
                            tags=["migrated", table_name],
                            metadata={"original_id": str(old_mem.get("id", ""))},
                            # created_at=old_mem.get('created_at'), # Memory dataclass doesn't have created_at in __init__?
                            # Wait, checking Memory dataclass... it doesn't have created_at in the definition above!
                            # It has expires_at.
                            tenant_id=tenant_id,
                        )
                        self.store(memory)
                        migrated += 1
                    except Exception:
                        # logger.warning(f"Failed to migrate a record: {e}")
                        continue

                return migrated

        except Exception as e:
            logger.error(f"Migration error for {table_name}: {e}")
            return 0

    def _find_duplicate(self, memory: Memory) -> Optional[dict]:
        """Find duplicate memory using content hash"""
        try:
            content_str = json.dumps(memory.content, sort_keys=True, cls=CustomJSONEncoder)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()

            with self._get_cursor() as cur:
                query = """
                SELECT id, importance_score, reinforcement_count
                FROM unified_ai_memory
                WHERE content_hash = %s
                    AND memory_type = %s
                    AND source_system = %s
                    AND tenant_id = %s
                LIMIT 1
                """
                cur.execute(
                    query,
                    (
                        content_hash,
                        memory.memory_type.value,
                        memory.source_system,
                        memory.tenant_id,
                    ),
                )
                return cur.fetchone()
        except (TypeError, ValueError, psycopg2.Error) as exc:
            logger.warning("Failed to find duplicate memory: %s", exc, exc_info=True)
            return None

    def _reinforce_memory(self, memory_id: str, new_memory: Memory) -> str:
        """Reinforce existing memory instead of duplicating"""
        with self._get_cursor() as cur:
            query = """
            UPDATE unified_ai_memory
            SET importance_score = LEAST(importance_score + 0.05, 1.0),
                reinforcement_count = reinforcement_count + 1,
                access_count = access_count + 1,
                last_accessed = NOW()
            WHERE id = %s
            RETURNING id
            """
            cur.execute(query, (memory_id,))
            self.conn.commit()
            logger.info(f"💪 Reinforced existing memory {memory_id}")
            return memory_id

    def _find_related_memories(self, content: dict, tenant_id: str, limit: int = 5) -> list[dict]:
        """Find memories related to the given content"""
        embedding = self._generate_embedding(content)
        if embedding is None:
            return []

        with self._get_cursor() as cur:
            # Convert to pgvector string format (psycopg2 lists → {..} but pgvector needs [..])
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            query = """
            SELECT id, memory_type, importance_score,
                   1 - (embedding <=> %s::vector) as similarity
            FROM unified_ai_memory
            WHERE embedding IS NOT NULL
                AND tenant_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            cur.execute(query, (embedding_str, tenant_id, embedding_str, limit))
            return cur.fetchall()

    @staticmethod
    def _to_pgvector(embedding: list[float]) -> str:
        """Convert embedding list to pgvector string format.

        psycopg2 serializes Python lists as PostgreSQL arrays ``{0.1,0.2,...}``
        but the pgvector ``::vector`` cast expects ``[0.1,0.2,...]``.
        """
        return "[" + ",".join(map(str, embedding)) + "]"

    def _generate_embedding(self, content: dict) -> Optional[list[float]]:
        """Generate embedding using configured provider order (no silent mixing)."""
        text_content = json.dumps(content, sort_keys=True, cls=CustomJSONEncoder)
        embedding = generate_embedding_sync(text_content, log=logger)
        if embedding is None:
            logger.error("❌ All embedding providers failed")
        return embedding

    def _generate_search_text(self, memory: Memory) -> str:
        """Generate searchable text from memory"""
        parts = []

        # Add content values
        if isinstance(memory.content, dict):
            for key, value in memory.content.items():
                if isinstance(value, str):
                    parts.append(value)

        # Add tags
        if memory.tags:
            parts.extend(memory.tags)

        # Add metadata values
        if memory.metadata:
            for value in memory.metadata.values():
                if isinstance(value, str):
                    parts.append(value)

        return " ".join(parts)

    def _update_access_counts(self, memory_ids: list[str]):
        """Update access counts for recalled memories"""
        with self._get_cursor() as cur:
            query = """
            UPDATE unified_ai_memory
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE id = ANY(%s::uuid[])
            """
            cur.execute(query, (memory_ids,))
            self.conn.commit()

    def _identify_patterns(self, memories: list[dict]) -> list[dict]:
        """Identify patterns in memories (simplified version)"""
        patterns = []

        # Group by source system
        system_groups = {}
        for mem in memories:
            system = mem.get("source_system", "unknown")
            if system not in system_groups:
                system_groups[system] = []
            system_groups[system].append(mem)

        # Look for patterns in each group
        for system, group_memories in system_groups.items():
            if len(group_memories) >= 3:
                # Pattern detected
                pattern = {
                    "confidence": min(0.9, len(group_memories) / 10),
                    "description": f"High activity in {system} system",
                    "memory_count": len(group_memories),
                    "action": f"Optimize {system} for increased load",
                    "estimated_impact": f"{len(group_memories) * 100} operations/day",
                }
                patterns.append(pattern)

        return patterns

    def _merge_memories(self, content1: dict, content2: dict) -> dict:
        """Merge two similar memories"""
        merged = content1.copy()

        # Add unique keys from content2
        for key, value in content2.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Merge dicts recursively
                merged[key] = {**merged[key], **value}

        return merged

    def apply_retention_policy(
        self, tenant_id: str = None, aggressive: bool = False
    ) -> dict[str, int]:
        """
        Apply importance-based retention policy
        Returns: statistics about retained/removed memories
        """
        tenant_id = tenant_id or self.tenant_id
        if not tenant_id:
            raise ValueError("tenant_id is required for retention policy")

        try:
            with self._get_cursor() as cur:
                stats = {"retained": 0, "removed": 0, "promoted": 0, "demoted": 0}

                # Calculate retention score: importance * access_frequency * recency
                cur.execute(
                    """
                    WITH retention_scores AS (
                        SELECT
                            id,
                            importance_score,
                            access_count,
                            EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 as age_days,
                            (importance_score *
                             LOG(GREATEST(access_count, 1) + 1) *
                             (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 2592000))) as retention_score
                        FROM unified_ai_memory
                        WHERE tenant_id = %s
                    )
                    SELECT
                        id,
                        retention_score,
                        importance_score,
                        access_count,
                        age_days
                    FROM retention_scores
                    ORDER BY retention_score ASC
                """,
                    (tenant_id,),
                )

                memories = cur.fetchall()

                # Determine thresholds
                low_threshold = 0.1 if aggressive else 0.05
                high_threshold = 0.8 if aggressive else 0.9

                for mem in memories:
                    retention_score = float(mem.get("retention_score") or 0.0)

                    # Remove very low value memories
                    if retention_score < low_threshold and mem["age_days"] > 30:
                        cur.execute(
                            """
                            UPDATE unified_ai_memory
                            SET expires_at = NOW() + INTERVAL '7 days'
                            WHERE id = %s
                        """,
                            (mem["id"],),
                        )
                        stats["removed"] += 1

                    # Promote high-value memories
                    elif retention_score > high_threshold and mem["importance_score"] < 0.9:
                        cur.execute(
                            """
                            UPDATE unified_ai_memory
                            SET importance_score = LEAST(importance_score + 0.1, 1.0)
                            WHERE id = %s
                        """,
                            (mem["id"],),
                        )
                        stats["promoted"] += 1

                    # Demote low-access old memories
                    elif (
                        mem["access_count"] < 2
                        and mem["age_days"] > 60
                        and mem["importance_score"] > 0.3
                    ):
                        cur.execute(
                            """
                            UPDATE unified_ai_memory
                            SET importance_score = GREATEST(importance_score - 0.1, 0.2)
                            WHERE id = %s
                        """,
                            (mem["id"],),
                        )
                        stats["demoted"] += 1

                    else:
                        stats["retained"] += 1

                self.conn.commit()
                logger.info(f"✅ Retention policy applied: {stats}")
                return stats

        except Exception as e:
            logger.error(f"❌ Failed to apply retention policy: {e}")
            # Note: Connection rollback is handled automatically by the pool
            return {"error": str(e)}

    def auto_garbage_collect(self, tenant_id: str = None, dry_run: bool = False) -> dict[str, int]:
        """
        Automatically garbage collect old, low-value memories
        """
        tenant_id = tenant_id or self.tenant_id
        if not tenant_id:
            raise ValueError("tenant_id is required for garbage collection")

        try:
            with self._get_cursor() as cur:
                stats = {"expired": 0, "low_value": 0, "total": 0}

                # Archive expired memories (soft delete for safety)
                if not dry_run:
                    cur.execute(
                        """
                        UPDATE unified_ai_memory
                        SET archived = TRUE, archived_at = NOW()
                        WHERE tenant_id = %s
                        AND archived = FALSE
                        AND expires_at IS NOT NULL
                        AND expires_at < NOW()
                    """,
                        (tenant_id,),
                    )
                    stats["expired"] = cur.rowcount
                else:
                    cur.execute(
                        """
                        SELECT COUNT(*) as count
                        FROM unified_ai_memory
                        WHERE tenant_id = %s
                        AND archived = FALSE
                        AND expires_at IS NOT NULL
                        AND expires_at < NOW()
                    """,
                        (tenant_id,),
                    )
                    stats["expired"] = cur.fetchone()["count"]

                # Archive old, low-value memories (soft delete for safety)
                if not dry_run:
                    cur.execute(
                        """
                        UPDATE unified_ai_memory
                        SET archived = TRUE, archived_at = NOW()
                        WHERE tenant_id = %s
                        AND archived = FALSE
                        AND importance_score < 0.3
                        AND access_count < 3
                        AND created_at < NOW() - INTERVAL '90 days'
                    """,
                        (tenant_id,),
                    )
                    stats["low_value"] = cur.rowcount
                else:
                    cur.execute(
                        """
                        SELECT COUNT(*) as count
                        FROM unified_ai_memory
                        WHERE tenant_id = %s
                        AND archived = FALSE
                        AND importance_score < 0.3
                        AND access_count < 3
                        AND created_at < NOW() - INTERVAL '90 days'
                    """,
                        (tenant_id,),
                    )
                    stats["low_value"] = cur.fetchone()["count"]

                stats["total"] = stats["expired"] + stats["low_value"]

                if not dry_run:
                    self.conn.commit()
                    logger.info(f"✅ Garbage collected {stats['total']} memories")
                else:
                    logger.info(f"📊 Dry run: would remove {stats['total']} memories")

                return stats

        except Exception as e:
            logger.error(f"❌ Failed to garbage collect: {e}")
            # Note: Connection rollback is handled automatically by the pool
            return {"error": str(e)}

    def get_stats(self, tenant_id: str = None) -> dict:
        """Get memory system statistics"""
        tenant_id = tenant_id or self.tenant_id
        if not tenant_id:
            raise ValueError("tenant_id is required for stats")

        with self._get_cursor() as cur:
            query = """
            SELECT
                COUNT(*) as total_memories,
                COUNT(DISTINCT source_system) as unique_systems,
                COUNT(DISTINCT source_agent) as unique_agents,
                AVG(importance_score) as avg_importance,
                MAX(access_count) as max_access_count,
                COUNT(DISTINCT context_id) as unique_contexts,
                COUNT(*) FILTER (WHERE importance_score >= 0.7) as high_importance,
                COUNT(*) FILTER (WHERE importance_score < 0.3) as low_importance,
                COUNT(*) FILTER (WHERE access_count > 10) as frequently_accessed,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL) as expiring
            FROM unified_ai_memory
            WHERE tenant_id = %s
            """
            cur.execute(query, (tenant_id,))
            return dict(cur.fetchone())

    # ──────────────────────────────────────────────────────────────
    # EPISODIC MEMORY — Persist strategies, queries, and outcomes
    # so agents learn what works over time.
    # Based on production-grade agentic AI patterns (2026-02-10).
    # ──────────────────────────────────────────────────────────────

    def store_episode(
        self,
        session_id: str,
        *,
        episode_type: str = "research",
        question: str = None,
        objective: str = None,
        plan: dict = None,
        retrieval_queries: list[str] = None,
        sources_used: list[str] = None,
        sources_ranked: dict = None,
        strategy_notes: str = None,
        outcome: str = None,
        outcome_score: float = 0.0,
        duration_ms: int = None,
        agent_id: str = None,
        tenant_id: str = None,
        metadata: dict = None,
    ) -> Optional[str]:
        """Store an episode capturing what strategy/queries/sources worked.

        This enables agents to recall successful strategies for similar
        future tasks (episodic memory pattern).
        """
        tid = tenant_id or self.tenant_id
        # Build searchable text for embedding generation
        search_parts = [question or "", objective or "", strategy_notes or ""]
        if retrieval_queries:
            search_parts.extend(retrieval_queries[:5])
        search_text = " ".join(p for p in search_parts if p).strip()
        embedding = self._generate_embedding({"text": search_text}) if search_text else None

        try:
            with self._get_cursor() as cur:
                if not cur:
                    logger.error("Failed to get cursor for episode storage")
                    return None

                cur.execute(
                    """
                    INSERT INTO episodic_memory (
                        session_id, episode_type, question, objective,
                        plan, retrieval_queries, sources_used, sources_ranked,
                        strategy_notes, outcome, outcome_score, duration_ms,
                        agent_id, tenant_id, metadata, embedding
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    ) RETURNING id
                    """,
                    (
                        session_id,
                        episode_type,
                        question,
                        objective,
                        Json(plan) if plan else None,
                        retrieval_queries,
                        sources_used,
                        Json(sources_ranked) if sources_ranked else None,
                        strategy_notes,
                        outcome,
                        outcome_score,
                        duration_ms,
                        agent_id,
                        tid,
                        Json(metadata or {}),
                        embedding,
                    ),
                )
                row = cur.fetchone()
                episode_id = str(row["id"]) if row else None

                self.log_to_brain(
                    "episodic_memory",
                    "episode_stored",
                    {
                        "episode_id": episode_id,
                        "session_id": session_id,
                        "episode_type": episode_type,
                        "outcome": outcome,
                        "outcome_score": outcome_score,
                    },
                )
                logger.info(
                    "Stored episode %s (type=%s, outcome=%s)", episode_id, episode_type, outcome
                )
                return episode_id

        except Exception as exc:
            logger.error("Failed to store episode: %s", exc, exc_info=True)
            return None

    def recall_episodes(
        self,
        query: str,
        *,
        episode_type: str = None,
        min_outcome_score: float = 0.0,
        limit: int = 5,
        tenant_id: str = None,
    ) -> list[dict]:
        """Recall past episodes similar to the current query.

        Returns episodes ordered by semantic similarity so the agent
        can reuse strategies that worked before.
        """
        tid = tenant_id or self.tenant_id
        embedding = self._generate_embedding({"text": query})

        try:
            with self._get_cursor() as cur:
                if not cur:
                    return []

                if embedding:
                    emb_str = self._to_pgvector(embedding)
                    cur.execute(
                        """
                        SELECT id, session_id, episode_type, question, objective,
                               retrieval_queries, sources_used, strategy_notes,
                               outcome, outcome_score, duration_ms, agent_id,
                               created_at,
                               1.0 - (embedding <=> %s::vector) as similarity
                        FROM episodic_memory
                        WHERE embedding IS NOT NULL
                          AND (%s IS NULL OR episode_type = %s)
                          AND outcome_score >= %s
                          AND (%s IS NULL OR tenant_id = %s::uuid)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            emb_str,
                            episode_type,
                            episode_type,
                            min_outcome_score,
                            tid,
                            tid,
                            emb_str,
                            limit,
                        ),
                    )
                else:
                    # Fallback: keyword-based recall
                    cur.execute(
                        """
                        SELECT id, session_id, episode_type, question, objective,
                               retrieval_queries, sources_used, strategy_notes,
                               outcome, outcome_score, duration_ms, agent_id,
                               created_at,
                               0.0 as similarity
                        FROM episodic_memory
                        WHERE (%s IS NULL OR episode_type = %s)
                          AND outcome_score >= %s
                          AND (%s IS NULL OR tenant_id = %s::uuid)
                        ORDER BY outcome_score DESC, created_at DESC
                        LIMIT %s
                        """,
                        (episode_type, episode_type, min_outcome_score, tid, tid, limit),
                    )

                rows = cur.fetchall()
                return [dict(r) for r in rows] if rows else []

        except Exception as exc:
            logger.error("Failed to recall episodes: %s", exc, exc_info=True)
            return []

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 20,
        tenant_id: str = None,
    ) -> list[dict]:
        """Hybrid search across unified_ai_memory using vector + FTS + RRF.

        Calls the hybrid_search_unified_memory() SQL function which does
        dense (vector) + sparse (FTS) retrieval with Reciprocal Rank Fusion.
        """
        tid = tenant_id or self.tenant_id
        embedding = self._generate_embedding({"text": query})
        emb_str = self._to_pgvector(embedding) if embedding else None

        try:
            with self._get_cursor() as cur:
                if not cur:
                    return []

                cur.execute(
                    """
                    SELECT * FROM hybrid_search_unified_memory(
                        p_query_text := %s,
                        p_query_embedding := %s::vector,
                        p_tenant_id := %s::uuid,
                        p_limit := %s
                    )
                    """,
                    (query, emb_str, tid, limit),
                )
                rows = cur.fetchall()
                return [dict(r) for r in rows] if rows else []

        except Exception as exc:
            logger.error("Hybrid search failed: %s", exc, exc_info=True)
            # Fallback to standard vector recall
            return self.recall(query, tenant_id=tid, limit=limit)

    def unified_retrieval(
        self,
        query: str,
        *,
        limit: int = 20,
        tenant_id: str = None,
    ) -> list[dict]:
        """Unified search across all memory tiers: Semantic, Episodic, and Documents.

        Aggregates results from:
        1. unified_ai_memory (via hybrid_search_unified_memory)
        2. document_chunks (via search_document_chunks)
        """
        tid = tenant_id or self.tenant_id
        embedding = self._generate_embedding({"text": query})
        emb_str = self._to_pgvector(embedding) if embedding else None

        results = []

        try:
            with self._get_cursor() as cur:
                if not cur:
                    return []

                # 1. Search Memory (Hybrid)
                cur.execute(
                    """
                    SELECT id, content, 'memory' as source_type, score_fused as score
                    FROM hybrid_search_unified_memory(%s, %s::vector, %s::uuid, %s)
                    """,
                    (query, emb_str, tid, limit),
                )
                memory_rows = cur.fetchall()
                for row in memory_rows:
                    results.append(
                        {
                            "id": str(row["id"]),
                            "content": row["content"],
                            "type": "memory",
                            "score": row["score"],
                        }
                    )

                # 2. Search Documents (Knowledge Base)
                # Check if function exists first to be safe, though we verified schema
                try:
                    cur.execute(
                        """
                        SELECT id, content, metadata, 1 - (embedding <=> %s::vector) as score
                        FROM document_chunks
                        WHERE embedding IS NOT NULL
                          AND (tenant_id IS NULL OR tenant_id = %s::uuid)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (emb_str, tid, emb_str, limit),
                    )
                    doc_rows = cur.fetchall()
                    for row in doc_rows:
                        results.append(
                            {
                                "id": str(row["id"]),
                                "content": row["content"],
                                "metadata": row["metadata"],
                                "type": "document",
                                "score": row["score"],
                            }
                        )
                except Exception as doc_exc:
                    logger.warning("Document search failed (table/func missing?): %s", doc_exc)

                # 3. Search Episodic Memory
                try:
                    cur.execute(
                        """
                        SELECT id, session_id, episode_type, question, objective,
                               outcome, outcome_score, created_at,
                               CASE WHEN %s::vector IS NOT NULL
                                    THEN 1 - (embedding <=> %s::vector)
                                    ELSE 0.5 END as score
                        FROM episodic_memory
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (emb_str, emb_str, emb_str, limit),
                    )
                    epi_rows = cur.fetchall()
                    for row in epi_rows:
                        results.append(
                            {
                                "id": str(row["id"]),
                                "content": {
                                    "question": row.get("question"),
                                    "objective": row.get("objective"),
                                    "outcome": row.get("outcome"),
                                },
                                "type": "episode",
                                "episode_type": row.get("episode_type"),
                                "session_id": row.get("session_id"),
                                "score": row["score"],
                            }
                        )
                except Exception as epi_exc:
                    logger.warning("Episodic recall in unified_retrieval failed: %s", epi_exc)

        except Exception as e:
            logger.error("Unified retrieval failed: %s", e, exc_info=True)
            return []

        # Sort combined results by score descending
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]


# Singleton instance
memory_manager = None


def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create the singleton memory manager"""
    global memory_manager
    if memory_manager is None:
        memory_manager = UnifiedMemoryManager()
    return memory_manager


if __name__ == "__main__":
    # Test the memory system
    manager = get_memory_manager()

    # Test with a dummy tenant ID for local execution
    TEST_TENANT = "test-tenant-id"
    manager.tenant_id = TEST_TENANT  # Ensure tenant ID is set for test

    # Store a test memory
    test_memory = Memory(
        memory_type=MemoryType.SEMANTIC,
        content={
            "test": "Unified memory system operational",
            "timestamp": datetime.now().isoformat(),
        },
        source_system="test",
        source_agent="initializer",
        created_by="system",
        importance_score=0.9,
        tags=["test", "initialization"],
        tenant_id=TEST_TENANT,
    )

    try:
        memory_id = manager.store(test_memory)
        print(f"✅ Stored test memory: {memory_id}")
    except Exception as e:
        print(f"❌ Storage failed (DB might be unreachable): {e}")

    # Get stats
    try:
        stats = manager.get_stats(TEST_TENANT)
        print(f"📊 Memory Stats: {json.dumps(stats, indent=2)}")
    except Exception as e:
        print(f"❌ Stats failed: {e}")

    print("✅ Unified Memory Manager operational!")
