"""
Embedded SQLite Memory System with RAG and Master Sync
Fast local memory access with bidirectional sync to master Postgres
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np

from safe_task import create_safe_task
from utils.embedding_provider import (
    generate_embedding_sync,
    get_gemini_model,
    get_local_model_name,
    get_openai_model,
    get_provider_order,
    iter_providers,
)

logger = logging.getLogger(__name__)

class EmbeddedMemorySystem:
    """
    Local SQLite memory cache with RAG capabilities and master Postgres sync

    Architecture:
    - Reads: Always from local SQLite (< 1ms)
    - Writes: Dual-write to local + async to master
    - Sync: Periodic background sync every 5 minutes
    - RAG: Local vector similarity search with embeddings
    """

    def __init__(self, local_db_path: str = None):
        # Use environment variable or fallback to project-relative path
        self.local_db_path = local_db_path or os.getenv(
            "EMBEDDED_MEMORY_PATH",
            os.path.join(os.path.dirname(__file__), "data", "ai-memory.db")
        )
        self.sqlite_conn = None
        self.pg_pool = None
        self.embedding_model = None
        self.sync_task = None
        self.last_sync = None
        self.initialized = False

    async def initialize(self):
        """Initialize local SQLite and master Postgres connections"""
        if self.initialized:
            return

        logger.info("🧠 Initializing Embedded Memory System...")

        # 1. Setup local SQLite
        await self._setup_local_db()

        # 2. Connect to master Postgres
        await self._connect_master()

        # 3. Load embedding model for RAG
        await self._load_embedding_model()

        # 4. Initial sync from master (will retry if pool not ready)
        await self.sync_from_master()

        # 5. Start background sync task (includes retry logic)
        self.sync_task = create_safe_task(self._background_sync())

        self.initialized = True
        logger.info("✅ Embedded Memory System initialized")

    async def _setup_local_db(self):
        """Create local SQLite database and tables"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)

        # Connect to SQLite
        self.sqlite_conn = sqlite3.connect(self.local_db_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row

        # Create schema
        cursor = self.sqlite_conn.cursor()

        # Unified AI Memory table

        # Task queue

        # Learning from mistakes

        # Sync metadata (tracks last sync time per table)

        # Indices for fast queries

        self.sqlite_conn.commit()
        logger.info("✅ Local SQLite database initialized")

    async def _connect_master(self):
        """Connect to master Postgres database - USE SHARED POOL"""
        try:
            # CRITICAL: Use the shared pool from database/async_connection.py
            # instead of creating our own pool to prevent pool exhaustion
            from database.async_connection import get_pool, using_fallback

            try:
                shared_pool = get_pool()
                if not using_fallback():
                    self.pg_pool = shared_pool
                    logger.info("✅ Embedded memory using SHARED database pool")
                    return
                else:
                    logger.warning("⚠️ Shared pool using fallback, master sync disabled")
                    self.pg_pool = None
                    return
            except RuntimeError:
                # Pool not yet initialized - this is OK, will retry in background
                logger.warning("⚠️ Shared pool not initialized yet, will retry sync later")
                self.pg_pool = None
                return

        except Exception as e:
            logger.error(f"❌ Failed to connect to master Postgres: {e}")
            self.pg_pool = None

    async def _ensure_pool_connection(self) -> bool:
        """Ensure we have a pool connection, retry if needed"""
        if self.pg_pool is not None:
            return True

        # Try to get the pool again
        try:
            from database.async_connection import get_pool, using_fallback

            shared_pool = get_pool()
            if not using_fallback():
                self.pg_pool = shared_pool
                logger.info("✅ Embedded memory connected to database pool")
                return True
            else:
                return False
        except RuntimeError:
            # Pool still not ready
            return False
        except Exception as e:
            logger.error(f"❌ Failed to get pool: {e}")
            return False

    async def _load_embedding_model(self):
        """Load embedding model configuration (no heavy local deps)."""
        providers = iter_providers(get_provider_order())
        primary = providers[0] if providers else "gemini"

        if primary in {"gemini", "google"}:
            self.embedding_model = f"gemini:{get_gemini_model()}"
        elif primary in {"openai", "oai"}:
            self.embedding_model = f"openai:{get_openai_model()}"
        elif primary == "local":
            self.embedding_model = f"local:{get_local_model_name()}"
        else:
            self.embedding_model = primary

        logger.info("✅ Embedding model configured (%s)", self.embedding_model)

    def _encode_embedding(self, text: str) -> Optional[bytes]:
        """Convert text to embedding vector using the shared embedding provider chain."""
        try:
            embedding_list = generate_embedding_sync(text, log=logger)
            if not embedding_list:
                return None
            embedding_np = np.array(embedding_list, dtype=np.float32)
            return embedding_np.tobytes()
        except Exception as e:
            logger.error("Embedding encoding failed: %s", e)
            return None

    def _decode_embedding(self, embedding_bytes: bytes) -> Optional[np.ndarray]:
        """Convert bytes back to numpy array"""
        try:
            return np.frombuffer(embedding_bytes, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding decoding failed: {e}")
            return None

    # ========== RAG: Retrieval Augmented Generation ==========

    async def _auto_sync_if_empty(self):
        """Auto-sync from master if local DB is empty (lazy loading)"""
        cursor = self.sqlite_conn.cursor()
        local_count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]

        if local_count == 0 and not self.last_sync:
            logger.info("🔄 Local DB empty on first access, triggering sync...")
            await self.sync_from_master(force=True)

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        RAG-powered memory search with semantic similarity

        Returns most relevant memories based on:
        1. Semantic similarity (vector search)
        2. Importance score
        3. Recency
        """
        if not self.sqlite_conn:
            return []

        # Auto-sync if empty (run in background to not block)
        cursor = self.sqlite_conn.cursor()
        local_count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]
        if local_count == 0 and not self.last_sync:
            create_safe_task(self._auto_sync_if_empty())

        # Get query embedding
        query_embedding = self._encode_embedding(query)

        # Build SQL query
        sql = """
            SELECT
                id, memory_type, source_agent, content,
                metadata, importance_score, access_count,
                created_at, last_accessed, embedding
            FROM unified_ai_memory
            WHERE importance_score >= ?
        """
        params = [min_importance]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY importance_score DESC, created_at DESC"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        if not query_embedding or not rows:
            # Fallback to simple query without vector similarity
            return [dict(row) for row in rows[:limit]]

        # Calculate semantic similarity
        query_vec = self._decode_embedding(query_embedding)
        results = []

        for row in rows:
            row_dict = dict(row)

            # Calculate similarity if embedding exists
            if row['embedding']:
                mem_vec = self._decode_embedding(row['embedding'])
                if mem_vec is not None and query_vec is not None:
                    # Cosine similarity
                    similarity = np.dot(query_vec, mem_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)
                    )
                    row_dict['similarity_score'] = float(similarity)
                else:
                    row_dict['similarity_score'] = 0.0
            else:
                row_dict['similarity_score'] = 0.0

            # Combined score: similarity (70%) + importance (30%)
            row_dict['combined_score'] = (
                row_dict['similarity_score'] * 0.7 +
                row_dict['importance_score'] * 0.3
            )

            results.append(row_dict)

        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)

        # Update access count and last_accessed for top results
        top_results = results[:limit]
        for result in top_results:
            cursor.execute("""
                UPDATE unified_ai_memory
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), result['id']))

        self.sqlite_conn.commit()

        return top_results

    def store_memory(
        self,
        content: str,
        memory_type: str = "general",
        memory_id: Optional[str] = None,
        source_agent: str = "system",
        metadata: Optional[dict] = None,
        importance_score: float = 0.5
    ) -> bool:
        """
        Store memory locally and async sync to master

        Dual-write pattern:
        1. Write to local SQLite (immediate)
        2. Queue write to master Postgres (async)
        """
        if not self.sqlite_conn:
            return False

        try:
            # Auto-generate memory_id if not provided
            if memory_id is None:
                memory_id = str(uuid.uuid4())

            # Generate embedding
            embedding = self._encode_embedding(content)

            # Store locally
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO unified_ai_memory (
                    id, memory_type, source_agent, content, embedding,
                    metadata, importance_score, access_count,
                    created_at, last_accessed, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)
            """, (
                memory_id,
                memory_type,
                source_agent,
                content,
                embedding,
                json.dumps(metadata) if metadata else None,
                importance_score,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            self.sqlite_conn.commit()

            # Async sync to master
            create_safe_task(self._sync_memory_to_master(memory_id))

            return True
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def _sync_memory_to_master(self, memory_id: str):
        """Sync single memory to master Postgres"""
        if not self.pg_pool:
            return

        try:
            # Get from local
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM unified_ai_memory WHERE id = ?", (memory_id,))
            row = cursor.fetchone()

            if not row:
                return

            # Write to master (include required source_system and created_by)
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO unified_ai_memory (
                        id, memory_type, source_system, source_agent, created_by, content,
                        metadata, importance_score, created_at, last_accessed
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        importance_score = EXCLUDED.importance_score,
                        last_accessed = EXCLUDED.last_accessed
                """,
                    row['id'], row['memory_type'],
                    dict(row).get('source_system', 'embedded_memory'),
                    row['source_agent'],
                    dict(row).get('created_by', 'embedded_memory'),
                    row['content'], row['metadata'], row['importance_score'],
                    row['created_at'], row['last_accessed']
                )

            # Mark as synced
            cursor.execute(
                "UPDATE unified_ai_memory SET synced_at = ? WHERE id = ?",
                (datetime.now().isoformat(), memory_id)
            )
            self.sqlite_conn.commit()

        except Exception as e:
            logger.error(f"Failed to sync memory to master: {e}")

    # ========== Task Queue Operations ==========

    def get_pending_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get pending tasks from local queue"""
        if not self.sqlite_conn:
            return []

        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT * FROM ai_autonomous_tasks
            WHERE status = 'pending'
            ORDER BY
                CASE priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    ELSE 4
                END,
                created_at ASC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error_log: Optional[str] = None
    ) -> bool:
        """Update task status locally and sync to master"""
        if not self.sqlite_conn:
            return False

        try:
            cursor = self.sqlite_conn.cursor()

            updates = {"status": status}
            if status == "in_progress":
                updates["started_at"] = datetime.now().isoformat()
            elif status in ["completed", "failed"]:
                updates["completed_at"] = datetime.now().isoformat()

            if result:
                updates["result"] = result
            if error_log:
                updates["error_log"] = error_log

            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [task_id]

            cursor.execute(
                f"UPDATE ai_autonomous_tasks SET {set_clause} WHERE id = ?",
                values
            )
            self.sqlite_conn.commit()

            # Async sync to master
            create_safe_task(self._sync_task_to_master(task_id))

            return True
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            return False

    async def _sync_task_to_master(self, task_id: str):
        """Sync task to master Postgres"""
        if not self.pg_pool:
            return

        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM ai_autonomous_tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()

            if not row:
                return

            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_autonomous_tasks (
                        id, task_type, status, priority, trigger_condition,
                        result, error_log, created_at, started_at, completed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        error_log = EXCLUDED.error_log,
                        started_at = EXCLUDED.started_at,
                        completed_at = EXCLUDED.completed_at
                """,
                    row['id'], row['task_type'], row['status'], row['priority'],
                    row['trigger_condition'], row['result'], row['error_log'],
                    row['created_at'], row['started_at'], row['completed_at']
                )

            cursor.execute(
                "UPDATE ai_autonomous_tasks SET synced_at = ? WHERE id = ?",
                (datetime.now().isoformat(), task_id)
            )
            self.sqlite_conn.commit()

        except Exception as e:
            logger.error(f"Failed to sync task to master: {e}")

    # ========== Bidirectional Sync ==========

    async def sync_from_master(self, force: bool = False):
        """Pull latest data from master to local

        Args:
            force: If True, force sync even if local DB has data
        """
        # Try to ensure pool connection (retry if it wasn't ready before)
        if not await self._ensure_pool_connection():
            logger.warning("⚠️ Master sync skipped (no pool connection available)")
            return

        # Check if local DB is empty and needs initial sync
        cursor = self.sqlite_conn.cursor()
        local_count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]

        if not force and local_count > 0 and self.last_sync:
            # Only sync if forced, or if DB is empty, or never synced
            logger.debug(f"📊 Local DB has {local_count} entries, skipping non-forced sync")
            return

        # Keep periodic sync bounded so it cannot starve the HTTP loop.
        memory_sync_limit = max(50, int(os.getenv("EMBEDDED_MEMORY_SYNC_MEMORY_LIMIT", "200")))
        task_sync_limit = max(10, int(os.getenv("EMBEDDED_MEMORY_SYNC_TASK_LIMIT", "50")))
        max_embeddings_per_sync = max(0, int(os.getenv("EMBEDDED_MEMORY_SYNC_MAX_EMBEDDINGS", "0")))

        logger.info(f"🔄 Syncing from master Postgres (local_count={local_count}, force={force})...")

        try:
            async with self.pg_pool.acquire() as conn:
                # Sync memories
                memories = await conn.fetch("""
                    SELECT * FROM unified_ai_memory
                    ORDER BY created_at DESC
                    LIMIT $1
                """, memory_sync_limit)

                cursor = self.sqlite_conn.cursor()
                generated_count = 0

                for mem in memories:
                    mem_id = str(mem['id'])

                    # Check if we already have a valid embedding locally
                    cursor.execute("SELECT embedding FROM unified_ai_memory WHERE id = ?", (mem_id,))
                    row = cursor.fetchone()

                    embedding = None
                    if row and row[0]:
                        # Reuse existing local embedding
                        embedding = row[0]
                    elif mem.get('content') and generated_count < max_embeddings_per_sync:
                        # Run sync embedding provider work off-loop to avoid
                        # blocking health checks under load.
                        embedding = await asyncio.to_thread(self._encode_embedding, mem['content'])
                        if embedding:
                            generated_count += 1

                    cursor.execute("""
                        INSERT OR REPLACE INTO unified_ai_memory (
                            id, memory_type, source_agent, content, embedding,
                            metadata, importance_score, access_count,
                            created_at, last_accessed, synced_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mem_id,
                        mem.get('memory_type'), mem.get('source_agent'),
                        mem.get('content'), embedding, mem.get('metadata'),
                        mem.get('importance_score', 0.5), 0,
                        str(mem.get('created_at')) if mem.get('created_at') else None,
                        str(mem.get('last_accessed')) if mem.get('last_accessed') else None,
                        datetime.now().isoformat()
                    ))

                # Sync tasks
                tasks = await conn.fetch("""
                    SELECT * FROM ai_autonomous_tasks
                    WHERE status IN ('pending', 'in_progress')
                    ORDER BY created_at DESC
                    LIMIT $1
                """, task_sync_limit)

                for task in tasks:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ai_autonomous_tasks (
                            id, task_type, status, priority, trigger_condition,
                            result, error_log, created_at, started_at,
                            completed_at, synced_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(task['id']),  # Convert UUID to string
                        task.get('task_type'), task.get('status'),
                        task.get('priority'), task.get('trigger_condition'),
                        task.get('result'), task.get('error_log'),
                        str(task.get('created_at')) if task.get('created_at') else None,
                        str(task.get('started_at')) if task.get('started_at') else None,
                        str(task.get('completed_at')) if task.get('completed_at') else None,
                        datetime.now().isoformat()
                    ))

                self.sqlite_conn.commit()

                # Update sync metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO sync_metadata (table_name, last_sync_time, last_sync_count, total_records)
                    VALUES ('unified_ai_memory', ?, ?, ?),
                           ('ai_autonomous_tasks', ?, ?, ?)
                """, (
                    datetime.now().isoformat(), len(memories),
                    cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0],
                    datetime.now().isoformat(), len(tasks),
                    cursor.execute("SELECT COUNT(*) FROM ai_autonomous_tasks").fetchone()[0]
                ))
                self.sqlite_conn.commit()

                self.last_sync = datetime.now()
                logger.info(f"✅ Synced {len(memories)} memories, {len(tasks)} tasks from master")

        except Exception as e:
            logger.error(f"❌ Sync from master failed: {e!r}")

    async def _background_sync(self):
        """Background task for periodic sync with retry logic

        - First 5 attempts: Every 30 seconds (for initial pool connection)
        - After that: Every 5 minutes (normal periodic sync)
        """
        retry_count = 0
        max_fast_retries = 5

        while True:
            try:
                # Use shorter interval for first few attempts (pool might not be ready yet)
                if retry_count < max_fast_retries:
                    await asyncio.sleep(30)  # 30 seconds
                    retry_count += 1
                else:
                    await asyncio.sleep(300)  # 5 minutes

                # Check if local DB is empty and force sync if needed
                cursor = self.sqlite_conn.cursor()
                local_count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]

                if local_count == 0:
                    logger.info("🔄 Local DB empty, attempting force sync...")
                    await self.sync_from_master(force=True)
                else:
                    # Normal periodic sync
                    await self.sync_from_master(force=False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background sync error: {e}")

    # ========== Learning Storage ==========

    def store_learning(
        self,
        learning_id: str,
        agent_id: str,
        task_id: str,
        mistake_description: str,
        root_cause: str,
        lesson_learned: str,
        impact_level: str = "medium"
    ) -> bool:
        """Store learning from mistakes"""
        if not self.sqlite_conn:
            return False

        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ai_learning_from_mistakes (
                    id, agent_id, task_id, mistake_description,
                    root_cause, lesson_learned, impact_level,
                    created_at, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (
                learning_id, agent_id, task_id, mistake_description,
                root_cause, lesson_learned, impact_level,
                datetime.now().isoformat()
            ))
            self.sqlite_conn.commit()

            # Also store as memory for RAG
            self.store_memory(
                memory_id=f"learning_{learning_id}",
                memory_type="learning",
                source_agent=agent_id,
                content=f"Lesson: {lesson_learned}. Context: {mistake_description}",
                metadata={"impact_level": impact_level, "task_id": task_id},
                importance_score=0.9 if impact_level == "high" else 0.7
            )

            return True
        except Exception as e:
            logger.error(f"Failed to store learning: {e}")
            return False

    # ========== Statistics ==========

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics"""
        if not self.sqlite_conn:
            return {}

        cursor = self.sqlite_conn.cursor()

        stats = {
            "total_memories": cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0],
            "total_tasks": cursor.execute("SELECT COUNT(*) FROM ai_autonomous_tasks").fetchone()[0],
            "pending_tasks": cursor.execute("SELECT COUNT(*) FROM ai_autonomous_tasks WHERE status = 'pending'").fetchone()[0],
            "total_learnings": cursor.execute("SELECT COUNT(*) FROM ai_learning_from_mistakes").fetchone()[0],
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "embedding_model": self.embedding_model
        }

        # Memory by type
        cursor.execute("SELECT memory_type, COUNT(*) as count FROM unified_ai_memory GROUP BY memory_type")
        stats["memories_by_type"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Tasks by status
        cursor.execute("SELECT status, COUNT(*) as count FROM ai_autonomous_tasks GROUP BY status")
        stats["tasks_by_status"] = {row[0]: row[1] for row in cursor.fetchall()}

        return stats

    async def close(self):
        """Cleanup connections"""
        if self.sync_task:
            self.sync_task.cancel()

        if self.sqlite_conn:
            self.sqlite_conn.close()

        # CRITICAL: Do NOT close self.pg_pool - it's a SHARED pool managed globally
        # Closing it here would break other modules using the same pool
        # The shared pool is managed by database/async_connection.py
        self.initialized = False


# Global instance
_embedded_memory = None

async def get_embedded_memory() -> EmbeddedMemorySystem:
    """Get or create global embedded memory instance"""
    global _embedded_memory

    if _embedded_memory is None or not getattr(_embedded_memory, "initialized", False):
        if _embedded_memory is not None:
            try:
                await _embedded_memory.close()
            except Exception:
                pass
        candidate = EmbeddedMemorySystem()
        try:
            await candidate.initialize()
            _embedded_memory = candidate
        except Exception:
            try:
                await candidate.close()
            except Exception:
                pass
            _embedded_memory = None
            raise

    return _embedded_memory
