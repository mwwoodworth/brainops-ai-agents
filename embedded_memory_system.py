"""
Embedded SQLite Memory System with RAG and Master Sync
Fast local memory access with bidirectional sync to master Postgres
"""

import sqlite3
import asyncpg
import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
# sentence_transformers removed - too heavy for Docker build
# Using simple hash-based embeddings instead
import hashlib

logger = logging.getLogger(__name__)

# Circuit breaker for OpenAI quota errors
_openai_disabled_until = 0
_OPENAI_BACKOFF_SECONDS = 3600  # 1 hour backoff on quota errors

class EmbeddedMemorySystem:
    """
    Local SQLite memory cache with RAG capabilities and master Postgres sync

    Architecture:
    - Reads: Always from local SQLite (< 1ms)
    - Writes: Dual-write to local + async to master
    - Sync: Periodic background sync every 5 minutes
    - RAG: Local vector similarity search with embeddings
    """

    def __init__(self, local_db_path: str = "/var/lib/ai-memory.db"):
        self.local_db_path = local_db_path
        self.sqlite_conn = None
        self.pg_pool = None
        self.embedding_model = None
        self.sync_task = None
        self.last_sync = None

    async def initialize(self):
        """Initialize local SQLite and master Postgres connections"""
        logger.info("🧠 Initializing Embedded Memory System...")

        # 1. Setup local SQLite
        await self._setup_local_db()

        # 2. Connect to master Postgres
        await self._connect_master()

        # 3. Load embedding model for RAG
        await self._load_embedding_model()

        # 4. Initial sync from master
        await self.sync_from_master()

        # 5. Start background sync task
        self.sync_task = asyncio.create_task(self._background_sync())

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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS unified_ai_memory (
                id TEXT PRIMARY KEY,
                memory_type TEXT,
                source_agent TEXT,
                content TEXT,
                embedding BLOB,
                metadata TEXT,
                importance_score REAL,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                synced_at TIMESTAMP
            )
        """)

        # Task queue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_autonomous_tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT,
                status TEXT,
                priority TEXT,
                trigger_condition TEXT,
                result TEXT,
                error_log TEXT,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                synced_at TIMESTAMP
            )
        """)

        # Learning from mistakes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_learning_from_mistakes (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                task_id TEXT,
                mistake_description TEXT,
                root_cause TEXT,
                lesson_learned TEXT,
                impact_level TEXT,
                created_at TIMESTAMP,
                synced_at TIMESTAMP
            )
        """)

        # Sync metadata (tracks last sync time per table)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_metadata (
                table_name TEXT PRIMARY KEY,
                last_sync_time TIMESTAMP,
                last_sync_count INTEGER,
                total_records INTEGER
            )
        """)

        # Indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON unified_ai_memory(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_agent ON unified_ai_memory(source_agent)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON unified_ai_memory(importance_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON ai_autonomous_tasks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON ai_autonomous_tasks(priority, created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_agent ON ai_learning_from_mistakes(agent_id)")

        self.sqlite_conn.commit()
        logger.info("✅ Local SQLite database initialized")

    async def _connect_master(self):
        """Connect to master Postgres database"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                logger.warning("⚠️ DATABASE_URL not set, master sync disabled")
                return

            self.pg_pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=2,  # Reduced to prevent pool exhaustion
                command_timeout=60,
                max_inactive_connection_lifetime=60,  # Recycle idle connections
                statement_cache_size=0  # Disable statement cache for session mode
            )
            logger.info("✅ Connected to master Postgres")
        except Exception as e:
            logger.error(f"❌ Failed to connect to master Postgres: {e}")
            self.pg_pool = None

    async def _load_embedding_model(self):
        """Load embedding model configuration"""
        # We are now using OpenAI API for embeddings
        self.embedding_model = "openai-text-embedding-3-small"
        logger.info("✅ Embedding model loaded (OpenAI text-embedding-3-small)")

    def _encode_embedding(self, text: str) -> Optional[bytes]:
        """Convert text to embedding vector using OpenAI"""
        global _openai_disabled_until
        import time as time_module

        # Circuit breaker check - skip OpenAI if quota exceeded recently
        if _openai_disabled_until > time_module.time():
            remaining = int(_openai_disabled_until - time_module.time())
            logger.debug(f"OpenAI embeddings disabled for {remaining}s more (quota exceeded)")
            return None

        try:
            import openai

            # Call OpenAI Embedding API
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )

            embedding_list = response.data[0].embedding

            # Convert to numpy array then bytes for SQLite
            embedding_np = np.array(embedding_list, dtype=np.float32)
            return embedding_np.tobytes()

        except Exception as e:
            error_str = str(e).lower()
            # Detect quota/rate limit errors and trigger circuit breaker
            if 'insufficient_quota' in error_str or '429' in error_str or 'rate_limit' in error_str:
                _openai_disabled_until = time_module.time() + _OPENAI_BACKOFF_SECONDS
                logger.warning(f"⚠️ OpenAI quota/rate limit hit - disabling embeddings for {_OPENAI_BACKOFF_SECONDS}s")
            else:
                logger.error(f"Embedding encoding failed: {e}")
            return None

    def _decode_embedding(self, embedding_bytes: bytes) -> Optional[np.ndarray]:
        """Convert bytes back to numpy array"""
        try:
            return np.frombuffer(embedding_bytes, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding decoding failed: {e}")
            return None

    # ========== RAG: Retrieval Augmented Generation ==========

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        RAG-powered memory search with semantic similarity

        Returns most relevant memories based on:
        1. Semantic similarity (vector search)
        2. Importance score
        3. Recency
        """
        if not self.sqlite_conn:
            return []

        cursor = self.sqlite_conn.cursor()

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
        memory_id: str,
        memory_type: str,
        source_agent: str,
        content: str,
        metadata: Optional[Dict] = None,
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
            asyncio.create_task(self._sync_memory_to_master(memory_id))

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

            # Write to master
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO unified_ai_memory (
                        id, memory_type, source_agent, content,
                        metadata, importance_score, created_at, last_accessed
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        importance_score = EXCLUDED.importance_score,
                        last_accessed = EXCLUDED.last_accessed
                """,
                    row['id'], row['memory_type'], row['source_agent'],
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

    def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
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
            asyncio.create_task(self._sync_task_to_master(task_id))

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

    async def sync_from_master(self):
        """Pull latest data from master to local"""
        if not self.pg_pool:
            logger.warning("⚠️ Master sync skipped (no connection)")
            return

        logger.info("🔄 Syncing from master Postgres...")

        try:
            async with self.pg_pool.acquire() as conn:
                # Sync memories
                memories = await conn.fetch("""
                    SELECT * FROM unified_ai_memory
                    ORDER BY created_at DESC
                    LIMIT 1000
                """)

                cursor = self.sqlite_conn.cursor()
                for mem in memories:
                    # Re-generate embedding locally
                    embedding = self._encode_embedding(mem['content']) if mem.get('content') else None

                    cursor.execute("""
                        INSERT OR REPLACE INTO unified_ai_memory (
                            id, memory_type, source_agent, content, embedding,
                            metadata, importance_score, access_count,
                            created_at, last_accessed, synced_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(mem['id']),  # Convert UUID to string
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
                    LIMIT 100
                """)

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
            logger.error(f"❌ Sync from master failed: {e}")

    async def _background_sync(self):
        """Background task for periodic sync (every 5 minutes)"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self.sync_from_master()
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

    def get_stats(self) -> Dict[str, Any]:
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
            "embedding_model": "all-MiniLM-L6-v2" if self.embedding_model else None
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

        if self.pg_pool:
            await self.pg_pool.close()


# Global instance
_embedded_memory = None

async def get_embedded_memory() -> EmbeddedMemorySystem:
    """Get or create global embedded memory instance"""
    global _embedded_memory

    if _embedded_memory is None:
        _embedded_memory = EmbeddedMemorySystem()
        await _embedded_memory.initialize()

    return _embedded_memory
