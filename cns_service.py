#!/usr/bin/env python3
"""
CNS (Central Nervous System) Service - Memory Intelligence Layer
================================================================
Provides intelligent memory operations for the BrainOps AI system with
REAL semantic embeddings (no random vectors).

This service:
1. Uses OpenAI text-embedding-3-small for production-quality embeddings
2. Falls back to Gemini or local sentence-transformers if needed
3. Provides semantic similarity search across the unified memory system
4. Handles the cns_memory table and migrates to unified_ai_memory

Author: BrainOps AI Team
Date: 2025-12-30
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database configuration - supports both individual env vars and DATABASE_URL
from urllib.parse import urlparse

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

# Fallback to DATABASE_URL if individual vars not set
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    _DATABASE_URL = os.getenv('DATABASE_URL', '')
    if _DATABASE_URL:
        _parsed = urlparse(_DATABASE_URL)
        DB_HOST = _parsed.hostname or ''
        DB_NAME = _parsed.path.lstrip('/') if _parsed.path else ''
        DB_USER = _parsed.username or ''
        DB_PASSWORD = _parsed.password or ''
        DB_PORT = str(_parsed.port) if _parsed.port else '5432'

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    raise RuntimeError(
        "Database configuration is incomplete. "
        "Set DB_HOST/DB_NAME/DB_USER/DB_PASSWORD or DATABASE_URL."
    )

DB_CONFIG = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": int(DB_PORT)
}

# Embedding dimension for OpenAI text-embedding-3-small
EMBEDDING_DIMENSION = 1536

# Default tenant for system operations
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")


# =============================================================================
# ENUMS AND MODELS
# =============================================================================

class MemoryType(str, Enum):
    """CNS Memory Types - maps to unified_ai_memory memory_type"""
    SYSTEM_INIT = "system_init"
    SYSTEM_KNOWLEDGE = "system_knowledge"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    META = "meta"


class MemoryCategory(str, Enum):
    """CNS Memory Categories"""
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    OPERATIONAL = "operational"
    LEARNING = "learning"
    DECISION = "decision"
    INSIGHT = "insight"
    ERROR = "error"


@dataclass
class CNSMemory:
    """Central Nervous System Memory Entry"""
    memory_type: MemoryType
    category: MemoryCategory
    title: str
    content: dict[str, Any]
    importance_score: float = 0.5
    tags: list[str] = None
    metadata: dict[str, Any] = None
    tenant_id: Optional[str] = None


# =============================================================================
# EMBEDDING SERVICE - REAL EMBEDDINGS ONLY
# =============================================================================

class EmbeddingService:
    """
    Production-grade embedding service with fallback chain.

    CRITICAL: This service generates REAL semantic embeddings.
    NO random vectors are ever used.

    Fallback order:
    1. OpenAI text-embedding-3-small (fastest, most accurate)
    2. Google Gemini embedding-001 (good alternative)
    3. Local sentence-transformers all-MiniLM-L6-v2 (always available)
    """

    def __init__(self):
        self._openai_client = None
        self._gemini_configured = False
        self._local_model = None
        self._stats = {
            "openai_calls": 0,
            "openai_errors": 0,
            "gemini_calls": 0,
            "gemini_errors": 0,
            "local_calls": 0,
            "local_errors": 0,
            "total_embeddings_generated": 0
        }

    def _init_openai(self):
        """Initialize OpenAI client lazily"""
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    import openai
                    self._openai_client = openai.OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized for embeddings")
                except ImportError:
                    logger.warning("openai package not installed")
        return self._openai_client

    def _init_gemini(self):
        """Initialize Gemini API lazily"""
        if not self._gemini_configured:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    self._gemini_configured = True
                    logger.info("Gemini API configured for embeddings")
                except ImportError:
                    logger.warning("google-generativeai package not installed")
        return self._gemini_configured

    def _init_local_model(self):
        """Initialize local sentence-transformer model lazily"""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Local embedding model loaded (all-MiniLM-L6-v2)")
            except ImportError:
                logger.warning("sentence-transformers package not installed")
        return self._local_model

    def generate_embedding(self, text: str) -> Optional[list[float]]:
        """
        Generate REAL embedding for text using fallback chain.

        IMPORTANT: This method NEVER returns random vectors.
        If all providers fail, it returns None.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats (1536 dimensions) or None if all providers fail
        """
        if not text or not text.strip():
            logger.warning("Cannot generate embedding for empty text")
            return None

        # Truncate very long text
        text = text[:30000] if len(text) > 30000 else text

        # Try OpenAI first (fastest, most accurate)
        embedding = self._try_openai(text)
        if embedding:
            self._stats["total_embeddings_generated"] += 1
            return embedding

        # Try Gemini as fallback
        embedding = self._try_gemini(text)
        if embedding:
            self._stats["total_embeddings_generated"] += 1
            return embedding

        # Try local model as last resort
        embedding = self._try_local(text)
        if embedding:
            self._stats["total_embeddings_generated"] += 1
            return embedding

        logger.error("All embedding providers failed - NO random vectors will be used")
        return None

    def _try_openai(self, text: str) -> Optional[list[float]]:
        """Try OpenAI embedding API"""
        client = self._init_openai()
        if not client:
            return None

        try:
            self._stats["openai_calls"] += 1
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            self._stats["openai_errors"] += 1
            logger.warning(f"OpenAI embedding failed: {e}")
            return None

    def _try_gemini(self, text: str) -> Optional[list[float]]:
        """Try Gemini embedding API with zero-padding to 1536d"""
        if not self._init_gemini():
            return None

        try:
            import google.generativeai as genai
            self._stats["gemini_calls"] += 1

            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']

            # Zero-pad to match OpenAI dimension (1536)
            if len(embedding) < EMBEDDING_DIMENSION:
                embedding = embedding + [0.0] * (EMBEDDING_DIMENSION - len(embedding))

            logger.debug("Used Gemini embedding (padded to 1536d)")
            return embedding
        except Exception as e:
            self._stats["gemini_errors"] += 1
            logger.warning(f"Gemini embedding failed: {e}")
            return None

    def _try_local(self, text: str) -> Optional[list[float]]:
        """Try local sentence-transformer model with zero-padding to 1536d"""
        model = self._init_local_model()
        if not model:
            return None

        try:
            self._stats["local_calls"] += 1
            embedding = model.encode(text).tolist()

            # Zero-pad to match OpenAI dimension (1536)
            if len(embedding) < EMBEDDING_DIMENSION:
                embedding = embedding + [0.0] * (EMBEDDING_DIMENSION - len(embedding))

            logger.debug("Used local embedding model (padded to 1536d)")
            return embedding
        except Exception as e:
            self._stats["local_errors"] += 1
            logger.warning(f"Local embedding failed: {e}")
            return None

    def get_stats(self) -> dict[str, int]:
        """Get embedding generation statistics"""
        return self._stats.copy()


# =============================================================================
# CNS MEMORY SERVICE
# =============================================================================

class CNSMemoryService:
    """
    Central Nervous System Memory Service.

    Manages intelligent memory operations with real semantic embeddings.
    All operations use the canonical unified_ai_memory table.
    """

    def __init__(self, tenant_id: str = None):
        self.tenant_id = tenant_id or DEFAULT_TENANT_ID
        self.embedding_service = EmbeddingService()
        self._conn = None
        logger.info(f"CNS Memory Service initialized for tenant: {self.tenant_id}")

    def _get_connection(self):
        """Get database connection"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**DB_CONFIG)
        return self._conn

    def store(self, memory: CNSMemory) -> Optional[str]:
        """
        Store a memory with real embedding in unified_ai_memory.

        Args:
            memory: CNSMemory object to store

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Prepare content - include title and category in content JSON (not separate columns)
            content_with_meta = dict(memory.content) if isinstance(memory.content, dict) else {"data": memory.content}
            content_with_meta["title"] = memory.title
            content_with_meta["category"] = memory.category.value if hasattr(memory.category, 'value') else str(memory.category)
            content_json = json.dumps(content_with_meta)
            content_text = json.dumps(memory.content, sort_keys=True)

            # Generate REAL embedding
            embedding = self.embedding_service.generate_embedding(content_text)
            embedding_str = None
            if embedding:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            else:
                logger.warning(f"Storing memory '{memory.title}' without embedding")

            # Check for duplicates using content hash
            content_hash = hashlib.sha256(content_text.encode()).hexdigest()
            cursor.execute("""
                SELECT id FROM unified_ai_memory
                WHERE content_hash = %s AND tenant_id = %s::uuid
            """, (content_hash, memory.tenant_id or self.tenant_id))

            existing = cursor.fetchone()
            if existing:
                # Reinforce existing memory
                cursor.execute("""
                    UPDATE unified_ai_memory
                    SET access_count = access_count + 1,
                        importance_score = LEAST(importance_score + 0.05, 1.0),
                        last_accessed = NOW()
                    WHERE id = %s
                    RETURNING id
                """, (existing['id'],))
                result = cursor.fetchone()
                conn.commit()
                logger.info(f"Reinforced existing memory: {result['id']}")
                return str(result['id'])

            # Generate search text
            category_str = memory.category.value if hasattr(memory.category, 'value') else str(memory.category)
            search_text = " ".join([
                memory.title or "",
                content_text[:1000],
                " ".join(memory.tags or []),
                category_str
            ])

            # Insert new memory (category/title are in content JSON, not separate columns)
            cursor.execute("""
                INSERT INTO unified_ai_memory (
                    memory_type, content, importance_score,
                    tags, source_system, source_agent, created_by, metadata,
                    embedding, search_text, tenant_id
                ) VALUES (
                    %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb,
                    %s::vector, %s, %s::uuid
                )
                RETURNING id
            """, (
                memory.memory_type.value,
                content_json,
                memory.importance_score,
                memory.tags or [],
                "cns_service",
                "cns_memory_service",
                "cns_service",
                json.dumps(memory.metadata or {}),
                embedding_str,
                search_text,
                memory.tenant_id or self.tenant_id
            ))

            result = cursor.fetchone()
            conn.commit()

            memory_id = str(result['id'])
            logger.info(f"Stored CNS memory: {memory_id} ({memory.title})")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store CNS memory: {e}")
            if self._conn:
                self._conn.rollback()
            return None

    def recall(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        category: Optional[MemoryCategory] = None,
        threshold: float = 0.5
    ) -> list[dict]:
        """
        Recall relevant memories using semantic similarity search.

        Args:
            query: Search query text
            limit: Maximum number of results
            memory_type: Filter by memory type
            category: Filter by category
            threshold: Minimum similarity threshold

        Returns:
            List of memory dictionaries with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            if not query_embedding:
                logger.warning("Cannot perform semantic search - embedding generation failed")
                return self._fallback_text_search(query, limit, memory_type, category)

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build query
            sql = """
                SELECT
                    id::text,
                    memory_type,
                    content,
                    importance_score,
                    category,
                    title,
                    tags,
                    created_at,
                    access_count,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM unified_ai_memory
                WHERE (tenant_id = %s::uuid OR tenant_id IS NULL)
                    AND embedding IS NOT NULL
            """
            params = [embedding_str, self.tenant_id]

            if memory_type:
                sql += " AND memory_type = %s"
                params.append(memory_type.value)

            if category:
                sql += " AND category = %s"
                params.append(category.value)

            sql += """
                AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY (1 - (embedding <=> %s::vector)) * importance_score DESC
                LIMIT %s
            """
            params.extend([embedding_str, threshold, embedding_str, limit])

            cursor.execute(sql, params)
            results = cursor.fetchall()

            # Update access counts
            if results:
                ids = [r['id'] for r in results]
                cursor.execute("""
                    UPDATE unified_ai_memory
                    SET access_count = access_count + 1,
                        last_accessed = NOW()
                    WHERE id = ANY(%s::uuid[])
                """, (ids,))
                conn.commit()

            return [dict(r) for r in results]

        except Exception as e:
            logger.error(f"Failed to recall CNS memories: {e}")
            return []

    def _fallback_text_search(
        self,
        query: str,
        limit: int,
        memory_type: Optional[MemoryType],
        category: Optional[MemoryCategory]
    ) -> list[dict]:
        """Fallback text search when embedding generation fails"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            sql = """
                SELECT
                    id::text,
                    memory_type,
                    content,
                    importance_score,
                    category,
                    title,
                    tags,
                    created_at,
                    access_count,
                    metadata,
                    NULL::float as similarity
                FROM unified_ai_memory
                WHERE (tenant_id = %s::uuid OR tenant_id IS NULL)
                    AND (search_text ILIKE %s OR content::text ILIKE %s OR title ILIKE %s)
            """
            params = [self.tenant_id, f"%{query}%", f"%{query}%", f"%{query}%"]

            if memory_type:
                sql += " AND memory_type = %s"
                params.append(memory_type.value)

            if category:
                sql += " AND category = %s"
                params.append(category.value)

            sql += " ORDER BY importance_score DESC, created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Fallback text search failed: {e}")
            return []

    def backfill_embeddings(self, batch_size: int = 100, dry_run: bool = False) -> dict[str, int]:
        """
        Backfill missing embeddings in unified_ai_memory.

        Args:
            batch_size: Number of records to process per batch
            dry_run: If True, only count records without updating

        Returns:
            Statistics about the backfill operation
        """
        stats = {
            "total_without_embedding": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0
        }

        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Count records without embeddings
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM unified_ai_memory
                WHERE embedding IS NULL
                AND (tenant_id = %s::uuid OR tenant_id IS NULL)
            """, (self.tenant_id,))
            stats["total_without_embedding"] = cursor.fetchone()["count"]

            if dry_run:
                logger.info(f"Dry run: {stats['total_without_embedding']} records need embeddings")
                return stats

            # Process in batches
            cursor.execute("""
                SELECT id, content, title
                FROM unified_ai_memory
                WHERE embedding IS NULL
                AND (tenant_id = %s::uuid OR tenant_id IS NULL)
                ORDER BY importance_score DESC
                LIMIT %s
            """, (self.tenant_id, batch_size))

            records = cursor.fetchall()

            for record in records:
                stats["processed"] += 1

                # Generate embedding
                content_text = json.dumps(record["content"], sort_keys=True)
                if record["title"]:
                    content_text = f"{record['title']}\n{content_text}"

                embedding = self.embedding_service.generate_embedding(content_text)

                if embedding:
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                    cursor.execute("""
                        UPDATE unified_ai_memory
                        SET embedding = %s::vector
                        WHERE id = %s
                    """, (embedding_str, record["id"]))
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1

            conn.commit()
            logger.info(f"Backfill complete: {stats['successful']}/{stats['processed']} successful")
            return stats

        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            if self._conn:
                self._conn.rollback()
            stats["error"] = str(e)
            return stats

    def migrate_from_cns_memory(self, limit: int = 1000) -> dict[str, int]:
        """
        Migrate records from legacy cns_memory table to unified_ai_memory.
        Generates real embeddings for all migrated records.

        Args:
            limit: Maximum records to migrate

        Returns:
            Migration statistics
        """
        stats = {
            "found": 0,
            "migrated": 0,
            "skipped_duplicate": 0,
            "failed": 0
        }

        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check if cns_memory table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'cns_memory'
                )
            """)
            if not cursor.fetchone()['exists']:
                logger.info("cns_memory table does not exist - nothing to migrate")
                return stats

            # Get records to migrate
            cursor.execute("""
                SELECT
                    memory_id,
                    memory_type,
                    category,
                    title,
                    content,
                    importance_score,
                    tags,
                    metadata,
                    created_at
                FROM cns_memory
                WHERE memory_id NOT IN (
                    SELECT original_id::uuid
                    FROM unified_ai_memory
                    WHERE migrated_from = 'cns_memory'
                    AND original_id IS NOT NULL
                )
                ORDER BY importance_score DESC
                LIMIT %s
            """, (limit,))

            records = cursor.fetchall()
            stats["found"] = len(records)

            for record in records:
                try:
                    # Generate REAL embedding
                    content_text = json.dumps(record["content"], sort_keys=True)
                    full_text = f"{record['title']}\n{content_text}" if record["title"] else content_text

                    embedding = self.embedding_service.generate_embedding(full_text)
                    embedding_str = None
                    if embedding:
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    # Generate search text
                    search_text = " ".join([
                        record["title"] or "",
                        content_text[:1000],
                        " ".join(record["tags"] or []),
                        record["category"] or ""
                    ])

                    # Map memory type
                    mem_type = record["memory_type"]
                    if mem_type not in ['episodic', 'semantic', 'procedural', 'working', 'meta']:
                        mem_type = 'semantic'

                    # Insert into unified_ai_memory
                    cursor.execute("""
                        INSERT INTO unified_ai_memory (
                            memory_type, content, importance_score, category, title,
                            tags, source_system, source_agent, created_by, metadata,
                            embedding, search_text, tenant_id,
                            migrated_from, original_id, migration_date, created_at
                        ) VALUES (
                            %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                            %s::vector, %s, %s::uuid,
                            %s, %s, NOW(), %s
                        )
                    """, (
                        mem_type,
                        json.dumps(record["content"]),
                        record["importance_score"],
                        record["category"],
                        record["title"],
                        record["tags"] or [],
                        "cns_memory_migration",
                        "migration_service",
                        "cns_migration",
                        json.dumps(record["metadata"] or {}),
                        embedding_str,
                        search_text,
                        self.tenant_id,
                        "cns_memory",
                        str(record["memory_id"]),
                        record["created_at"]
                    ))

                    stats["migrated"] += 1

                except Exception as e:
                    logger.warning(f"Failed to migrate record {record['memory_id']}: {e}")
                    stats["failed"] += 1

            conn.commit()
            logger.info(f"CNS migration complete: {stats['migrated']}/{stats['found']} migrated")
            return stats

        except Exception as e:
            logger.error(f"CNS migration failed: {e}")
            if self._conn:
                self._conn.rollback()
            stats["error"] = str(e)
            return stats

    def get_stats(self) -> dict[str, Any]:
        """Get CNS memory system statistics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                    COUNT(DISTINCT memory_type) as memory_types,
                    COUNT(DISTINCT category) as categories,
                    AVG(importance_score) as avg_importance,
                    MAX(created_at) as latest_memory
                FROM unified_ai_memory
                WHERE tenant_id = %s::uuid OR tenant_id IS NULL
            """, (self.tenant_id,))

            db_stats = cursor.fetchone()

            return {
                "database": dict(db_stats),
                "embedding_service": self.embedding_service.get_stats(),
                "tenant_id": self.tenant_id
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_cns_service: Optional[CNSMemoryService] = None


def get_cns_service(tenant_id: str = None) -> CNSMemoryService:
    """Get or create CNS service singleton"""
    global _cns_service
    if _cns_service is None:
        _cns_service = CNSMemoryService(tenant_id)
    return _cns_service


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CNS Memory Service CLI")
    parser.add_argument("command", choices=["stats", "backfill", "migrate", "test"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=1000)

    args = parser.parse_args()

    service = get_cns_service()

    if args.command == "stats":
        stats = service.get_stats()
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "backfill":
        print(f"Backfilling embeddings (batch_size={args.batch_size}, dry_run={args.dry_run})")
        result = service.backfill_embeddings(batch_size=args.batch_size, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))

    elif args.command == "migrate":
        print(f"Migrating from cns_memory (limit={args.limit})")
        result = service.migrate_from_cns_memory(limit=args.limit)
        print(json.dumps(result, indent=2))

    elif args.command == "test":
        print("Testing CNS Memory Service...")

        # Test store
        memory = CNSMemory(
            memory_type=MemoryType.SEMANTIC,
            category=MemoryCategory.OPERATIONAL,
            title="CNS Service Test",
            content={"test": "Real embedding generation", "timestamp": datetime.now().isoformat()},
            importance_score=0.8,
            tags=["test", "cns"]
        )

        memory_id = service.store(memory)
        print(f"Stored memory: {memory_id}")

        # Test recall
        results = service.recall("embedding generation test", limit=5)
        print(f"Recalled {len(results)} memories")

        # Get stats
        stats = service.get_stats()
        print(f"Total embeddings generated: {stats['embedding_service']['total_embeddings_generated']}")

        print("CNS Service test complete!")

    service.close()
