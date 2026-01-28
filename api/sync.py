#!/usr/bin/env python3
"""
Memory Sync API - One-time migration and ongoing synchronization
================================================================
Migrates data from legacy memory tables to unified_ai_memory:
- memories (239k+ rows)
- production_memory (56k+ rows)
- cns_memory (191 rows)

Generates REAL embeddings using OpenAI text-embedding-3-small.
Deduplicates by content hash.

Author: BrainOps AI Team
Date: 2025-12-30
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database.async_connection import get_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory/sync", tags=["memory-sync"])


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")
BATCH_SIZE = 100
EMBEDDING_DIMENSION = 1536

# Track migration progress (in-memory for now, should be persisted)
_migration_status: dict[str, Any] = {
    "running": False,
    "current_table": None,
    "progress": {},
    "errors": [],
    "started_at": None,
    "completed_at": None
}


# =============================================================================
# MODELS
# =============================================================================

class MigrationStatus(BaseModel):
    """Migration status response"""
    running: bool
    current_table: Optional[str]
    progress: dict[str, Any]
    errors: list[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class MigrationRequest(BaseModel):
    """Migration request parameters"""
    tables: list[str] = Field(
        default=["memories", "production_memory", "cns_memory"],
        description="Tables to migrate"
    )
    batch_size: int = Field(default=100, ge=10, le=1000)
    generate_embeddings: bool = Field(default=True, description="Generate real embeddings")
    dry_run: bool = Field(default=False, description="Preview without writing")
    limit_per_table: Optional[int] = Field(default=None, description="Limit records per table")


class BackfillRequest(BaseModel):
    """Embedding backfill request"""
    batch_size: int = Field(default=100, ge=10, le=1000)
    dry_run: bool = Field(default=False)


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> str:
    """Extract tenant ID from header or use default"""
    return x_tenant_id or DEFAULT_TENANT_ID


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

async def generate_embedding(text: str) -> Optional[list[float]]:
    """
    Generate REAL embedding using OpenAI with fallbacks.
    NEVER returns random vectors.
    """
    if not text or not text.strip():
        return None

    # Truncate if too long
    text = text[:30000] if len(text) > 30000 else text

    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")

    # Try Gemini as fallback
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embedding = list(result['embedding'])
            # Truncate or pad to 1536d
            original_len = len(embedding)
            if len(embedding) > EMBEDDING_DIMENSION:
                embedding = embedding[:EMBEDDING_DIMENSION]
                logger.info(f"Gemini embedding truncated from {original_len} to {EMBEDDING_DIMENSION}d")
            elif len(embedding) < EMBEDDING_DIMENSION:
                embedding = embedding + [0.0] * (EMBEDDING_DIMENSION - len(embedding))
                logger.info(f"Gemini embedding padded from {original_len} to {EMBEDDING_DIMENSION}d")
            return embedding
        except Exception as e:
            logger.warning(f"Gemini embedding failed: {e}")

    # Try local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text).tolist()
        # Zero-pad to 1536d
        if len(embedding) < EMBEDDING_DIMENSION:
            embedding = embedding + [0.0] * (EMBEDDING_DIMENSION - len(embedding))
        return embedding
    except Exception as e:
        logger.warning(f"Local embedding failed: {e}")

    logger.error("All embedding providers failed")
    return None


# =============================================================================
# MIGRATION FUNCTIONS
# =============================================================================

async def migrate_memories_table(
    pool,
    tenant_id: str,
    batch_size: int = 100,
    generate_embeddings: bool = True,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict[str, int]:
    """
    Migrate from 'memories' table to unified_ai_memory.

    memories table schema:
    - id (uuid)
    - tenant_id (text)
    - entity_type (text)
    - entity_id (text)
    - kind (text) - default 'fact'
    - content (text)
    - source (text)
    - created_by (text)
    - created_at (timestamptz)
    - metadata (jsonb)
    """
    stats = {"found": 0, "migrated": 0, "skipped_duplicate": 0, "failed": 0}

    try:
        # Check for already migrated records
        offset = 0
        total_limit = limit or 1_000_000

        while True:
            # Use parameterized query for batch_size and offset
            query = """
                SELECT
                    m.id::text as original_id,
                    m.tenant_id,
                    m.entity_type,
                    m.entity_id,
                    m.kind,
                    m.content,
                    m.source,
                    m.created_by,
                    m.created_at,
                    m.metadata
                FROM memories m
                WHERE NOT EXISTS (
                    SELECT 1 FROM unified_ai_memory u
                    WHERE u.original_id = m.id::text
                    AND u.migrated_from = 'memories'
                )
                ORDER BY m.created_at DESC
                LIMIT $1 OFFSET $2
            """

            rows = await pool.fetch(query, batch_size, offset)
            if not rows:
                break

            stats["found"] += len(rows)

            for row in rows:
                if stats["migrated"] + stats["skipped_duplicate"] >= total_limit:
                    break

                try:
                    # Prepare content - include migration metadata in content JSON
                    content_text = row["content"] or ""
                    content_json = json.dumps({
                        "text": content_text,
                        "entity_type": row["entity_type"],
                        "entity_id": row["entity_id"],
                        "kind": row["kind"],
                        "source": row["source"],
                        "migrated_from": "memories",
                        "original_id": row["original_id"]
                    })

                    # Check for duplicate by content hash
                    content_hash = hashlib.sha256(content_json.encode()).hexdigest()

                    existing = await pool.fetchrow("""
                        SELECT id FROM unified_ai_memory WHERE content_hash = $1
                    """, content_hash)

                    if existing:
                        stats["skipped_duplicate"] += 1
                        continue

                    if dry_run:
                        stats["migrated"] += 1
                        continue

                    # Generate embedding
                    embedding_str = None
                    if generate_embeddings and content_text:
                        embedding = await generate_embedding(content_text)
                        if embedding:
                            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    # Map kind to memory_type
                    kind_to_type = {
                        "fact": "semantic",
                        "event": "episodic",
                        "procedure": "procedural",
                        "observation": "semantic"
                    }
                    memory_type = kind_to_type.get(row["kind"], "semantic")

                    # Generate search text
                    search_text = " ".join(filter(None, [
                        content_text[:1000],
                        row["entity_type"],
                        row["entity_id"],
                        row["source"]
                    ]))

                    # Insert (migration metadata stored in content JSON, use source_entity_* columns)
                    await pool.execute("""
                        INSERT INTO unified_ai_memory (
                            memory_type, content, source_system, source_agent, created_by,
                            source_entity_type, source_entity_id, metadata,
                            embedding, search_text, tenant_id, created_at
                        ) VALUES (
                            $1, $2::jsonb, $3, $4, $5,
                            $6, $7, $8::jsonb,
                            $9::vector, $10, $11::uuid, $12
                        )
                    """,
                        memory_type,
                        content_json,
                        "memories_migration",
                        "sync_service",
                        row["created_by"] or "migration",
                        row["entity_type"],
                        row["entity_id"],
                        json.dumps(row["metadata"] or {}),
                        embedding_str,
                        search_text,
                        tenant_id,
                        row["created_at"]
                    )

                    stats["migrated"] += 1

                except Exception as e:
                    logger.warning(f"Failed to migrate memory {row['original_id']}: {e}")
                    stats["failed"] += 1

            offset += batch_size

            if stats["migrated"] + stats["skipped_duplicate"] >= total_limit:
                break

        return stats

    except Exception as e:
        logger.error(f"memories migration failed: {e}")
        stats["error"] = str(e)
        return stats


async def migrate_production_memory_table(
    pool,
    tenant_id: str,
    batch_size: int = 100,
    generate_embeddings: bool = True,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict[str, int]:
    """
    Migrate from 'production_memory' table to unified_ai_memory.

    production_memory table schema:
    - id (integer, serial)
    - memory_type (text)
    - content (text)
    - context (jsonb)
    - session_id (text)
    - system_name (text)
    - importance (float, 0-1)
    - embedding (float[]) - NOTE: may be random vectors!
    - created_at (timestamptz)
    - last_accessed (timestamptz)
    - access_count (integer)
    - metadata (jsonb)
    """
    stats = {"found": 0, "migrated": 0, "skipped_duplicate": 0, "regenerated_embeddings": 0, "failed": 0}

    try:
        offset = 0
        total_limit = limit or 1_000_000

        while True:
            # Use parameterized query for batch_size and offset
            query = """
                SELECT
                    p.id::text as original_id,
                    p.memory_type,
                    p.content,
                    p.context,
                    p.session_id,
                    p.system_name,
                    p.importance,
                    p.created_at,
                    p.last_accessed,
                    p.access_count,
                    p.metadata
                FROM production_memory p
                WHERE NOT EXISTS (
                    SELECT 1 FROM unified_ai_memory u
                    WHERE u.original_id = p.id::text
                    AND u.migrated_from = 'production_memory'
                )
                ORDER BY p.importance DESC, p.created_at DESC
                LIMIT $1 OFFSET $2
            """

            rows = await pool.fetch(query, batch_size, offset)
            if not rows:
                break

            stats["found"] += len(rows)

            for row in rows:
                if stats["migrated"] + stats["skipped_duplicate"] >= total_limit:
                    break

                try:
                    # Prepare content - include all migration metadata in content JSON
                    content_text = row["content"] or ""
                    content_data = {
                        "text": content_text,
                        "context": row["context"] or {},
                        "session_id": row["session_id"],
                        "system_name": row["system_name"],
                        "migrated_from": "production_memory",
                        "original_id": row["original_id"]
                    }
                    content_json = json.dumps(content_data)

                    # Check for duplicate by content hash
                    content_hash = hashlib.sha256(content_json.encode()).hexdigest()

                    existing = await pool.fetchrow("""
                        SELECT id FROM unified_ai_memory WHERE content_hash = $1
                    """, content_hash)

                    if existing:
                        stats["skipped_duplicate"] += 1
                        continue

                    if dry_run:
                        stats["migrated"] += 1
                        continue

                    # ALWAYS regenerate embeddings (production_memory may have random vectors)
                    embedding_str = None
                    if generate_embeddings and content_text:
                        embedding = await generate_embedding(content_text)
                        if embedding:
                            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                            stats["regenerated_embeddings"] += 1

                    # Map memory_type
                    valid_types = ['episodic', 'semantic', 'procedural', 'working', 'meta']
                    memory_type = row["memory_type"] if row["memory_type"] in valid_types else "semantic"

                    # Generate search text
                    search_text = " ".join(filter(None, [
                        content_text[:1000],
                        row["session_id"],
                        row["system_name"]
                    ]))

                    # Insert (migration metadata in content JSON, use only valid columns)
                    await pool.execute("""
                        INSERT INTO unified_ai_memory (
                            memory_type, content, importance_score, source_system, source_agent,
                            created_by, metadata,
                            embedding, search_text, tenant_id, access_count, last_accessed, created_at
                        ) VALUES (
                            $1, $2::jsonb, $3, $4, $5,
                            $6, $7::jsonb,
                            $8::vector, $9, $10::uuid, $11, $12, $13
                        )
                    """,
                        memory_type,
                        content_json,
                        row["importance"] or 0.5,
                        "production_memory_migration",
                        "sync_service",
                        "migration",
                        json.dumps(row["metadata"] or {}),
                        embedding_str,
                        search_text,
                        tenant_id,
                        row["access_count"] or 0,
                        row["last_accessed"],
                        row["created_at"]
                    )

                    stats["migrated"] += 1

                except Exception as e:
                    logger.warning(f"Failed to migrate production_memory {row['original_id']}: {e}")
                    stats["failed"] += 1

            offset += batch_size

            if stats["migrated"] + stats["skipped_duplicate"] >= total_limit:
                break

        return stats

    except Exception as e:
        logger.error(f"production_memory migration failed: {e}")
        stats["error"] = str(e)
        return stats


async def migrate_cns_memory_table(
    pool,
    tenant_id: str,
    batch_size: int = 100,
    generate_embeddings: bool = True,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> dict[str, int]:
    """
    Migrate from 'cns_memory' table to unified_ai_memory.

    cns_memory table schema:
    - memory_id (uuid)
    - memory_type (varchar)
    - category (varchar)
    - title (text)
    - content (jsonb)
    - embedding (vector) - may be NULL or have invalid data
    - importance_score (float)
    - tags (text[])
    - metadata (jsonb)
    - created_at, updated_at, accessed_count, last_accessed, expires_at
    """
    stats = {"found": 0, "migrated": 0, "skipped_duplicate": 0, "failed": 0}

    try:
        offset = 0
        total_limit = limit or 1_000_000

        while True:
            # Use parameterized query for batch_size and offset
            query = """
                SELECT
                    c.memory_id::text as original_id,
                    c.memory_type,
                    c.category,
                    c.title,
                    c.content,
                    c.importance_score,
                    c.tags,
                    c.metadata,
                    c.created_at,
                    c.updated_at,
                    c.accessed_count,
                    c.last_accessed,
                    c.expires_at
                FROM cns_memory c
                WHERE NOT EXISTS (
                    SELECT 1 FROM unified_ai_memory u
                    WHERE u.original_id = c.memory_id::text
                    AND u.migrated_from = 'cns_memory'
                )
                ORDER BY c.importance_score DESC
                LIMIT $1 OFFSET $2
            """

            rows = await pool.fetch(query, batch_size, offset)
            if not rows:
                break

            stats["found"] += len(rows)

            for row in rows:
                if stats["migrated"] + stats["skipped_duplicate"] >= total_limit:
                    break

                try:
                    # Prepare content - include category, title, migration metadata
                    content_base = row["content"] if row["content"] else {}
                    if isinstance(content_base, dict):
                        content_with_meta = dict(content_base)
                    else:
                        content_with_meta = {"data": content_base}
                    content_with_meta["title"] = row["title"]
                    content_with_meta["category"] = row["category"]
                    content_with_meta["migrated_from"] = "cns_memory"
                    content_with_meta["original_id"] = row["original_id"]
                    content_json = json.dumps(content_with_meta)
                    content_text = json.dumps(row["content"], sort_keys=True) if row["content"] else ""

                    # Check for duplicate
                    content_hash = hashlib.sha256(content_json.encode()).hexdigest()
                    existing = await pool.fetchrow("""
                        SELECT id FROM unified_ai_memory WHERE content_hash = $1
                    """, content_hash)

                    if existing:
                        stats["skipped_duplicate"] += 1
                        continue

                    if dry_run:
                        stats["migrated"] += 1
                        continue

                    # Generate REAL embedding (CNS table may have NULL or invalid embeddings)
                    embedding_str = None
                    if generate_embeddings:
                        full_text = f"{row['title'] or ''}\n{content_text}"
                        embedding = await generate_embedding(full_text)
                        if embedding:
                            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    # Map memory_type
                    valid_types = ['episodic', 'semantic', 'procedural', 'working', 'meta', 'system_init', 'system_knowledge']
                    memory_type = row["memory_type"] if row["memory_type"] in valid_types else "semantic"

                    # Generate search text
                    search_text = " ".join(filter(None, [
                        row["title"] or "",
                        content_text[:1000],
                        " ".join(row["tags"] or []),
                        row["category"] or ""
                    ]))

                    # Insert (category/title/migration metadata in content JSON)
                    await pool.execute("""
                        INSERT INTO unified_ai_memory (
                            memory_type, content, importance_score,
                            tags, source_system, source_agent, created_by, metadata,
                            embedding, search_text, tenant_id, access_count, last_accessed, expires_at,
                            created_at, updated_at
                        ) VALUES (
                            $1, $2::jsonb, $3,
                            $4, $5, $6, $7, $8::jsonb,
                            $9::vector, $10, $11::uuid, $12, $13, $14,
                            $15, $16
                        )
                    """,
                        memory_type,
                        content_json,
                        row["importance_score"] or 0.5,
                        row["tags"] or [],
                        "cns_memory_migration",
                        "sync_service",
                        "migration",
                        json.dumps(row["metadata"] or {}),
                        embedding_str,
                        search_text,
                        tenant_id,
                        row["accessed_count"] or 0,
                        row["last_accessed"],
                        row["expires_at"],
                        row["created_at"],
                        row["updated_at"]
                    )

                    stats["migrated"] += 1

                except Exception as e:
                    logger.warning(f"Failed to migrate cns_memory {row['original_id']}: {e}")
                    stats["failed"] += 1

            offset += batch_size

            if stats["migrated"] + stats["skipped_duplicate"] >= total_limit:
                break

        return stats

    except Exception as e:
        logger.error(f"cns_memory migration failed: {e}")
        stats["error"] = str(e)
        return stats


async def backfill_missing_embeddings(
    pool,
    tenant_id: str,
    batch_size: int = 100,
    dry_run: bool = False
) -> dict[str, int]:
    """Backfill embeddings for records that don't have them."""
    stats = {"total_without": 0, "processed": 0, "successful": 0, "failed": 0}

    try:
        # Count records without embeddings
        count = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_ai_memory
            WHERE embedding IS NULL
            AND (tenant_id = $1::uuid OR tenant_id IS NULL)
        """, tenant_id)
        stats["total_without"] = count

        if dry_run:
            return stats

        # Process in batches
        rows = await pool.fetch("""
            SELECT id, content, title
            FROM unified_ai_memory
            WHERE embedding IS NULL
            AND (tenant_id = $1::uuid OR tenant_id IS NULL)
            ORDER BY importance_score DESC
            LIMIT $2
        """, tenant_id, batch_size)

        for row in rows:
            stats["processed"] += 1
            try:
                content_text = json.dumps(row["content"], sort_keys=True)
                if row["title"]:
                    content_text = f"{row['title']}\n{content_text}"

                embedding = await generate_embedding(content_text)
                if embedding:
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                    await pool.execute("""
                        UPDATE unified_ai_memory
                        SET embedding = $1::vector
                        WHERE id = $2
                    """, embedding_str, row["id"])
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                logger.warning(f"Failed to backfill embedding for {row['id']}: {e}")
                stats["failed"] += 1

        return stats

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        stats["error"] = str(e)
        return stats


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_migration_status() -> MigrationStatus:
    """Get current migration status"""
    return MigrationStatus(**_migration_status)


@router.get("/preview")
async def preview_migration(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Preview migration without executing - shows what would be migrated"""
    pool = get_pool()

    try:
        # Count records in each source table
        memories_count = await pool.fetchval("SELECT COUNT(*) FROM memories")
        production_count = await pool.fetchval("SELECT COUNT(*) FROM production_memory")
        cns_count = await pool.fetchval("SELECT COUNT(*) FROM cns_memory")
        unified_count = await pool.fetchval("SELECT COUNT(*) FROM unified_ai_memory")

        # Count already migrated
        migrated_from_memories = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_ai_memory WHERE migrated_from = 'memories'
        """)
        migrated_from_production = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_ai_memory WHERE migrated_from = 'production_memory'
        """)
        migrated_from_cns = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_ai_memory WHERE migrated_from = 'cns_memory'
        """)

        # Count without embeddings
        without_embeddings = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_ai_memory WHERE embedding IS NULL
        """)

        return {
            "source_tables": {
                "memories": {
                    "total": memories_count,
                    "already_migrated": migrated_from_memories,
                    "remaining": memories_count - migrated_from_memories
                },
                "production_memory": {
                    "total": production_count,
                    "already_migrated": migrated_from_production,
                    "remaining": production_count - migrated_from_production
                },
                "cns_memory": {
                    "total": cns_count,
                    "already_migrated": migrated_from_cns,
                    "remaining": cns_count - migrated_from_cns
                }
            },
            "unified_ai_memory": {
                "total": unified_count,
                "without_embeddings": without_embeddings,
                "with_embeddings": unified_count - without_embeddings
            },
            "estimated_embedding_cost": f"~${(memories_count + production_count + cns_count - migrated_from_memories - migrated_from_production - migrated_from_cns) * 0.00002:.2f} (OpenAI)"
        }

    except Exception as e:
        logger.error(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/migrate")
async def start_migration(
    request: MigrationRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Start migration from legacy tables to unified_ai_memory.
    Runs in background to avoid timeout.
    """
    global _migration_status

    if _migration_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Migration already in progress"
        )

    async def run_migration():
        global _migration_status
        _migration_status["running"] = True
        _migration_status["started_at"] = datetime.now()
        _migration_status["errors"] = []
        _migration_status["progress"] = {}

        pool = get_pool()

        try:
            for table in request.tables:
                _migration_status["current_table"] = table

                if table == "memories":
                    stats = await migrate_memories_table(
                        pool, tenant_id,
                        batch_size=request.batch_size,
                        generate_embeddings=request.generate_embeddings,
                        dry_run=request.dry_run,
                        limit=request.limit_per_table
                    )
                elif table == "production_memory":
                    stats = await migrate_production_memory_table(
                        pool, tenant_id,
                        batch_size=request.batch_size,
                        generate_embeddings=request.generate_embeddings,
                        dry_run=request.dry_run,
                        limit=request.limit_per_table
                    )
                elif table == "cns_memory":
                    stats = await migrate_cns_memory_table(
                        pool, tenant_id,
                        batch_size=request.batch_size,
                        generate_embeddings=request.generate_embeddings,
                        dry_run=request.dry_run,
                        limit=request.limit_per_table
                    )
                else:
                    continue

                _migration_status["progress"][table] = stats

                if "error" in stats:
                    _migration_status["errors"].append(f"{table}: {stats['error']}")

        except Exception as e:
            _migration_status["errors"].append(str(e))

        finally:
            _migration_status["running"] = False
            _migration_status["current_table"] = None
            _migration_status["completed_at"] = datetime.now()

    background_tasks.add_task(run_migration)

    return {
        "message": "Migration started in background",
        "tables": request.tables,
        "dry_run": request.dry_run,
        "check_status_at": "/api/memory/sync/status"
    }


@router.post("/backfill-embeddings")
async def backfill_embeddings(
    request: BackfillRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Backfill missing embeddings in unified_ai_memory"""
    pool = get_pool()

    if request.dry_run:
        stats = await backfill_missing_embeddings(
            pool, tenant_id,
            batch_size=request.batch_size,
            dry_run=True
        )
        return {
            "dry_run": True,
            "records_without_embeddings": stats["total_without"],
            "would_process": min(stats["total_without"], request.batch_size)
        }

    async def run_backfill():
        await backfill_missing_embeddings(
            pool, tenant_id,
            batch_size=request.batch_size,
            dry_run=False
        )

    background_tasks.add_task(run_backfill)

    return {
        "message": "Embedding backfill started in background",
        "batch_size": request.batch_size
    }


@router.post("/migrate-single/{table_name}")
async def migrate_single_table(
    table_name: str,
    batch_size: int = Query(default=100, ge=10, le=1000),
    limit: Optional[int] = Query(default=None),
    generate_embeddings: bool = Query(default=True),
    dry_run: bool = Query(default=False),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Migrate a single table synchronously (use for smaller tables)"""
    pool = get_pool()

    valid_tables = ["memories", "production_memory", "cns_memory"]
    if table_name not in valid_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table. Must be one of: {valid_tables}")

    try:
        if table_name == "memories":
            stats = await migrate_memories_table(
                pool, tenant_id, batch_size, generate_embeddings, dry_run, limit
            )
        elif table_name == "production_memory":
            stats = await migrate_production_memory_table(
                pool, tenant_id, batch_size, generate_embeddings, dry_run, limit
            )
        elif table_name == "cns_memory":
            stats = await migrate_cns_memory_table(
                pool, tenant_id, batch_size, generate_embeddings, dry_run, limit
            )

        return {
            "table": table_name,
            "dry_run": dry_run,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Migration of {table_name} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
