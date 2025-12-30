"""
Memory System API Endpoints - Production Ready
CANONICAL: All operations use unified_ai_memory as the single source of truth
Includes tenant isolation, semantic search, and backward compatibility
"""
import logging
import json
import os
from datetime import datetime
from typing import Any, Optional, List
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field

from database.async_connection import get_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


# =============================================================================
# CONSTANTS
# =============================================================================

# The canonical memory table - all operations go here
CANONICAL_TABLE = "unified_ai_memory"

# Default tenant for backward compatibility
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")


# =============================================================================
# MODELS
# =============================================================================

class MemoryType(str, Enum):
    """Valid memory types in unified_ai_memory"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    META = "meta"


class MemoryStatus(BaseModel):
    """Memory system status model"""
    status: str = Field(..., description="Operational status")
    total_memories: int = Field(default=0, ge=0)
    unique_contexts: int = Field(default=0, ge=0)
    avg_importance: float = Field(default=0.0, ge=0.0)
    memories_with_embeddings: int = Field(default=0, ge=0)
    table_used: str = Field(default=CANONICAL_TABLE)
    tenant_id: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryEntry(BaseModel):
    """Memory entry model for API responses"""
    id: str
    memory_type: str
    content: Any
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    category: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_system: Optional[str] = None
    source_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    similarity: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreMemoryRequest(BaseModel):
    """Request model for storing memories"""
    content: Any = Field(..., description="Memory content (text or JSON)")
    memory_type: str = Field(default="semantic", description="Type: episodic, semantic, procedural, working, meta")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    category: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_system: str = Field(default="api")
    source_agent: str = Field(default="user")
    metadata: dict[str, Any] = Field(default_factory=dict)
    context_id: Optional[str] = None


class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query")
    memory_type: Optional[str] = None
    category: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    importance_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_semantic: bool = Field(default=True, description="Use vector similarity search")


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

async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate real embedding using OpenAI text-embedding-3-small.
    Falls back gracefully if API unavailable.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set - semantic search unavailable")
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)

        # Truncate if too long (8191 token limit for text-embedding-3-small)
        text_truncated = text[:30000] if len(text) > 30000 else text

        response = client.embeddings.create(
            input=text_truncated,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/status", response_model=MemoryStatus)
async def get_memory_status(
    tenant_id: str = Depends(get_tenant_id)
) -> MemoryStatus:
    """
    Get memory system status from canonical unified_ai_memory table.
    Includes tenant-specific statistics.
    """
    pool = get_pool()

    try:
        # Get comprehensive stats from unified_ai_memory
        stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(DISTINCT context_id) as unique_contexts,
                COALESCE(AVG(importance_score), 0.0) as avg_importance,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(DISTINCT source_system) as unique_systems,
                COUNT(DISTINCT memory_type) as memory_types
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        return MemoryStatus(
            status="operational",
            table_used=CANONICAL_TABLE,
            total_memories=stats["total_memories"] or 0,
            unique_contexts=stats["unique_contexts"] or 0,
            avg_importance=float(stats["avg_importance"] or 0.0),
            memories_with_embeddings=stats["with_embeddings"] or 0,
            tenant_id=tenant_id,
            message=f"Canonical table: {CANONICAL_TABLE} | {stats['unique_systems']} systems | {stats['memory_types']} types"
        )

    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        return MemoryStatus(
            status="error",
            message=f"Unable to retrieve memory statistics: {str(e)}",
            total_memories=0,
            unique_contexts=0,
            avg_importance=0.0,
            memories_with_embeddings=0
        )


@router.get("/search")
async def search_memories(
    query: str = Query(..., description="Search query"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    importance_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance"),
    use_semantic: bool = Query(True, description="Use vector similarity search"),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Search memories using semantic vector search (if available) or text search.
    All searches go to the canonical unified_ai_memory table.
    """
    pool = get_pool()

    try:
        results = []

        if use_semantic:
            # Try semantic search first
            query_embedding = await generate_embedding(query)

            if query_embedding:
                # Semantic vector search
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                sql = """
                    SELECT
                        id::text,
                        memory_type,
                        content,
                        importance_score,
                        category,
                        title,
                        tags,
                        source_system,
                        source_agent,
                        created_at,
                        last_accessed,
                        access_count,
                        metadata,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM unified_ai_memory
                    WHERE (tenant_id = $2::uuid OR tenant_id IS NULL)
                        AND embedding IS NOT NULL
                        AND importance_score >= $3
                """
                params = [embedding_str, tenant_id, importance_threshold]
                param_idx = 4

                if memory_type:
                    sql += f" AND memory_type = ${param_idx}"
                    params.append(memory_type)
                    param_idx += 1

                if category:
                    sql += f" AND category = ${param_idx}"
                    params.append(category)
                    param_idx += 1

                sql += f"""
                    ORDER BY (1 - (embedding <=> $1::vector)) * importance_score DESC
                    LIMIT ${param_idx}
                """
                params.append(limit)

                rows = await pool.fetch(sql, *params)
                results = [_row_to_memory_entry(r) for r in rows]

                # Update access counts for retrieved memories
                if results:
                    memory_ids = [r["id"] for r in results]
                    await pool.execute("""
                        UPDATE unified_ai_memory
                        SET access_count = access_count + 1,
                            last_accessed = NOW()
                        WHERE id = ANY($1::uuid[])
                    """, memory_ids)

                return {
                    "results": results,
                    "total": len(results),
                    "query": query,
                    "search_method": "semantic_vector",
                    "table": CANONICAL_TABLE,
                    "tenant_id": tenant_id,
                    "filters": {
                        "memory_type": memory_type,
                        "category": category,
                        "importance_threshold": importance_threshold
                    }
                }

        # Fallback to text search
        sql = """
            SELECT
                id::text,
                memory_type,
                content,
                importance_score,
                category,
                title,
                tags,
                source_system,
                source_agent,
                created_at,
                last_accessed,
                access_count,
                metadata,
                NULL::float as similarity
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND importance_score >= $2
                AND (
                    search_text ILIKE $3
                    OR content::text ILIKE $3
                    OR title ILIKE $3
                )
        """
        params = [tenant_id, importance_threshold, f"%{query}%"]
        param_idx = 4

        if memory_type:
            sql += f" AND memory_type = ${param_idx}"
            params.append(memory_type)
            param_idx += 1

        if category:
            sql += f" AND category = ${param_idx}"
            params.append(category)
            param_idx += 1

        sql += f" ORDER BY importance_score DESC, created_at DESC LIMIT ${param_idx}"
        params.append(limit)

        rows = await pool.fetch(sql, *params)
        results = [_row_to_memory_entry(r) for r in rows]

        return {
            "results": results,
            "total": len(results),
            "query": query,
            "search_method": "text_search",
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id,
            "filters": {
                "memory_type": memory_type,
                "category": category,
                "importance_threshold": importance_threshold
            }
        }

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.post("/store")
async def store_memory(
    request: StoreMemoryRequest,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Store a new memory in the canonical unified_ai_memory table.
    Generates real embeddings for semantic search.
    """
    pool = get_pool()

    try:
        # Prepare content as JSON
        if isinstance(request.content, str):
            content_json = json.dumps({"text": request.content})
            content_text = request.content
        else:
            content_json = json.dumps(request.content)
            content_text = json.dumps(request.content)

        # Generate embedding
        embedding = await generate_embedding(content_text)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None

        # Generate search text
        search_text = " ".join([
            content_text,
            request.title or "",
            " ".join(request.tags),
            request.category or ""
        ])

        # Validate memory_type
        try:
            mem_type = MemoryType(request.memory_type.lower())
        except ValueError:
            mem_type = MemoryType.SEMANTIC

        # Insert into unified_ai_memory
        result = await pool.fetchrow("""
            INSERT INTO unified_ai_memory (
                memory_type, content, importance_score, category, title,
                tags, source_system, source_agent, created_by, metadata,
                context_id, embedding, search_text, tenant_id
            ) VALUES (
                $1, $2::jsonb, $3, $4, $5, $6, $7, $8, $9, $10::jsonb,
                $11::uuid, $12::vector, $13, $14::uuid
            )
            RETURNING id::text, content_hash, created_at
        """,
            mem_type.value,
            content_json,
            request.importance_score,
            request.category,
            request.title,
            request.tags,
            request.source_system,
            request.source_agent,
            "api_user",
            json.dumps(request.metadata),
            request.context_id,
            embedding_str,
            search_text,
            tenant_id
        )

        return {
            "success": True,
            "id": result["id"],
            "content_hash": result["content_hash"],
            "created_at": result["created_at"].isoformat() if result["created_at"] else None,
            "has_embedding": embedding is not None,
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id
        }

    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=f"Store failed: {str(e)}") from e


@router.post("/semantic-search")
async def semantic_search(
    request: SearchRequest,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Dedicated semantic search endpoint using vector similarity.
    Requires embeddings to be present in the database.
    """
    return await search_memories(
        query=request.query,
        memory_type=request.memory_type,
        category=request.category,
        limit=request.limit,
        importance_threshold=request.importance_threshold,
        use_semantic=request.use_semantic,
        tenant_id=tenant_id
    )


@router.get("/by-type/{memory_type}")
async def get_memories_by_type(
    memory_type: str,
    limit: int = Query(20, ge=1, le=100),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memories filtered by type from canonical table"""
    pool = get_pool()

    try:
        rows = await pool.fetch("""
            SELECT
                id::text,
                memory_type,
                content,
                importance_score,
                category,
                title,
                tags,
                source_system,
                source_agent,
                created_at,
                last_accessed,
                access_count,
                metadata,
                NULL::float as similarity
            FROM unified_ai_memory
            WHERE memory_type = $1
                AND (tenant_id = $2::uuid OR tenant_id IS NULL)
            ORDER BY importance_score DESC, created_at DESC
            LIMIT $3
        """, memory_type, tenant_id, limit)

        return {
            "results": [_row_to_memory_entry(r) for r in rows],
            "total": len(rows),
            "memory_type": memory_type,
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id
        }

    except Exception as e:
        logger.error(f"Failed to get memories by type: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/by-category/{category}")
async def get_memories_by_category(
    category: str,
    limit: int = Query(20, ge=1, le=100),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memories filtered by category from canonical table"""
    pool = get_pool()

    try:
        rows = await pool.fetch("""
            SELECT
                id::text,
                memory_type,
                content,
                importance_score,
                category,
                title,
                tags,
                source_system,
                source_agent,
                created_at,
                last_accessed,
                access_count,
                metadata,
                NULL::float as similarity
            FROM unified_ai_memory
            WHERE category = $1
                AND (tenant_id = $2::uuid OR tenant_id IS NULL)
            ORDER BY importance_score DESC, created_at DESC
            LIMIT $3
        """, category, tenant_id, limit)

        return {
            "results": [_row_to_memory_entry(r) for r in rows],
            "total": len(rows),
            "category": category,
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id
        }

    except Exception as e:
        logger.error(f"Failed to get memories by category: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{memory_id}")
async def get_memory(
    memory_id: str,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get a specific memory by ID"""
    pool = get_pool()

    try:
        row = await pool.fetchrow("""
            SELECT
                id::text,
                memory_type,
                content,
                importance_score,
                category,
                title,
                tags,
                source_system,
                source_agent,
                created_at,
                last_accessed,
                access_count,
                metadata,
                content_hash,
                context_id::text,
                parent_memory_id::text,
                related_memories,
                embedding IS NOT NULL as has_embedding
            FROM unified_ai_memory
            WHERE id = $1::uuid
                AND (tenant_id = $2::uuid OR tenant_id IS NULL)
        """, memory_id, tenant_id)

        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Update access count
        await pool.execute("""
            UPDATE unified_ai_memory
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE id = $1::uuid
        """, memory_id)

        return {
            "success": True,
            "memory": dict(row),
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Delete a memory by ID (soft delete by setting expires_at)"""
    pool = get_pool()

    try:
        result = await pool.execute("""
            UPDATE unified_ai_memory
            SET expires_at = NOW()
            WHERE id = $1::uuid
                AND (tenant_id = $2::uuid OR tenant_id IS NULL)
        """, memory_id, tenant_id)

        if "UPDATE 0" in result:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {
            "success": True,
            "message": "Memory marked for deletion",
            "id": memory_id,
            "table": CANONICAL_TABLE
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats/by-system")
async def get_stats_by_system(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memory statistics grouped by source system"""
    pool = get_pool()

    try:
        rows = await pool.fetch("""
            SELECT
                source_system,
                COUNT(*) as count,
                AVG(importance_score) as avg_importance,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                MAX(created_at) as latest
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY source_system
            ORDER BY count DESC
        """, tenant_id)

        return {
            "stats": [dict(r) for r in rows],
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id
        }

    except Exception as e:
        logger.error(f"Failed to get stats by system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats/by-type")
async def get_stats_by_type(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memory statistics grouped by memory type"""
    pool = get_pool()

    try:
        rows = await pool.fetch("""
            SELECT
                memory_type,
                COUNT(*) as count,
                AVG(importance_score) as avg_importance,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                MAX(created_at) as latest
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY memory_type
            ORDER BY count DESC
        """, tenant_id)

        return {
            "stats": [dict(r) for r in rows],
            "table": CANONICAL_TABLE,
            "tenant_id": tenant_id
        }

    except Exception as e:
        logger.error(f"Failed to get stats by type: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _row_to_memory_entry(row) -> dict:
    """Convert database row to memory entry dict"""
    return {
        "id": row["id"],
        "memory_type": row["memory_type"],
        "content": row["content"],
        "importance": float(row["importance_score"] or 0.5),
        "category": row["category"],
        "title": row["title"],
        "tags": row["tags"] or [],
        "source_system": row["source_system"],
        "source_agent": row["source_agent"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "last_accessed": row["last_accessed"].isoformat() if row["last_accessed"] else None,
        "access_count": row["access_count"] or 0,
        "similarity": float(row["similarity"]) if row["similarity"] else None,
        "metadata": row["metadata"] or {}
    }
