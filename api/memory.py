"""
Memory System API Endpoints - Production Ready
CANONICAL: All operations use unified_ai_memory as the single source of truth
Includes tenant isolation, semantic search, and backward compatibility
"""
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database.async_connection import DatabaseUnavailableError, get_pool, using_fallback

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
    tags: list[str] = Field(default_factory=list)
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
    tags: list[str] = Field(default_factory=list)
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


def _require_real_database(operation: str) -> None:
    if using_fallback():
        message = (
            "Database unavailable; in-memory fallback active. "
            "Configure DATABASE_URL and disable ALLOW_INMEMORY_FALLBACK for production."
        )
        logger.error("Refusing %s: %s", operation, message)
        raise HTTPException(status_code=503, detail=message)


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

async def generate_embedding(text: str) -> Optional[list[float]]:
    """
    Generate embedding with OpenAI primary, Gemini fallback.
    OpenAI: text-embedding-3-small (1536 dims)
    Gemini: gemini-embedding-001 (configurable to 1536 dims for compatibility)
    Returns None if both providers fail.
    """
    # Truncate if too long
    text_truncated = text[:30000] if len(text) > 30000 else text

    # Try OpenAI first (primary)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.embeddings.create(
                input=text_truncated,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed, trying Gemini fallback: {e}")

    # Fallback to Gemini
    gemini_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text_truncated,
                task_type="retrieval_query"  # Optimized for search queries
            )
            if result and "embedding" in result:
                embedding = list(result["embedding"])
                # Gemini text-embedding-004 produces 768 dims, but our DB has 1536 (OpenAI)
                # Pad with zeros to match dimensions for compatibility
                if len(embedding) < 1536:
                    padding = [0.0] * (1536 - len(embedding))
                    embedding = embedding + padding
                    logger.info(f"Gemini embedding padded from {len(result['embedding'])} to {len(embedding)} dims")
                else:
                    logger.info(f"Gemini embedding generated: {len(embedding)} dims")
                return embedding
        except Exception as e:
            logger.error(f"Gemini embedding also failed: {e}")

    logger.error("All embedding providers failed - semantic search unavailable")
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
    try:
        _require_real_database("memory status")
        pool = get_pool()

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

        if not stats:
            return MemoryStatus(
                status="operational",
                table_used=CANONICAL_TABLE,
                message="No memory stats available",
                tenant_id=tenant_id
            )

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

    except HTTPException:
        raise
    except DatabaseUnavailableError as exc:
        logger.error("Memory status unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        logger.error("Failed to get memory status: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to retrieve memory statistics") from e


@router.get("/stats", response_model=MemoryStatus)
async def get_memory_stats(
    tenant_id: str = Depends(get_tenant_id)
) -> MemoryStatus:
    """
    Get memory system statistics (Alias for /status).
    """
    return await get_memory_status(tenant_id)


@router.get("/embedding-status")
async def get_embedding_status():
    """
    Diagnostic endpoint to check embedding capability status.
    Tests both OpenAI (primary) and Gemini (fallback) providers.
    """
    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_status = "configured" if openai_key else "missing"
    openai_test = "untested"
    openai_error = None
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.embeddings.create(input="test", model="text-embedding-3-small")
            if response.data and len(response.data[0].embedding) > 0:
                openai_test = "working"
        except Exception as e:
            openai_test = "failed"
            openai_error = str(e)

    # Check Gemini fallback
    gemini_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    gemini_status = "configured" if gemini_key else "missing"
    gemini_test = "untested"
    gemini_error = None
    gemini_dims = None
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content="test",
                task_type="retrieval_query"
            )
            if result and "embedding" in result and len(result["embedding"]) > 0:
                gemini_dims = len(result["embedding"])
                gemini_test = "working"
        except Exception as e:
            gemini_test = "failed"
            gemini_error = str(e)

    # Semantic search available if either provider works
    semantic_available = openai_test == "working" or gemini_test == "working"
    active_provider = "openai" if openai_test == "working" else ("gemini" if gemini_test == "working" else None)

    return {
        "openai": {"status": openai_status, "test": openai_test, "error": openai_error},
        "gemini": {"status": gemini_status, "test": gemini_test, "error": gemini_error, "native_dims": gemini_dims, "padded_to": 1536 if gemini_test == "working" else None},
        "semantic_search_available": semantic_available,
        "active_provider": active_provider,
        "db_embedding_dims": 1536
    }


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
    try:
        _require_real_database("memory search")
        pool = get_pool()
        results = []

        if use_semantic:
            # Try semantic search first
            query_embedding = await generate_embedding(query)

            if not query_embedding:
                raise HTTPException(
                    status_code=503,
                    detail="Semantic search unavailable. Configure OPENAI_API_KEY or set use_semantic=false."
                )

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory search unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Memory search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}: {str(e)[:200]}") from e


@router.post("/store")
async def store_memory(
    request: StoreMemoryRequest,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Store a new memory in the canonical unified_ai_memory table.
    Generates real embeddings for semantic search.
    """
    try:
        _require_real_database("memory store")
        pool = get_pool()

        # Prepare content as JSON - include category and title in content (not separate columns)
        if isinstance(request.content, str):
            content_json = json.dumps({
                "text": request.content,
                "category": request.category,
                "title": request.title
            })
            content_text = request.content
        else:
            content_dict = dict(request.content) if isinstance(request.content, dict) else {"data": request.content}
            content_dict["category"] = request.category
            content_dict["title"] = request.title
            content_json = json.dumps(content_dict)
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

        # Insert into unified_ai_memory (category/title are in content JSON, not separate columns)
        result = await pool.fetchrow("""
            INSERT INTO unified_ai_memory (
                memory_type, content, importance_score,
                tags, source_system, source_agent, created_by, metadata,
                context_id, embedding, search_text, tenant_id
            ) VALUES (
                $1, $2::jsonb, $3, $4, $5, $6, $7, $8::jsonb,
                $9::uuid, $10::vector, $11, $12::uuid
            )
            RETURNING id::text, content_hash, created_at
        """,
            mem_type.value,
            content_json,
            request.importance_score,
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

        if not result:
            logger.error("Store memory returned no result from database")
            raise HTTPException(
                status_code=503,
                detail="Memory store failed: database did not return a record"
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

    except DatabaseUnavailableError as exc:
        logger.error("Memory store unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to store memory: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Store failed: {type(e).__name__}: {str(e)[:200]}") from e


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
    try:
        _require_real_database("memory by-type")
        pool = get_pool()

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory by-type unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memories by type: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memories by type") from e


@router.get("/by-category/{category}")
async def get_memories_by_category(
    category: str,
    limit: int = Query(20, ge=1, le=100),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memories filtered by category from canonical table"""
    try:
        _require_real_database("memory by-category")
        pool = get_pool()

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory by-category unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memories by category: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memories by category") from e


@router.get("/health")
async def memory_health(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Health check endpoint for memory system.
    Returns operational status and basic statistics.
    MUST be defined before /{memory_id} to avoid route collision.
    """
    try:
        _require_real_database("memory health")
        pool = get_pool()

        stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as recent_hour,
                MAX(created_at) as last_memory_at
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        return {
            "status": "healthy",
            "operational": True,
            "table": CANONICAL_TABLE,
            "total_memories": stats["total_memories"] or 0,
            "memories_with_embeddings": stats["with_embeddings"] or 0,
            "memories_last_hour": stats["recent_hour"] or 0,
            "last_memory_at": stats["last_memory_at"].isoformat() if stats["last_memory_at"] else None,
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except DatabaseUnavailableError as exc:
        logger.error("Memory health unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        logger.error("Memory health check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="Memory health check failed") from e


@router.get("/stats/by-system")
async def get_stats_by_system(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memory statistics grouped by source system"""
    try:
        _require_real_database("memory stats by-system")
        pool = get_pool()

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory stats by-system unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get stats by system: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get stats by system") from e


@router.get("/stats/by-type")
async def get_stats_by_type(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get memory statistics grouped by memory type"""
    try:
        _require_real_database("memory stats by-type")
        pool = get_pool()

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory stats by-type unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get stats by type: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get stats by type") from e


@router.get("/id/{memory_id}")
async def get_memory(
    memory_id: str,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Get a specific memory by ID"""
    # Validate that memory_id looks like a UUID
    try:
        uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Invalid memory ID format: {memory_id}")

    try:
        _require_real_database("memory by-id")
        pool = get_pool()

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory by-id unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memory: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memory") from e


@router.delete("/id/{memory_id}")
async def delete_memory(
    memory_id: str,
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Delete a memory by ID (soft delete by setting expires_at)"""
    # Validate that memory_id looks like a UUID
    try:
        uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Invalid memory ID format: {memory_id}")

    try:
        _require_real_database("memory delete")
        pool = get_pool()

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

    except DatabaseUnavailableError as exc:
        logger.error("Memory delete unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete memory: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete memory") from e


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
