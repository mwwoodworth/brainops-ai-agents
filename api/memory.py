"""
Memory System API Endpoints - Production Ready
Fixed to work with actual database schema
"""
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database.async_connection import get_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])

# Table schema mappings - actual column names in each table
TABLE_SCHEMAS = {
    "ai_persistent_memory": {
        "id_col": "context_key",
        "content_col": "content",  # jsonb
        "importance_col": "importance_score",
        "created_col": "created_at",
        "content_is_jsonb": True,
    },
    "memory_entries": {
        "id_col": "owner_id",
        "content_col": "content",  # text
        "importance_col": "importance",
        "created_col": "created_at",
        "tags_col": "tags",
        "content_is_jsonb": False,
    },
    "memories": {
        "id_col": "entity_id",
        "content_col": "content",  # text
        "importance_col": None,
        "created_col": "created_at",
        "content_is_jsonb": False,
    },
}


class MemoryStatus(BaseModel):
    """Memory system status model"""
    status: str = Field(..., description="Operational status")
    total_memories: int = Field(default=0, ge=0)
    unique_contexts: int = Field(default=0, ge=0)
    avg_importance: float = Field(default=0.0, ge=0.0)
    table_used: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryEntry(BaseModel):
    """Memory entry model"""
    id: str
    context_key: Optional[str] = None
    content: Any  # Can be text or jsonb
    importance: float = Field(default=0.0, ge=0.0)
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("/status", response_model=MemoryStatus)
async def get_memory_status() -> MemoryStatus:
    """Get memory system status with proper error handling"""
    pool = get_pool()

    try:
        # Check which memory tables exist
        existing_tables = await pool.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ai_persistent_memory', 'memory_entries', 'memories')
        """)

        if not existing_tables:
            return MemoryStatus(
                status="not_configured",
                message="Memory system tables not found",
                total_memories=0,
                unique_contexts=0,
                avg_importance=0.0
            )

        # Priority order: ai_persistent_memory > memory_entries > memories
        table_priority = ["ai_persistent_memory", "memory_entries", "memories"]
        table_names = [t["table_name"] for t in existing_tables]

        table_name = None
        for t in table_priority:
            if t in table_names:
                table_name = t
                break

        if not table_name:
            table_name = table_names[0]

        schema = TABLE_SCHEMAS.get(table_name, {})
        id_col = schema.get("id_col", "id")
        importance_col = schema.get("importance_col")

        # Build stats query based on table schema
        if importance_col:
            stats_query = f"""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT {id_col}) as unique_contexts,
                    COALESCE(AVG({importance_col}::numeric), 0.0) as avg_importance
                FROM {table_name}
            """
        else:
            stats_query = f"""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT {id_col}) as unique_contexts,
                    0.0 as avg_importance
                FROM {table_name}
            """

        stats = await pool.fetchrow(stats_query)

        return MemoryStatus(
            status="operational",
            table_used=table_name,
            total_memories=stats["total_memories"] or 0,
            unique_contexts=stats["unique_contexts"] or 0,
            avg_importance=float(stats["avg_importance"] or 0.0),
            message=f"Using table: {table_name}"
        )

    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        return MemoryStatus(
            status="error",
            message=f"Unable to retrieve memory statistics: {str(e)}",
            total_memories=0,
            unique_contexts=0,
            avg_importance=0.0
        )


@router.get("/search")
async def search_memories(
    query: str = Query(..., description="Search query"),
    context_key: Optional[str] = Query(None, description="Filter by context key"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    importance_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance")
) -> dict[str, Any]:
    """Search memory entries across available memory tables"""
    pool = get_pool()

    try:
        # Determine which table to use
        existing_tables = await pool.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ai_persistent_memory', 'memory_entries', 'memories')
        """)

        if not existing_tables:
            return {
                "results": [],
                "total": 0,
                "query": query,
                "message": "Memory system not initialized"
            }

        table_names = [t["table_name"] for t in existing_tables]

        # Try memory_entries first (has better schema for searching)
        if "memory_entries" in table_names:
            return await _search_memory_entries(pool, query, context_key, limit, importance_threshold)
        elif "ai_persistent_memory" in table_names:
            return await _search_ai_persistent_memory(pool, query, context_key, limit, importance_threshold)
        elif "memories" in table_names:
            return await _search_memories(pool, query, context_key, limit)

        return {
            "results": [],
            "total": 0,
            "query": query,
            "message": "No compatible memory table found"
        }

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


async def _search_memory_entries(
    pool, query: str, context_key: Optional[str], limit: int, importance_threshold: float
) -> dict[str, Any]:
    """Search in memory_entries table"""
    base_query = """
        SELECT
            id::text,
            owner_id,
            content,
            importance,
            created_at,
            tags
        FROM memory_entries
        WHERE importance >= $1
    """
    params: list[Any] = [int(importance_threshold * 100)]  # importance is integer 0-100

    if context_key:
        base_query += " AND owner_id = $2"
        params.append(context_key)

    if query:
        param_num = len(params) + 1
        base_query += f" AND content ILIKE ${param_num}"
        params.append(f"%{query}%")

    base_query += f" ORDER BY importance DESC, created_at DESC LIMIT {limit}"

    results = await pool.fetch(base_query, *params)

    return {
        "results": [
            {
                "id": r["id"],
                "context_key": r["owner_id"],
                "content": r["content"],
                "importance": float(r["importance"] or 0) / 100.0,
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "tags": r["tags"] or []
            }
            for r in results
        ],
        "total": len(results),
        "query": query,
        "table": "memory_entries",
        "filters": {
            "context_key": context_key,
            "importance_threshold": importance_threshold
        }
    }


async def _search_ai_persistent_memory(
    pool, query: str, context_key: Optional[str], limit: int, importance_threshold: float
) -> dict[str, Any]:
    """Search in ai_persistent_memory table (content is jsonb)"""
    base_query = """
        SELECT
            id::text,
            context_key,
            content,
            importance_score,
            created_at
        FROM ai_persistent_memory
        WHERE importance_score >= $1
    """
    params: list[Any] = [importance_threshold]

    if context_key:
        base_query += " AND context_key = $2"
        params.append(context_key)

    if query:
        param_num = len(params) + 1
        # content is jsonb, search in the text representation
        base_query += f" AND content::text ILIKE ${param_num}"
        params.append(f"%{query}%")

    base_query += f" ORDER BY importance_score DESC, created_at DESC LIMIT {limit}"

    results = await pool.fetch(base_query, *params)

    return {
        "results": [
            {
                "id": r["id"],
                "context_key": r["context_key"],
                "content": r["content"],  # jsonb
                "importance": float(r["importance_score"] or 0.0),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "tags": []
            }
            for r in results
        ],
        "total": len(results),
        "query": query,
        "table": "ai_persistent_memory",
        "filters": {
            "context_key": context_key,
            "importance_threshold": importance_threshold
        }
    }


async def _search_memories(
    pool, query: str, context_key: Optional[str], limit: int
) -> dict[str, Any]:
    """Search in memories table (no importance field)"""
    base_query = """
        SELECT
            id::text,
            entity_id,
            content,
            created_at,
            tenant_id
        FROM memories
        WHERE 1=1
    """
    params: list[Any] = []

    if context_key:
        base_query += f" AND entity_id = ${len(params) + 1}"
        params.append(context_key)

    if query:
        base_query += f" AND content ILIKE ${len(params) + 1}"
        params.append(f"%{query}%")

    base_query += f" ORDER BY created_at DESC LIMIT {limit}"

    results = await pool.fetch(base_query, *params)

    return {
        "results": [
            {
                "id": r["id"],
                "context_key": r["entity_id"],
                "content": r["content"],
                "importance": 0.0,
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "tags": [],
                "tenant_id": r["tenant_id"]
            }
            for r in results
        ],
        "total": len(results),
        "query": query,
        "table": "memories",
        "filters": {
            "context_key": context_key
        }
    }
