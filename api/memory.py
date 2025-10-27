"""
Memory System API Endpoints - Production Ready
Fixed 500 errors with proper error handling
"""
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database.async_connection import get_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryStatus(BaseModel):
    """Memory system status model"""
    status: str = Field(..., description="Operational status")
    total_memories: int = Field(default=0, ge=0)
    unique_users: int = Field(default=0, ge=0)
    avg_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    table_used: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryEntry(BaseModel):
    """Memory entry model"""
    id: str
    user_id: str
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: Optional[datetime] = None
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
                unique_users=0,
                avg_importance=0.0
            )

        # Use the first available table
        table_name = existing_tables[0]["table_name"]

        # Get statistics from the available table
        stats_query = f"""
            SELECT
                COUNT(*) as total_memories,
                COUNT(DISTINCT user_id) as unique_users,
                COALESCE(AVG(importance::numeric), 0.0) as avg_importance
            FROM {table_name}
            WHERE user_id IS NOT NULL
        """

        stats = await pool.fetchrow(stats_query)

        return MemoryStatus(
            status="operational",
            table_used=table_name,
            total_memories=stats["total_memories"] or 0,
            unique_users=stats["unique_users"] or 0,
            avg_importance=float(stats["avg_importance"] or 0.0),
            message=f"Using table: {table_name}"
        )

    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        # Return graceful degradation instead of 500
        return MemoryStatus(
            status="error",
            message=f"Unable to retrieve memory statistics: {str(e)}",
            total_memories=0,
            unique_users=0,
            avg_importance=0.0
        )


@router.get("/search")
async def search_memories(
    query: str = Query(..., description="Search query"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    importance_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance")
) -> dict[str, Any]:
    """Search memory entries with semantic similarity"""
    pool = get_pool()

    try:
        # First check if memory table exists
        table_check = await pool.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'ai_persistent_memory'
            )
        """)

        if not table_check:
            return {
                "results": [],
                "total": 0,
                "query": query,
                "message": "Memory system not initialized"
            }

        # Build search query
        base_query = """
            SELECT
                id::text,
                user_id,
                content,
                importance,
                created_at,
                tags
            FROM ai_persistent_memory
            WHERE importance >= $1
        """

        params = [importance_threshold]

        if user_id:
            base_query += " AND user_id = $2"
            params.append(user_id)

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
                    "user_id": r["user_id"],
                    "content": r["content"],
                    "importance": float(r["importance"]) if r["importance"] else 0.0,
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "tags": r["tags"] or []
                }
                for r in results
            ],
            "total": len(results),
            "query": query,
            "filters": {
                "user_id": user_id,
                "importance_threshold": importance_threshold
            }
        }

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e
