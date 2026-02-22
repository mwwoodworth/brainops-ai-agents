"""
Memory Observability API
========================
Comprehensive observability endpoints for monitoring AI memory systems.

Provides:
- Memory usage statistics by type (episodic, procedural, semantic, working, meta)
- Memory access patterns (hot vs cold memories)
- Memory decay tracking
- Memory consolidation metrics
- Storage efficiency metrics
- Query performance on memory operations

Part of BrainOps AI OS Memory Observability Suite.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database.async_connection import DatabaseUnavailableError, get_pool, using_fallback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory/observability", tags=["Memory Observability"])


# =============================================================================
# CONSTANTS
# =============================================================================

CANONICAL_TABLE = "unified_ai_memory"
DEFAULT_TENANT_ID = "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"


# =============================================================================
# MODELS
# =============================================================================

class MemoryTypeStats(BaseModel):
    """Statistics for a specific memory type"""
    memory_type: str
    total_count: int
    avg_importance: float
    avg_access_count: float
    avg_age_days: float
    with_embeddings: int
    high_importance_count: int
    low_importance_count: int
    last_created: Optional[datetime] = None
    last_accessed: Optional[datetime] = None


class MemoryHealthMetrics(BaseModel):
    """Overall memory system health metrics"""
    status: str = Field(..., description="healthy, degraded, critical")
    health_score: int = Field(..., ge=0, le=100)
    total_memories: int
    memories_with_embeddings: int
    embedding_coverage_pct: float
    avg_importance_score: float
    avg_access_count: float
    hot_memories_count: int
    cold_memories_count: int
    stale_memories_count: int
    expired_memories_count: int
    recommendations: list[str]
    timestamp: datetime


class HotMemory(BaseModel):
    """Frequently accessed memory entry"""
    id: str
    memory_type: str
    access_count: int
    importance_score: float
    last_accessed: Optional[datetime]
    created_at: Optional[datetime]
    content_preview: str
    source_system: Optional[str]


class DecayMetrics(BaseModel):
    """Memory decay statistics"""
    total_decayed: int
    avg_decay_rate: float
    memories_by_age_bucket: dict[str, int]
    importance_distribution: dict[str, int]
    decay_candidates: int
    recently_promoted: int
    recently_demoted: int


class ConsolidationMetrics(BaseModel):
    """Memory consolidation statistics"""
    total_consolidations: int
    avg_similarity_threshold: float
    duplicate_pairs_detected: int
    memories_merged_last_24h: int
    memories_pending_consolidation: int
    consolidation_efficiency: float


class StorageMetrics(BaseModel):
    """Storage efficiency metrics"""
    total_memories: int
    total_with_embeddings: int
    embedding_storage_estimate_mb: float
    content_storage_estimate_mb: float
    avg_content_size_bytes: float
    unique_systems: int
    unique_agents: int
    unique_contexts: int
    storage_by_type: dict[str, int]
    storage_by_system: dict[str, int]


class QueryPerformanceMetrics(BaseModel):
    """Memory query performance metrics"""
    avg_search_latency_ms: float
    avg_store_latency_ms: float
    total_accesses_today: int
    total_stores_today: int
    cache_hit_rate: float
    slow_queries_count: int
    query_distribution_by_type: dict[str, int]


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> str:
    """Extract tenant ID from header or use default"""
    return x_tenant_id or DEFAULT_TENANT_ID


def _require_real_database(operation: str) -> None:
    """Ensure we have a real database connection"""
    if using_fallback():
        message = (
            "Database unavailable; in-memory fallback active. "
            "Configure DATABASE_URL for production."
        )
        logger.error("Refusing %s: %s", operation, message)
        raise HTTPException(status_code=503, detail=message)


# =============================================================================
# COMPREHENSIVE STATISTICS ENDPOINT
# =============================================================================

@router.get("/stats")
async def get_memory_stats(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get comprehensive memory statistics.

    Returns detailed statistics including:
    - Total memories by type
    - Memory access frequency
    - Average memory age
    - Consolidation rate
    - Storage utilization
    """
    try:
        _require_real_database("memory stats")
        pool = get_pool()

        # Get overall stats
        overall = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(DISTINCT source_system) as unique_systems,
                COUNT(DISTINCT source_agent) as unique_agents,
                COUNT(DISTINCT context_id) as unique_contexts,
                COALESCE(AVG(importance_score), 0.0) as avg_importance,
                COALESCE(AVG(access_count), 0.0) as avg_access_count,
                COALESCE(AVG(EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400), 0.0) as avg_age_days,
                MAX(created_at) as last_created,
                MAX(last_accessed) as last_accessed,
                COUNT(*) FILTER (WHERE access_count >= 10) as hot_count,
                COUNT(*) FILTER (WHERE access_count <= 1 AND created_at < NOW() - INTERVAL '30 days') as cold_count,
                COUNT(*) FILTER (WHERE importance_score >= 0.7) as high_importance,
                COUNT(*) FILTER (WHERE importance_score < 0.3) as low_importance,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at < NOW()) as expired_count,
                COUNT(*) FILTER (WHERE reinforcement_count > 0) as reinforced_count,
                COALESCE(SUM(reinforcement_count), 0) as total_reinforcements
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        # Get stats by type
        type_stats = await pool.fetch("""
            SELECT
                memory_type,
                COUNT(*) as count,
                COALESCE(AVG(importance_score), 0.0) as avg_importance,
                COALESCE(AVG(access_count), 0.0) as avg_access_count,
                COALESCE(AVG(EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400), 0.0) as avg_age_days,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(*) FILTER (WHERE importance_score >= 0.7) as high_importance,
                COUNT(*) FILTER (WHERE importance_score < 0.3) as low_importance,
                MAX(created_at) as last_created,
                MAX(last_accessed) as last_accessed
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY memory_type
            ORDER BY count DESC
        """, tenant_id)

        # Get stats by source system
        system_stats = await pool.fetch("""
            SELECT
                source_system,
                COUNT(*) as count,
                COALESCE(AVG(importance_score), 0.0) as avg_importance,
                MAX(created_at) as last_activity
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY source_system
            ORDER BY count DESC
            LIMIT 20
        """, tenant_id)

        # Calculate derived metrics
        total = overall["total_memories"] or 1
        embedding_coverage = (overall["with_embeddings"] or 0) / total * 100
        hot_ratio = (overall["hot_count"] or 0) / total * 100
        cold_ratio = (overall["cold_count"] or 0) / total * 100

        return {
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "summary": {
                "total_memories": overall["total_memories"] or 0,
                "memories_with_embeddings": overall["with_embeddings"] or 0,
                "embedding_coverage_pct": round(embedding_coverage, 2),
                "unique_systems": overall["unique_systems"] or 0,
                "unique_agents": overall["unique_agents"] or 0,
                "unique_contexts": overall["unique_contexts"] or 0,
                "avg_importance_score": round(float(overall["avg_importance"] or 0), 3),
                "avg_access_count": round(float(overall["avg_access_count"] or 0), 2),
                "avg_age_days": round(float(overall["avg_age_days"] or 0), 1),
                "last_memory_created": overall["last_created"].isoformat() if overall["last_created"] else None,
                "last_memory_accessed": overall["last_accessed"].isoformat() if overall["last_accessed"] else None,
            },
            "access_patterns": {
                "hot_memories": overall["hot_count"] or 0,
                "hot_ratio_pct": round(hot_ratio, 2),
                "cold_memories": overall["cold_count"] or 0,
                "cold_ratio_pct": round(cold_ratio, 2),
                "high_importance": overall["high_importance"] or 0,
                "low_importance": overall["low_importance"] or 0,
                "expired_count": overall["expired_count"] or 0,
                "reinforced_count": overall["reinforced_count"] or 0,
                "total_reinforcements": overall["total_reinforcements"] or 0,
            },
            "by_type": [
                {
                    "memory_type": row["memory_type"],
                    "count": row["count"],
                    "avg_importance": round(float(row["avg_importance"] or 0), 3),
                    "avg_access_count": round(float(row["avg_access_count"] or 0), 2),
                    "avg_age_days": round(float(row["avg_age_days"] or 0), 1),
                    "with_embeddings": row["with_embeddings"],
                    "high_importance": row["high_importance"],
                    "low_importance": row["low_importance"],
                    "last_created": row["last_created"].isoformat() if row["last_created"] else None,
                }
                for row in type_stats
            ],
            "by_system": [
                {
                    "source_system": row["source_system"],
                    "count": row["count"],
                    "avg_importance": round(float(row["avg_importance"] or 0), 3),
                    "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
                }
                for row in system_stats
            ],
        }

    except DatabaseUnavailableError as exc:
        logger.error("Memory stats unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memory stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@router.get("/health")
async def get_memory_health(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Health check for memory systems.

    Returns operational status and health score based on:
    - Embedding coverage
    - Memory freshness
    - Access patterns
    - Decay status
    """
    try:
        _require_real_database("memory health")
        pool = get_pool()

        # Get health metrics
        metrics = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COALESCE(AVG(importance_score), 0.0) as avg_importance,
                COALESCE(AVG(access_count), 0.0) as avg_access_count,
                COUNT(*) FILTER (WHERE access_count >= 10) as hot_count,
                COUNT(*) FILTER (WHERE access_count <= 1 AND created_at < NOW() - INTERVAL '30 days') as cold_count,
                COUNT(*) FILTER (WHERE created_at < NOW() - INTERVAL '90 days' AND access_count < 3) as stale_count,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at < NOW()) as expired_count,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as recent_hour,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as recent_day
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        total = metrics["total_memories"] or 1

        # Calculate health score (0-100)
        embedding_coverage = (metrics["with_embeddings"] or 0) / total
        hot_ratio = (metrics["hot_count"] or 0) / total
        stale_ratio = (metrics["stale_count"] or 0) / total
        expired_ratio = (metrics["expired_count"] or 0) / total

        health_score = int(
            embedding_coverage * 30 +  # 30 points for embedding coverage
            min(1, hot_ratio * 5) * 20 +  # 20 points for activity
            (1 - stale_ratio) * 25 +  # 25 points for freshness
            (1 - expired_ratio) * 15 +  # 15 points for no expired
            min(1, float(metrics["avg_importance"] or 0)) * 10  # 10 points for quality
        )

        health_score = max(0, min(100, health_score))

        # Determine status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        else:
            status = "critical"

        # Generate recommendations
        recommendations = []
        if embedding_coverage < 0.8:
            recommendations.append(f"Low embedding coverage ({embedding_coverage*100:.1f}%) - run embedding backfill")
        if stale_ratio > 0.3:
            recommendations.append(f"High stale memory ratio ({stale_ratio*100:.1f}%) - consider garbage collection")
        if expired_ratio > 0.1:
            recommendations.append(f"Expired memories detected ({metrics['expired_count']}) - run cleanup")
        if metrics["recent_day"] == 0:
            recommendations.append("No new memories in 24h - check memory ingestion pipeline")
        if hot_ratio < 0.05:
            recommendations.append("Low memory utilization - consider memory consolidation")

        if not recommendations:
            recommendations.append("Memory system is healthy - no immediate action required")

        return {
            "status": status,
            "health_score": health_score,
            "operational": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "metrics": {
                "total_memories": metrics["total_memories"] or 0,
                "memories_with_embeddings": metrics["with_embeddings"] or 0,
                "embedding_coverage_pct": round(embedding_coverage * 100, 2),
                "avg_importance_score": round(float(metrics["avg_importance"] or 0), 3),
                "avg_access_count": round(float(metrics["avg_access_count"] or 0), 2),
                "hot_memories": metrics["hot_count"] or 0,
                "cold_memories": metrics["cold_count"] or 0,
                "stale_memories": metrics["stale_count"] or 0,
                "expired_memories": metrics["expired_count"] or 0,
                "memories_last_hour": metrics["recent_hour"] or 0,
                "memories_last_24h": metrics["recent_day"] or 0,
            },
            "recommendations": recommendations,
        }

    except DatabaseUnavailableError as exc:
        logger.error("Memory health unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Memory health check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="Memory health check failed") from e


# =============================================================================
# HOT MEMORIES ENDPOINT
# =============================================================================

@router.get("/hot")
async def get_hot_memories(
    limit: int = Query(20, ge=1, le=100, description="Number of hot memories to return"),
    min_access_count: int = Query(5, ge=1, description="Minimum access count threshold"),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get most frequently accessed memories.

    Hot memories are those with high access counts, indicating
    they are frequently retrieved and likely important for operations.
    """
    try:
        _require_real_database("hot memories")
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT
                id::text,
                memory_type,
                access_count,
                importance_score,
                last_accessed,
                created_at,
                CASE
                    WHEN jsonb_typeof(content) = 'string' THEN LEFT(content #>> '{}', 200)
                    WHEN content ? 'text' THEN LEFT(content->>'text', 200)
                    WHEN content ? 'summary' THEN LEFT(content->>'summary', 200)
                    ELSE LEFT(content::text, 200)
                END as content_preview,
                source_system,
                source_agent,
                reinforcement_count,
                tags
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND access_count >= $2
            ORDER BY access_count DESC, importance_score DESC
            LIMIT $3
        """, tenant_id, min_access_count, limit)

        # Calculate additional metrics
        if rows:
            total_hot_accesses = sum(r["access_count"] for r in rows)
            avg_hot_importance = sum(r["importance_score"] for r in rows) / len(rows)
        else:
            total_hot_accesses = 0
            avg_hot_importance = 0

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "summary": {
                "hot_memories_count": len(rows),
                "total_hot_accesses": total_hot_accesses,
                "avg_hot_importance": round(avg_hot_importance, 3),
                "min_access_threshold": min_access_count,
            },
            "hot_memories": [
                {
                    "id": row["id"],
                    "memory_type": row["memory_type"],
                    "access_count": row["access_count"],
                    "importance_score": round(float(row["importance_score"] or 0), 3),
                    "last_accessed": row["last_accessed"].isoformat() if row["last_accessed"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "content_preview": row["content_preview"],
                    "source_system": row["source_system"],
                    "source_agent": row["source_agent"],
                    "reinforcement_count": row["reinforcement_count"] or 0,
                    "tags": row["tags"] or [],
                }
                for row in rows
            ],
        }

    except DatabaseUnavailableError as exc:
        logger.error("Hot memories unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get hot memories: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# COLD MEMORIES ENDPOINT
# =============================================================================

@router.get("/cold")
async def get_cold_memories(
    limit: int = Query(50, ge=1, le=200, description="Number of cold memories to return"),
    days_old: int = Query(30, ge=1, description="Minimum age in days"),
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get rarely accessed cold memories.

    Cold memories are old, rarely accessed entries that may be
    candidates for consolidation, archival, or garbage collection.
    """
    try:
        _require_real_database("cold memories")
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT
                id::text,
                memory_type,
                access_count,
                importance_score,
                last_accessed,
                created_at,
                EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 as age_days,
                CASE
                    WHEN jsonb_typeof(content) = 'string' THEN LEFT(content #>> '{}', 150)
                    WHEN content ? 'text' THEN LEFT(content->>'text', 150)
                    ELSE LEFT(content::text, 150)
                END as content_preview,
                source_system
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND access_count <= 1
                AND created_at < NOW() - ($2 || ' days')::interval
            ORDER BY created_at ASC, access_count ASC
            LIMIT $3
        """, tenant_id, str(days_old), limit)

        # Get total cold count for context
        cold_total = await pool.fetchval("""
            SELECT COUNT(*)
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND access_count <= 1
                AND created_at < NOW() - ($2 || ' days')::interval
        """, tenant_id, str(days_old))

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "summary": {
                "cold_memories_shown": len(rows),
                "cold_memories_total": cold_total or 0,
                "age_threshold_days": days_old,
                "recommendation": "Consider archiving or removing cold memories to improve performance"
                if cold_total and cold_total > 1000 else "Cold memory count is acceptable",
            },
            "cold_memories": [
                {
                    "id": row["id"],
                    "memory_type": row["memory_type"],
                    "access_count": row["access_count"],
                    "importance_score": round(float(row["importance_score"] or 0), 3),
                    "age_days": round(float(row["age_days"] or 0), 1),
                    "last_accessed": row["last_accessed"].isoformat() if row["last_accessed"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "content_preview": row["content_preview"],
                    "source_system": row["source_system"],
                }
                for row in rows
            ],
        }

    except DatabaseUnavailableError as exc:
        logger.error("Cold memories unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get cold memories: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# DECAY METRICS ENDPOINT
# =============================================================================

@router.get("/decay")
async def get_decay_metrics(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get memory decay statistics.

    Tracks how memories age and lose relevance over time,
    including importance score distribution and age buckets.
    """
    try:
        _require_real_database("decay metrics")
        pool = get_pool()

        # Age bucket distribution
        age_buckets = await pool.fetch("""
            SELECT
                CASE
                    WHEN created_at > NOW() - INTERVAL '1 day' THEN '0-1 days'
                    WHEN created_at > NOW() - INTERVAL '7 days' THEN '1-7 days'
                    WHEN created_at > NOW() - INTERVAL '30 days' THEN '7-30 days'
                    WHEN created_at > NOW() - INTERVAL '90 days' THEN '30-90 days'
                    ELSE '90+ days'
                END as age_bucket,
                COUNT(*) as count,
                AVG(importance_score) as avg_importance,
                AVG(access_count) as avg_access
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY age_bucket
            ORDER BY
                CASE age_bucket
                    WHEN '0-1 days' THEN 1
                    WHEN '1-7 days' THEN 2
                    WHEN '7-30 days' THEN 3
                    WHEN '30-90 days' THEN 4
                    ELSE 5
                END
        """, tenant_id)

        # Importance distribution
        importance_dist = await pool.fetch("""
            SELECT
                CASE
                    WHEN importance_score >= 0.9 THEN 'critical (0.9-1.0)'
                    WHEN importance_score >= 0.7 THEN 'high (0.7-0.9)'
                    WHEN importance_score >= 0.5 THEN 'medium (0.5-0.7)'
                    WHEN importance_score >= 0.3 THEN 'low (0.3-0.5)'
                    ELSE 'minimal (0-0.3)'
                END as importance_level,
                COUNT(*) as count
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY importance_level
            ORDER BY
                CASE importance_level
                    WHEN 'critical (0.9-1.0)' THEN 1
                    WHEN 'high (0.7-0.9)' THEN 2
                    WHEN 'medium (0.5-0.7)' THEN 3
                    WHEN 'low (0.3-0.5)' THEN 4
                    ELSE 5
                END
        """, tenant_id)

        # Decay candidates (old, low importance, low access)
        decay_candidates = await pool.fetchval("""
            SELECT COUNT(*)
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND importance_score < 0.3
                AND access_count < 3
                AND created_at < NOW() - INTERVAL '60 days'
        """, tenant_id)

        # Recently promoted/demoted (if tracking exists)
        # Note: This assumes metadata tracking of promotions/demotions
        recently_changed = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE importance_score > 0.7 AND created_at > NOW() - INTERVAL '7 days') as promoted,
                COUNT(*) FILTER (WHERE importance_score < 0.3 AND last_accessed > NOW() - INTERVAL '7 days') as demoted
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        # Calculate average decay rate (importance decrease per day for old memories)
        avg_decay_info = await pool.fetchrow("""
            SELECT
                AVG(importance_score) as avg_old_importance,
                AVG(importance_score) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as avg_new_importance
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND created_at < NOW() - INTERVAL '30 days'
        """, tenant_id)

        old_imp = float(avg_decay_info["avg_old_importance"] or 0.5)
        new_imp = float(avg_decay_info["avg_new_importance"] or 0.5)
        decay_rate = max(0, (new_imp - old_imp) / 30) if new_imp > 0 else 0

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "decay_metrics": {
                "avg_decay_rate_per_day": round(decay_rate, 4),
                "decay_candidates": decay_candidates or 0,
                "recently_promoted_7d": recently_changed["promoted"] or 0,
                "recently_demoted_7d": recently_changed["demoted"] or 0,
            },
            "age_distribution": {
                row["age_bucket"]: {
                    "count": row["count"],
                    "avg_importance": round(float(row["avg_importance"] or 0), 3),
                    "avg_access": round(float(row["avg_access"] or 0), 2),
                }
                for row in age_buckets
            },
            "importance_distribution": {
                row["importance_level"]: row["count"]
                for row in importance_dist
            },
            "recommendations": _get_decay_recommendations(decay_candidates, age_buckets, importance_dist),
        }

    except DatabaseUnavailableError as exc:
        logger.error("Decay metrics unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get decay metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


def _get_decay_recommendations(decay_candidates: int, age_buckets: list, importance_dist: list) -> list[str]:
    """Generate recommendations based on decay metrics"""
    recommendations = []

    if decay_candidates and decay_candidates > 500:
        recommendations.append(f"High decay candidate count ({decay_candidates}) - schedule garbage collection")

    # Check for too many old memories
    old_count = sum(
        row["count"] for row in age_buckets
        if row["age_bucket"] in ["30-90 days", "90+ days"]
    )
    if old_count > 5000:
        recommendations.append(f"Many old memories ({old_count}) - consider archival policy")

    # Check for low quality memories
    low_quality = sum(
        row["count"] for row in importance_dist
        if "minimal" in row["importance_level"] or "low" in row["importance_level"]
    )
    if low_quality > 1000:
        recommendations.append(f"Many low-importance memories ({low_quality}) - review retention policy")

    if not recommendations:
        recommendations.append("Memory decay metrics are healthy")

    return recommendations


# =============================================================================
# CONSOLIDATION METRICS ENDPOINT
# =============================================================================

@router.get("/consolidation")
async def get_consolidation_metrics(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get memory consolidation statistics.

    Shows metrics about memory deduplication, merging,
    and consolidation efficiency.
    """
    try:
        _require_real_database("consolidation metrics")
        pool = get_pool()

        # Check for memories with consolidation metadata
        consolidation_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE metadata ? 'consolidated_into') as total_consolidated,
                COUNT(*) FILTER (WHERE metadata ? 'consolidated_into' AND created_at > NOW() - INTERVAL '24 hours') as consolidated_24h,
                COUNT(*) FILTER (WHERE reinforcement_count > 0) as reinforced_count,
                SUM(COALESCE(reinforcement_count, 0)) as total_reinforcements
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        # Find potential duplicates (same content hash)
        duplicate_pairs = await pool.fetchval("""
            SELECT COUNT(*) / 2
            FROM (
                SELECT content_hash, COUNT(*) as cnt
                FROM unified_ai_memory
                WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                    AND content_hash IS NOT NULL
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            ) dups
        """, tenant_id)

        # Memories pending consolidation (similar but not merged)
        # This is an approximation based on same source_system and similar timestamps
        pending_consolidation = await pool.fetchval("""
            SELECT COUNT(DISTINCT m1.id)
            FROM unified_ai_memory m1
            JOIN unified_ai_memory m2 ON
                m1.source_system = m2.source_system
                AND m1.memory_type = m2.memory_type
                AND m1.id < m2.id
                AND ABS(EXTRACT(EPOCH FROM (m1.created_at - m2.created_at))) < 3600
            WHERE (m1.tenant_id = $1::uuid OR m1.tenant_id IS NULL)
            LIMIT 1000
        """, tenant_id)

        # Calculate consolidation efficiency
        total = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        consolidated = consolidation_stats["total_consolidated"] or 0
        efficiency = (1 - (consolidated / max(total, 1))) * 100 if total else 100

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "consolidation_metrics": {
                "total_consolidated": consolidated,
                "consolidated_last_24h": consolidation_stats["consolidated_24h"] or 0,
                "reinforced_memories": consolidation_stats["reinforced_count"] or 0,
                "total_reinforcements": consolidation_stats["total_reinforcements"] or 0,
                "duplicate_pairs_detected": duplicate_pairs or 0,
                "pending_consolidation_estimate": min(pending_consolidation or 0, 1000),
                "consolidation_efficiency_pct": round(efficiency, 2),
            },
            "recommendations": [
                f"Found {duplicate_pairs or 0} duplicate content hashes - run deduplication"
                if (duplicate_pairs or 0) > 50 else "Duplicate levels acceptable",
                "Consider running consolidation job" if (pending_consolidation or 0) > 100 else "Consolidation status healthy",
            ],
        }

    except DatabaseUnavailableError as exc:
        logger.error("Consolidation metrics unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get consolidation metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# STORAGE METRICS ENDPOINT
# =============================================================================

@router.get("/storage")
async def get_storage_metrics(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get storage efficiency metrics.

    Provides insights into memory storage usage,
    including estimated sizes and distribution.
    """
    try:
        _require_real_database("storage metrics")
        pool = get_pool()

        # Get storage statistics
        storage_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(DISTINCT source_system) as unique_systems,
                COUNT(DISTINCT source_agent) as unique_agents,
                COUNT(DISTINCT context_id) as unique_contexts,
                AVG(LENGTH(content::text)) as avg_content_length,
                SUM(LENGTH(content::text)) as total_content_length
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        # Storage by type
        type_storage = await pool.fetch("""
            SELECT
                memory_type,
                COUNT(*) as count,
                SUM(LENGTH(content::text)) as content_bytes
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY memory_type
            ORDER BY count DESC
        """, tenant_id)

        # Storage by system
        system_storage = await pool.fetch("""
            SELECT
                source_system,
                COUNT(*) as count,
                SUM(LENGTH(content::text)) as content_bytes
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
            GROUP BY source_system
            ORDER BY count DESC
            LIMIT 15
        """, tenant_id)

        # Estimate embedding storage (1536 dims * 4 bytes per float * count)
        embedding_count = storage_stats["with_embeddings"] or 0
        embedding_storage_mb = (embedding_count * 1536 * 4) / (1024 * 1024)

        # Content storage in MB
        content_storage_mb = (storage_stats["total_content_length"] or 0) / (1024 * 1024)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "storage_metrics": {
                "total_memories": storage_stats["total_memories"] or 0,
                "memories_with_embeddings": embedding_count,
                "embedding_storage_estimate_mb": round(embedding_storage_mb, 2),
                "content_storage_estimate_mb": round(content_storage_mb, 2),
                "total_storage_estimate_mb": round(embedding_storage_mb + content_storage_mb, 2),
                "avg_content_size_bytes": round(float(storage_stats["avg_content_length"] or 0), 0),
                "unique_systems": storage_stats["unique_systems"] or 0,
                "unique_agents": storage_stats["unique_agents"] or 0,
                "unique_contexts": storage_stats["unique_contexts"] or 0,
            },
            "storage_by_type": {
                row["memory_type"]: {
                    "count": row["count"],
                    "storage_bytes": row["content_bytes"] or 0,
                    "storage_mb": round((row["content_bytes"] or 0) / (1024 * 1024), 3),
                }
                for row in type_storage
            },
            "storage_by_system": {
                row["source_system"]: {
                    "count": row["count"],
                    "storage_bytes": row["content_bytes"] or 0,
                    "storage_mb": round((row["content_bytes"] or 0) / (1024 * 1024), 3),
                }
                for row in system_storage
            },
        }

    except DatabaseUnavailableError as exc:
        logger.error("Storage metrics unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get storage metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# QUERY PERFORMANCE METRICS ENDPOINT
# =============================================================================

@router.get("/performance")
async def get_query_performance(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get memory query performance metrics.

    Tracks query latencies, access patterns, and
    provides performance insights.
    """
    try:
        _require_real_database("performance metrics")
        pool = get_pool()

        # Get access log statistics if the table exists
        access_stats = None
        try:
            access_stats = await pool.fetchrow("""
                SELECT
                    COUNT(*) as total_accesses,
                    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') as accesses_today,
                    COUNT(*) FILTER (WHERE access_type = 'write') as writes_today,
                    COUNT(*) FILTER (WHERE access_type = 'read') as reads_today,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(latency_ms) FILTER (WHERE access_type = 'read') as avg_read_latency,
                    AVG(latency_ms) FILTER (WHERE access_type = 'write') as avg_write_latency,
                    COUNT(*) FILTER (WHERE latency_ms > 500) as slow_queries,
                    COUNT(*) FILTER (WHERE hit_cache = true)::float / NULLIF(COUNT(*), 0) as cache_hit_rate
                FROM memory_context_access_log
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
        except Exception:
            # Table may not exist
            logger.debug("memory_context_access_log table not available")

        # Get memory operation statistics from unified_ai_memory
        memory_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as created_today,
                COUNT(*) FILTER (WHERE last_accessed > NOW() - INTERVAL '24 hours') as accessed_today,
                SUM(access_count) FILTER (WHERE last_accessed > NOW() - INTERVAL '24 hours') as total_accesses_today
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid OR tenant_id IS NULL
        """, tenant_id)

        # Query distribution by type
        type_distribution = await pool.fetch("""
            SELECT
                memory_type,
                SUM(access_count) as total_accesses,
                COUNT(*) as memory_count
            FROM unified_ai_memory
            WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                AND last_accessed > NOW() - INTERVAL '7 days'
            GROUP BY memory_type
            ORDER BY total_accesses DESC
        """, tenant_id)

        # Build response
        response = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "performance_metrics": {
                "memories_created_today": memory_stats["created_today"] or 0,
                "memories_accessed_today": memory_stats["accessed_today"] or 0,
                "total_accesses_today": memory_stats["total_accesses_today"] or 0,
            },
            "query_distribution_7d": {
                row["memory_type"]: {
                    "accesses": row["total_accesses"] or 0,
                    "memory_count": row["memory_count"],
                }
                for row in type_distribution
            },
        }

        # Add access log metrics if available
        if access_stats:
            response["performance_metrics"].update({
                "avg_latency_ms": round(float(access_stats["avg_latency_ms"] or 0), 2),
                "avg_read_latency_ms": round(float(access_stats["avg_read_latency"] or 0), 2),
                "avg_write_latency_ms": round(float(access_stats["avg_write_latency"] or 0), 2),
                "slow_queries_24h": access_stats["slow_queries"] or 0,
                "cache_hit_rate": round(float(access_stats["cache_hit_rate"] or 0), 3),
            })
        else:
            response["note"] = "Detailed access logging not available - consider enabling memory_context_access_log"

        return response

    except DatabaseUnavailableError as exc:
        logger.error("Performance metrics unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get performance metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# =============================================================================
# COMPREHENSIVE DASHBOARD ENDPOINT
# =============================================================================

@router.get("/dashboard")
async def get_memory_dashboard(
    tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """
    Get comprehensive memory observability dashboard.

    Aggregates all memory metrics into a single dashboard view
    for monitoring and operational awareness.
    """
    try:
        _require_real_database("memory dashboard")

        # Fetch all metrics in parallel where possible
        stats = await get_memory_stats(tenant_id)
        health = await get_memory_health(tenant_id)
        decay = await get_decay_metrics(tenant_id)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "dashboard": {
                "health": {
                    "status": health["status"],
                    "score": health["health_score"],
                    "recommendations": health["recommendations"][:3],
                },
                "summary": stats["summary"],
                "access_patterns": stats["access_patterns"],
                "memory_types": stats["by_type"][:5],
                "top_systems": stats["by_system"][:5],
                "decay_status": {
                    "decay_candidates": decay["decay_metrics"]["decay_candidates"],
                    "decay_rate": decay["decay_metrics"]["avg_decay_rate_per_day"],
                },
                "age_distribution": decay["age_distribution"],
            },
        }

    except DatabaseUnavailableError as exc:
        logger.error("Dashboard unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dashboard: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e
