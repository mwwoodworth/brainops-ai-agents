"""
DATABASE INTELLIGENCE API
=========================

AI-powered database monitoring, optimization, and health analysis.

Features:
- Comprehensive database health monitoring
- Slow query detection and analysis
- Index recommendations
- Query optimization with AI
- Maintenance recommendations

Created: 2026-01-27
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/db-intelligence", tags=["database-intelligence"])

# Import Database Intelligence
try:
    from db_intelligence import (
        get_db_intelligence,
        get_db_health,
        DatabaseIntelligence
    )
    DB_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    DB_INTELLIGENCE_AVAILABLE = False
    logger.warning(f"Database Intelligence not available: {e}")


class QueryOptimizeRequest(BaseModel):
    query: str


@router.get("/")
async def root():
    """
    DATABASE INTELLIGENCE ROOT

    AI-powered database monitoring and optimization.
    """
    return {
        "service": "BrainOps Database Intelligence",
        "description": "AI-powered database monitoring, optimization, and health analysis",
        "available": DB_INTELLIGENCE_AVAILABLE,
        "endpoints": {
            "GET /db-intelligence/health": "Comprehensive database health check",
            "GET /db-intelligence/slow-queries": "Detect slow queries",
            "GET /db-intelligence/index-recommendations": "Get index recommendations",
            "POST /db-intelligence/optimize-query": "AI-powered query optimization",
            "GET /db-intelligence/maintenance": "Get maintenance recommendations",
            "GET /db-intelligence/trend": "Get health trend over time"
        }
    }


@router.get("/health")
async def get_health():
    """
    COMPREHENSIVE DATABASE HEALTH

    Returns:
    - Connection pool status
    - Cache hit ratio
    - Slow query count
    - Dead tuple count (bloat)
    - Index usage ratio
    - Issues and recommendations
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    health = await get_db_health()
    return {
        "status": "healthy" if health["healthy"] else "unhealthy",
        "health": health
    }


@router.get("/slow-queries")
async def get_slow_queries(threshold_ms: float = 100):
    """
    DETECT SLOW QUERIES

    Identifies queries exceeding the threshold and provides optimization suggestions.

    Args:
        threshold_ms: Minimum query time to be considered slow (default: 100ms)
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    slow_queries = await db.detect_slow_queries(threshold_ms)

    return {
        "threshold_ms": threshold_ms,
        "count": len(slow_queries),
        "queries": [
            {
                "query": sq.query[:200] + "..." if len(sq.query) > 200 else sq.query,
                "calls": sq.calls,
                "avg_ms": round(sq.avg_duration_ms, 2),
                "max_ms": round(sq.max_duration_ms, 2),
                "table": sq.table_name,
                "suggestions": sq.optimization_suggestions
            }
            for sq in slow_queries
        ]
    }


@router.get("/index-recommendations")
async def get_index_recommendations():
    """
    GET INDEX RECOMMENDATIONS

    Analyzes table access patterns and recommends new indexes
    to improve query performance.
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    recommendations = await db.get_index_recommendations()

    return {
        "count": len(recommendations),
        "recommendations": [
            {
                "table": rec.table_name,
                "column": rec.column_name,
                "reason": rec.reason,
                "improvement": rec.estimated_improvement,
                "create_statement": rec.create_statement,
                "priority": rec.priority
            }
            for rec in recommendations
        ]
    }


@router.post("/optimize-query")
async def optimize_query(request: QueryOptimizeRequest):
    """
    AI-POWERED QUERY OPTIMIZATION

    Uses AI to analyze a query and suggest optimizations.

    Returns:
    - Identified issues
    - Optimized query
    - Recommended indexes
    - Estimated improvement
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    result = await db.optimize_query(request.query)

    return {
        "original_query": request.query[:500],
        "optimization": result
    }


@router.get("/maintenance")
async def get_maintenance_recommendations():
    """
    DATABASE MAINTENANCE RECOMMENDATIONS

    Identifies tables needing:
    - VACUUM ANALYZE (dead tuple cleanup)
    - REINDEX (index bloat)
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    maintenance = await db.run_maintenance()

    return {
        "recommendations": maintenance
    }


@router.get("/trend")
async def get_health_trend(hours: int = 24):
    """
    DATABASE HEALTH TREND

    Shows health trend over the specified time period.

    Args:
        hours: Number of hours to analyze (default: 24)
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    trend = db.get_health_trend(hours)

    return {
        "period_hours": hours,
        "trend": trend
    }


@router.get("/connection-stats")
async def get_connection_stats():
    """
    CONNECTION POOL STATISTICS

    Real-time connection pool status.
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    pool = await db._get_pool()

    if not pool:
        return {"error": "Database pool not available"}

    stats = await db._get_connection_stats(pool)
    return {
        "connections": stats
    }


@router.get("/cache-stats")
async def get_cache_stats():
    """
    DATABASE CACHE STATISTICS

    Shows buffer cache hit ratio and effectiveness.
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    pool = await db._get_pool()

    if not pool:
        return {"error": "Database pool not available"}

    stats = await db._get_cache_stats(pool)
    return {
        "cache": {
            "hits": stats["hits"],
            "reads": stats["reads"],
            "hit_ratio": f"{stats['hit_ratio']:.1%}"
        }
    }


@router.get("/table-stats")
async def get_table_stats():
    """
    TABLE STATISTICS

    Shows table-level statistics including dead tuples and index usage.
    """
    if not DB_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database Intelligence not available")

    db = get_db_intelligence()
    pool = await db._get_pool()

    if not pool:
        return {"error": "Database pool not available"}

    stats = await db._get_table_stats(pool)
    return {
        "tables": {
            "dead_tuples": stats["total_dead_tuples"],
            "estimated_bloat_mb": round(stats["bloat_mb"], 2),
            "index_usage_ratio": f"{stats['index_usage_ratio']:.1%}"
        }
    }
