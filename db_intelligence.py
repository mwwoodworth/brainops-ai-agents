"""
DATABASE INTELLIGENCE ENGINE
============================

AI-powered database monitoring, optimization, and management.

Features:
- Query performance analysis
- Index recommendations
- Connection pool monitoring
- Slow query detection
- Schema optimization suggestions
- Real-time database health
- Automatic query optimization
- Dead tuple and bloat detection

Created: 2026-01-27
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryPerformance(Enum):
    """Query performance classification"""
    EXCELLENT = "excellent"  # < 10ms
    GOOD = "good"           # 10-100ms
    SLOW = "slow"           # 100-1000ms
    CRITICAL = "critical"   # > 1000ms


@dataclass
class SlowQuery:
    """A detected slow query"""
    query: str
    duration_ms: float
    calls: int
    avg_duration_ms: float
    max_duration_ms: float
    table_name: Optional[str]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class IndexRecommendation:
    """A recommended index"""
    table_name: str
    column_name: str
    reason: str
    estimated_improvement: str
    create_statement: str
    priority: str  # high, medium, low


@dataclass
class DatabaseHealth:
    """Overall database health status"""
    healthy: bool
    score: float  # 0.0 - 1.0
    connection_count: int
    max_connections: int
    active_queries: int
    slow_queries_count: int
    dead_tuples: int
    table_bloat_mb: float
    index_usage_ratio: float
    cache_hit_ratio: float
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DatabaseIntelligence:
    """
    DATABASE INTELLIGENCE ENGINE

    AI-powered database monitoring and optimization.
    """

    def __init__(self):
        self.slow_queries: List[SlowQuery] = []
        self.index_recommendations: List[IndexRecommendation] = []
        self.health_history: List[DatabaseHealth] = []
        self.query_stats: Dict[str, Dict] = {}
        self._pool = None

    async def _get_pool(self):
        """Get database connection pool"""
        if self._pool is None:
            try:
                from database.async_connection import get_pool
                self._pool = get_pool()
            except Exception as e:
                logger.warning(f"Could not get pool: {e}")
        return self._pool

    async def get_comprehensive_health(self) -> DatabaseHealth:
        """
        Get comprehensive database health status.
        """
        pool = await self._get_pool()
        if not pool:
            return DatabaseHealth(
                healthy=False,
                score=0,
                connection_count=0,
                max_connections=0,
                active_queries=0,
                slow_queries_count=0,
                dead_tuples=0,
                table_bloat_mb=0,
                index_usage_ratio=0,
                cache_hit_ratio=0,
                issues=["Database connection not available"],
                recommendations=["Check database connectivity"]
            )

        issues = []
        recommendations = []

        try:
            # Get connection stats
            conn_stats = await self._get_connection_stats(pool)

            # Get query stats
            query_stats = await self._get_query_stats(pool)

            # Get table stats
            table_stats = await self._get_table_stats(pool)

            # Get cache stats
            cache_stats = await self._get_cache_stats(pool)

            # Analyze and build health report
            score = 1.0

            # Connection health
            if conn_stats["usage_ratio"] > 0.8:
                issues.append(f"High connection usage: {conn_stats['usage_ratio']:.0%}")
                recommendations.append("Consider increasing max_connections or using connection pooling")
                score -= 0.2

            # Slow queries
            if query_stats["slow_count"] > 10:
                issues.append(f"High slow query count: {query_stats['slow_count']}")
                recommendations.append("Review and optimize slow queries")
                score -= 0.15

            # Dead tuples (bloat)
            if table_stats["total_dead_tuples"] > 100000:
                issues.append(f"High dead tuple count: {table_stats['total_dead_tuples']:,}")
                recommendations.append("Run VACUUM ANALYZE on affected tables")
                score -= 0.1

            # Cache hit ratio
            if cache_stats["hit_ratio"] < 0.9:
                issues.append(f"Low cache hit ratio: {cache_stats['hit_ratio']:.0%}")
                recommendations.append("Consider increasing shared_buffers")
                score -= 0.1

            # Index usage
            if table_stats["index_usage_ratio"] < 0.8:
                issues.append(f"Low index usage: {table_stats['index_usage_ratio']:.0%}")
                recommendations.append("Review missing indexes")
                score -= 0.1

            health = DatabaseHealth(
                healthy=score >= 0.7,
                score=max(0, score),
                connection_count=conn_stats["active"],
                max_connections=conn_stats["max"],
                active_queries=query_stats["active"],
                slow_queries_count=query_stats["slow_count"],
                dead_tuples=table_stats["total_dead_tuples"],
                table_bloat_mb=table_stats["bloat_mb"],
                index_usage_ratio=table_stats["index_usage_ratio"],
                cache_hit_ratio=cache_stats["hit_ratio"],
                issues=issues,
                recommendations=recommendations
            )

            self.health_history.append(health)
            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return DatabaseHealth(
                healthy=False,
                score=0.5,
                connection_count=0,
                max_connections=0,
                active_queries=0,
                slow_queries_count=0,
                dead_tuples=0,
                table_bloat_mb=0,
                index_usage_ratio=0,
                cache_hit_ratio=0,
                issues=[f"Health check error: {str(e)}"],
                recommendations=["Investigate database connectivity"]
            )

    async def _get_connection_stats(self, pool) -> Dict[str, Any]:
        """Get connection pool statistics"""
        try:
            result = await pool.fetchrow("""
                SELECT
                    count(*) as total,
                    count(*) FILTER (WHERE state = 'active') as active,
                    count(*) FILTER (WHERE state = 'idle') as idle,
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_conn
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)

            if result:
                return {
                    "total": result["total"],
                    "active": result["active"],
                    "idle": result["idle"],
                    "max": result["max_conn"],
                    "usage_ratio": result["total"] / result["max_conn"] if result["max_conn"] else 0
                }
        except Exception as e:
            logger.warning(f"Connection stats query failed: {e}")

        return {"total": 0, "active": 0, "idle": 0, "max": 100, "usage_ratio": 0}

    async def _get_query_stats(self, pool) -> Dict[str, Any]:
        """Get query performance statistics"""
        try:
            # Check if pg_stat_statements is available
            check = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_extension WHERE extname = 'pg_stat_statements'
                )
            """)

            if check:
                # Get slow queries from pg_stat_statements
                result = await pool.fetchrow("""
                    SELECT
                        count(*) as total_queries,
                        count(*) FILTER (WHERE mean_exec_time > 100) as slow_count,
                        COALESCE(sum(calls), 0) as total_calls,
                        COALESCE(avg(mean_exec_time), 0) as avg_time_ms
                    FROM pg_stat_statements
                    WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
                """)

                if result:
                    return {
                        "total": result["total_queries"],
                        "slow_count": result["slow_count"] or 0,
                        "total_calls": result["total_calls"],
                        "avg_time_ms": result["avg_time_ms"],
                        "active": 0
                    }

            # Fallback: just count active queries
            active = await pool.fetchval("""
                SELECT count(*) FROM pg_stat_activity
                WHERE state = 'active' AND datname = current_database()
            """)

            return {"total": 0, "slow_count": 0, "total_calls": 0, "avg_time_ms": 0, "active": active or 0}

        except Exception as e:
            logger.warning(f"Query stats failed: {e}")
            return {"total": 0, "slow_count": 0, "total_calls": 0, "avg_time_ms": 0, "active": 0}

    async def _get_table_stats(self, pool) -> Dict[str, Any]:
        """Get table statistics including bloat"""
        try:
            result = await pool.fetchrow("""
                SELECT
                    COALESCE(sum(n_dead_tup), 0) as dead_tuples,
                    COALESCE(sum(pg_total_relation_size(relid)) / 1024 / 1024, 0) as total_size_mb,
                    COALESCE(
                        avg(CASE WHEN idx_scan + seq_scan > 0
                            THEN idx_scan::float / (idx_scan + seq_scan)
                            ELSE 1 END
                        ), 0
                    ) as index_usage
                FROM pg_stat_user_tables
            """)

            if result:
                return {
                    "total_dead_tuples": int(result["dead_tuples"]),
                    "bloat_mb": float(result["total_size_mb"]) * 0.1,  # Estimate bloat as 10%
                    "index_usage_ratio": float(result["index_usage"])
                }
        except Exception as e:
            logger.warning(f"Table stats failed: {e}")

        return {"total_dead_tuples": 0, "bloat_mb": 0, "index_usage_ratio": 1.0}

    async def _get_cache_stats(self, pool) -> Dict[str, Any]:
        """Get cache hit statistics"""
        try:
            result = await pool.fetchrow("""
                SELECT
                    COALESCE(sum(blks_hit), 0) as hits,
                    COALESCE(sum(blks_read), 0) as reads
                FROM pg_stat_database
                WHERE datname = current_database()
            """)

            if result:
                total = result["hits"] + result["reads"]
                hit_ratio = result["hits"] / total if total > 0 else 1.0
                return {"hits": result["hits"], "reads": result["reads"], "hit_ratio": hit_ratio}
        except Exception as e:
            logger.warning(f"Cache stats failed: {e}")

        return {"hits": 0, "reads": 0, "hit_ratio": 1.0}

    async def detect_slow_queries(self, threshold_ms: float = 100) -> List[SlowQuery]:
        """Detect and analyze slow queries"""
        pool = await self._get_pool()
        if not pool:
            return []

        slow_queries = []

        try:
            # Check if pg_stat_statements is available
            check = await pool.fetchval("""
                SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'pg_stat_statements')
            """)

            if check:
                results = await pool.fetch(f"""
                    SELECT
                        query,
                        calls,
                        mean_exec_time as avg_ms,
                        max_exec_time as max_ms,
                        total_exec_time as total_ms
                    FROM pg_stat_statements
                    WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
                    AND mean_exec_time > {threshold_ms}
                    ORDER BY mean_exec_time DESC
                    LIMIT 20
                """)

                for row in results:
                    # Extract table name from query
                    table_name = self._extract_table_name(row["query"])

                    sq = SlowQuery(
                        query=row["query"][:500],  # Truncate long queries
                        duration_ms=row["total_ms"],
                        calls=row["calls"],
                        avg_duration_ms=row["avg_ms"],
                        max_duration_ms=row["max_ms"],
                        table_name=table_name,
                        optimization_suggestions=self._generate_query_suggestions(row["query"])
                    )
                    slow_queries.append(sq)

        except Exception as e:
            logger.warning(f"Slow query detection failed: {e}")

        self.slow_queries = slow_queries
        return slow_queries

    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract table name from query"""
        query_upper = query.upper()

        # FROM clause
        if "FROM" in query_upper:
            parts = query_upper.split("FROM")
            if len(parts) > 1:
                table_part = parts[1].strip().split()[0]
                return table_part.strip('"').lower()

        # UPDATE
        if query_upper.startswith("UPDATE"):
            parts = query_upper.split()
            if len(parts) > 1:
                return parts[1].strip('"').lower()

        # INSERT INTO
        if "INSERT INTO" in query_upper:
            parts = query_upper.split("INSERT INTO")
            if len(parts) > 1:
                table_part = parts[1].strip().split()[0]
                return table_part.strip('"').lower()

        return None

    def _generate_query_suggestions(self, query: str) -> List[str]:
        """Generate optimization suggestions for a query"""
        suggestions = []
        query_upper = query.upper()

        # Check for SELECT *
        if "SELECT *" in query_upper:
            suggestions.append("Avoid SELECT * - specify only needed columns")

        # Check for missing WHERE
        if "SELECT" in query_upper and "WHERE" not in query_upper and "LIMIT" not in query_upper:
            suggestions.append("Consider adding WHERE clause or LIMIT to reduce result set")

        # Check for LIKE with leading wildcard
        if "LIKE '%'" in query_upper or "LIKE '%" in query:
            suggestions.append("Leading wildcard in LIKE prevents index usage - consider full-text search")

        # Check for OR conditions
        if " OR " in query_upper:
            suggestions.append("OR conditions may prevent index usage - consider UNION instead")

        # Check for NOT IN with subquery
        if "NOT IN (SELECT" in query_upper:
            suggestions.append("NOT IN with subquery can be slow - consider LEFT JOIN with NULL check")

        # Check for large OFFSET
        if "OFFSET" in query_upper:
            suggestions.append("Large OFFSET is slow - consider keyset pagination")

        # Check for ORDER BY without index
        if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
            suggestions.append("ORDER BY without LIMIT may sort entire result set")

        if not suggestions:
            suggestions.append("Consider adding an index on frequently filtered columns")

        return suggestions

    async def get_index_recommendations(self) -> List[IndexRecommendation]:
        """Get AI-powered index recommendations"""
        pool = await self._get_pool()
        if not pool:
            return []

        recommendations = []

        try:
            # Find tables with sequential scans
            results = await pool.fetch("""
                SELECT
                    schemaname,
                    relname as table_name,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    n_live_tup
                FROM pg_stat_user_tables
                WHERE seq_scan > idx_scan
                AND n_live_tup > 1000
                ORDER BY seq_tup_read DESC
                LIMIT 10
            """)

            for row in results:
                table = row["table_name"]

                # Get frequently used columns from index stats
                cols = await pool.fetch(f"""
                    SELECT
                        attname
                    FROM pg_stats
                    WHERE tablename = '{table}'
                    AND n_distinct > 10
                    ORDER BY n_distinct DESC
                    LIMIT 3
                """)

                if cols:
                    col = cols[0]["attname"]
                    rec = IndexRecommendation(
                        table_name=table,
                        column_name=col,
                        reason=f"Table has {row['seq_scan']} sequential scans vs {row['idx_scan']} index scans",
                        estimated_improvement="Could reduce sequential scans by 50-80%",
                        create_statement=f"CREATE INDEX CONCURRENTLY idx_{table}_{col} ON {table} ({col})",
                        priority="high" if row["seq_scan"] > 1000 else "medium"
                    )
                    recommendations.append(rec)

        except Exception as e:
            logger.warning(f"Index recommendation failed: {e}")

        self.index_recommendations = recommendations
        return recommendations

    async def optimize_query(self, query: str) -> Dict[str, Any]:
        """Use AI to optimize a query"""
        try:
            from ai_intelligence import get_ai_intelligence

            ai = get_ai_intelligence()

            prompt = f"""You are a PostgreSQL query optimization expert.
Analyze this query and provide an optimized version:

ORIGINAL QUERY:
{query}

Provide:
1. Analysis of performance issues
2. Optimized query
3. Recommended indexes
4. Estimated improvement

Respond in JSON format:
{{
    "issues": ["list of issues"],
    "optimized_query": "the optimized SQL",
    "indexes_needed": ["CREATE INDEX statements"],
    "estimated_improvement": "percentage or description"
}}"""

            result = await ai._call_ai(prompt, model="standard")

            # Parse response
            import json
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0:
                return json.loads(result[json_start:json_end])

        except Exception as e:
            logger.warning(f"AI query optimization failed: {e}")

        # Fallback: return basic suggestions
        return {
            "issues": self._generate_query_suggestions(query),
            "optimized_query": query,
            "indexes_needed": [],
            "estimated_improvement": "Unknown - AI optimization unavailable"
        }

    async def run_maintenance(self) -> Dict[str, Any]:
        """Run database maintenance tasks"""
        pool = await self._get_pool()
        if not pool:
            return {"error": "Database not available"}

        results = {
            "vacuum_analyze": [],
            "reindex": [],
            "errors": []
        }

        try:
            # Get tables needing vacuum
            tables = await pool.fetch("""
                SELECT relname as table_name
                FROM pg_stat_user_tables
                WHERE n_dead_tup > 10000
                ORDER BY n_dead_tup DESC
                LIMIT 5
            """)

            for table in tables:
                try:
                    # Note: VACUUM cannot run in transaction
                    # This is informational - actual vacuum should be run by DBA
                    results["vacuum_analyze"].append({
                        "table": table["table_name"],
                        "recommendation": f"VACUUM ANALYZE {table['table_name']}"
                    })
                except Exception as e:
                    results["errors"].append(f"Vacuum {table['table_name']}: {e}")

        except Exception as e:
            results["errors"].append(f"Maintenance check failed: {e}")

        return results

    def get_health_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trend over time"""
        if not self.health_history:
            return {"trend": "unknown", "data_points": 0}

        recent = [h for h in self.health_history if (datetime.now(timezone.utc) - h.timestamp).total_seconds() < hours * 3600]

        if len(recent) < 2:
            return {"trend": "insufficient_data", "data_points": len(recent)}

        # Calculate trend
        scores = [h.score for h in recent]
        first_half_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)

        if second_half_avg > first_half_avg + 0.05:
            trend = "improving"
        elif second_half_avg < first_half_avg - 0.05:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "data_points": len(recent),
            "current_score": scores[-1],
            "average_score": sum(scores) / len(scores)
        }


# Global instance
_db_intelligence: Optional[DatabaseIntelligence] = None


def get_db_intelligence() -> DatabaseIntelligence:
    """Get the global database intelligence instance"""
    global _db_intelligence
    if _db_intelligence is None:
        _db_intelligence = DatabaseIntelligence()
    return _db_intelligence


async def get_db_health() -> Dict[str, Any]:
    """Convenience function to get database health"""
    db = get_db_intelligence()
    health = await db.get_comprehensive_health()

    return {
        "healthy": health.healthy,
        "score": health.score,
        "connections": f"{health.connection_count}/{health.max_connections}",
        "cache_hit_ratio": f"{health.cache_hit_ratio:.1%}",
        "slow_queries": health.slow_queries_count,
        "dead_tuples": health.dead_tuples,
        "issues": health.issues,
        "recommendations": health.recommendations,
        "timestamp": health.timestamp.isoformat()
    }
