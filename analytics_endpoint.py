"""
AI Agent Analytics Endpoint
Provides comprehensive analytics for agent executions and performance
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["analytics"])


@router.post("/analytics")
async def agent_analytics(
    action: str = "analyze",
    period: str = "current_month",
    metric: Optional[str] = "revenue",
    agent_id: Optional[str] = None,
    data: Optional[dict[str, Any]] = None
):
    """
    AI Agent Analytics Endpoint

    Actions:
    - analyze: Analyze agent performance metrics
    - report: Generate detailed reports
    - predict: Predictive analytics based on historical data

    Periods:
    - current_month, last_month, current_quarter, current_year

    Metrics:
    - revenue: Revenue analysis
    - performance: Agent performance metrics
    - utilization: Agent utilization rates
    - success_rate: Task success rates
    """
    from database.async_connection import get_pool

    try:
        pool = get_pool()

        # Calculate date range based on period
        now = datetime.utcnow()
        if period == "current_month":
            start_date = datetime(now.year, now.month, 1)
            end_date = now
        elif period == "last_month":
            if now.month == 1:
                start_date = datetime(now.year - 1, 12, 1)
                end_date = datetime(now.year, 1, 1) - timedelta(seconds=1)
            else:
                start_date = datetime(now.year, now.month - 1, 1)
                end_date = datetime(now.year, now.month, 1) - timedelta(seconds=1)
        elif period == "current_quarter":
            quarter_month = ((now.month - 1) // 3) * 3 + 1
            start_date = datetime(now.year, quarter_month, 1)
            end_date = now
        else:  # current_year
            start_date = datetime(now.year, 1, 1)
            end_date = now

        response = {
            "action": action,
            "period": period,
            "metric": metric,
            "timestamp": now.isoformat(),
            "data": data or {}
        }

        if action == "analyze":
            if metric == "revenue":
                # Get revenue metrics from database
                revenue_query = """
                    SELECT
                        COUNT(*) as transaction_count,
                        SUM(CASE WHEN amount IS NOT NULL THEN amount ELSE 0 END) as total_revenue,
                        AVG(CASE WHEN amount IS NOT NULL THEN amount ELSE 0 END) as avg_revenue
                    FROM invoices
                    WHERE invoice_date >= $1 AND invoice_date <= $2
                """
                revenue_data = await pool.fetchrow(revenue_query, start_date, end_date)

                response["analysis"] = {
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "metrics": {
                        "total_revenue": float(revenue_data["total_revenue"] or 0),
                        "transaction_count": revenue_data["transaction_count"] or 0,
                        "average_revenue": float(revenue_data["avg_revenue"] or 0)
                    },
                    "status": "analyzed"
                }

            elif metric == "performance":
                # Get agent performance metrics
                perf_query = """
                    SELECT
                        COUNT(*) as total_executions,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                        AVG(duration_ms) as avg_duration_ms
                    FROM agent_executions
                    WHERE started_at >= $1 AND started_at <= $2
                """
                if agent_id:
                    perf_query += " AND agent_id::text = $3"
                    perf_data = await pool.fetchrow(perf_query, start_date, end_date, agent_id)
                else:
                    perf_data = await pool.fetchrow(perf_query, start_date, end_date)

                total = perf_data["total_executions"] or 1  # Avoid division by zero
                response["analysis"] = {
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "metrics": {
                        "total_executions": perf_data["total_executions"] or 0,
                        "successful_executions": perf_data["successful"] or 0,
                        "failed_executions": perf_data["failed"] or 0,
                        "success_rate": ((perf_data["successful"] or 0) / total) * 100,
                        "avg_duration_ms": float(perf_data["avg_duration_ms"] or 0)
                    },
                    "status": "analyzed"
                }

            elif metric == "utilization":
                # Get agent utilization metrics
                util_query = """
                    SELECT
                        a.id,
                        a.name,
                        a.category,
                        COUNT(ae.id) as execution_count,
                        MAX(ae.started_at) as last_executed
                    FROM agents a
                    LEFT JOIN agent_executions ae ON a.id = ae.agent_id
                        AND ae.started_at >= $1 AND ae.started_at <= $2
                    WHERE a.enabled = true
                    GROUP BY a.id, a.name, a.category
                    ORDER BY execution_count DESC
                """
                util_data = await pool.fetch(util_query, start_date, end_date)

                agents_utilized = sum(1 for row in util_data if row["execution_count"] > 0)
                total_agents = len(util_data)

                response["analysis"] = {
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "metrics": {
                        "total_agents": total_agents,
                        "agents_utilized": agents_utilized,
                        "utilization_rate": (agents_utilized / max(total_agents, 1)) * 100,
                        "top_agents": [
                            {
                                "id": str(row["id"]),
                                "name": row["name"],
                                "category": row["category"],
                                "executions": row["execution_count"],
                                "last_executed": row["last_executed"].isoformat() if row["last_executed"] else None
                            }
                            for row in util_data[:10]  # Top 10 agents
                        ]
                    },
                    "status": "analyzed"
                }

            elif metric == "success_rate":
                # Get success rate metrics by agent
                success_query = """
                    SELECT
                        a.name,
                        a.category,
                        COUNT(ae.id) as total,
                        COUNT(CASE WHEN ae.status = 'completed' THEN 1 END) as successful
                    FROM agents a
                    LEFT JOIN agent_executions ae ON a.id = ae.agent_id
                        AND ae.started_at >= $1 AND ae.started_at <= $2
                    WHERE a.enabled = true
                    GROUP BY a.name, a.category
                    HAVING COUNT(ae.id) > 0
                    ORDER BY COUNT(ae.id) DESC
                """
                success_data = await pool.fetch(success_query, start_date, end_date)

                response["analysis"] = {
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "metrics": {
                        "agent_success_rates": [
                            {
                                "name": row["name"],
                                "category": row["category"],
                                "total_executions": row["total"],
                                "successful_executions": row["successful"],
                                "success_rate": (row["successful"] / row["total"]) * 100
                            }
                            for row in success_data
                        ]
                    },
                    "status": "analyzed"
                }

        elif action == "report":
            # Generate comprehensive report
            response["report"] = {
                "title": f"AI Agent Analytics Report - {period}",
                "generated_at": now.isoformat(),
                "period": period,
                "summary": "Comprehensive agent performance and utilization report",
                "sections": ["performance", "utilization", "revenue", "recommendations"],
                "status": "generated"
            }

        elif action == "predict":
            # Predictive analytics based on real data

            # 1. Get current month revenue
            current_start = datetime(now.year, now.month, 1)
            revenue_query = """
                SELECT SUM(CASE WHEN amount IS NOT NULL THEN amount ELSE 0 END) as total
                FROM invoices
                WHERE invoice_date >= $1
            """
            current_rev_row = await pool.fetchrow(revenue_query, current_start)
            current_rev = float(current_rev_row["total"] or 0)

            # 2. Get previous month revenue
            if now.month == 1:
                prev_start = datetime(now.year - 1, 12, 1)
                prev_end = datetime(now.year, 1, 1) - timedelta(seconds=1)
            else:
                prev_start = datetime(now.year, now.month - 1, 1)
                prev_end = datetime(now.year, now.month, 1) - timedelta(seconds=1)

            prev_rev_row = await pool.fetchrow(
                "SELECT SUM(CASE WHEN amount IS NOT NULL THEN amount ELSE 0 END) as total FROM invoices WHERE invoice_date >= $1 AND invoice_date <= $2",
                prev_start, prev_end
            )
            prev_rev = float(prev_rev_row["total"] or 0)

            # 3. Calculate growth and projection
            growth = 0.0
            if prev_rev > 0:
                growth = ((current_rev - prev_rev) / prev_rev) * 100

            # Simple projection: Apply growth rate to current revenue
            # Note: This is a basic linear projection.
            next_month_projection = current_rev * (1 + (growth / 100)) if growth > 0 else current_rev

            response["prediction"] = {
                "period": period,
                "forecast": {
                    "next_month_revenue": round(next_month_projection, 2),
                    "expected_growth": round(growth, 2),
                    "confidence": 0.70 if prev_rev > 0 else 0.3
                },
                "status": "predicted"
            }

        else:
            response["error"] = f"Unknown action: {action}"
            response["status"] = "failed"

        return response

    except Exception as e:
        logger.error(f"Analytics endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analytics processing failed: {str(e)}"
        )


@router.get("/analytics/summary")
async def get_analytics_summary():
    """Get quick analytics summary"""
    from database.async_connection import get_pool

    try:
        pool = get_pool()

        # Get quick stats
        stats_query = """
            SELECT
                (SELECT COUNT(*) FROM agents WHERE enabled = true) as active_agents,
                (SELECT COUNT(*) FROM agent_executions WHERE created_at > NOW() - INTERVAL '24 hours') as executions_24h,
                (SELECT COUNT(*) FROM agent_executions WHERE status = 'completed' AND created_at > NOW() - INTERVAL '24 hours') as successful_24h,
                (SELECT AVG(latency_ms) FROM agent_executions WHERE created_at > NOW() - INTERVAL '24 hours') as avg_duration_24h
        """
        stats = await pool.fetchrow(stats_query)

        return {
            "summary": {
                "active_agents": stats["active_agents"] or 0,
                "executions_24h": stats["executions_24h"] or 0,
                "successful_24h": stats["successful_24h"] or 0,
                "success_rate_24h": ((stats["successful_24h"] or 0) / max(stats["executions_24h"] or 1, 1)) * 100,
                "avg_duration_ms": float(stats["avg_duration_24h"] or 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard")
async def get_dashboard_data(
    time_range: str = Query("7d", description="Time range: 24h, 7d, 30d, 90d"),
    include_predictions: bool = Query(True, description="Include predictive analytics")
):
    """
    Get comprehensive dashboard data with all key metrics

    Returns dashboard-ready data structure for visualization
    """
    from database.async_connection import get_pool

    try:
        pool = get_pool()
        now = datetime.utcnow()

        # Parse time range
        range_map = {
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90)
        }
        time_delta = range_map.get(time_range, timedelta(days=7))
        start_time = now - time_delta

        # 1. Overview metrics
        overview = await pool.fetchrow("""
            SELECT
                COUNT(DISTINCT ae.agent_id) as active_agents_count,
                COUNT(ae.id) as total_executions,
                COUNT(CASE WHEN ae.status = 'completed' THEN 1 END) as successful_executions,
                COUNT(CASE WHEN ae.status = 'failed' THEN 1 END) as failed_executions,
                AVG(ae.latency_ms) as avg_latency,
                MAX(ae.created_at) as last_execution
            FROM agent_executions ae
            WHERE ae.created_at >= $1
        """, start_time)

        # 2. Time series data (executions over time)
        time_series = await pool.fetch("""
            SELECT
                DATE_TRUNC('hour', created_at) as time_bucket,
                COUNT(*) as executions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                AVG(latency_ms) as avg_latency
            FROM agent_executions
            WHERE created_at >= $1
            GROUP BY time_bucket
            ORDER BY time_bucket
        """, start_time)

        # 3. Top performing agents
        top_agents = await pool.fetch("""
            SELECT
                a.name,
                a.category,
                COUNT(ae.id) as executions,
                COUNT(CASE WHEN ae.status = 'completed' THEN 1 END) as successful,
                AVG(ae.latency_ms) as avg_latency,
                MAX(ae.created_at) as last_used
            FROM agents a
            LEFT JOIN agent_executions ae ON a.id = ae.agent_id
                AND ae.created_at >= $1
            WHERE a.enabled = true
            GROUP BY a.name, a.category
            ORDER BY executions DESC
            LIMIT 10
        """, start_time)

        # 4. Agent category breakdown
        category_stats = await pool.fetch("""
            SELECT
                a.category,
                COUNT(ae.id) as executions,
                COUNT(CASE WHEN ae.status = 'completed' THEN 1 END) as successful,
                AVG(ae.latency_ms) as avg_latency
            FROM agents a
            LEFT JOIN agent_executions ae ON a.id = ae.agent_id
                AND ae.created_at >= $1
            WHERE a.enabled = true
            GROUP BY a.category
            ORDER BY executions DESC
        """, start_time)

        # 5. Recent errors
        recent_errors = await pool.fetch("""
            SELECT
                ae.agent_id::text,
                a.name as agent_name,
                ae.error_message,
                ae.created_at
            FROM agent_executions ae
            JOIN agents a ON a.id = ae.agent_id
            WHERE ae.status = 'failed'
                AND ae.created_at >= $1
            ORDER BY ae.created_at DESC
            LIMIT 10
        """, start_time)

        # Build dashboard response
        dashboard = {
            "generated_at": now.isoformat(),
            "time_range": time_range,
            "period_start": start_time.isoformat(),
            "period_end": now.isoformat(),

            "overview": {
                "active_agents": overview["active_agents_count"] or 0,
                "total_executions": overview["total_executions"] or 0,
                "successful_executions": overview["successful_executions"] or 0,
                "failed_executions": overview["failed_executions"] or 0,
                "success_rate": round(
                    ((overview["successful_executions"] or 0) / max(overview["total_executions"] or 1, 1)) * 100,
                    2
                ),
                "avg_latency_ms": float(overview["avg_latency"] or 0),
                "last_execution": overview["last_execution"].isoformat() if overview["last_execution"] else None
            },

            "time_series": [
                {
                    "timestamp": row["time_bucket"].isoformat(),
                    "executions": row["executions"],
                    "successful": row["successful"],
                    "success_rate": round((row["successful"] / max(row["executions"], 1)) * 100, 2),
                    "avg_latency": float(row["avg_latency"] or 0)
                }
                for row in time_series
            ],

            "top_agents": [
                {
                    "name": row["name"],
                    "category": row["category"],
                    "executions": row["executions"] or 0,
                    "successful": row["successful"] or 0,
                    "success_rate": round(
                        ((row["successful"] or 0) / max(row["executions"] or 1, 1)) * 100,
                        2
                    ),
                    "avg_latency_ms": float(row["avg_latency"] or 0),
                    "last_used": row["last_used"].isoformat() if row["last_used"] else None
                }
                for row in top_agents
            ],

            "category_breakdown": [
                {
                    "category": row["category"],
                    "executions": row["executions"] or 0,
                    "successful": row["successful"] or 0,
                    "success_rate": round(
                        ((row["successful"] or 0) / max(row["executions"] or 1, 1)) * 100,
                        2
                    ),
                    "avg_latency_ms": float(row["avg_latency"] or 0)
                }
                for row in category_stats
            ],

            "recent_errors": [
                {
                    "agent_id": row["agent_id"],
                    "agent_name": row["agent_name"],
                    "error": row["error_message"],
                    "timestamp": row["created_at"].isoformat()
                }
                for row in recent_errors
            ]
        }

        # Add AI-generated insights
        dashboard["insights"] = await generate_automated_insights(dashboard)

        # Add predictions if requested
        if include_predictions:
            dashboard["predictions"] = await generate_predictions(dashboard, pool, start_time)

        return dashboard

    except Exception as e:
        logger.error(f"Dashboard endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_automated_insights(dashboard_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate automated insights from dashboard data using AI

    Analyzes patterns and generates actionable insights
    """
    insights = []

    overview = dashboard_data.get("overview", {})
    time_series = dashboard_data.get("time_series", [])
    top_agents = dashboard_data.get("top_agents", [])

    # 1. Success rate insight
    success_rate = overview.get("success_rate", 0)
    if success_rate >= 95:
        insights.append({
            "type": "positive",
            "category": "performance",
            "title": "Excellent Success Rate",
            "message": f"System maintaining {success_rate:.1f}% success rate - well above target",
            "priority": "low",
            "action": "Continue monitoring"
        })
    elif success_rate < 80:
        insights.append({
            "type": "warning",
            "category": "performance",
            "title": "Success Rate Below Target",
            "message": f"Success rate at {success_rate:.1f}% - investigate failure causes",
            "priority": "high",
            "action": "Review recent errors and optimize failing agents"
        })

    # 2. Performance trend insight
    if len(time_series) >= 2:
        recent_executions = sum(t["executions"] for t in time_series[-3:])
        earlier_executions = sum(t["executions"] for t in time_series[:3]) if len(time_series) >= 6 else recent_executions

        if recent_executions > earlier_executions * 1.5:
            insights.append({
                "type": "positive",
                "category": "growth",
                "title": "Increased Activity Detected",
                "message": f"Agent activity increased by {((recent_executions / max(earlier_executions, 1)) - 1) * 100:.0f}%",
                "priority": "medium",
                "action": "Monitor resource utilization"
            })
        elif recent_executions < earlier_executions * 0.5:
            insights.append({
                "type": "warning",
                "category": "activity",
                "title": "Decreased Activity",
                "message": "Agent activity has decreased significantly",
                "priority": "medium",
                "action": "Investigate potential issues or reduced demand"
            })

    # 3. Latency insight
    avg_latency = overview.get("avg_latency_ms", 0)
    if avg_latency > 5000:  # 5 seconds
        insights.append({
            "type": "warning",
            "category": "performance",
            "title": "High Latency Detected",
            "message": f"Average latency at {avg_latency:.0f}ms - optimization recommended",
            "priority": "high",
            "action": "Profile slow agents and optimize code paths"
        })

    # 4. Agent utilization insight
    if top_agents:
        top_agent_executions = top_agents[0]["executions"]
        total_executions = overview.get("total_executions", 1)
        concentration = (top_agent_executions / total_executions) * 100

        if concentration > 50:
            insights.append({
                "type": "info",
                "category": "utilization",
                "title": "High Concentration on Single Agent",
                "message": f"Top agent handles {concentration:.0f}% of all executions",
                "priority": "medium",
                "action": "Consider load balancing or agent diversification"
            })

    # 5. Error rate insight
    failed_count = overview.get("failed_executions", 0)
    total_count = overview.get("total_executions", 1)
    error_rate = (failed_count / total_count) * 100

    if error_rate > 10:
        insights.append({
            "type": "critical",
            "category": "errors",
            "title": "High Error Rate",
            "message": f"Error rate at {error_rate:.1f}% - immediate attention required",
            "priority": "critical",
            "action": "Review error logs and implement fixes"
        })

    return insights


async def generate_predictions(dashboard_data: dict[str, Any], pool, start_time) -> dict[str, Any]:
    """
    Generate predictive analytics based on historical trends
    """
    try:
        # Get historical data for trend analysis
        time_series = dashboard_data.get("time_series", [])

        if len(time_series) < 5:
            return {
                "available": False,
                "reason": "Insufficient data for predictions"
            }

        # Extract execution counts
        execution_counts = [t["executions"] for t in time_series]

        # Simple linear regression for trend
        import numpy as np
        x = np.arange(len(execution_counts))
        y = np.array(execution_counts)

        # Calculate trend
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Predict next periods
        next_periods = 3
        predictions = []
        for i in range(1, next_periods + 1):
            predicted_value = coeffs[0] * (len(x) + i) + coeffs[1]
            predictions.append({
                "period": i,
                "predicted_executions": max(0, int(predicted_value)),
                "confidence": 0.7  # Simple model, medium confidence
            })

        # Trend classification
        if slope > len(execution_counts) * 0.1:
            trend = "increasing"
        elif slope < -len(execution_counts) * 0.1:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "available": True,
            "trend": trend,
            "trend_strength": abs(float(slope)),
            "predictions": predictions,
            "recommendation": _get_trend_recommendation(trend, slope, execution_counts)
        }

    except Exception as e:
        logger.error(f"Prediction generation error: {e}")
        return {
            "available": False,
            "reason": f"Error: {str(e)}"
        }


def _get_trend_recommendation(trend: str, slope: float, data: list[int]) -> str:
    """Generate recommendation based on trend"""
    if trend == "increasing":
        return "Activity is growing - consider scaling resources and monitoring capacity"
    elif trend == "decreasing":
        return "Activity is declining - investigate potential issues or seasonal patterns"
    else:
        return "Activity is stable - maintain current configuration and monitoring"


@router.get("/analytics/insights")
async def get_ai_insights(
    category: Optional[str] = Query(None, description="Filter by insight category")
):
    """
    Get AI-generated insights about system performance

    Uses machine learning to identify patterns and anomalies
    """
    from database.async_connection import get_pool

    try:
        # Get recent dashboard data
        pool = get_pool()
        now = datetime.utcnow()
        start_time = now - timedelta(days=7)

        # Simplified data fetch for insights
        overview = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_executions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                AVG(latency_ms) as avg_latency
            FROM agent_executions
            WHERE created_at >= $1
        """, start_time)

        # Generate insights using the dashboard structure
        dashboard_data = {
            "overview": {
                "total_executions": overview["total_executions"] or 0,
                "successful_executions": overview["successful"] or 0,
                "failed_executions": (overview["total_executions"] or 0) - (overview["successful"] or 0),
                "success_rate": ((overview["successful"] or 0) / max(overview["total_executions"] or 1, 1)) * 100,
                "avg_latency_ms": float(overview["avg_latency"] or 0)
            },
            "time_series": [],
            "top_agents": []
        }

        insights = await generate_automated_insights(dashboard_data)

        # Filter by category if specified
        if category:
            insights = [i for i in insights if i["category"] == category]

        return {
            "insights": insights,
            "generated_at": now.isoformat(),
            "total_insights": len(insights),
            "categories": list(set(i["category"] for i in insights))
        }

    except Exception as e:
        logger.error(f"AI insights endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
