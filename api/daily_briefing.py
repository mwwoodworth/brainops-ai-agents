"""
DAILY BRIEFING API
==================

Generates an AI-powered daily briefing summarizing overnight agent activity,
learning insights, memory growth, and system health.

Endpoints:
- GET /briefing - Get today's AI-generated daily briefing
- GET /briefing/stats - Get raw statistics without AI summary

Author: BrainOps AI System
Version: 1.0.0 (2026-02-02)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/briefing",
    tags=["daily-briefing"],
    dependencies=[Depends(verify_api_key)]
)


# Lazy imports to avoid circular dependencies
def _get_pool():
    """Get database pool with fallback check"""
    from database.async_connection import get_pool, using_fallback
    if using_fallback():
        raise HTTPException(status_code=503, detail="Database unavailable")
    return get_pool()


async def _get_learning_bridge():
    """Get learning action bridge"""
    try:
        from learning_action_bridge import get_learning_bridge
        return await get_learning_bridge()
    except ImportError:
        logger.warning("Learning action bridge not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get learning bridge: {e}")
        return None


def _get_ai_intelligence():
    """Get AI intelligence module"""
    try:
        from ai_intelligence import get_ai_intelligence
        return get_ai_intelligence()
    except ImportError:
        logger.warning("AI Intelligence not available")
        return None


def _get_circuit_breaker_health():
    """Get circuit breaker health status"""
    try:
        from service_circuit_breakers import get_circuit_breaker_health
        return get_circuit_breaker_health()
    except ImportError:
        logger.warning("Circuit breaker module not available")
        return None


class BriefingResponse(BaseModel):
    """Daily briefing response model"""
    briefing_date: str
    summary: str
    agent_activity: dict[str, Any]
    learning_insights: dict[str, Any]
    memory_growth: dict[str, Any]
    system_health: dict[str, Any]
    recommendations: list[str]
    generated_at: str


async def _get_agent_activity_stats(pool, hours: int = 24) -> dict[str, Any]:
    """Query agent execution statistics for the last N hours"""
    try:
        # Get execution counts and success rate
        stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_executions,
                COUNT(*) FILTER (WHERE status = 'completed' OR status = 'success') as successful,
                COUNT(*) FILTER (WHERE status = 'failed' OR status = 'error') as failed,
                COUNT(DISTINCT agent_name) as unique_agents
            FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '%s hours'
        """ % hours)

        # Get top performing agents
        top_agents_rows = await pool.fetch("""
            SELECT
                agent_name,
                COUNT(*) as execution_count,
                COUNT(*) FILTER (WHERE status = 'completed' OR status = 'success') as successful
            FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '%s hours'
            GROUP BY agent_name
            ORDER BY execution_count DESC
            LIMIT 5
        """ % hours)

        # Get agents with failures
        failed_agents_rows = await pool.fetch("""
            SELECT
                agent_name,
                COUNT(*) as failure_count,
                MAX(error_message) as last_error
            FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '%s hours'
              AND (status = 'failed' OR status = 'error')
            GROUP BY agent_name
            ORDER BY failure_count DESC
            LIMIT 3
        """ % hours)

        total = stats['total_executions'] or 0
        successful = stats['successful'] or 0
        failed = stats['failed'] or 0
        success_rate = (successful / total * 100) if total > 0 else 0

        top_agents = [row['agent_name'] for row in top_agents_rows]
        failed_agents = [
            {
                "agent": row['agent_name'],
                "failures": row['failure_count'],
                "last_error": (row['last_error'] or "")[:100]
            }
            for row in failed_agents_rows
        ]

        return {
            "executions_last_24h": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(success_rate, 1),
            "unique_agents": stats['unique_agents'] or 0,
            "top_agents": top_agents,
            "failed_agents": failed_agents if failed_agents else []
        }

    except Exception as e:
        logger.error(f"Failed to get agent activity stats: {e}")
        return {
            "executions_last_24h": 0,
            "success_rate": 0,
            "top_agents": [],
            "error": str(e)
        }


async def _get_learning_insights(pool, bridge) -> dict[str, Any]:
    """Get learning bridge status and behavior rules"""
    try:
        # Get recent learning patterns/insights from database
        insights_row = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_insights,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as new_insights_24h
            FROM ai_learning_insights
        """)

        # Get recent improvement proposals
        proposals_row = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_proposals,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as new_proposals_24h,
                COUNT(*) FILTER (WHERE status = 'proposed') as pending_approval,
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                COUNT(*) FILTER (WHERE auto_approved = true AND created_at > NOW() - INTERVAL '24 hours') as auto_approved_24h
            FROM ai_improvement_proposals
        """)

        # Get behavior rules from bridge if available
        rules = []
        rules_count = 0
        if bridge:
            status = bridge.get_status()
            rules_count = status.get('total_rules', 0)

            # Get recent rules (simplified - from bridge's internal state)
            for rule_id, rule in list(bridge.behavior_rules.items())[:5]:
                rules.append({
                    "type": rule.rule_type.value,
                    "action": rule.action[:80] + "..." if len(rule.action) > 80 else rule.action,
                    "confidence": round(rule.confidence, 2)
                })

        return {
            "new_insights_24h": insights_row['new_insights_24h'] if insights_row else 0,
            "total_insights": insights_row['total_insights'] if insights_row else 0,
            "new_proposals_24h": proposals_row['new_proposals_24h'] if proposals_row else 0,
            "pending_approval": proposals_row['pending_approval'] if proposals_row else 0,
            "auto_approved_24h": proposals_row['auto_approved_24h'] if proposals_row else 0,
            "behavior_rules_count": rules_count,
            "recent_rules": rules
        }

    except Exception as e:
        logger.error(f"Failed to get learning insights: {e}")
        return {
            "new_rules_created": 0,
            "rules": [],
            "error": str(e)
        }


async def _get_memory_growth(pool) -> dict[str, Any]:
    """Get memory growth statistics from unified_ai_memory"""
    try:
        # Get memory stats
        stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as new_memories_24h,
                COUNT(DISTINCT source_system) as unique_systems,
                COUNT(DISTINCT memory_type) as memory_types,
                AVG(importance_score) as avg_importance
            FROM unified_ai_memory
        """)

        # Get memory by type breakdown
        type_breakdown = await pool.fetch("""
            SELECT
                memory_type,
                COUNT(*) as count
            FROM unified_ai_memory
            GROUP BY memory_type
            ORDER BY count DESC
            LIMIT 5
        """)

        # Get memory by source system
        source_breakdown = await pool.fetch("""
            SELECT
                source_system,
                COUNT(*) as count
            FROM unified_ai_memory
            GROUP BY source_system
            ORDER BY count DESC
            LIMIT 5
        """)

        return {
            "new_memories_24h": stats['new_memories_24h'] or 0 if stats else 0,
            "total_memories": stats['total_memories'] or 0 if stats else 0,
            "unique_systems": stats['unique_systems'] or 0 if stats else 0,
            "memory_types": stats['memory_types'] or 0 if stats else 0,
            "avg_importance": round(float(stats['avg_importance'] or 0), 2) if stats else 0,
            "by_type": {row['memory_type']: row['count'] for row in type_breakdown},
            "by_source": {row['source_system']: row['count'] for row in source_breakdown}
        }

    except Exception as e:
        logger.error(f"Failed to get memory growth stats: {e}")
        return {
            "new_memories": 0,
            "total_memories": 0,
            "error": str(e)
        }


async def _get_system_health(pool) -> dict[str, Any]:
    """Get overall system health status"""
    try:
        # Get circuit breaker health
        cb_health = _get_circuit_breaker_health()

        if cb_health:
            total_circuits = cb_health.get('total_circuits', 0)
            closed_count = cb_health.get('by_state', {}).get('closed', 0)
            open_circuits = cb_health.get('open_circuits', [])
            overall = cb_health.get('overall_health', 'unknown')
            circuit_summary = f"{closed_count}/{total_circuits} closed"
        else:
            total_circuits = 0
            closed_count = 0
            open_circuits = []
            overall = "unknown"
            circuit_summary = "unavailable"

        # Get recent errors/alerts from unified_brain_logs
        alerts = []
        try:
            recent_errors = await pool.fetch("""
                SELECT
                    action,
                    data->>'error' as error,
                    created_at
                FROM unified_brain_logs
                WHERE action LIKE '%error%' OR action LIKE '%fail%' OR action LIKE '%alert%'
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 5
            """)

            for row in recent_errors:
                alerts.append({
                    "action": row['action'],
                    "error": (row['error'] or "")[:100],
                    "time": row['created_at'].isoformat() if row['created_at'] else None
                })
        except Exception as log_err:
            logger.debug(f"Could not fetch alerts from logs: {log_err}")

        # Get database health
        db_healthy = True
        try:
            await pool.fetchval("SELECT 1")
        except Exception:
            db_healthy = False

        return {
            "overall": "healthy" if overall == "healthy" and db_healthy else "degraded" if db_healthy else "critical",
            "database": "healthy" if db_healthy else "unhealthy",
            "circuit_breakers": circuit_summary,
            "open_circuits": open_circuits,
            "alerts": alerts
        }

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {
            "overall": "unknown",
            "circuit_breakers": "error",
            "alerts": [],
            "error": str(e)
        }


async def _generate_ai_summary(
    agent_activity: dict,
    learning_insights: dict,
    memory_growth: dict,
    system_health: dict,
    ai_intelligence
) -> tuple[str, list[str]]:
    """Generate AI-powered summary and recommendations"""

    if not ai_intelligence:
        # Fallback: generate basic summary without AI
        summary_parts = []
        recommendations = []

        # Agent activity summary
        exec_count = agent_activity.get('executions_last_24h', 0)
        success_rate = agent_activity.get('success_rate', 0)
        summary_parts.append(f"Processed {exec_count} agent executions with {success_rate}% success rate.")

        if success_rate < 90 and exec_count > 0:
            recommendations.append("Investigate failing agents to improve success rate")

        # Learning summary
        new_insights = learning_insights.get('new_insights_24h', 0)
        if new_insights > 0:
            summary_parts.append(f"Generated {new_insights} new learning insights.")

        pending = learning_insights.get('pending_approval', 0)
        if pending > 0:
            recommendations.append(f"Review {pending} pending improvement proposals")

        # Memory summary
        new_memories = memory_growth.get('new_memories_24h', 0)
        total_memories = memory_growth.get('total_memories', 0)
        summary_parts.append(f"Added {new_memories} new memories (total: {total_memories}).")

        # Health summary
        overall_health = system_health.get('overall', 'unknown')
        if overall_health != 'healthy':
            summary_parts.append(f"System health: {overall_health}.")
            recommendations.append("Address system health issues")
        else:
            summary_parts.append("All systems healthy.")

        alerts = system_health.get('alerts', [])
        if alerts:
            recommendations.append(f"Review {len(alerts)} recent alerts")

        summary = " ".join(summary_parts)

        if not recommendations:
            recommendations.append("Continue monitoring system performance")

        return summary, recommendations

    # Use AI to generate intelligent summary
    try:
        context = {
            "agent_activity": agent_activity,
            "learning_insights": learning_insights,
            "memory_growth": memory_growth,
            "system_health": system_health
        }

        prompt = f"""You are the BrainOps AI Operating System generating a daily briefing.

Analyze this overnight activity data and provide:
1. A 2-3 sentence executive summary of the key events
2. 3-5 actionable recommendations for the operator

DATA:
- Agent Executions: {agent_activity.get('executions_last_24h', 0)} total, {agent_activity.get('success_rate', 0)}% success rate
- Top Agents: {', '.join(agent_activity.get('top_agents', [])[:3]) or 'None'}
- Failed Agents: {len(agent_activity.get('failed_agents', []))} agents with failures
- New Learning Insights: {learning_insights.get('new_insights_24h', 0)}
- Pending Proposals: {learning_insights.get('pending_approval', 0)}
- New Memories: {memory_growth.get('new_memories_24h', 0)} (Total: {memory_growth.get('total_memories', 0)})
- System Health: {system_health.get('overall', 'unknown')}
- Alerts: {len(system_health.get('alerts', []))}

Respond in JSON format:
{{
    "summary": "Your executive summary here",
    "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]
}}"""

        import json
        response = await ai_intelligence._call_ai(prompt, model="standard")

        # Parse JSON response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(response[json_start:json_end])
            return data.get("summary", "Analysis complete."), data.get("recommendations", [])

    except Exception as e:
        logger.warning(f"AI summary generation failed: {e}")

    # Fallback
    return (
        f"Processed {agent_activity.get('executions_last_24h', 0)} agent executions overnight with "
        f"{agent_activity.get('success_rate', 0)}% success rate. System health: {system_health.get('overall', 'unknown')}.",
        ["Review system logs for details", "Monitor agent performance"]
    )


@router.get("/", response_model=BriefingResponse)
async def get_daily_briefing(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back (default 24, max 168)")
) -> BriefingResponse:
    """
    Generate an AI-powered daily briefing.

    Summarizes:
    - Overnight agent executions and success rates
    - Learning bridge status and new behavior rules
    - Memory growth in unified_ai_memory
    - System health and circuit breaker status

    Returns an AI-generated summary with actionable recommendations.
    """
    pool = _get_pool()
    bridge = await _get_learning_bridge()
    ai = _get_ai_intelligence()

    # Gather all statistics
    agent_activity = await _get_agent_activity_stats(pool, hours)
    learning_insights = await _get_learning_insights(pool, bridge)
    memory_growth = await _get_memory_growth(pool)
    system_health = await _get_system_health(pool)

    # Generate AI summary
    summary, recommendations = await _generate_ai_summary(
        agent_activity,
        learning_insights,
        memory_growth,
        system_health,
        ai
    )

    now = datetime.now(timezone.utc)

    return BriefingResponse(
        briefing_date=now.strftime("%Y-%m-%d"),
        summary=summary,
        agent_activity=agent_activity,
        learning_insights=learning_insights,
        memory_growth=memory_growth,
        system_health=system_health,
        recommendations=recommendations,
        generated_at=now.isoformat()
    )


@router.get("/stats")
async def get_briefing_stats(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back")
) -> dict[str, Any]:
    """
    Get raw briefing statistics without AI summary.

    Useful for dashboards and custom analysis.
    """
    pool = _get_pool()
    bridge = await _get_learning_bridge()

    agent_activity = await _get_agent_activity_stats(pool, hours)
    learning_insights = await _get_learning_insights(pool, bridge)
    memory_growth = await _get_memory_growth(pool)
    system_health = await _get_system_health(pool)

    return {
        "hours_analyzed": hours,
        "agent_activity": agent_activity,
        "learning_insights": learning_insights,
        "memory_growth": memory_growth,
        "system_health": system_health,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health")
async def briefing_health_check() -> dict[str, Any]:
    """
    Health check for the daily briefing endpoint.

    Verifies database connectivity and module availability.
    """
    status = {
        "endpoint": "daily_briefing",
        "database": "unknown",
        "ai_intelligence": "unknown",
        "learning_bridge": "unknown",
        "circuit_breakers": "unknown"
    }

    # Check database
    try:
        pool = _get_pool()
        await pool.fetchval("SELECT 1")
        status["database"] = "healthy"
    except Exception as e:
        status["database"] = f"error: {str(e)[:50]}"

    # Check AI Intelligence
    ai = _get_ai_intelligence()
    status["ai_intelligence"] = "available" if ai else "unavailable"

    # Check Learning Bridge
    try:
        bridge = await _get_learning_bridge()
        status["learning_bridge"] = "available" if bridge else "unavailable"
    except Exception:
        status["learning_bridge"] = "error"

    # Check Circuit Breakers
    cb_health = _get_circuit_breaker_health()
    status["circuit_breakers"] = "available" if cb_health else "unavailable"

    all_healthy = (
        status["database"] == "healthy" and
        status["ai_intelligence"] == "available"
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": status,
        "checked_at": datetime.now(timezone.utc).isoformat()
    }
