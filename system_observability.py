"""
BrainOps AI OS Observability Layer
==================================
Provides visibility into what the AI OS is actually doing.

This module exposes:
- Agent execution results (what did they find/produce?)
- Self-healing history (what was fixed automatically?)
- Active problems (what needs attention?)
- System awareness (natural language queries)
- Learning insights (what has the system learned?)

Created: 2026-02-02
Purpose: Make the AI OS transparent and queryable
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from database.async_connection import DatabaseUnavailableError, get_pool

logger = logging.getLogger(__name__)

# Router for all observability endpoints
router = APIRouter(prefix="/observe", tags=["Observability"])


class AgentResult(BaseModel):
    agent_name: str
    execution_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    result_summary: Optional[str]
    result_data: Optional[Dict[str, Any]]
    error: Optional[str]


class HealingAction(BaseModel):
    id: str
    timestamp: datetime
    issue_type: str
    issue_description: str
    action_taken: str
    outcome: str
    affected_component: str
    auto_resolved: bool


class ActiveProblem(BaseModel):
    id: str
    severity: str  # critical, warning, info
    component: str
    description: str
    detected_at: datetime
    suggested_action: str
    auto_fixable: bool


class SystemInsight(BaseModel):
    category: str
    insight: str
    confidence: float
    learned_at: datetime
    source: str


class SystemStatus(BaseModel):
    overall_health: str
    active_problems: List[ActiveProblem]
    recent_healing_actions: List[HealingAction]
    agent_health_summary: Dict[str, Any]
    recommendations: List[str]


def _get_pool_or_none():
    """Get the shared database pool, or None if unavailable."""
    try:
        return get_pool()
    except DatabaseUnavailableError:
        return None


@router.get("/agents/{agent_name}/results", response_model=List[AgentResult])
async def get_agent_results(
    agent_name: str,
    limit: int = Query(default=10, le=100),
    include_errors: bool = Query(default=True),
):
    """
    Get actual results/outputs from a specific agent.

    This answers: "What did this agent actually find or produce?"
    """
    pool = _get_pool_or_none()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            # Query agent executions with their results
            query = """
                SELECT
                    agent_name,
                    id::text as execution_id,
                    status,
                    started_at,
                    completed_at,
                    EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000 as duration_ms,
                    result->>'summary' as result_summary,
                    result as result_data,
                    error_message as error
                FROM ai_agent_executions
                WHERE agent_name = $1
            """
            if not include_errors:
                query += " AND status = 'completed'"
            query += " ORDER BY started_at DESC LIMIT $2"

            rows = await conn.fetch(query, agent_name, limit)

            results = []
            for row in rows:
                results.append(
                    AgentResult(
                        agent_name=row["agent_name"],
                        execution_id=row["execution_id"],
                        status=row["status"],
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        duration_ms=int(row["duration_ms"]) if row["duration_ms"] else None,
                        result_summary=row["result_summary"],
                        result_data=row["result_data"] if row["result_data"] else None,
                        error=row["error"],
                    )
                )

            return results
    except Exception as e:
        logger.error(f"Failed to get agent results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/healing/history", response_model=List[HealingAction])
async def get_healing_history(
    limit: int = Query(default=20, le=100), hours: int = Query(default=24, le=168)
):
    """
    Get history of self-healing actions taken by the system.

    This answers: "What has the AI OS automatically fixed?"
    """
    pool = _get_pool_or_none()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            since = datetime.utcnow() - timedelta(hours=hours)

            # Check multiple possible tables for healing data
            tables_to_try = [
                "ai_recovery_actions",
                "self_healing_actions",
                "ai_healing_log",
                "ai_system_events",
            ]

            for table in tables_to_try:
                try:
                    # Check if table exists
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                        table,
                    )
                    if not exists:
                        continue

                    if table == "ai_recovery_actions":
                        query = """
                            SELECT
                                id::text,
                                created_at as timestamp,
                                error_type as issue_type,
                                error_message as issue_description,
                                recovery_action as action_taken,
                                outcome,
                                component as affected_component,
                                COALESCE(auto_resolved, true) as auto_resolved
                            FROM ai_recovery_actions
                            WHERE created_at > $1
                            ORDER BY created_at DESC
                            LIMIT $2
                        """
                        rows = await conn.fetch(query, since, limit)
                        if rows:
                            return [HealingAction(**dict(row)) for row in rows]

                    elif table == "ai_system_events":
                        query = """
                            SELECT
                                id::text,
                                created_at as timestamp,
                                event_type as issue_type,
                                message as issue_description,
                                COALESCE(metadata->>'action', 'auto-recovery') as action_taken,
                                COALESCE(metadata->>'outcome', 'resolved') as outcome,
                                COALESCE(metadata->>'component', 'system') as affected_component,
                                true as auto_resolved
                            FROM ai_system_events
                            WHERE event_type IN ('healing', 'recovery', 'auto_fix', 'self_heal')
                            AND created_at > $1
                            ORDER BY created_at DESC
                            LIMIT $2
                        """
                        rows = await conn.fetch(query, since, limit)
                        if rows:
                            return [HealingAction(**dict(row)) for row in rows]

                except Exception as e:
                    logger.warning(f"Failed to query {table}: {e}")
                    continue

            # If no healing history found, return empty but check system health
            return []

    except Exception as e:
        logger.error(f"Failed to get healing history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/problems/active", response_model=List[ActiveProblem])
async def get_active_problems():
    """
    Get currently active problems that need attention.

    This answers: "What's wrong right now?"
    """
    problems = []

    pool = _get_pool_or_none()

    # 1. Check for failing agents
    try:
        if pool:
            async with pool.acquire() as conn:
                # Find agents with high failure rates in last hour
                query = """
                    SELECT
                        agent_name,
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed,
                        MAX(error_message) as last_error
                    FROM ai_agent_executions
                    WHERE started_at > NOW() - INTERVAL '1 hour'
                    GROUP BY agent_name
                    HAVING COUNT(*) FILTER (WHERE status = 'failed') > 0
                """
                rows = await conn.fetch(query)
                for row in rows:
                    failure_rate = row["failed"] / row["total"] * 100 if row["total"] > 0 else 0
                    if failure_rate > 30:
                        problems.append(
                            ActiveProblem(
                                id=f"agent-failure-{row['agent_name']}",
                                severity="warning" if failure_rate < 50 else "critical",
                                component=f"Agent: {row['agent_name']}",
                                description=f"{failure_rate:.0f}% failure rate ({row['failed']}/{row['total']} failed). Last error: {row['last_error'] or 'Unknown'}",
                                detected_at=datetime.utcnow(),
                                suggested_action=f"Review agent {row['agent_name']} logs and restart if needed",
                                auto_fixable=True,
                            )
                        )
    except Exception as e:
        logger.warning(f"Failed to check agent failures: {e}")

    # 2. Check for API quota issues (from recent logs)
    try:
        if pool:
            async with pool.acquire() as conn:
                # Check for rate limit errors
                query = """
                    SELECT COUNT(*) as count
                    FROM ai_unified_logs
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    AND (message ILIKE '%429%' OR message ILIKE '%quota%' OR message ILIKE '%rate limit%')
                """
                try:
                    count = await conn.fetchval(query)
                    if count and count > 5:
                        problems.append(
                            ActiveProblem(
                                id="api-quota-issue",
                                severity="critical",
                                component="AI Providers (OpenAI/Anthropic)",
                                description=f"{count} rate limit or quota errors in the last hour. API calls may be failing.",
                                detected_at=datetime.utcnow(),
                                suggested_action="Check API billing and increase quotas or reduce request frequency",
                                auto_fixable=False,
                            )
                        )
                except:
                    pass
    except Exception as e:
        logger.warning(f"Failed to check API quotas: {e}")

    # 3. Check circuit breakers
    try:
        from service_circuit_breakers import get_all_circuit_statuses

        if get_all_circuit_statuses:
            statuses = get_all_circuit_statuses()
            for service, status in statuses.items():
                if status.get("state") == "open":
                    problems.append(
                        ActiveProblem(
                            id=f"circuit-open-{service}",
                            severity="critical",
                            component=f"Service: {service}",
                            description=f"Circuit breaker OPEN for {service}. Service is being bypassed due to failures.",
                            detected_at=datetime.utcnow(),
                            suggested_action=f"Check {service} service health and wait for circuit to close",
                            auto_fixable=True,
                        )
                    )
    except Exception as e:
        logger.warning(f"Failed to check circuit breakers: {e}")

    # 4. Check memory system health
    try:
        from embedded_memory_system import get_embedded_memory

        if get_embedded_memory:
            mem = get_embedded_memory()
            stats = await mem.get_stats() if hasattr(mem, "get_stats") else None
            if stats and stats.get("pending_tasks", 0) > 10:
                problems.append(
                    ActiveProblem(
                        id="memory-backlog",
                        severity="warning",
                        component="Embedded Memory System",
                        description=f"{stats['pending_tasks']} memory tasks pending. Memory sync may be lagging.",
                        detected_at=datetime.utcnow(),
                        suggested_action="Memory system will catch up automatically, or trigger force sync",
                        auto_fixable=True,
                    )
                )
    except Exception as e:
        logger.warning(f"Failed to check memory system: {e}")

    return problems


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get comprehensive system status with problems, healing, and recommendations.

    This is the main "how is my AI OS doing?" endpoint.
    """
    problems = await get_active_problems()
    healing = await get_healing_history(limit=5, hours=24)

    # Calculate overall health
    critical_count = len([p for p in problems if p.severity == "critical"])
    warning_count = len([p for p in problems if p.severity == "warning"])

    if critical_count > 0:
        overall_health = "critical"
    elif warning_count > 2:
        overall_health = "degraded"
    elif warning_count > 0:
        overall_health = "warning"
    else:
        overall_health = "healthy"

    # Get agent health summary
    agent_summary = {}
    pool = _get_pool_or_none()
    if pool:
        try:
            async with pool.acquire() as conn:
                query = """
                    SELECT
                        agent_name,
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'completed') as success,
                        MAX(started_at) as last_run
                    FROM ai_agent_executions
                    WHERE started_at > NOW() - INTERVAL '24 hours'
                    GROUP BY agent_name
                    ORDER BY total DESC
                    LIMIT 10
                """
                rows = await conn.fetch(query)
                for row in rows:
                    agent_summary[row["agent_name"]] = {
                        "executions_24h": row["total"],
                        "success_rate": f"{(row['success']/row['total']*100):.0f}%"
                        if row["total"] > 0
                        else "N/A",
                        "last_run": row["last_run"].isoformat() if row["last_run"] else None,
                    }
        except Exception as e:
            logger.warning(f"Failed to get agent summary: {e}")

    # Generate recommendations
    recommendations = []
    if critical_count > 0:
        recommendations.append("⚠️ Address critical issues immediately")
    if any("quota" in str(p.description).lower() for p in problems):
        recommendations.append("💰 Review API billing - quota issues detected")
    if not healing:
        recommendations.append(
            "ℹ️ No recent self-healing actions - system may be stable or healing not triggered"
        )
    if len(agent_summary) < 5:
        recommendations.append("📊 Few agents active - consider activating more scheduled tasks")
    if overall_health == "healthy":
        recommendations.append("✅ System operating normally")

    return SystemStatus(
        overall_health=overall_health,
        active_problems=problems,
        recent_healing_actions=healing,
        agent_health_summary=agent_summary,
        recommendations=recommendations,
    )


@router.get("/learning/insights", response_model=List[SystemInsight])
async def get_learning_insights(
    limit: int = Query(default=20, le=100), category: Optional[str] = None
):
    """
    Get insights the system has learned over time.

    This answers: "What has the AI OS learned?"
    """
    pool = _get_pool_or_none()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    insights = []

    try:
        async with pool.acquire() as conn:
            # Check ai_learnings table
            try:
                query = """
                    SELECT
                        COALESCE(category, 'general') as category,
                        content as insight,
                        COALESCE(confidence, 0.8) as confidence,
                        created_at as learned_at,
                        COALESCE(source, 'system') as source
                    FROM ai_learnings
                """
                if category:
                    query += " WHERE category = $1"
                    query += " ORDER BY created_at DESC LIMIT $2"
                    rows = await conn.fetch(query, category, limit)
                else:
                    query += " ORDER BY created_at DESC LIMIT $1"
                    rows = await conn.fetch(query, limit)

                for row in rows:
                    insights.append(SystemInsight(**dict(row)))
            except Exception as e:
                logger.warning(f"ai_learnings table not available: {e}")

            # Also check ai_knowledge_graph for synthesized knowledge
            try:
                query = """
                    SELECT
                        'knowledge' as category,
                        content as insight,
                        0.85 as confidence,
                        created_at as learned_at,
                        'knowledge_graph' as source
                    FROM ai_knowledge_graph
                    WHERE node_type = 'insight'
                    ORDER BY created_at DESC
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)
                for row in rows:
                    insights.append(SystemInsight(**dict(row)))
            except Exception as e:
                logger.warning(f"ai_knowledge_graph not available: {e}")

    except Exception as e:
        logger.error(f"Failed to get learning insights: {e}")

    return sorted(insights, key=lambda x: x.learned_at, reverse=True)[:limit]


@router.post("/ask")
async def ask_system(
    question: str = Query(..., description="Natural language question about the system")
):
    """
    Ask the AI OS a question in natural language.

    Examples:
    - "What's wrong right now?"
    - "How are the agents performing?"
    - "What did CustomerIntelligence find today?"
    - "Are there any problems I should know about?"

    This is the conversational interface to the AI OS.
    """
    question_lower = question.lower()

    # Route to appropriate handler based on question
    if any(word in question_lower for word in ["wrong", "problem", "issue", "broken", "failing"]):
        problems = await get_active_problems()
        if not problems:
            return {
                "answer": "No active problems detected. All systems appear healthy.",
                "confidence": 0.9,
                "data": {"problems": []},
            }
        else:
            problem_list = "\n".join(
                [f"- [{p.severity.upper()}] {p.component}: {p.description}" for p in problems]
            )
            return {
                "answer": f"Found {len(problems)} issue(s):\n\n{problem_list}",
                "confidence": 0.95,
                "data": {"problems": [p.dict() for p in problems]},
            }

    elif any(word in question_lower for word in ["agent", "performing", "running", "status"]):
        status = await get_system_status()
        agent_info = "\n".join(
            [
                f"- {name}: {info['executions_24h']} runs, {info['success_rate']} success"
                for name, info in status.agent_health_summary.items()
            ]
        )
        return {
            "answer": f"System Status: {status.overall_health.upper()}\n\nAgent Performance (24h):\n{agent_info or 'No agent data available'}",
            "confidence": 0.9,
            "data": status.dict(),
        }

    elif any(word in question_lower for word in ["heal", "fix", "repair", "recover"]):
        healing = await get_healing_history(limit=10, hours=24)
        if not healing:
            return {
                "answer": "No self-healing actions in the last 24 hours. Either the system is stable or no auto-fixable issues occurred.",
                "confidence": 0.85,
                "data": {"healing_actions": []},
            }
        else:
            heal_list = "\n".join(
                [
                    f"- {h.timestamp.strftime('%H:%M')}: Fixed {h.issue_type} in {h.affected_component}"
                    for h in healing[:5]
                ]
            )
            return {
                "answer": f"{len(healing)} self-healing action(s) in the last 24h:\n\n{heal_list}",
                "confidence": 0.9,
                "data": {"healing_actions": [h.dict() for h in healing]},
            }

    elif any(word in question_lower for word in ["learn", "insight", "know"]):
        insights = await get_learning_insights(limit=5)
        if not insights:
            return {
                "answer": "No learning insights recorded yet. The system will learn from interactions over time.",
                "confidence": 0.7,
                "data": {"insights": []},
            }
        else:
            insight_list = "\n".join([f"- [{i.category}] {i.insight[:100]}..." for i in insights])
            return {
                "answer": f"Recent learnings:\n\n{insight_list}",
                "confidence": 0.85,
                "data": {"insights": [i.dict() for i in insights]},
            }

    else:
        # General status response
        status = await get_system_status()
        return {
            "answer": f"System Health: {status.overall_health.upper()}\n"
            f"Active Problems: {len(status.active_problems)}\n"
            f"Recent Healing: {len(status.recent_healing_actions)} actions\n"
            f"Active Agents: {len(status.agent_health_summary)}\n\n"
            f"Recommendations:\n" + "\n".join(status.recommendations),
            "confidence": 0.8,
            "data": status.dict(),
        }


# Convenience endpoint for quick health check
@router.get("/quick")
async def quick_status():
    """
    Quick one-line status check.

    Returns: "HEALTHY", "WARNING", "DEGRADED", or "CRITICAL"
    """
    problems = await get_active_problems()
    critical = len([p for p in problems if p.severity == "critical"])
    warnings = len([p for p in problems if p.severity == "warning"])

    if critical > 0:
        return {"status": "CRITICAL", "problems": critical, "warnings": warnings}
    elif warnings > 2:
        return {"status": "DEGRADED", "problems": 0, "warnings": warnings}
    elif warnings > 0:
        return {"status": "WARNING", "problems": 0, "warnings": warnings}
    else:
        return {"status": "HEALTHY", "problems": 0, "warnings": 0}
