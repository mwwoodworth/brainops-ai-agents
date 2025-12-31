"""
AI AWARENESS API - Complete System Intelligence
================================================
THE single endpoint ANY AI can call to understand EVERYTHING about the AI OS.
Designed for Claude, Gemini, Codex, and all future AI agents.

This is not a dashboard - it's INTELLIGENCE.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/awareness", tags=["AI Awareness"])

# Database pool
from database.async_connection import get_pool

class SystemAlert(BaseModel):
    severity: str  # critical, warning, info
    system: str
    message: str
    timestamp: str
    auto_remediation: Optional[str] = None

class AwarenessResponse(BaseModel):
    timestamp: str
    overall_health: str
    alerts: List[Dict[str, Any]]
    systems: Dict[str, Any]
    metrics: Dict[str, Any]
    recent_activity: Dict[str, Any]
    knowledge_summary: Dict[str, Any]
    recommendations: List[str]


async def _safe_query(pool, query: str, *args, default=None):
    """Execute query safely, return default on error"""
    try:
        return await pool.fetch(query, *args)
    except Exception as e:
        logger.warning(f"Query failed: {e}")
        return default or []


@router.get("/complete")
async def get_complete_awareness() -> Dict[str, Any]:
    """
    COMPLETE AI AWARENESS - Everything an AI needs to know

    Call this endpoint to understand:
    - Current system health and any problems
    - What's running and what's not
    - Recent activity and decisions
    - Alerts that need attention
    - Recommendations for action

    This is THE endpoint for AI context loading.
    """
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    now = datetime.utcnow()
    alerts: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    # ============================================
    # 1. AGENT HEALTH - Are agents actually working?
    # ============================================
    agent_health = {"status": "unknown", "details": {}}
    try:
        # Recent executions
        recent_execs = await _safe_query(pool, """
            SELECT
                agent_name,
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                COUNT(CASE WHEN status = 'running' THEN 1 END) as running,
                MAX(created_at) as last_run
            FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY agent_name
            ORDER BY total DESC
        """)

        total_runs = sum(r['total'] for r in recent_execs)
        total_completed = sum(r['completed'] for r in recent_execs)
        total_failed = sum(r['failed'] for r in recent_execs)
        stuck_agents = [r for r in recent_execs if r['running'] > 0]

        success_rate = (total_completed / total_runs * 100) if total_runs > 0 else 0

        agent_health = {
            "status": "healthy" if success_rate > 90 else "degraded" if success_rate > 70 else "critical",
            "executions_1hr": total_runs,
            "success_rate": round(success_rate, 1),
            "failed_1hr": total_failed,
            "active_agents": len(recent_execs),
            "stuck_agents": len(stuck_agents)
        }

        if total_failed > 5:
            alerts.append({
                "severity": "warning",
                "system": "agents",
                "message": f"{total_failed} agent executions failed in the last hour",
                "timestamp": now.isoformat()
            })

        if stuck_agents:
            alerts.append({
                "severity": "warning",
                "system": "agents",
                "message": f"{len(stuck_agents)} agents appear stuck (running status)",
                "timestamp": now.isoformat()
            })

    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        alerts.append({"severity": "critical", "system": "agents", "message": str(e), "timestamp": now.isoformat()})

    # ============================================
    # 2. AUREA ORCHESTRATOR - Is the brain working?
    # ============================================
    aurea_health = {"status": "unknown"}
    try:
        aurea_decisions = await _safe_query(pool, """
            SELECT COUNT(*) as count FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)

        decision_count = aurea_decisions[0]['count'] if aurea_decisions else 0
        aurea_health = {
            "status": "active" if decision_count > 10 else "idle" if decision_count > 0 else "inactive",
            "decisions_1hr": decision_count,
            "ooda_active": decision_count > 0
        }

        if decision_count == 0:
            alerts.append({
                "severity": "warning",
                "system": "aurea",
                "message": "No AUREA decisions in the last hour - orchestrator may be stalled",
                "timestamp": now.isoformat()
            })

    except Exception as e:
        logger.warning(f"AUREA check failed: {e}")

    # ============================================
    # 3. MEMORY SYSTEM - Is context being preserved?
    # ============================================
    memory_health = {"status": "unknown"}
    try:
        memory_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as recent,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings
            FROM unified_ai_memory
        """)

        brain_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN priority = 'critical' THEN 1 END) as critical
            FROM unified_brain
        """)

        logs_stats = await _safe_query(pool, """
            SELECT COUNT(*) as count FROM unified_brain_logs
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)

        mem = memory_stats[0] if memory_stats else {}
        brain = brain_stats[0] if brain_stats else {}
        logs = logs_stats[0] if logs_stats else {}

        memory_health = {
            "status": "active" if mem.get('recent', 0) > 0 else "stale",
            "unified_memory_total": mem.get('total', 0),
            "memory_writes_1hr": mem.get('recent', 0),
            "embeddings_available": mem.get('with_embeddings', 0),
            "brain_knowledge": brain.get('total', 0),
            "critical_knowledge": brain.get('critical', 0),
            "brain_logs_1hr": logs.get('count', 0)
        }

        if mem.get('with_embeddings', 0) == 0:
            alerts.append({
                "severity": "info",
                "system": "memory",
                "message": "No embeddings in memory - semantic search unavailable",
                "timestamp": now.isoformat()
            })
            recommendations.append("Configure OPENAI_API_KEY for semantic memory search")

    except Exception as e:
        logger.warning(f"Memory check failed: {e}")

    # ============================================
    # 4. SCHEMA ISSUES - Are queries working?
    # ============================================
    schema_issues = []
    try:
        # Check for recent errors in agent outputs mentioning schema
        schema_errors = await _safe_query(pool, """
            SELECT agent_name, error_message
            FROM ai_agent_executions
            WHERE status = 'failed'
            AND error_message ILIKE '%column%does not exist%'
            AND created_at > NOW() - INTERVAL '24 hours'
            LIMIT 5
        """)

        for err in schema_errors:
            schema_issues.append({
                "agent": err['agent_name'],
                "error": err['error_message'][:100]
            })

        if schema_issues:
            alerts.append({
                "severity": "warning",
                "system": "database",
                "message": f"{len(schema_issues)} schema mismatch errors detected",
                "timestamp": now.isoformat()
            })
            recommendations.append("Fix schema mismatches in agent queries")

    except Exception as e:
        logger.warning(f"Schema check failed: {e}")

    # ============================================
    # 5. REVENUE PIPELINE - Is money flowing?
    # ============================================
    revenue_health = {"status": "unknown"}
    try:
        revenue_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total_leads,
                COUNT(CASE WHEN stage = 'WON' THEN 1 END) as won,
                COALESCE(SUM(CASE WHEN stage = 'WON' THEN estimated_value ELSE 0 END), 0) as won_value
            FROM ai_revenue_leads
        """)

        rev = revenue_stats[0] if revenue_stats else {}
        revenue_health = {
            "status": "active" if rev.get('total_leads', 0) > 0 else "empty",
            "total_leads": rev.get('total_leads', 0),
            "won_deals": rev.get('won', 0),
            "revenue_tracked": float(rev.get('won_value', 0))
        }

    except Exception as e:
        logger.warning(f"Revenue check failed: {e}")

    # ============================================
    # 6. RECENT ACTIVITY - What just happened?
    # ============================================
    recent_activity = {}
    try:
        # Recent thoughts
        thoughts = await _safe_query(pool, """
            SELECT COUNT(*) as count FROM ai_thought_stream
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)

        # Recent brain writes
        brain_writes = await _safe_query(pool, """
            SELECT key, category FROM unified_brain
            WHERE created_at > NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC LIMIT 5
        """)

        recent_activity = {
            "thoughts_1hr": thoughts[0]['count'] if thoughts else 0,
            "brain_writes_1hr": [{"key": r['key'], "category": r['category']} for r in brain_writes]
        }

    except Exception as e:
        logger.warning(f"Activity check failed: {e}")

    # ============================================
    # 7. DETERMINE OVERALL HEALTH
    # ============================================
    critical_alerts = len([a for a in alerts if a['severity'] == 'critical'])
    warning_alerts = len([a for a in alerts if a['severity'] == 'warning'])

    if critical_alerts > 0:
        overall_health = "CRITICAL"
        recommendations.insert(0, "IMMEDIATE ACTION REQUIRED - Check critical alerts")
    elif warning_alerts > 2:
        overall_health = "DEGRADED"
        recommendations.insert(0, "Multiple systems need attention")
    elif agent_health.get('success_rate', 0) > 90 and aurea_health.get('status') == 'active':
        overall_health = "HEALTHY"
    else:
        overall_health = "OPERATIONAL"

    # ============================================
    # BUILD RESPONSE
    # ============================================
    return {
        "timestamp": now.isoformat(),
        "overall_health": overall_health,
        "alert_summary": {
            "critical": critical_alerts,
            "warning": warning_alerts,
            "info": len([a for a in alerts if a['severity'] == 'info'])
        },
        "alerts": alerts,
        "systems": {
            "agents": agent_health,
            "aurea_orchestrator": aurea_health,
            "memory": memory_health,
            "revenue": revenue_health
        },
        "recent_activity": recent_activity,
        "schema_issues": schema_issues,
        "recommendations": recommendations,
        "quick_actions": {
            "fix_stuck_agents": "/scheduler/restart-stuck",
            "trigger_health_check": "/execute/HealthMonitor",
            "search_brain": "POST /brain/search",
            "store_memory": "POST /memory/store",
            "aurea_status": "/aurea/status"
        }
    }


@router.get("/alerts")
async def get_active_alerts() -> Dict[str, Any]:
    """Get only active alerts that need attention"""
    awareness = await get_complete_awareness()
    return {
        "timestamp": awareness["timestamp"],
        "overall_health": awareness["overall_health"],
        "alerts": awareness["alerts"],
        "recommendations": awareness["recommendations"]
    }


@router.get("/quick")
async def get_quick_status() -> Dict[str, Any]:
    """Ultra-fast status check - just the essentials"""
    pool = get_pool()
    if not pool:
        return {"status": "database_unavailable"}

    try:
        # Quick counts only
        agents_1hr = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)

        decisions_1hr = await pool.fetchval("""
            SELECT COUNT(*) FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)

        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "agents_active": agents_1hr > 0,
            "agent_executions_1hr": agents_1hr,
            "aurea_active": decisions_1hr > 0,
            "aurea_decisions_1hr": decisions_1hr
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
