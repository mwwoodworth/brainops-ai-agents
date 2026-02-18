"""
AI AWARENESS API - Complete System Intelligence
================================================
THE single endpoint ANY AI can call to understand EVERYTHING about the AI OS.
Designed for Claude, Gemini, Codex, and all future AI agents.

This is not a dashboard - it's INTELLIGENCE.
"""

import logging
from datetime import datetime
from typing import Any, Optional

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
    alerts: list[dict[str, Any]]
    systems: dict[str, Any]
    metrics: dict[str, Any]
    recent_activity: dict[str, Any]
    knowledge_summary: dict[str, Any]
    recommendations: list[str]


async def _safe_query(pool, query: str, *args, default=None):
    """Execute query safely, return default on error"""
    try:
        return await pool.fetch(query, *args)
    except Exception as e:
        logger.warning(f"Query failed: {e}")
        return default or []


@router.get("/complete")
async def get_complete_awareness() -> dict[str, Any]:
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
    alerts: list[dict[str, Any]] = []
    recommendations: list[str] = []

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
    # CRITICAL: Distinguish REAL vs TEST data for accurate DevOps context
    revenue_health = {"status": "unknown"}
    try:
        # Test email patterns for filtering
        test_filter = """
            email NOT ILIKE '%test%' AND email NOT ILIKE '%example%'
            AND email NOT ILIKE '%demo%' AND email NOT ILIKE '%sample%'
            AND email NOT ILIKE '%fake%' AND email NOT ILIKE '%placeholder%'
            AND email NOT ILIKE '@test.%' AND email NOT ILIKE '@example.%'
            AND email NOT ILIKE '%localhost%'
        """

        # REAL-ONLY revenue stats (ground truth)
        real_revenue_stats = await _safe_query(pool, f"""
            SELECT
                COUNT(*) as real_leads,
                COUNT(CASE WHEN stage = 'won' THEN 1 END) as real_won,
                COALESCE(SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END), 0) as real_revenue
            FROM revenue_leads
            WHERE {test_filter}
        """)

        # ALL data stats (for comparison)
        all_revenue_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total_leads,
                COUNT(CASE WHEN stage = 'won' THEN 1 END) as won,
                COALESCE(SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END), 0) as won_value
            FROM revenue_leads
        """)

        real = real_revenue_stats[0] if real_revenue_stats else {}
        all_rev = all_revenue_stats[0] if all_revenue_stats else {}

        real_leads = real.get('real_leads', 0)
        total_leads = all_rev.get('total_leads', 0)
        test_leads = total_leads - real_leads

        revenue_health = {
            "status": "active" if total_leads > 0 else "empty",
            # GROUND TRUTH - REAL ONLY
            "real_leads": real_leads,
            "real_won": real.get('real_won', 0),
            "real_revenue": float(real.get('real_revenue', 0)),
            # Test/Demo data (for awareness only)
            "test_leads": test_leads,
            "total_leads": total_leads,
            # Classification breakdown
            "data_breakdown": {
                "real": real_leads,
                "test_demo": test_leads,
                "real_percentage": round(real_leads / total_leads * 100, 1) if total_leads > 0 else 0
            },
            # Warning if all revenue is test data
            "warning": "ALL REVENUE IS TEST DATA - $0 REAL" if real.get('real_revenue', 0) == 0 and total_leads > 0 else None
        }

        # Add alert if $0 real revenue
        if real.get('real_revenue', 0) == 0 and real_leads > 0:
            alerts.append({
                "severity": "critical",
                "system": "revenue",
                "message": f"$0 real revenue from {real_leads} real leads - conversion needed!",
                "timestamp": now.isoformat()
            })
            recommendations.insert(0, f"CRITICAL: Convert {real_leads} real leads to revenue")

    except Exception as e:
        logger.warning(f"Revenue check failed: {e}")

    # ============================================
    # 5.5 DSPY REVENUE OPTIMIZER - Is the reward loop wired?
    # ============================================
    dspy_optimizer: dict[str, Any] = {"enabled": False, "compiled": False}
    try:
        from optimization.revenue_prompt_optimizer import get_revenue_prompt_optimizer

        optimizer = get_revenue_prompt_optimizer()
        dspy_optimizer = optimizer.status()

        # Summarize recent compile tasks (if task queue is being used)
        compile_stats = await _safe_query(
            pool,
            """
            SELECT status, COUNT(*) as count, MAX(created_at) as last_created
            FROM ai_task_queue
            WHERE task_type = 'revenue_prompt_compile'
              AND created_at > NOW() - INTERVAL '24 hours'
            GROUP BY status
            """,
            default=[],
        )
        dspy_optimizer["compile_tasks_24h"] = [
            {
                "status": r.get("status"),
                "count": int(r.get("count") or 0),
                "last_created": r.get("last_created").isoformat() if r.get("last_created") else None,
            }
            for r in (compile_stats or [])
        ]

        last_task_rows = await _safe_query(
            pool,
            """
            SELECT id, status, created_at, completed_at, error_message
            FROM ai_task_queue
            WHERE task_type = 'revenue_prompt_compile'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            default=[],
        )
        if last_task_rows:
            r0 = last_task_rows[0]
            dspy_optimizer["last_compile_task"] = {
                "id": str(r0.get("id")),
                "status": r0.get("status"),
                "created_at": r0.get("created_at").isoformat() if r0.get("created_at") else None,
                "completed_at": r0.get("completed_at").isoformat() if r0.get("completed_at") else None,
                "error_message": r0.get("error_message"),
            }

        if dspy_optimizer.get("enabled") and not dspy_optimizer.get("compiled"):
            recommendations.append(
                "DSPy optimizer enabled but not compiled yet (need more training samples or POST /api/v1/revenue/prompt-optimizer/compile)"
            )

        if dspy_optimizer.get("auto_recompile") and not dspy_optimizer.get("enabled"):
            alerts.append(
                {
                    "severity": "info",
                    "system": "dspy_optimizer",
                    "message": "DSPY_REVENUE_AUTO_RECOMPILE is enabled but optimizer is disabled/unavailable",
                    "timestamp": now.isoformat(),
                }
            )

    except Exception as e:
        logger.debug(f"DSPy optimizer status unavailable: {e}")

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
    # 7. MEMORY ENFORCEMENT (Total Completion Protocol)
    # ============================================
    enforcement_health = {"status": "unknown"}
    try:
        # Memory verification stats
        verification_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE verification_state = 'VERIFIED') as verified,
                COUNT(*) FILTER (WHERE verification_state = 'UNVERIFIED') as unverified,
                COUNT(*) FILTER (WHERE verification_state = 'DEGRADED') as degraded,
                AVG(confidence_score) as avg_confidence
            FROM unified_ai_memory
            WHERE expires_at IS NULL OR expires_at > NOW()
        """)

        # RBA/WBA audit stats (last 24 hours)
        audit_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total_ops,
                COUNT(*) FILTER (WHERE rba_enforced = true) as rba_enforced,
                COUNT(*) FILTER (WHERE wba_enforced = true) as wba_enforced,
                COUNT(*) FILTER (WHERE operation_result = 'blocked') as blocked
            FROM memory_operation_audit
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)

        # Truth backlog
        backlog_stats = await _safe_query(pool, """
            SELECT COUNT(*) as backlog_count FROM memory_truth_backlog
        """)

        # Conflict count
        conflict_stats = await _safe_query(pool, """
            SELECT COUNT(*) as open_conflicts FROM memory_conflicts
            WHERE resolution_status = 'open'
        """)

        ver = verification_stats[0] if verification_stats else {}
        aud = audit_stats[0] if audit_stats else {}
        backlog = backlog_stats[0] if backlog_stats else {}
        conflicts = conflict_stats[0] if conflict_stats else {}

        total_memories = ver.get('total', 0) or 0
        verified_count = ver.get('verified', 0) or 0
        verification_rate = (verified_count / max(total_memories, 1)) * 100

        rba_rate = ((aud.get('rba_enforced', 0) or 0) / max(aud.get('total_ops', 0) or 1, 1)) * 100
        wba_rate = ((aud.get('wba_enforced', 0) or 0) / max(aud.get('total_ops', 0) or 1, 1)) * 100

        enforcement_health = {
            "status": "active" if rba_rate > 0 or wba_rate > 0 else "inactive",
            "total_memories": total_memories,
            "verified": verified_count,
            "unverified": ver.get('unverified', 0) or 0,
            "degraded": ver.get('degraded', 0) or 0,
            "verification_rate": round(verification_rate, 1),
            "avg_confidence": round(float(ver.get('avg_confidence', 0) or 0), 3),
            "rba_enforcement_rate_24h": round(rba_rate, 1),
            "wba_enforcement_rate_24h": round(wba_rate, 1),
            "truth_backlog": backlog.get('backlog_count', 0) or 0,
            "open_conflicts": conflicts.get('open_conflicts', 0) or 0
        }

        if (conflicts.get('open_conflicts', 0) or 0) > 100:
            alerts.append({
                "severity": "warning",
                "system": "memory_enforcement",
                "message": f"{conflicts.get('open_conflicts', 0)} open memory conflicts need resolution",
                "timestamp": now.isoformat()
            })

    except Exception as e:
        logger.warning(f"Enforcement check failed: {e}")

    # ============================================
    # 8. LEARNING FEEDBACK LOOP (Total Completion Protocol)
    # ============================================
    learning_health = {"status": "unknown"}
    try:
        # Insight stats
        insight_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE applied = false) as unapplied,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as recent
            FROM ai_learning_insights
        """)

        # Proposal stats
        proposal_stats = await _safe_query(pool, """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'proposed') as pending,
                COUNT(*) FILTER (WHERE status = 'approved') as approved,
                COUNT(*) FILTER (WHERE status = 'queued_for_self_build') as queued_for_self_build,
                COUNT(*) FILTER (WHERE status = 'pr_opened') as pr_opened,
                COUNT(*) FILTER (WHERE status = 'completed') as completed
            FROM ai_improvement_proposals
        """)

        ins = insight_stats[0] if insight_stats else {}
        prop = proposal_stats[0] if proposal_stats else {}

        learning_health = {
            "status": "active" if (ins.get('recent', 0) or 0) > 0 else "idle",
            "total_insights": ins.get('total', 0) or 0,
            "unapplied_insights": ins.get('unapplied', 0) or 0,
            "insights_24hr": ins.get('recent', 0) or 0,
            "pending_proposals": prop.get('pending', 0) or 0,
            "approved_proposals": prop.get('approved', 0) or 0,
            "queued_for_self_build": prop.get('queued_for_self_build', 0) or 0,
            "pr_opened": prop.get('pr_opened', 0) or 0,
            "completed_proposals": prop.get('completed', 0) or 0
        }

        if (ins.get('unapplied', 0) or 0) > 1000:
            alerts.append({
                "severity": "info",
                "system": "learning",
                "message": f"{ins.get('unapplied', 0)} insights haven't been applied - run feedback loop",
                "timestamp": now.isoformat()
            })
            recommendations.append("Run /learning/feedback/run to process unapplied insights")

    except Exception as e:
        logger.warning(f"Learning check failed: {e}")

    # ============================================
    # 9. DETERMINE OVERALL HEALTH
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
            "revenue": revenue_health,
            "dspy_revenue_optimizer": dspy_optimizer,
            "memory_enforcement": enforcement_health,
            "learning_feedback": learning_health
        },
        "recent_activity": recent_activity,
        "schema_issues": schema_issues,
        "recommendations": recommendations,
        "quick_actions": {
            "fix_stuck_agents": "/resolver/fix/stuck-agents",
            "trigger_health_check": "/execute/HealthMonitor",
            "search_brain": "POST /brain/search",
            "store_memory": "POST /memory/store",
            "aurea_status": "/aurea/status",
            "enforcement_stats": "/enforcement/stats",
            "hygiene_health": "/hygiene/health",
            "run_feedback_loop": "POST /learning/feedback/run"
        }
    }


@router.get("/alerts")
async def get_active_alerts() -> dict[str, Any]:
    """Get only active alerts that need attention"""
    awareness = await get_complete_awareness()
    return {
        "timestamp": awareness["timestamp"],
        "overall_health": awareness["overall_health"],
        "alerts": awareness["alerts"],
        "recommendations": awareness["recommendations"]
    }


@router.get("/quick")
async def get_quick_status() -> dict[str, Any]:
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
