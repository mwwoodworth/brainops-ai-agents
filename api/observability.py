"""
BrainOps AI OS - Comprehensive Observability API
TRUE real-time visibility into every aspect of the system.
Perfect observability for perfect operations.
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/observability", tags=["observability"])


class SystemHealth(BaseModel):
    """Complete system health snapshot"""
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: float
    timestamp: str


class AUREAStatus(BaseModel):
    """AUREA orchestrator status"""
    running: bool
    mode: str
    cycles_last_hour: int
    decisions_pending: int
    decisions_completed: int
    health_score: float


class AgentStatus(BaseModel):
    """Agent execution status"""
    total_scheduled: int
    executions_last_hour: int
    success_rate: float
    top_agents: List[Dict[str, Any]]


class MemoryStatus(BaseModel):
    """Memory system status"""
    unified_brain_entries: int
    vector_memories: int
    knowledge_nodes: int
    last_sync: Optional[str]


class RevenueStatus(BaseModel):
    """Revenue pipeline status"""
    leads_in_pipeline: int
    proposals_pending: int
    mrr_estimate: float
    conversion_rate: float


class FullDashboard(BaseModel):
    """Complete system dashboard"""
    timestamp: str
    overall_status: str
    systems: Dict[str, Dict[str, Any]]
    aurea: Dict[str, Any]
    agents: Dict[str, Any]
    memory: Dict[str, Any]
    learning: Dict[str, Any]
    self_healing: Dict[str, Any]
    revenue: Dict[str, Any]
    database: Dict[str, Any]
    mcp: Dict[str, Any]


# Track startup time
_startup_time = time.time()


async def _get_database_stats() -> Dict[str, Any]:
    """Get database connection and table stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        pool = get_pool()

        if using_fallback():
            return {
                "status": "fallback",
                "connected": False,
                "using_in_memory": True,
                "reason": "Database connection failed, using in-memory fallback"
            }

        # Test connection
        connected = await pool.test_connection()
        if not connected:
            return {
                "status": "disconnected",
                "connected": False,
                "error": "Connection test failed"
            }

        # Get table counts
        try:
            result = await pool.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public') as total_tables,
                    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'ai_%') as ai_tables
            """)
            return {
                "status": "connected",
                "connected": True,
                "total_tables": result["total_tables"] if result else 0,
                "ai_tables": result["ai_tables"] if result else 0
            }
        except Exception as e:
            return {
                "status": "connected",
                "connected": True,
                "error": f"Could not get table stats: {e}"
            }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }


async def _get_aurea_stats() -> Dict[str, Any]:
    """Get AUREA orchestrator stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable", "running": False}

        pool = get_pool()

        # Get cycle count in last hour
        cycles = await pool.fetchval("""
            SELECT COUNT(*) FROM aurea_state
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """) or 0

        # Get decision stats
        decision_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN execution_status = 'pending' THEN 1 END) as pending,
                COUNT(CASE WHEN execution_status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN execution_status = 'failed' THEN 1 END) as failed
            FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)

        total_decisions = decision_stats["total"] if decision_stats else 0
        pending = decision_stats["pending"] if decision_stats else 0
        completed = decision_stats["completed"] if decision_stats else 0
        failed = decision_stats["failed"] if decision_stats else 0

        success_rate = (completed / total_decisions * 100) if total_decisions > 0 else 0

        return {
            "status": "running" if cycles > 0 else "stopped",
            "running": cycles > 0,
            "mode": "FULL_AUTO",
            "cycles_last_hour": cycles,
            "decisions_24h": {
                "total": total_decisions,
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "success_rate": round(success_rate, 2)
            },
            "health_score": round(success_rate, 2)
        }
    except Exception as e:
        logger.error(f"Error getting AUREA stats: {e}")
        return {"status": "error", "error": str(e), "running": False}


async def _get_agent_stats() -> Dict[str, Any]:
    """Get agent execution stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Get execution stats
        executions = await pool.fetch("""
            SELECT
                agent_name,
                COUNT(*) as executions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                MAX(created_at) as last_run
            FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY agent_name
            ORDER BY executions DESC
            LIMIT 10
        """)

        total_executions = sum(r["executions"] for r in executions) if executions else 0
        total_successful = sum(r["successful"] for r in executions) if executions else 0
        success_rate = (total_successful / total_executions * 100) if total_executions > 0 else 0

        # Get scheduled agent count
        scheduled_count = await pool.fetchval("""
            SELECT COUNT(*) FROM agent_schedules WHERE enabled = true
        """) or 0

        top_agents = [
            {
                "name": r["agent_name"],
                "executions": r["executions"],
                "successful": r["successful"],
                "last_run": r["last_run"].isoformat() if r["last_run"] else None
            }
            for r in (executions or [])
        ]

        return {
            "status": "active" if total_executions > 0 else "idle",
            "total_scheduled": scheduled_count,
            "executions_24h": total_executions,
            "success_rate": round(success_rate, 2),
            "top_agents": top_agents
        }
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        return {"status": "error", "error": str(e)}


async def _get_learning_stats() -> Dict[str, Any]:
    """Get learning system stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Get insight counts
        insights_24h = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_learning_insights
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """) or 0

        insights_7d = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_learning_insights
            WHERE created_at > NOW() - INTERVAL '7 days'
        """) or 0

        # Get knowledge node count
        knowledge_nodes = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_knowledge_graph
        """) or 0

        return {
            "status": "active" if insights_24h > 0 else "idle",
            "insights_24h": insights_24h,
            "insights_7d": insights_7d,
            "knowledge_nodes": knowledge_nodes,
            "learning_rate": round(insights_24h / 24, 2) if insights_24h else 0  # insights per hour
        }
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return {"status": "error", "error": str(e)}


async def _get_self_healing_stats() -> Dict[str, Any]:
    """Get self-healing system stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Get remediation stats
        remediations = await pool.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
            FROM remediation_history
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)

        total = remediations["total"] if remediations else 0
        successful = remediations["successful"] if remediations else 0
        failed = remediations["failed"] if remediations else 0

        return {
            "status": "active" if total > 0 else "idle",
            "remediations_24h": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / total * 100, 2) if total > 0 else 100
        }
    except Exception as e:
        logger.error(f"Error getting self-healing stats: {e}")
        return {"status": "error", "error": str(e)}


async def _get_memory_stats() -> Dict[str, Any]:
    """Get memory system stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Check for unified_brain table
        brain_count = await pool.fetchval("""
            SELECT COUNT(*) FROM unified_brain
        """) or 0

        # Check for vector memories
        vector_count = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_persistent_memory
        """) or 0

        # Check for conversations
        conversation_count = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_conversations
        """) or 0

        return {
            "status": "active",
            "unified_brain_entries": brain_count,
            "vector_memories": vector_count,
            "conversations": conversation_count,
            "total_memories": brain_count + vector_count
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"status": "error", "error": str(e)}


async def _get_revenue_stats() -> Dict[str, Any]:
    """Get revenue pipeline stats"""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Get lead pipeline
        leads = await pool.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN stage = 'NEW' THEN 1 END) as new,
                COUNT(CASE WHEN stage = 'QUALIFIED' THEN 1 END) as qualified,
                COUNT(CASE WHEN stage = 'PROPOSAL_SENT' THEN 1 END) as proposal_sent,
                COUNT(CASE WHEN stage = 'WON' THEN 1 END) as won
            FROM ai_leads
        """)

        # Get tenant/customer stats (actual revenue)
        tenant_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_tenants,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_tenants
            FROM tenants
        """)

        total_leads = leads["total"] if leads else 0
        won = leads["won"] if leads else 0
        active_tenants = tenant_stats["active_tenants"] if tenant_stats else 0

        # Estimate MRR (rough calculation)
        # Assuming average $75/mo per active tenant
        estimated_mrr = active_tenants * 75

        return {
            "status": "active" if total_leads > 0 else "idle",
            "leads_total": total_leads,
            "leads_new": leads["new"] if leads else 0,
            "leads_qualified": leads["qualified"] if leads else 0,
            "leads_proposal_sent": leads["proposal_sent"] if leads else 0,
            "leads_won": won,
            "conversion_rate": round(won / total_leads * 100, 2) if total_leads > 0 else 0,
            "active_tenants": active_tenants,
            "estimated_mrr": estimated_mrr
        }
    except Exception as e:
        logger.error(f"Error getting revenue stats: {e}")
        return {"status": "error", "error": str(e)}


async def _get_mcp_stats() -> Dict[str, Any]:
    """Get MCP integration stats"""
    try:
        # Check MCP availability
        try:
            from mcp_integration import MCPClient
            return {
                "status": "available",
                "servers": 13,  # Known count
                "tools": 358,   # Known count
                "last_check": datetime.utcnow().isoformat()
            }
        except ImportError:
            return {
                "status": "not_available",
                "error": "MCP integration not loaded"
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/dashboard")
async def get_full_dashboard() -> Dict[str, Any]:
    """
    Get COMPLETE system dashboard with real-time metrics.
    This is THE endpoint for full system observability.
    """
    timestamp = datetime.utcnow().isoformat()
    uptime = time.time() - _startup_time

    # Gather all stats in parallel for speed
    db_task = asyncio.create_task(_get_database_stats())
    aurea_task = asyncio.create_task(_get_aurea_stats())
    agent_task = asyncio.create_task(_get_agent_stats())
    learning_task = asyncio.create_task(_get_learning_stats())
    healing_task = asyncio.create_task(_get_self_healing_stats())
    memory_task = asyncio.create_task(_get_memory_stats())
    revenue_task = asyncio.create_task(_get_revenue_stats())
    mcp_task = asyncio.create_task(_get_mcp_stats())

    # Wait for all
    db_stats = await db_task
    aurea_stats = await aurea_task
    agent_stats = await agent_task
    learning_stats = await learning_task
    healing_stats = await healing_task
    memory_stats = await memory_task
    revenue_stats = await revenue_task
    mcp_stats = await mcp_task

    # Determine overall status
    issues = []
    if not db_stats.get("connected"):
        issues.append("database_disconnected")
    if not aurea_stats.get("running"):
        issues.append("aurea_not_running")
    if agent_stats.get("executions_24h", 0) < 10:
        issues.append("low_agent_activity")

    if len(issues) == 0:
        overall_status = "healthy"
    elif len(issues) <= 2:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    # Get version from environment or config
    try:
        from config import config
        version = getattr(config, 'version', '9.8.1')
    except:
        version = "9.8.1"

    return {
        "timestamp": timestamp,
        "overall_status": overall_status,
        "issues": issues,
        "version": version,
        "uptime_seconds": round(uptime, 2),
        "uptime_human": str(timedelta(seconds=int(uptime))),

        "database": db_stats,
        "aurea": aurea_stats,
        "agents": agent_stats,
        "learning": learning_stats,
        "self_healing": healing_stats,
        "memory": memory_stats,
        "revenue": revenue_stats,
        "mcp": mcp_stats,

        "systems": {
            "ai_agents": {
                "url": "https://brainops-ai-agents.onrender.com",
                "status": overall_status,
                "version": version
            },
            "backend": {
                "url": "https://brainops-backend-prod.onrender.com",
                "status": "check_external"
            },
            "mrg_app": {
                "url": "https://myroofgenius.com",
                "status": "check_external"
            },
            "erp": {
                "url": "https://weathercraft-erp.vercel.app",
                "status": "check_external"
            },
            "mcp_bridge": {
                "url": "https://brainops-mcp-bridge.onrender.com",
                "status": "check_external"
            }
        }
    }


@router.get("/health/deep")
async def deep_health_check() -> Dict[str, Any]:
    """
    Deep health check that verifies ALL system components.
    Use for monitoring and alerting.
    """
    checks = {}
    all_healthy = True

    # Check database
    db_stats = await _get_database_stats()
    checks["database"] = {
        "healthy": db_stats.get("connected", False),
        "details": db_stats
    }
    if not db_stats.get("connected"):
        all_healthy = False

    # Check AUREA
    aurea_stats = await _get_aurea_stats()
    checks["aurea"] = {
        "healthy": aurea_stats.get("running", False),
        "details": aurea_stats
    }
    if not aurea_stats.get("running"):
        all_healthy = False

    # Check agents
    agent_stats = await _get_agent_stats()
    agent_healthy = agent_stats.get("executions_24h", 0) > 0
    checks["agents"] = {
        "healthy": agent_healthy,
        "details": agent_stats
    }

    # Check memory
    memory_stats = await _get_memory_stats()
    checks["memory"] = {
        "healthy": memory_stats.get("status") != "error",
        "details": memory_stats
    }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_healthy": all_healthy,
        "checks": checks
    }


@router.get("/alerts")
async def get_active_alerts() -> Dict[str, Any]:
    """
    Get active system alerts based on health checks.
    """
    alerts = []

    # Check database
    db_stats = await _get_database_stats()
    if not db_stats.get("connected"):
        alerts.append({
            "severity": "critical",
            "component": "database",
            "message": "Database connection lost - using in-memory fallback",
            "timestamp": datetime.utcnow().isoformat()
        })

    # Check AUREA
    aurea_stats = await _get_aurea_stats()
    if not aurea_stats.get("running"):
        alerts.append({
            "severity": "high",
            "component": "aurea",
            "message": "AUREA orchestrator not running",
            "timestamp": datetime.utcnow().isoformat()
        })

    pending_decisions = aurea_stats.get("decisions_24h", {}).get("pending", 0)
    if pending_decisions > 50:
        alerts.append({
            "severity": "medium",
            "component": "aurea",
            "message": f"{pending_decisions} decisions pending execution",
            "timestamp": datetime.utcnow().isoformat()
        })

    # Check agents
    agent_stats = await _get_agent_stats()
    if agent_stats.get("executions_24h", 0) < 10:
        alerts.append({
            "severity": "medium",
            "component": "agents",
            "message": "Low agent activity - only {} executions in 24h".format(
                agent_stats.get("executions_24h", 0)
            ),
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_count": len(alerts),
        "critical": len([a for a in alerts if a["severity"] == "critical"]),
        "high": len([a for a in alerts if a["severity"] == "high"]),
        "medium": len([a for a in alerts if a["severity"] == "medium"]),
        "alerts": alerts
    }


@router.get("/diagnostics")
async def get_diagnostics() -> Dict[str, Any]:
    """
    Get detailed diagnostics for debugging.
    """
    import sys
    import platform

    # Get environment info
    env_vars = {
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "not_set"),
        "DB_HOST": "***" if os.getenv("DB_HOST") else "NOT_SET",
        "DB_NAME": os.getenv("DB_NAME", "NOT_SET"),
        "DB_USER": "***" if os.getenv("DB_USER") else "NOT_SET",
        "DB_PASSWORD": "***" if os.getenv("DB_PASSWORD") else "NOT_SET",
        "DATABASE_URL": "***" if os.getenv("DATABASE_URL") else "NOT_SET",
        "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else "NOT_SET",
        "ANTHROPIC_API_KEY": "***" if os.getenv("ANTHROPIC_API_KEY") else "NOT_SET",
    }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_vars": env_vars,
        "database": await _get_database_stats()
    }
