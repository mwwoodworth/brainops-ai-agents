"""
AI OS UNIFIED DASHBOARD
=======================

THE ULTIMATE COMMAND CENTER FOR THE AI OS

This provides a single endpoint to see EVERYTHING:
- Neural Core status (self-awareness)
- Code Quality status (code-level health)
- All system health
- Active capabilities
- Intelligence engine status
- Optimization suggestions
- Recent activity
- Comprehensive observability metrics (NEW)

ONE ENDPOINT TO RULE THEM ALL.

Created: 2026-01-27
Updated: 2026-01-27 - Added /observability endpoint
"""

import asyncio
import logging
import time

from safe_task import create_safe_task
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import aiohttp
from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-os", tags=["ai-os-dashboard"])

# Track startup time for uptime calculation
_startup_time = time.time()


@router.get("/")
async def ai_os_root():
    """
    🌟 AI OS UNIFIED DASHBOARD ROOT

    The one endpoint to see everything.
    """
    return {
        "service": "BrainOps AI Operating System",
        "description": "TRUE AI OS - Self-Aware, Self-Healing, Intelligent",
        "version": "v10.3.0-observability",
        "philosophy": "You don't check the AI OS - you ASK it",
        "endpoints": {
            "GET /ai-os/status": "Complete AI OS status dashboard",
            "GET /ai-os/health": "Quick health check",
            "GET /ai-os/capabilities": "All AI OS capabilities",
            "GET /ai-os/issues": "All detected issues across systems",
            "GET /ai-os/activity": "Recent AI OS activity",
            "GET /ai-os/observability": "COMPREHENSIVE observability dashboard - ALL metrics",
            "GET /ai-os/observability/history": "Historical observability data for trend analysis",
            "GET /ai-os/observability/summary": "Quick observability summary for dashboards"
        }
    }


@router.get("/status")
async def get_complete_status() -> Dict[str, Any]:
    """
    📊 COMPLETE AI OS STATUS DASHBOARD

    Returns comprehensive status of the entire AI OS:
    - Neural Core (self-awareness)
    - Code Quality (code-level health)
    - All systems health
    - Active capabilities
    - Intelligence summary
    - Optimization suggestions
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ai_os_state": "OPERATIONAL",
        "neural_core": {},
        "code_quality": {},
        "systems": {},
        "capabilities": {},
        "intelligence": {},
        "optimizations": [],
        "issues": [],
        "overall_health": 1.0
    }

    # Get Neural Core status
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        report = core.generate_self_report()
        result["neural_core"] = {
            "state": core.state.value,
            "message": report.message,
            "health": report.overall_health,
            "systems_healthy": report.systems_healthy,
            "systems_total": report.systems_aware_of,
            "uptime_seconds": report.uptime_seconds,
            "decisions_made": report.recent_decisions,
            "active_healings": report.active_healings
        }
        result["issues"].extend(report.issues)
        result["systems"] = core.get_system_details()
    except Exception as e:
        result["neural_core"] = {"error": str(e)}
        logger.warning(f"Failed to get Neural Core status: {e}")

    # Get Code Quality status
    try:
        from code_quality_monitor import get_code_quality_monitor
        monitor = get_code_quality_monitor()
        summary = monitor.get_summary()
        result["code_quality"] = {
            "last_scan": summary.get("last_scan"),
            "total_issues": summary.get("total_issues", 0),
            "unresolved_issues": summary.get("unresolved_issues", 0),
            "critical_count": summary.get("critical_count", 0),
            "error_count": summary.get("error_count", 0),
            "auto_fixable": summary.get("auto_fixable", 0)
        }
        # Add code quality issues
        issues = monitor.get_all_issues(unresolved_only=True)
        for issue in issues[:10]:  # Limit to 10
            result["issues"].append(f"[CODE] {issue['source']}: {issue['message']}")
    except Exception as e:
        result["code_quality"] = {"error": str(e)}
        logger.warning(f"Failed to get Code Quality status: {e}")

    # Get Bleeding Edge capabilities
    try:
        from api.bleeding_edge import get_bleeding_edge_status
        be_status = await get_bleeding_edge_status()
        result["capabilities"] = {
            "bleeding_edge": be_status.get("capabilities", []),
            "total_capabilities": be_status.get("total_capabilities", 0),
            "consciousness_active": be_status.get("consciousness", {}).get("active", False),
            "ooda_active": be_status.get("ooda", {}).get("active", False)
        }
    except Exception as e:
        result["capabilities"] = {"error": str(e)}
        logger.warning(f"Failed to get capabilities: {e}")

    # Get Intelligence Engine status
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        intel_summary = core.get_intelligence_summary()
        result["intelligence"] = {
            "total_analyses": intel_summary.get("total_analyses", 0),
            "total_fix_attempts": intel_summary.get("total_fix_attempts", 0),
            "fix_success_rate": intel_summary.get("fix_success_rate", 0),
            "learning_active": True
        }
    except Exception as e:
        result["intelligence"] = {"error": str(e)}

    # Get optimization suggestions
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        suggestions = await core.get_optimization_suggestions()
        result["optimizations"] = suggestions[:5]  # Top 5
    except Exception as e:
        logger.warning(f"Failed to get optimizations: {e}")

    # Get Brain Memory stats
    try:
        from unified_brain import get_brain_stats
        brain_stats = await get_brain_stats()
        result["brain_memory"] = {
            "total_entries": brain_stats.get("total_entries", 0),
            "last_update": brain_stats.get("last_update")
        }
    except Exception as e:
        result["brain_memory"] = {"entries": "unknown"}

    # Calculate overall health
    neural_health = result["neural_core"].get("health", 1.0) if isinstance(result["neural_core"], dict) else 0.5
    code_issues = result["code_quality"].get("critical_count", 0) if isinstance(result["code_quality"], dict) else 0

    if code_issues > 0:
        result["overall_health"] = min(neural_health, 0.7)
    else:
        result["overall_health"] = neural_health

    # Determine AI OS state
    if result["overall_health"] >= 0.9:
        result["ai_os_state"] = "FULLY_OPERATIONAL"
    elif result["overall_health"] >= 0.7:
        result["ai_os_state"] = "OPERATIONAL_WITH_WARNINGS"
    elif result["overall_health"] >= 0.5:
        result["ai_os_state"] = "DEGRADED"
    else:
        result["ai_os_state"] = "CRITICAL"

    return result


@router.get("/health")
async def quick_health() -> Dict[str, Any]:
    """
    💚 QUICK HEALTH CHECK

    Fast health check for monitoring systems.
    """
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        report = core.generate_self_report()

        return {
            "healthy": report.overall_health >= 0.8,
            "health_score": report.overall_health,
            "state": core.state.value,
            "systems": f"{report.systems_healthy}/{report.systems_aware_of}",
            "message": report.message
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """
    🚀 ALL AI OS CAPABILITIES

    List all active AI OS capabilities.
    """
    capabilities = {
        "core": [
            "Self-Awareness (Neural Core)",
            "Continuous Monitoring",
            "Self-Healing (Auto-Restart)",
            "Intelligent Issue Analysis",
            "Pattern-Based Root Cause Detection",
            "Optimization Suggestions",
            "Code Quality Monitoring",
            "Integration Testing"
        ],
        "intelligence": [
            "Deep Issue Analysis",
            "Fix Strategy Recommendation",
            "Confidence Scoring",
            "Learning from Past Fixes",
            "Proactive Optimization"
        ],
        "agents": [],
        "bleeding_edge": []
    }

    # Get registered agents
    try:
        from agent_executor import get_all_agents
        agents = get_all_agents()
        capabilities["agents"] = [a.name for a in agents[:20]]  # First 20
    except Exception:
        capabilities["agents"] = ["Unable to enumerate"]

    # Get bleeding edge
    try:
        from api.bleeding_edge import get_bleeding_edge_status
        be_status = await get_bleeding_edge_status()
        capabilities["bleeding_edge"] = be_status.get("capabilities", [])
    except Exception:
        capabilities["bleeding_edge"] = ["Unable to enumerate"]

    capabilities["total"] = (
        len(capabilities["core"]) +
        len(capabilities["intelligence"]) +
        len(capabilities.get("agents", [])) +
        len(capabilities.get("bleeding_edge", []))
    )

    return capabilities


@router.get("/issues")
async def get_all_issues() -> Dict[str, Any]:
    """
    ⚠️ ALL DETECTED ISSUES

    Aggregated issues from all monitoring systems.
    """
    issues = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": 0,
        "critical": 0,
        "error": 0,
        "warning": 0,
        "by_source": {},
        "issues": []
    }

    # Get Neural Core issues
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        report = core.generate_self_report()
        for issue in report.issues:
            issues["issues"].append({
                "source": "neural_core",
                "severity": "error",
                "message": issue
            })
            issues["error"] += 1
    except Exception:
        pass

    # Get Code Quality issues
    try:
        from code_quality_monitor import get_code_quality_monitor
        monitor = get_code_quality_monitor()
        code_issues = monitor.get_all_issues(unresolved_only=True)
        for issue in code_issues:
            issues["issues"].append({
                "source": "code_quality",
                "severity": issue.get("severity", "error"),
                "message": f"{issue['source']}: {issue['message']}"
            })
            if issue.get("severity") == "critical":
                issues["critical"] += 1
            elif issue.get("severity") == "error":
                issues["error"] += 1
            else:
                issues["warning"] += 1
    except Exception:
        pass

    issues["total"] = len(issues["issues"])

    # Group by source
    for issue in issues["issues"]:
        source = issue["source"]
        if source not in issues["by_source"]:
            issues["by_source"][source] = 0
        issues["by_source"][source] += 1

    return issues


@router.get("/activity")
async def get_recent_activity() -> Dict[str, Any]:
    """
    📈 RECENT AI OS ACTIVITY

    Recent signals, decisions, and healings.
    """
    activity = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recent_signals": [],
        "recent_decisions": [],
        "recent_healings": []
    }

    try:
        from neural_core import get_neural_core
        core = get_neural_core()

        # Get recent signals
        signals = core.get_recent_signals(20)
        activity["recent_signals"] = signals

        # Get intelligence summary for decisions
        intel = core.get_intelligence_summary()
        activity["recent_decisions"] = intel.get("recent_analyses", [])
        activity["recent_healings"] = intel.get("recent_fixes", [])

    except Exception as e:
        activity["error"] = str(e)

    return activity


# =============================================================================
# COMPREHENSIVE OBSERVABILITY DASHBOARD
# =============================================================================


async def _get_system_health_metrics() -> Dict[str, Any]:
    """Get health metrics for all 9+ monitored systems."""
    systems = {}

    # Define all systems to check
    system_urls = {
        "brainops_ai_agents": "https://brainops-ai-agents.onrender.com/health",
        "brainops_backend": "https://brainops-backend-prod.onrender.com/health",
        "mcp_bridge": "https://brainops-mcp-bridge.onrender.com/health",
        "myroofgenius": "https://myroofgenius.com",
        "weathercraft_erp": "https://weathercraft-erp.vercel.app",
    }

    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for name, url in system_urls.items():
            try:
                async with session.get(url) as resp:
                    systems[name] = {
                        "status": "healthy" if resp.status == 200 else "degraded",
                        "response_code": resp.status,
                        "latency_ms": int(resp.headers.get("X-Response-Time", 0)) or None
                    }
            except Exception as e:
                systems[name] = {
                    "status": "unreachable",
                    "error": str(e)[:100]
                }

    # Add internal systems status
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            systems["database"] = {"status": "fallback", "connected": False}
        else:
            pool = get_pool()
            connected = await pool.test_connection()
            systems["database"] = {"status": "healthy" if connected else "degraded", "connected": connected}
    except Exception as e:
        systems["database"] = {"status": "error", "error": str(e)[:100]}

    # AUREA status
    try:
        from database.async_connection import get_pool, using_fallback
        if not using_fallback():
            pool = get_pool()
            cycles = await pool.fetchval("""
                SELECT COUNT(*) FROM aurea_state
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """) or 0
            systems["aurea"] = {"status": "running" if cycles > 0 else "idle", "cycles_last_hour": cycles}
    except Exception:
        systems["aurea"] = {"status": "unknown"}

    # Neural Core status
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        systems["neural_core"] = {"status": core.state.value, "healthy": True}
    except Exception:
        systems["neural_core"] = {"status": "unavailable"}

    # Calculate overall health
    healthy_count = sum(1 for s in systems.values() if s.get("status") in ["healthy", "running"])
    total_count = len(systems)

    return {
        "systems": systems,
        "healthy_count": healthy_count,
        "total_count": total_count,
        "health_percentage": round(healthy_count / total_count * 100, 2) if total_count > 0 else 0
    }


async def _get_ai_model_usage() -> Dict[str, Any]:
    """Get AI model usage statistics from database."""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Get model usage from ai_api_usage table
        usage_stats = await pool.fetch("""
            SELECT
                model,
                COUNT(*) as calls,
                SUM(COALESCE(tokens_used, 0)) as total_tokens,
                AVG(COALESCE(total_cost, 0)) as avg_cost_cents,
                SUM(COALESCE(total_cost, 0)) as total_cost_cents,
                MAX(created_at) as last_used
            FROM ai_api_usage
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY model
            ORDER BY calls DESC
        """)

        models = {}
        total_calls = 0
        total_tokens = 0
        total_cost = 0.0

        for row in usage_stats or []:
            model_name = row["model"] or "unknown"
            models[model_name] = {
                "calls_24h": row["calls"] or 0,
                "tokens_24h": row["total_tokens"] or 0,
                "avg_cost_cents": round(float(row["avg_cost_cents"] or 0), 4),
                "total_cost_cents": round(float(row["total_cost_cents"] or 0), 2),
                "last_used": row["last_used"].isoformat() if row["last_used"] else None
            }
            total_calls += row["calls"] or 0
            total_tokens += row["total_tokens"] or 0
            total_cost += float(row["total_cost_cents"] or 0)

        return {
            "status": "active" if total_calls > 0 else "idle",
            "total_calls_24h": total_calls,
            "total_tokens_24h": total_tokens,
            "total_cost_24h_cents": round(total_cost, 2),
            "by_model": models
        }
    except Exception as e:
        logger.error(f"Error getting AI model usage: {e}")
        return {"status": "error", "error": str(e)[:100]}


async def _get_latency_percentiles() -> Dict[str, Any]:
    """Get response latency percentiles from metrics registry."""
    try:
        from ai_observability import MetricsRegistry
        registry = MetricsRegistry.get_instance()

        histograms = registry.get_all_metrics().get("histograms", {})

        latency_metrics = {}
        for name, histogram_data in histograms.items():
            if "latency" in name.lower() or "duration" in name.lower():
                for label_key, stats in histogram_data.items():
                    metric_key = f"{name}_{label_key}" if label_key != "default" else name
                    latency_metrics[metric_key] = {
                        "count": stats.get("count", 0),
                        "avg_ms": round(stats.get("avg", 0), 2),
                        "p50_ms": round(stats.get("p50", 0), 2),
                        "p95_ms": round(stats.get("p95", 0), 2),
                        "p99_ms": round(stats.get("p99", 0), 2)
                    }

        return {
            "status": "active" if latency_metrics else "no_data",
            "metrics": latency_metrics
        }
    except Exception as e:
        logger.error(f"Error getting latency percentiles: {e}")
        return {"status": "error", "error": str(e)[:100]}


async def _get_error_rates() -> Dict[str, Any]:
    """Get error rates by endpoint from various sources."""
    error_data = {
        "by_endpoint": {},
        "by_severity": {"critical": 0, "error": 0, "warning": 0, "info": 0},
        "recent_errors": []
    }

    try:
        from ai_observability import MetricsRegistry
        registry = MetricsRegistry.get_instance()

        counters = registry.get_all_metrics().get("counters", {})

        # Find error counters
        for name, values in counters.items():
            if "error" in name.lower() or "failure" in name.lower():
                for label_key, count in values.items():
                    error_data["by_endpoint"][f"{name}_{label_key}"] = int(count)
    except Exception:
        pass

    # Get recent errors from database
    try:
        from database.async_connection import get_pool, using_fallback
        if not using_fallback():
            pool = get_pool()
            recent = await pool.fetch("""
                SELECT level, message, timestamp
                FROM observability.logs
                WHERE level IN ('ERROR', 'CRITICAL', 'WARNING')
                    AND timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 20
            """)

            for row in recent or []:
                error_data["recent_errors"].append({
                    "level": row["level"],
                    "message": row["message"][:200] if row["message"] else "",
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                })
                level = row["level"].lower() if row["level"] else "info"
                if level in error_data["by_severity"]:
                    error_data["by_severity"][level] += 1
    except Exception as e:
        logger.debug(f"Could not fetch error logs: {e}")

    total_errors = sum(error_data["by_severity"].values())
    error_data["total_errors_1h"] = total_errors
    error_data["status"] = "healthy" if total_errors == 0 else "degraded" if total_errors < 10 else "critical"

    return error_data


async def _get_memory_metrics() -> Dict[str, Any]:
    """Get unified AI memory system metrics."""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Unified AI Memory stats
        memory_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(DISTINCT memory_type) as memory_types,
                COUNT(DISTINCT source_system) as unique_systems,
                COALESCE(AVG(importance_score), 0) as avg_importance,
                COALESCE(AVG(access_count), 0) as avg_access_count,
                COUNT(*) FILTER (WHERE access_count >= 10) as hot_memories,
                COUNT(*) FILTER (WHERE access_count <= 1 AND created_at < NOW() - INTERVAL '30 days') as cold_memories,
                MAX(created_at) as last_memory_created
            FROM unified_ai_memory
        """)

        # Unified brain entries
        brain_count = await pool.fetchval("SELECT COUNT(*) FROM unified_brain") or 0

        # Conversations (table was dropped during cleanup - gracefully default to 0)
        try:
            conversation_count = await pool.fetchval("""
                SELECT COUNT(*) FROM ai_conversations
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """) or 0
        except Exception:
            conversation_count = 0

        return {
            "status": "active",
            "unified_ai_memory": {
                "total_memories": memory_stats["total_memories"] or 0,
                "with_embeddings": memory_stats["with_embeddings"] or 0,
                "embedding_coverage_pct": round(
                    (memory_stats["with_embeddings"] or 0) / max(memory_stats["total_memories"] or 1, 1) * 100, 2
                ),
                "memory_types": memory_stats["memory_types"] or 0,
                "unique_systems": memory_stats["unique_systems"] or 0,
                "avg_importance": round(float(memory_stats["avg_importance"] or 0), 3),
                "avg_access_count": round(float(memory_stats["avg_access_count"] or 0), 2),
                "hot_memories": memory_stats["hot_memories"] or 0,
                "cold_memories": memory_stats["cold_memories"] or 0,
                "last_memory_created": memory_stats["last_memory_created"].isoformat() if memory_stats["last_memory_created"] else None
            },
            "unified_brain_entries": brain_count,
            "conversations_24h": conversation_count
        }
    except Exception as e:
        logger.error(f"Error getting memory metrics: {e}")
        return {"status": "error", "error": str(e)[:100]}


async def _get_agent_execution_stats() -> Dict[str, Any]:
    """Get agent execution statistics."""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Agent execution stats
        executions = await pool.fetch("""
            SELECT
                agent_name,
                COUNT(*) as executions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_duration_sec,
                MAX(created_at) as last_run
            FROM ai_agent_executions
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY agent_name
            ORDER BY executions DESC
            LIMIT 15
        """)

        total_executions = sum(r["executions"] for r in executions) if executions else 0
        total_successful = sum(r["successful"] for r in executions) if executions else 0
        total_failed = sum(r["failed"] for r in executions) if executions else 0

        # Scheduled agent count
        scheduled_count = await pool.fetchval("""
            SELECT COUNT(*) FROM agent_schedules WHERE enabled = true
        """) or 0

        return {
            "status": "active" if total_executions > 0 else "idle",
            "total_executions_24h": total_executions,
            "successful_24h": total_successful,
            "failed_24h": total_failed,
            "success_rate": round(total_successful / max(total_executions, 1) * 100, 2),
            "scheduled_agents": scheduled_count,
            "top_agents": [
                {
                    "name": r["agent_name"],
                    "executions": r["executions"],
                    "successful": r["successful"],
                    "failed": r["failed"],
                    "avg_duration_sec": round(float(r["avg_duration_sec"] or 0), 2),
                    "last_run": r["last_run"].isoformat() if r["last_run"] else None
                }
                for r in (executions or [])[:10]
            ]
        }
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        return {"status": "error", "error": str(e)[:100]}


async def _get_prediction_accuracy() -> Dict[str, Any]:
    """Get prediction accuracy from learning systems."""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # AUREA decision accuracy
        decision_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_decisions,
                COUNT(CASE WHEN execution_status = 'completed' THEN 1 END) as successful,
                COUNT(CASE WHEN execution_status = 'failed' THEN 1 END) as failed,
                AVG(CASE WHEN confidence IS NOT NULL THEN confidence ELSE 0.5 END) as avg_confidence
            FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '7 days'
        """)

        # AI predictions accuracy (if tracked)
        prediction_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN actual_outcome = predicted_outcome THEN 1 END) as accurate,
                AVG(confidence) as avg_confidence
            FROM ai_learning_insights
            WHERE created_at > NOW() - INTERVAL '7 days'
                AND actual_outcome IS NOT NULL
        """) if await pool.fetchval("SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_learning_insights' AND column_name = 'actual_outcome')") else None

        total_decisions = decision_stats["total_decisions"] or 0 if decision_stats else 0
        successful = decision_stats["successful"] or 0 if decision_stats else 0

        return {
            "status": "active" if total_decisions > 0 else "no_data",
            "aurea_decisions": {
                "total_7d": total_decisions,
                "successful": successful,
                "failed": decision_stats["failed"] or 0 if decision_stats else 0,
                "success_rate": round(successful / max(total_decisions, 1) * 100, 2),
                "avg_confidence": round(float(decision_stats["avg_confidence"] or 0.5) * 100, 2) if decision_stats else 50.0
            },
            "prediction_accuracy": {
                "total_7d": prediction_stats["total_predictions"] or 0 if prediction_stats else 0,
                "accurate": prediction_stats["accurate"] or 0 if prediction_stats else 0,
                "accuracy_rate": round(
                    (prediction_stats["accurate"] or 0) / max(prediction_stats["total_predictions"] or 1, 1) * 100, 2
                ) if prediction_stats else 0,
                "avg_confidence": round(float(prediction_stats["avg_confidence"] or 0) * 100, 2) if prediction_stats else 0
            } if prediction_stats else {"status": "no_data"}
        }
    except Exception as e:
        logger.error(f"Error getting prediction accuracy: {e}")
        return {"status": "error", "error": str(e)[:100]}


async def _get_learning_progress() -> Dict[str, Any]:
    """Get learning system progress metrics."""
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable"}

        pool = get_pool()

        # Learning insights
        insights_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_insights,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as insights_24h,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as insights_7d,
                COUNT(DISTINCT insight_type) as insight_types
            FROM ai_learning_insights
        """)

        # Knowledge graph size
        knowledge_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_nodes,
                COUNT(DISTINCT node_type) as node_types
            FROM ai_knowledge_graph
        """)

        # Recent learning events
        recent_insights = await pool.fetch("""
            SELECT insight_type, insight_content, confidence, created_at
            FROM ai_learning_insights
            ORDER BY created_at DESC
            LIMIT 5
        """)

        return {
            "status": "active" if (insights_stats and insights_stats["insights_24h"]) else "idle",
            "insights": {
                "total": insights_stats["total_insights"] or 0 if insights_stats else 0,
                "last_24h": insights_stats["insights_24h"] or 0 if insights_stats else 0,
                "last_7d": insights_stats["insights_7d"] or 0 if insights_stats else 0,
                "insight_types": insights_stats["insight_types"] or 0 if insights_stats else 0,
                "learning_rate_per_hour": round((insights_stats["insights_24h"] or 0) / 24, 2) if insights_stats else 0
            },
            "knowledge_graph": {
                "total_nodes": knowledge_stats["total_nodes"] or 0 if knowledge_stats else 0,
                "node_types": knowledge_stats["node_types"] or 0 if knowledge_stats else 0
            },
            "recent_insights": [
                {
                    "type": r["insight_type"],
                    "content": r["insight_content"][:100] if r["insight_content"] else "",
                    "confidence": round(float(r["confidence"] or 0), 2),
                    "timestamp": r["created_at"].isoformat() if r["created_at"] else None
                }
                for r in (recent_insights or [])
            ]
        }
    except Exception as e:
        logger.error(f"Error getting learning progress: {e}")
        return {"status": "error", "error": str(e)[:100]}


async def _get_resource_utilization() -> Dict[str, Any]:
    """Get resource utilization metrics."""
    import psutil

    try:
        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "status": "active",
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round(disk.percent, 2)
            },
            "uptime_seconds": round(time.time() - _startup_time, 2),
            "uptime_human": str(timedelta(seconds=int(time.time() - _startup_time)))
        }
    except ImportError:
        # psutil not available
        return {
            "status": "limited",
            "uptime_seconds": round(time.time() - _startup_time, 2),
            "uptime_human": str(timedelta(seconds=int(time.time() - _startup_time))),
            "note": "psutil not installed - limited metrics available"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:100]}


@router.get("/observability")
async def get_observability_dashboard() -> Dict[str, Any]:
    """
    COMPREHENSIVE AI OBSERVABILITY DASHBOARD

    Aggregates ALL system metrics into a single unified endpoint:
    - System health metrics (all 9+ systems)
    - AI model usage statistics
    - Token consumption by model
    - Response latency percentiles
    - Error rates by endpoint
    - Memory system metrics
    - Agent execution statistics
    - Prediction accuracy
    - Learning system progress
    - Resource utilization

    This is THE endpoint for complete AI OS observability.
    """
    timestamp = datetime.now(timezone.utc)

    # Gather all metrics in parallel for maximum speed
    tasks = {
        "health": create_safe_task(_get_system_health_metrics()),
        "models": create_safe_task(_get_ai_model_usage()),
        "performance": create_safe_task(_get_latency_percentiles()),
        "errors": create_safe_task(_get_error_rates()),
        "memory": create_safe_task(_get_memory_metrics()),
        "agents": create_safe_task(_get_agent_execution_stats()),
        "predictions": create_safe_task(_get_prediction_accuracy()),
        "learning": create_safe_task(_get_learning_progress()),
        "resources": create_safe_task(_get_resource_utilization()),
    }

    results = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as e:
            logger.error(f"Failed to get {name} metrics: {e}")
            results[name] = {"status": "error", "error": str(e)[:100]}

    # Calculate overall observability status
    statuses = [r.get("status", "unknown") for r in results.values() if isinstance(r, dict)]
    error_count = sum(1 for s in statuses if s in ["error", "unavailable", "unreachable"])
    healthy_count = sum(1 for s in statuses if s in ["healthy", "active", "running"])

    if error_count == 0 and healthy_count >= len(statuses) * 0.7:
        overall_status = "healthy"
    elif error_count <= 2:
        overall_status = "degraded"
    else:
        overall_status = "critical"

    return {
        "timestamp": timestamp.isoformat(),
        "overall_status": overall_status,
        "health": results.get("health", {}),
        "models": results.get("models", {}),
        "performance": results.get("performance", {}),
        "errors": results.get("errors", {}),
        "memory": results.get("memory", {}),
        "agents": results.get("agents", {}),
        "predictions": results.get("predictions", {}),
        "learning": results.get("learning", {}),
        "resources": results.get("resources", {})
    }


@router.get("/observability/history")
async def get_observability_history(
    metric: str = Query("all", description="Metric category: health, models, agents, errors, memory, or all"),
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve")
) -> Dict[str, Any]:
    """
    Historical observability data for trend analysis.

    Retrieves time-series data for the specified metric category
    over the requested time period.
    """
    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            return {"status": "database_unavailable", "data": []}

        pool = get_pool()
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
            "hours": hours,
            "data": {}
        }

        if metric in ["all", "agents"]:
            # Agent execution history
            agent_history = await pool.fetch("""
                SELECT
                    DATE_TRUNC('hour', created_at) as hour,
                    COUNT(*) as total_executions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                FROM ai_agent_executions
                WHERE created_at > NOW() - ($1 || ' hours')::interval
                GROUP BY DATE_TRUNC('hour', created_at)
                ORDER BY hour
            """, str(hours))

            result["data"]["agents"] = [
                {
                    "hour": r["hour"].isoformat() if r["hour"] else None,
                    "total": r["total_executions"],
                    "successful": r["successful"],
                    "failed": r["failed"]
                }
                for r in (agent_history or [])
            ]

        if metric in ["all", "models"]:
            # Model usage history
            model_history = await pool.fetch("""
                SELECT
                    DATE_TRUNC('hour', created_at) as hour,
                    model,
                    COUNT(*) as calls,
                    SUM(COALESCE(tokens_used, 0)) as tokens
                FROM ai_api_usage
                WHERE created_at > NOW() - ($1 || ' hours')::interval
                GROUP BY DATE_TRUNC('hour', created_at), model
                ORDER BY hour
            """, str(hours))

            result["data"]["models"] = [
                {
                    "hour": r["hour"].isoformat() if r["hour"] else None,
                    "model": r["model"],
                    "calls": r["calls"],
                    "tokens": r["tokens"]
                }
                for r in (model_history or [])
            ]

        if metric in ["all", "errors"]:
            # Error history
            error_history = await pool.fetch("""
                SELECT
                    DATE_TRUNC('hour', timestamp) as hour,
                    level,
                    COUNT(*) as count
                FROM observability.logs
                WHERE timestamp > NOW() - ($1 || ' hours')::interval
                    AND level IN ('ERROR', 'CRITICAL', 'WARNING')
                GROUP BY DATE_TRUNC('hour', timestamp), level
                ORDER BY hour
            """, str(hours))

            result["data"]["errors"] = [
                {
                    "hour": r["hour"].isoformat() if r["hour"] else None,
                    "level": r["level"],
                    "count": r["count"]
                }
                for r in (error_history or [])
            ]

        if metric in ["all", "memory"]:
            # Memory stats history (if available)
            try:
                memory_history = await pool.fetch("""
                    SELECT
                        DATE_TRUNC('hour', created_at) as hour,
                        COUNT(*) as memories_created
                    FROM unified_ai_memory
                    WHERE created_at > NOW() - ($1 || ' hours')::interval
                    GROUP BY DATE_TRUNC('hour', created_at)
                    ORDER BY hour
                """, str(hours))

                result["data"]["memory"] = [
                    {
                        "hour": r["hour"].isoformat() if r["hour"] else None,
                        "memories_created": r["memories_created"]
                    }
                    for r in (memory_history or [])
                ]
            except Exception:
                result["data"]["memory"] = []

        if metric in ["all", "learning"]:
            # Learning insights history
            learning_history = await pool.fetch("""
                SELECT
                    DATE_TRUNC('hour', created_at) as hour,
                    COUNT(*) as insights,
                    AVG(confidence) as avg_confidence
                FROM ai_learning_insights
                WHERE created_at > NOW() - ($1 || ' hours')::interval
                GROUP BY DATE_TRUNC('hour', created_at)
                ORDER BY hour
            """, str(hours))

            result["data"]["learning"] = [
                {
                    "hour": r["hour"].isoformat() if r["hour"] else None,
                    "insights": r["insights"],
                    "avg_confidence": round(float(r["avg_confidence"] or 0), 3)
                }
                for r in (learning_history or [])
            ]

        return result

    except Exception as e:
        logger.error(f"Error getting observability history: {e}")
        return {"status": "error", "error": str(e), "data": {}}


@router.get("/observability/summary")
async def get_observability_summary() -> Dict[str, Any]:
    """
    Quick observability summary for dashboards.

    Returns key metrics in a compact format suitable for
    status indicators and quick health checks.
    """
    timestamp = datetime.now(timezone.utc)

    summary = {
        "timestamp": timestamp.isoformat(),
        "status": "unknown",
        "quick_stats": {}
    }

    try:
        from database.async_connection import get_pool, using_fallback
        if using_fallback():
            summary["status"] = "database_unavailable"
            return summary

        pool = get_pool()

        # Quick stats queries
        stats = await pool.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM ai_agent_executions WHERE created_at > NOW() - INTERVAL '24 hours') as agent_executions_24h,
                (SELECT COUNT(*) FROM ai_agent_executions WHERE status = 'completed' AND created_at > NOW() - INTERVAL '24 hours') as agent_success_24h,
                (SELECT COUNT(*) FROM ai_api_usage WHERE created_at > NOW() - INTERVAL '24 hours') as api_calls_24h,
                (SELECT SUM(COALESCE(tokens_used, 0)) FROM ai_api_usage WHERE created_at > NOW() - INTERVAL '24 hours') as tokens_24h,
                (SELECT COUNT(*) FROM unified_ai_memory) as total_memories,
                (SELECT COUNT(*) FROM unified_brain) as brain_entries,
                (SELECT COUNT(*) FROM ai_learning_insights WHERE created_at > NOW() - INTERVAL '24 hours') as learning_insights_24h
        """)

        agent_executions = stats["agent_executions_24h"] or 0
        agent_success = stats["agent_success_24h"] or 0

        summary["quick_stats"] = {
            "agent_executions_24h": agent_executions,
            "agent_success_rate": round(agent_success / max(agent_executions, 1) * 100, 1),
            "api_calls_24h": stats["api_calls_24h"] or 0,
            "tokens_consumed_24h": stats["tokens_24h"] or 0,
            "total_memories": stats["total_memories"] or 0,
            "brain_entries": stats["brain_entries"] or 0,
            "learning_insights_24h": stats["learning_insights_24h"] or 0,
            "uptime_seconds": round(time.time() - _startup_time, 0)
        }

        # Determine overall status
        if agent_executions > 0 and (stats["api_calls_24h"] or 0) > 0:
            summary["status"] = "healthy"
        elif agent_executions > 0 or (stats["api_calls_24h"] or 0) > 0:
            summary["status"] = "degraded"
        else:
            summary["status"] = "idle"

    except Exception as e:
        logger.error(f"Error getting observability summary: {e}")
        summary["status"] = "error"
        summary["error"] = str(e)[:100]

    return summary


@router.get("/invariant-engine")
async def get_invariant_engine_status() -> Dict[str, Any]:
    """
    Invariant Engine daemon status and last run results.

    Returns the singleton engine's run history, violation state,
    and check configuration for the OS Status dashboard.
    """
    try:
        from invariant_monitor import get_invariant_engine
        engine = get_invariant_engine()
        status = engine.get_status()

        # Also fetch recent violations from DB for dashboard
        from database.async_connection import get_pool, using_fallback
        if not using_fallback():
            pool = get_pool()
            recent = await pool.fetch("""
                SELECT check_name, severity, message,
                       created_at::text, resolved
                FROM invariant_violations
                WHERE resolved = false
                ORDER BY created_at DESC
                LIMIT 20
            """)
            status["open_violations"] = [dict(r) for r in recent]

            # RLS coverage snapshot
            rls_stats = await pool.fetchrow("""
                SELECT
                    count(*) FILTER (WHERE c.relrowsecurity = true) as rls_enabled,
                    count(*) FILTER (WHERE c.relrowsecurity = false) as rls_disabled,
                    count(*) as total
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relkind = 'r'
            """)
            status["rls_coverage"] = {
                "enabled": rls_stats["rls_enabled"],
                "disabled": rls_stats["rls_disabled"],
                "total": rls_stats["total"],
                "percent": round(rls_stats["rls_enabled"] / max(rls_stats["total"], 1) * 100, 1),
            }

        return {"status": "ok", "engine": status}
    except Exception as e:
        logger.error("Invariant engine status error: %s", e)
        return {"status": "error", "error": str(e)[:200]}
