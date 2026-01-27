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

ONE ENDPOINT TO RULE THEM ALL.

Created: 2026-01-27
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-os", tags=["ai-os-dashboard"])


@router.get("/")
async def ai_os_root():
    """
    🌟 AI OS UNIFIED DASHBOARD ROOT

    The one endpoint to see everything.
    """
    return {
        "service": "BrainOps AI Operating System",
        "description": "TRUE AI OS - Self-Aware, Self-Healing, Intelligent",
        "version": "v10.2.0-intelligence",
        "philosophy": "You don't check the AI OS - you ASK it",
        "endpoints": {
            "GET /ai-os/status": "Complete AI OS status dashboard",
            "GET /ai-os/health": "Quick health check",
            "GET /ai-os/capabilities": "All AI OS capabilities",
            "GET /ai-os/issues": "All detected issues across systems",
            "GET /ai-os/activity": "Recent AI OS activity"
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
    except:
        capabilities["agents"] = ["Unable to enumerate"]

    # Get bleeding edge
    try:
        from api.bleeding_edge import get_bleeding_edge_status
        be_status = await get_bleeding_edge_status()
        capabilities["bleeding_edge"] = be_status.get("capabilities", [])
    except:
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
    except:
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
    except:
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
