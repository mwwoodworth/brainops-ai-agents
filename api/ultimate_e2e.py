"""
Ultimate E2E API Router
========================
Exposes the COMPLETE e2e awareness system via API.
Build logs, database state, UI tests, issue detection - EVERYTHING.
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ultimate-e2e", tags=["Ultimate E2E System"])

# Lazy singleton
_system = None
_monitoring_task = None


async def _get_system():
    """Lazy load the Ultimate E2E System"""
    global _system
    if _system is None:
        try:
            from ultimate_e2e_system import UltimateE2ESystem
            _system = UltimateE2ESystem()
            await _system.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize Ultimate E2E System: {e}")
            raise HTTPException(status_code=503, detail=f"System not available: {str(e)}") from e
    return _system


@router.get("/status")
async def get_comprehensive_status():
    """
    Get COMPLETE system status - everything we know.
    Includes: builds, database, services, frontends, issues.
    """
    system = await _get_system()
    return await system.get_comprehensive_status()


@router.get("/builds")
async def monitor_builds():
    """
    Monitor Render build logs for all services.
    Detects build/deploy failures and analyzes with AI.
    """
    system = await _get_system()
    return await system.monitor_all_builds()


@router.get("/builds/{service_name}")
async def get_service_builds(service_name: str):
    """Get build history for a specific service"""
    system = await _get_system()

    if service_name not in ["ai-agents", "backend", "mcp-bridge"]:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}")

    deploys = await system.get_render_deploys(service_name, limit=10)
    return {
        "service": service_name,
        "deploys": deploys,
        "count": len(deploys)
    }


@router.get("/builds/{service_name}/{deploy_id}/logs")
async def get_deploy_logs(service_name: str, deploy_id: str):
    """Get logs for a specific deploy"""
    system = await _get_system()

    if service_name not in ["ai-agents", "backend", "mcp-bridge"]:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}")

    logs = await system.get_deploy_logs(service_name, deploy_id)
    return {
        "service": service_name,
        "deploy_id": deploy_id,
        "logs": logs,
        "log_count": len(logs)
    }


@router.get("/database")
async def get_database_state():
    """
    Get comprehensive database state.
    Includes: table counts, recent activity, health checks.
    """
    system = await _get_system()
    return await system.get_database_state()


# SECURITY: /database/query endpoint DISABLED - Raw SQL execution is a security risk
# @router.post("/database/query")
# async def query_database(query: str):
#     """Execute a read-only database query - DISABLED FOR SECURITY."""
#     pass
# Use specific, parameterized endpoints instead of raw SQL access


@router.post("/ui-tests")
async def run_ui_tests(background_tasks: BackgroundTasks, run_in_background: bool = False):
    """
    Run comprehensive UI tests on all frontends.
    Tests: page loads, navigation, auth flows, forms, performance.
    """
    system = await _get_system()

    if run_in_background:
        background_tasks.add_task(system.run_comprehensive_ui_tests)
        return {"status": "started", "message": "UI tests running in background"}

    return await system.run_comprehensive_ui_tests()


@router.post("/chatgpt-tests")
async def run_chatgpt_tests(background_tasks: BackgroundTasks, run_in_background: bool = False):
    """
    Run human-like ChatGPT agent tests.
    Performs actual logins, form fills, and complex user flows.
    """
    system = await _get_system()

    if run_in_background:
        background_tasks.add_task(system.run_chatgpt_agent_tests)
        return {"status": "started", "message": "ChatGPT agent tests running in background"}

    return await system.run_chatgpt_agent_tests()


@router.get("/issues")
async def get_issues(include_resolved: bool = False):
    """
    Get all detected system issues.
    Issues are auto-detected from build failures, service outages, UI test failures, etc.
    """
    system = await _get_system()
    issues = system.get_all_issues(include_resolved=include_resolved)
    return {
        "issues": issues,
        "total": len(issues),
        "critical": len([i for i in issues if i.get("severity") == "critical"]),
        "unresolved": len([i for i in issues if not i.get("resolved")])
    }


@router.post("/issues/{issue_id}/resolve")
async def resolve_issue(issue_id: str, resolution_note: str = ""):
    """Mark an issue as resolved"""
    system = await _get_system()
    success = await system.resolve_issue(issue_id, resolution_note)

    if not success:
        raise HTTPException(status_code=404, detail=f"Issue not found: {issue_id}")

    return {"success": True, "issue_id": issue_id, "status": "resolved"}


@router.post("/start-monitoring")
async def start_continuous_monitoring(background_tasks: BackgroundTasks, interval_seconds: int = 60):
    """
    Start 24/7 continuous monitoring.
    Monitors: builds, services, database, issues.
    Runs UI tests hourly.
    """
    global _monitoring_task

    system = await _get_system()

    if _monitoring_task and not _monitoring_task.done():
        return {"status": "already_running", "message": "Continuous monitoring is already active"}

    async def run_monitoring():
        await system.start_continuous_monitoring(interval_seconds)

    _monitoring_task = asyncio.create_task(run_monitoring())

    return {
        "status": "started",
        "interval_seconds": interval_seconds,
        "message": "24/7 continuous monitoring activated"
    }


@router.post("/stop-monitoring")
async def stop_monitoring():
    """Stop continuous monitoring"""
    global _monitoring_task

    system = await _get_system()
    system.stop_monitoring()

    if _monitoring_task:
        _monitoring_task.cancel()
        _monitoring_task = None

    return {"status": "stopped", "message": "Continuous monitoring stopped"}


@router.get("/monitoring-status")
async def get_monitoring_status():
    """Get current monitoring status"""
    global _monitoring_task

    is_running = _monitoring_task is not None and not _monitoring_task.done()

    return {
        "monitoring_active": is_running,
        "status": "running" if is_running else "stopped"
    }


@router.get("/health")
async def health_check():
    """Quick health check for the Ultimate E2E System"""
    try:
        system = await _get_system()
        status = await system.get_comprehensive_status()
        return {
            "healthy": status.get("health_score", 0) >= 70,
            "health_score": status.get("health_score", 0),
            "health_status": status.get("health_status", "unknown"),
            "issue_count": status.get("issue_count", {}).get("unresolved", 0)
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}
