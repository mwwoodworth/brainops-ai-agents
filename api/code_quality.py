"""
CODE QUALITY API - Deep Code-Level Monitoring
=============================================

Endpoints for comprehensive code quality and integration monitoring.

Created: 2026-01-27
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/code-quality", tags=["code-quality"])

# Import code quality monitor
try:
    from code_quality_monitor import get_code_quality_monitor, run_full_scan
    CODE_QUALITY_AVAILABLE = True
except ImportError as e:
    CODE_QUALITY_AVAILABLE = False
    logger.warning(f"Code quality monitor not available: {e}")


@router.get("/")
async def root():
    """
    🔬 CODE QUALITY MONITOR ROOT

    Deep code-level monitoring beyond service health.
    """
    return {
        "service": "BrainOps Code Quality Monitor",
        "description": "Deep monitoring for code issues, integrations, and runtime errors",
        "available": CODE_QUALITY_AVAILABLE,
        "endpoints": {
            "GET /code-quality/scan": "Run full code quality scan",
            "GET /code-quality/issues": "Get all detected issues",
            "GET /code-quality/summary": "Get summary of code quality status",
            "GET /code-quality/integrations": "Check integration health",
            "GET /code-quality/deployments": "Check deployment status"
        }
    }


@router.get("/scan")
async def full_scan():
    """
    🔬 RUN FULL CODE QUALITY SCAN

    Comprehensive scan including:
    - Integration endpoint tests
    - Runtime error detection from logs
    - Deployment status check
    - Database health check
    """
    if not CODE_QUALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Code quality monitor not available")

    results = await run_full_scan()
    return results


@router.get("/issues")
async def get_issues(unresolved_only: bool = True):
    """
    📋 GET ALL DETECTED ISSUES

    Returns all issues found during scans.
    """
    if not CODE_QUALITY_AVAILABLE:
        return {"issues": [], "message": "Code quality monitor not available"}

    monitor = get_code_quality_monitor()
    issues = monitor.get_all_issues(unresolved_only)

    return {
        "total": len(issues),
        "unresolved_only": unresolved_only,
        "issues": issues
    }


@router.get("/summary")
async def get_summary():
    """
    📊 GET CODE QUALITY SUMMARY

    High-level summary of code quality status.
    """
    if not CODE_QUALITY_AVAILABLE:
        return {"message": "Code quality monitor not available"}

    monitor = get_code_quality_monitor()
    return monitor.get_summary()


@router.get("/integrations")
async def check_integrations():
    """
    🔗 CHECK INTEGRATION HEALTH

    Test all integration endpoints.
    """
    if not CODE_QUALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Code quality monitor not available")

    monitor = get_code_quality_monitor()
    return await monitor.check_integrations()


@router.get("/deployments")
async def check_deployments():
    """
    🚀 CHECK DEPLOYMENT STATUS

    Get recent deployment status for all services.
    """
    if not CODE_QUALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Code quality monitor not available")

    monitor = get_code_quality_monitor()
    return await monitor.check_deployment_status()


@router.get("/runtime-errors")
async def check_runtime_errors():
    """
    ⚠️ CHECK RUNTIME ERRORS

    Scan logs for runtime errors.
    """
    if not CODE_QUALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Code quality monitor not available")

    monitor = get_code_quality_monitor()
    return await monitor.check_runtime_errors()


@router.get("/database")
async def check_database():
    """
    🗄️ CHECK DATABASE HEALTH

    Verify database connectivity and health.
    """
    if not CODE_QUALITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Code quality monitor not available")

    monitor = get_code_quality_monitor()
    return await monitor.check_database_health()
