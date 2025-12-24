"""
E2E Verification API Router
============================
API endpoints for comprehensive end-to-end system verification.
Ensures 100% operational status across ALL BrainOps systems.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/e2e", tags=["E2E Verification"])

# Background verification task
_background_verification_running = False
_last_background_result = None


@router.get("/verify")
async def run_verification(
    quick: bool = Query(False, description="Run quick health check only (critical endpoints)")
):
    """
    Run E2E verification of ALL systems.

    Returns whether the system is 100% operational.
    Any failure in critical endpoints means the system is NOT 100% operational.

    Use ?quick=true for a faster check of critical endpoints only.
    """
    try:
        from e2e_system_verification import run_full_e2e_verification, run_quick_health_check

        if quick:
            result = await run_quick_health_check()
            return {
                "verification_type": "quick",
                "is_100_percent_operational": result["is_healthy"],
                **result
            }
        else:
            result = await run_full_e2e_verification()
            return {
                "verification_type": "full",
                **result
            }
    except Exception as e:
        logger.error(f"E2E verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/status")
async def get_verification_status():
    """Get the status of the E2E verification system"""
    try:
        from e2e_system_verification import e2e_verification

        return {
            "system": "e2e_verification",
            "status": "operational",
            "total_tests_configured": len(e2e_verification.tests),
            "categories": {
                "core_api": sum(1 for t in e2e_verification.tests if t.category.value == "core_api"),
                "bleeding_edge": sum(1 for t in e2e_verification.tests if t.category.value == "bleeding_edge"),
                "frontend": sum(1 for t in e2e_verification.tests if t.category.value == "frontend"),
                "mcp": sum(1 for t in e2e_verification.tests if t.category.value == "mcp"),
            },
            "critical_tests": sum(1 for t in e2e_verification.tests if t.critical),
            "last_verification": e2e_verification.get_report_summary(),
            "capabilities": [
                "full_e2e_verification",
                "quick_health_check",
                "category_breakdown",
                "actionable_recommendations",
                "100_percent_operational_check"
            ]
        }
    except Exception as e:
        logger.error(f"Could not get verification status: {e}")
        return {
            "system": "e2e_verification",
            "status": "error",
            "error": str(e)
        }


@router.get("/last-report")
async def get_last_report():
    """Get the last verification report"""
    try:
        from e2e_system_verification import get_last_verification_report

        report = await get_last_verification_report()
        if not report:
            return {
                "message": "No verification report available. Run /e2e/verify first.",
                "has_report": False
            }

        return {
            "has_report": True,
            **report
        }
    except Exception as e:
        logger.error(f"Could not get last report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-matrix")
async def get_health_matrix():
    """
    Get a comprehensive health matrix of all systems.
    This is a quick reference view without running full verification.
    """
    try:
        from e2e_system_verification import e2e_verification

        # Group tests by category
        matrix = {}
        for test in e2e_verification.tests:
            cat = test.category.value
            if cat not in matrix:
                matrix[cat] = {
                    "endpoints": [],
                    "critical_count": 0,
                    "total_count": 0
                }
            matrix[cat]["endpoints"].append({
                "name": test.name,
                "url": test.url,
                "critical": test.critical,
                "timeout": test.timeout_seconds
            })
            matrix[cat]["total_count"] += 1
            if test.critical:
                matrix[cat]["critical_count"] += 1

        return {
            "health_matrix": matrix,
            "total_endpoints": len(e2e_verification.tests),
            "total_critical": sum(1 for t in e2e_verification.tests if t.critical),
            "message": "Use /e2e/verify to run actual verification"
        }
    except Exception as e:
        logger.error(f"Could not get health matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify/background")
async def start_background_verification(background_tasks: BackgroundTasks):
    """Start E2E verification in the background"""
    global _background_verification_running, _last_background_result

    if _background_verification_running:
        return {
            "status": "already_running",
            "message": "Background verification is already in progress"
        }

    async def run_bg_verification():
        global _background_verification_running, _last_background_result
        try:
            from e2e_system_verification import run_full_e2e_verification
            _background_verification_running = True
            _last_background_result = await run_full_e2e_verification()
        finally:
            _background_verification_running = False

    # Schedule background task
    background_tasks.add_task(lambda: asyncio.run(run_bg_verification()))

    return {
        "status": "started",
        "message": "Background verification started. Check /e2e/verify/background/status for results."
    }


@router.get("/verify/background/status")
async def get_background_verification_status():
    """Get status of background verification"""
    global _background_verification_running, _last_background_result

    if _background_verification_running:
        return {
            "status": "running",
            "message": "Verification in progress..."
        }

    if _last_background_result:
        return {
            "status": "completed",
            "result": _last_background_result
        }

    return {
        "status": "not_started",
        "message": "No background verification has been run. Use POST /e2e/verify/background to start."
    }


@router.get("/systems")
async def list_verified_systems():
    """List all systems being verified"""
    try:
        from e2e_system_verification import e2e_verification

        systems = {}
        for test in e2e_verification.tests:
            # Extract base URL
            url_parts = test.url.split("/")
            base_url = "/".join(url_parts[:3])

            if base_url not in systems:
                systems[base_url] = {
                    "base_url": base_url,
                    "endpoints": [],
                    "critical_endpoints": 0
                }

            systems[base_url]["endpoints"].append(test.name)
            if test.critical:
                systems[base_url]["critical_endpoints"] += 1

        return {
            "systems": list(systems.values()),
            "total_systems": len(systems),
            "total_endpoints": len(e2e_verification.tests)
        }
    except Exception as e:
        logger.error(f"Could not list systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))
