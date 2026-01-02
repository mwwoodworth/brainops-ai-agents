"""
DevOps Loop API Router
======================
Exposes the ultimate self-healing DevOps loop via API.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devops-loop", tags=["DevOps Loop"])

# Lazy initialization
_loop = None
_continuous_task = None


def _get_loop():
    """Lazy load the DevOps loop"""
    global _loop
    if _loop is None:
        try:
            from devops_loop import get_devops_loop
            _loop = get_devops_loop()
        except Exception as e:
            logger.error(f"Failed to initialize DevOps loop: {e}")
            raise HTTPException(status_code=503, detail="DevOps loop not available")
    return _loop


@router.get("/status")
async def get_devops_status():
    """Get DevOps loop status"""
    loop = _get_loop()
    return {
        "system": "devops_loop",
        "status": "operational",
        "cycle_count": loop.cycle_count,
        "is_running": loop.is_running,
        "capabilities": [
            "continuous_monitoring",
            "ui_testing",
            "backend_health_checks",
            "auto_remediation",
            "consciousness_integration",
            "ooda_decision_making",
            "pattern_learning"
        ],
        "monitored_systems": {
            "backends": ["brainops-ai-agents", "brainops-backend", "mcp-bridge"],
            "frontends": ["myroofgenius", "weathercraft-erp"]
        }
    }


@router.post("/run-cycle")
async def run_devops_cycle():
    """Run a single DevOps OODA cycle"""
    try:
        from devops_loop import run_devops_cycle as run_cycle
        result = await run_cycle()
        return result
    except Exception as e:
        logger.error(f"DevOps cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_continuous(background_tasks: BackgroundTasks, interval_seconds: int = 60):
    """Start continuous DevOps monitoring"""
    global _continuous_task

    if _continuous_task and not _continuous_task.done():
        return {"status": "already_running", "cycle_count": _get_loop().cycle_count}

    try:
        from devops_loop import start_continuous_devops

        async def run_continuous():
            await start_continuous_devops(interval_seconds)

        _continuous_task = asyncio.create_task(run_continuous())

        return {
            "status": "started",
            "interval_seconds": interval_seconds,
            "message": "Continuous DevOps monitoring started"
        }
    except Exception as e:
        logger.error(f"Failed to start continuous DevOps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_continuous():
    """Stop continuous DevOps monitoring"""
    global _continuous_task

    loop = _get_loop()
    loop.is_running = False

    if _continuous_task:
        _continuous_task.cancel()
        _continuous_task = None

    return {"status": "stopped", "final_cycle_count": loop.cycle_count}


@router.get("/health-summary")
async def get_health_summary():
    """Get current health summary of all monitored systems"""
    try:
        from devops_loop import run_devops_cycle as run_cycle
        result = await run_cycle()
        return {
            "timestamp": result.get("completed"),
            "health_summary": result.get("health_summary"),
            "observations": {
                "backends": {
                    k: {"health": v.get("health"), "latency_ms": v.get("latency_ms")}
                    for k, v in result.get("observations", {}).get("backends", {}).items()
                },
                "frontends": {
                    k: {"health": v.get("health"), "latency_ms": v.get("latency_ms")}
                    for k, v in result.get("observations", {}).get("frontends", {}).items()
                }
            },
            "anomalies_count": len(result.get("anomalies", [])),
            "actions_taken": len(result.get("actions_taken", []))
        }
    except Exception as e:
        logger.error(f"Health summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_current_anomalies():
    """Get current anomalies from last cycle"""
    try:
        from devops_loop import run_devops_cycle as run_cycle
        result = await run_cycle()
        return {
            "timestamp": result.get("completed"),
            "anomalies": result.get("anomalies", []),
            "total": len(result.get("anomalies", []))
        }
    except Exception as e:
        logger.error(f"Anomalies check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comprehensive-e2e")
async def run_comprehensive_e2e(app_name: Optional[str] = None):
    """Run COMPREHENSIVE e2e tests - not basic, COMPLETE coverage"""
    try:
        from comprehensive_e2e_tests import run_comprehensive_e2e as run_e2e
        result = await run_e2e(app_name)
        return result
    except Exception as e:
        logger.error(f"Comprehensive e2e failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comprehensive-e2e/status")
async def get_e2e_status():
    """Get comprehensive e2e testing capabilities"""
    return {
        "available": True,
        "applications": ["myroofgenius", "weathercraft-erp", "command-center", "brainstack-studio"],
        "test_categories": [
            "page_load", "navigation", "auth", "cta",
            "layout", "content", "performance", "responsive"
        ],
        "tests_per_app": {
            "myroofgenius": 14,
            "weathercraft-erp": 6,
            "command-center": 4,
            "brainstack-studio": 5
        },
        "total_tests": 29,
        "description": "Comprehensive e2e tests covering ALL UI elements across ALL 4 frontends"
    }
