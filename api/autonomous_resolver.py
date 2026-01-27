"""
AUTONOMOUS ISSUE RESOLVER API
=============================

API endpoints for the autonomous issue resolution system.

Endpoints:
- GET  /resolver/issues      - Get current system issues
- POST /resolver/fix         - Run a resolution cycle
- POST /resolver/fix-all     - Run resolution and return detailed results
- GET  /resolver/stats       - Get resolution statistics
- POST /resolver/start       - Start continuous resolution
- POST /resolver/stop        - Stop continuous resolution

Created: 2026-01-27
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resolver", tags=["autonomous-resolver"])

# Lazy load resolver
_resolver = None
_background_task = None


def _get_resolver():
    """Lazy load the resolver"""
    global _resolver
    if _resolver is None:
        try:
            from autonomous_issue_resolver import get_resolver
            _resolver = get_resolver()
        except Exception as e:
            logger.error(f"Failed to load resolver: {e}")
            raise HTTPException(status_code=503, detail=f"Resolver not available: {e}")
    return _resolver


class ResolutionResponse(BaseModel):
    success: bool
    total_fixed: int
    message: str
    details: dict[str, Any] = {}


@router.get("/")
async def resolver_root():
    """Autonomous Issue Resolver root"""
    return {
        "service": "Autonomous Issue Resolver",
        "description": "Detects and FIXES AI OS issues automatically",
        "capabilities": [
            "Fix stuck agents (auto-cancel after threshold)",
            "Resolve memory conflicts (merge/dedupe)",
            "Verify unverified memories (confidence-based)",
            "Apply unapplied insights (low-risk auto-apply)",
            "Process pending proposals (auto-approve safe ones)"
        ],
        "endpoints": {
            "GET /resolver/issues": "Get current system issues",
            "POST /resolver/fix": "Run one resolution cycle",
            "POST /resolver/fix-all": "Run full resolution with details",
            "GET /resolver/stats": "Get resolution statistics",
            "POST /resolver/start": "Start continuous resolution",
            "POST /resolver/stop": "Stop continuous resolution"
        }
    }


@router.get("/issues")
async def get_current_issues():
    """
    GET CURRENT SYSTEM ISSUES

    Returns all detected issues that need resolution:
    - Stuck agents (running too long)
    - Failed agents (in last hour)
    - Memory conflicts (unresolved)
    - Unverified memories
    - Unapplied insights
    - Pending proposals
    """
    try:
        resolver = _get_resolver()
        issues = await resolver.get_current_issues()
        return {
            "success": True,
            "timestamp": issues.get("timestamp"),
            "issues": issues,
            "needs_attention": issues["total_issues"] > 0,
            "summary": {
                "stuck_agents": len(issues["stuck_agents"]),
                "failed_agents": len(issues["failed_agents"]),
                "memory_conflicts": issues["memory_conflicts"],
                "unverified_memories": issues["unverified_memories"],
                "unapplied_insights": issues["unapplied_insights"],
                "pending_proposals": issues["pending_proposals"],
                "total": issues["total_issues"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix")
async def run_resolution_cycle():
    """
    RUN RESOLUTION CYCLE

    Executes one full resolution cycle that:
    1. Cancels stuck agents
    2. Resolves memory conflicts
    3. Verifies/degrades memories
    4. Applies insights
    5. Processes proposals

    Returns summary of fixes applied.
    """
    try:
        resolver = _get_resolver()
        result = await resolver.run_full_resolution_cycle()

        return {
            "success": result["success"],
            "total_fixed": result["total_fixed"],
            "duration_seconds": result["duration_seconds"],
            "issues_reduced": result["issues_reduced"],
            "message": f"Fixed {result['total_fixed']} issues in {result['duration_seconds']}s",
            "resolutions": result["resolutions"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolution cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix-all")
async def run_full_resolution():
    """
    RUN FULL RESOLUTION WITH DETAILS

    Like /fix but returns complete before/after comparison.
    """
    try:
        resolver = _get_resolver()
        result = await resolver.run_full_resolution_cycle()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full resolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_resolution_stats():
    """
    GET RESOLUTION STATISTICS

    Returns statistics about resolution cycles:
    - Total cycles run
    - Total issues fixed
    - Average cycle duration
    - Last cycle details
    """
    try:
        resolver = _get_resolver()
        stats = resolver.get_resolution_stats()
        return {
            "success": True,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_continuous_resolution(background_tasks: BackgroundTasks):
    """
    START CONTINUOUS RESOLUTION

    Starts the autonomous resolver in the background.
    It will continuously monitor and fix issues.
    """
    global _background_task

    try:
        resolver = _get_resolver()

        if resolver._running:
            return {
                "success": False,
                "message": "Resolver is already running",
                "status": "running"
            }

        # Start in background
        background_tasks.add_task(resolver.start_continuous_resolution)

        return {
            "success": True,
            "message": "Autonomous resolution started",
            "status": "starting",
            "interval_seconds": resolver.resolve_interval_seconds
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start resolver: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_continuous_resolution():
    """
    STOP CONTINUOUS RESOLUTION

    Stops the autonomous resolver background task.
    """
    try:
        resolver = _get_resolver()

        if not resolver._running:
            return {
                "success": False,
                "message": "Resolver is not running",
                "status": "stopped"
            }

        resolver.stop()

        return {
            "success": True,
            "message": "Autonomous resolution stopped",
            "status": "stopped"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop resolver: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def resolver_health():
    """Health check for the resolver"""
    try:
        resolver = _get_resolver()
        stats = resolver.get_resolution_stats()

        return {
            "status": "healthy",
            "running": resolver._running,
            "total_cycles": stats.get("total_cycles", 0),
            "total_fixed": stats.get("total_fixed", 0)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Quick fix endpoints for specific issue types

@router.post("/fix/stuck-agents")
async def fix_stuck_agents():
    """Fix only stuck agents"""
    try:
        resolver = _get_resolver()
        result = await resolver.fix_stuck_agents()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix/memory-conflicts")
async def fix_memory_conflicts():
    """Fix only memory conflicts"""
    try:
        resolver = _get_resolver()
        result = await resolver.resolve_memory_conflicts()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix/stale-memories")
async def fix_stale_memories():
    """Cleanup stale memories"""
    try:
        resolver = _get_resolver()
        result = await resolver.cleanup_stale_memories()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix/insights")
async def fix_unapplied_insights():
    """Apply unapplied insights"""
    try:
        resolver = _get_resolver()
        result = await resolver.apply_insights()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix/proposals")
async def fix_pending_proposals():
    """Process pending proposals"""
    try:
        resolver = _get_resolver()
        result = await resolver.process_proposals()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
