"""
Memory Hygiene API Router
===============================
API endpoints for memory hygiene operations and scheduled maintenance.

Part of BrainOps OS Total Completion Protocol.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from memory_hygiene import MemoryHygieneSystem, get_hygiene_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hygiene", tags=["Memory Hygiene"])


# =============================================================================
# HYGIENE EXECUTION ENDPOINTS
# =============================================================================

@router.post("/run")
async def run_full_hygiene() -> dict[str, Any]:
    """
    Run the full memory hygiene cycle.

    This includes:
    - Degrade stale verifications
    - Detect conflicts
    - Deduplicate memories
    - Apply confidence decay
    - Mark superseded items
    - Create re-verification tasks
    - Clean expired memories
    """
    try:
        hygiene = get_hygiene_system()
        report = await hygiene.run_full_hygiene()

        return {
            "success": True,
            "report": report
        }

    except Exception as e:
        logger.error(f"Hygiene run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/degrade-stale")
async def degrade_stale_verifications() -> dict[str, Any]:
    """
    Mark verified memories as DEGRADED if past expiration.
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.degrade_stale_verifications()

        return {
            "success": True,
            "degraded_count": count
        }

    except Exception as e:
        logger.error(f"Failed to degrade stale verifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-conflicts")
async def detect_conflicts() -> dict[str, Any]:
    """
    Detect and record memory conflicts (duplicates, contradictions).
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.detect_conflicts()

        return {
            "success": True,
            "new_conflicts_detected": count
        }

    except Exception as e:
        logger.error(f"Failed to detect conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deduplicate")
async def deduplicate_memories() -> dict[str, Any]:
    """
    Find and merge near-duplicate memories.
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.deduplicate_memories()

        return {
            "success": True,
            "duplicates_merged": count
        }

    except Exception as e:
        logger.error(f"Failed to deduplicate memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-decay")
async def apply_confidence_decay() -> dict[str, Any]:
    """
    Apply time-based confidence decay to unverified memories.
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.apply_confidence_decay()

        return {
            "success": True,
            "confidence_decayed": count
        }

    except Exception as e:
        logger.error(f"Failed to apply confidence decay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-superseded")
async def mark_superseded_items() -> dict[str, Any]:
    """
    Mark items as superseded based on supersedes relationships.
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.mark_superseded_items()

        return {
            "success": True,
            "items_superseded": count
        }

    except Exception as e:
        logger.error(f"Failed to mark superseded items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-reverification-tasks")
async def create_reverification_tasks() -> dict[str, Any]:
    """
    Create tasks for memories needing re-verification.
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.create_reverification_tasks()

        return {
            "success": True,
            "tasks_created": count
        }

    except Exception as e:
        logger.error(f"Failed to create re-verification tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean-expired")
async def clean_expired_memories() -> dict[str, Any]:
    """
    Remove (soft-delete) expired memories.
    """
    try:
        hygiene = get_hygiene_system()
        count = await hygiene.clean_expired_memories()

        return {
            "success": True,
            "expired_cleaned": count
        }

    except Exception as e:
        logger.error(f"Failed to clean expired memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BACKLOG AND STATS ENDPOINTS
# =============================================================================

@router.get("/backlog-stats")
async def get_backlog_stats() -> dict[str, Any]:
    """
    Get statistics about the truth backlog.
    """
    try:
        hygiene = get_hygiene_system()
        stats = await hygiene.get_backlog_stats()

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Failed to get backlog stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_hygiene_health() -> dict[str, Any]:
    """
    Get the overall health status of the memory hygiene system.
    """
    try:
        hygiene = get_hygiene_system()
        await hygiene.initialize()

        # Get memory health metrics
        health = await hygiene.pool.fetchrow("""
            SELECT
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE verification_state = 'VERIFIED') as verified,
                COUNT(*) FILTER (WHERE verification_state = 'UNVERIFIED') as unverified,
                COUNT(*) FILTER (WHERE verification_state = 'DEGRADED') as degraded,
                COUNT(*) FILTER (WHERE verification_state = 'BROKEN') as broken,
                COUNT(*) FILTER (WHERE expires_at < NOW()) as expired,
                COUNT(*) FILTER (WHERE verification_expires_at < NOW()) as stale_verification,
                COUNT(*) FILTER (WHERE confidence_score < 0.3) as low_confidence,
                AVG(confidence_score) as avg_confidence
            FROM unified_ai_memory
        """)

        # Get conflict count
        conflict_count = await hygiene.pool.fetchval("""
            SELECT COUNT(*) FROM memory_conflicts WHERE resolution_status = 'open'
        """)

        # Calculate health score (0-100)
        total = health["total_memories"] or 1
        verified_pct = (health["verified"] or 0) / total
        degraded_pct = (health["degraded"] or 0) / total
        broken_pct = (health["broken"] or 0) / total
        low_confidence_pct = (health["low_confidence"] or 0) / total

        health_score = max(0, min(100, int(
            100 * verified_pct
            - 20 * degraded_pct
            - 50 * broken_pct
            - 10 * low_confidence_pct
            - min(10, (conflict_count or 0) * 2)
        )))

        # Determine status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        elif health_score >= 40:
            status = "degraded"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "metrics": {
                "total_memories": health["total_memories"] or 0,
                "verified": health["verified"] or 0,
                "unverified": health["unverified"] or 0,
                "degraded": health["degraded"] or 0,
                "broken": health["broken"] or 0,
                "expired": health["expired"] or 0,
                "stale_verification": health["stale_verification"] or 0,
                "low_confidence": health["low_confidence"] or 0,
                "avg_confidence": round(float(health["avg_confidence"] or 0), 3),
                "open_conflicts": conflict_count or 0
            },
            "recommendations": _get_health_recommendations(health, conflict_count)
        }

    except Exception as e:
        logger.error(f"Failed to get hygiene health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_health_recommendations(health: dict, conflict_count: int) -> list[str]:
    """Generate actionable recommendations based on health metrics."""
    recommendations = []

    total = health["total_memories"] or 1

    if (health["unverified"] or 0) / total > 0.5:
        recommendations.append("High unverified rate - schedule bulk verification job")

    if (health["degraded"] or 0) / total > 0.1:
        recommendations.append("Degraded memories need re-verification")

    if (health["broken"] or 0) > 100:
        recommendations.append("Review and cleanup broken memories")

    if health["stale_verification"] or 0 > 50:
        recommendations.append("Run stale verification degradation job")

    if conflict_count and conflict_count > 10:
        recommendations.append(f"Resolve {conflict_count} open conflicts")

    if float(health["avg_confidence"] or 0) < 0.5:
        recommendations.append("Average confidence is low - consider verification campaign")

    if not recommendations:
        recommendations.append("Memory system is healthy - no immediate actions required")

    return recommendations


# =============================================================================
# SCHEDULED JOB STATUS
# =============================================================================

@router.get("/schedule")
async def get_hygiene_schedule() -> dict[str, Any]:
    """
    Get the status of scheduled hygiene jobs.
    """
    return {
        "scheduled_jobs": [
            {
                "name": "full_hygiene",
                "description": "Complete hygiene cycle",
                "schedule": "0 */6 * * *",  # Every 6 hours
                "enabled": True
            },
            {
                "name": "degrade_stale",
                "description": "Degrade stale verifications",
                "schedule": "0 * * * *",  # Every hour
                "enabled": True
            },
            {
                "name": "detect_conflicts",
                "description": "Detect memory conflicts",
                "schedule": "*/30 * * * *",  # Every 30 minutes
                "enabled": True
            },
            {
                "name": "apply_decay",
                "description": "Apply confidence decay",
                "schedule": "0 0 * * *",  # Daily at midnight
                "enabled": True
            }
        ],
        "note": "Jobs run automatically via agent_scheduler"
    }
