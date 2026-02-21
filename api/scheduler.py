"""
Scheduler & scheduling management endpoints.

Extracted from app.py during Phase 2 Wave 2B.
Covers email scheduler stats, agent scheduling, and scheduler diagnostics.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from database.async_connection import get_pool
from services.scheduler_queries import (
    fetch_email_queue_counts,
    fetch_active_agents,
    fetch_scheduled_agent_ids,
    insert_agent_schedule,
    upsert_agent_schedule,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scheduler"])


# ---------------------------------------------------------------------------
# Lazy accessors for app-level singletons
# ---------------------------------------------------------------------------
def _get_app():
    import app as _app

    return _app.app


def _scheduler_available():
    import app as _app

    return getattr(_app, "SCHEDULER_AVAILABLE", False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/email/scheduler-stats")
async def email_scheduler_stats():
    """Get email scheduler daemon statistics."""
    try:
        from email_scheduler_daemon import get_email_scheduler

        daemon = get_email_scheduler()
        stats = daemon.get_stats()

        queue_counts = await fetch_email_queue_counts()

        return {
            "daemon_stats": stats,
            "queue_counts": queue_counts,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except ImportError:
        return {"error": "email_scheduler_daemon module not available"}
    except Exception as e:
        logger.error(f"Failed to get email scheduler stats: {e}")
        return {"error": str(e)}


@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get detailed scheduler status and diagnostics (requires auth)"""
    try:
        app = _get_app()
        if (
            not _scheduler_available()
            or not hasattr(app.state, "scheduler")
            or not app.state.scheduler
        ):
            return {
                "enabled": False,
                "message": "Scheduler not available",
                "timestamp": datetime.utcnow().isoformat(),
            }

        scheduler = app.state.scheduler
        apscheduler_jobs = scheduler.scheduler.get_jobs()

        return {
            "enabled": True,
            "running": scheduler.scheduler.running,
            "state": scheduler.scheduler.state,
            "registered_jobs_count": len(scheduler.registered_jobs),
            "apscheduler_jobs_count": len(apscheduler_jobs),
            "registered_jobs": list(scheduler.registered_jobs.values()),
            "apscheduler_jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger),
                }
                for job in apscheduler_jobs
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "enabled": False,
                "error": str(e),
                "message": "Failed to retrieve scheduler status",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.post("/scheduler/restart-stuck")
async def restart_stuck_executions():
    """
    Cleanup stuck execution records across the AI OS.

    Why this exists:
    - Some execution logs can remain in `running` state (e.g., timeouts/cancellations).
    - This endpoint normalizes those stale rows to a failure state so dashboards/metrics stay honest.

    NOTE: The canonical resolver endpoint is `POST /resolver/fix/stuck-agents`.
    This route is kept as a stable alias for callers that reference the old path.
    """
    try:
        from autonomous_issue_resolver import get_resolver

        resolver = get_resolver()
        result = await resolver.fix_stuck_agents()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details,
            "timestamp": result.timestamp.isoformat(),
        }
    except Exception as e:
        logger.error("Failed to restart stuck executions: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/scheduler/activate-all")
async def activate_all_agents_scheduler():
    """
    Schedule ALL agents that don't have active schedules.
    This activates the full AI OS by ensuring every agent runs on a schedule.
    """
    app = _get_app()
    if not _scheduler_available() or not hasattr(app.state, "scheduler") or not app.state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    scheduler = app.state.scheduler
    pool = get_pool()

    try:
        agents_result = await fetch_active_agents(pool)
        existing_agent_ids = await fetch_scheduled_agent_ids(pool)

        scheduled_count = 0
        already_scheduled = 0
        errors = []

        for agent in agents_result:
            agent_id = str(agent["id"])
            agent_name = agent["name"]
            agent_type = agent.get("type", "general").lower()

            if agent_id in existing_agent_ids:
                already_scheduled += 1
                continue

            # Determine frequency based on agent type
            if agent_type in ["analytics", "revenue", "customer"]:
                frequency = 30  # High-value agents: every 30 min
            elif agent_type in ["monitor", "security"]:
                frequency = 15  # Critical agents: every 15 min
            elif agent_type in ["learning", "optimization"]:
                frequency = 60  # Learning agents: every hour
            else:
                frequency = 60  # Default: every hour

            try:
                await insert_agent_schedule(pool, agent_id, frequency)
                scheduler.add_schedule(agent_id, agent_name, frequency)
                scheduled_count += 1
                logger.info(f"Scheduled agent {agent_name} every {frequency} min")

            except Exception as e:
                errors.append(f"{agent_name}: {str(e)}")
                logger.error(f"Failed to schedule {agent_name}: {e}")

        return {
            "success": True,
            "message": f"Activated {scheduled_count} new agent schedules",
            "new_schedules": scheduled_count,
            "already_scheduled": already_scheduled,
            "total_agents": len(agents_result),
            "errors": errors if errors else None,
        }

    except Exception as e:
        logger.error(f"Failed to activate all agents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/agents/schedule")
async def schedule_agent(
    agent_id: str = Body(..., embed=True),
    frequency_minutes: int = Body(60, embed=True),
    enabled: bool = Body(True, embed=True),
    run_at: Optional[str] = Body(None, embed=True),
):
    """
    Schedule an agent for future execution.
    Can set recurring schedule (frequency_minutes) or one-time execution (run_at).
    """
    pool = get_pool()
    schedule_id = str(uuid.uuid4())

    try:
        result = await upsert_agent_schedule(
            pool, agent_id, frequency_minutes, enabled, schedule_id
        )

        if result is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        action = result["action"]
        schedule_id = result["schedule_id"]
        agent_name = result["agent_name"]

        # Add to runtime scheduler if available
        app = _get_app()
        if _scheduler_available() and hasattr(app.state, "scheduler") and app.state.scheduler:
            app.state.scheduler.add_schedule(agent_id, agent_name, frequency_minutes)

        return {
            "success": True,
            "action": action,
            "schedule_id": schedule_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "frequency_minutes": frequency_minutes,
            "enabled": enabled,
            "next_run": run_at or f"In {frequency_minutes} minutes",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule agent: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
