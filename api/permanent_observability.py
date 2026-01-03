"""
PERMANENT OBSERVABILITY API
Endpoints for accessing live system observability data.
Never miss anything - all events captured and queryable.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    from fastapi import HTTPException
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/visibility",
    tags=["permanent-observability"],
    dependencies=[Depends(verify_api_key)]
)


class EventFilter(BaseModel):
    """Filter for querying events"""
    service: Optional[str] = None
    severity: Optional[str] = None
    event_type: Optional[str] = None
    limit: int = 100


@router.get("/status")
async def get_visibility_status():
    """
    Get current visibility daemon status.
    Shows if permanent observability is running and healthy.
    """
    try:
        from permanent_observability_daemon import get_observability_daemon

        daemon = get_observability_daemon()
        stats = daemon.get_stats()

        return {
            "status": "running" if stats["running"] else "stopped",
            "daemon": stats,
            "message": "Permanent observability is active" if stats["running"] else "Daemon not started"
        }
    except Exception as e:
        logger.error(f"Failed to get visibility status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/health-summary")
async def get_health_summary():
    """
    Get real-time health summary of all monitored services.
    This is the primary endpoint for system-wide health visibility.
    """
    try:
        from permanent_observability_daemon import get_system_health_summary

        summary = await get_system_health_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        return {
            "overall": "unknown",
            "error": str(e)
        }


@router.get("/events")
async def get_recent_events(
    limit: int = Query(100, ge=1, le=1000),
    service: Optional[str] = Query(None),
    severity: Optional[str] = Query(None)
):
    """
    Get recent observability events.
    Events are permanently stored - nothing is lost.
    """
    try:
        from permanent_observability_daemon import get_observability_daemon, AlertSeverity

        daemon = get_observability_daemon()

        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
            except ValueError:
                pass

        events = await daemon.get_recent_events(
            limit=limit,
            service=service,
            severity=severity_enum
        )

        return {
            "count": len(events),
            "events": events
        }
    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        return {
            "count": 0,
            "events": [],
            "error": str(e)
        }


@router.get("/timeline/{service}")
async def get_service_timeline(
    service: str,
    hours: int = Query(24, ge=1, le=168)
):
    """
    Get event timeline for a specific service.
    Perfect for debugging and understanding service history.
    """
    try:
        from permanent_observability_daemon import get_observability_daemon

        daemon = get_observability_daemon()
        timeline = await daemon.get_service_timeline(service, hours)

        return {
            "service": service,
            "hours": hours,
            "event_count": len(timeline),
            "timeline": timeline
        }
    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        return {
            "service": service,
            "event_count": 0,
            "timeline": [],
            "error": str(e)
        }


@router.get("/alerts")
async def get_active_alerts(
    hours: int = Query(24, ge=1, le=168)
):
    """
    Get all alerts from the past N hours.
    Critical alerts are highlighted.
    """
    try:
        from permanent_observability_daemon import get_observability_daemon

        daemon = get_observability_daemon()

        # Get alert events
        all_events = await daemon.get_recent_events(limit=500)

        # Filter to alerts only
        alerts = [
            e for e in all_events
            if e["event_type"] == "alert"
        ]

        # Group by severity
        by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }

        for alert in alerts:
            sev = alert.get("severity", "info")
            if sev in by_severity:
                by_severity[sev].append(alert)

        return {
            "total_alerts": len(alerts),
            "by_severity": {
                k: len(v) for k, v in by_severity.items()
            },
            "critical": by_severity["critical"][:10],
            "high": by_severity["high"][:10],
            "all_alerts": alerts[:50]
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return {
            "total_alerts": 0,
            "error": str(e)
        }


@router.post("/capture")
async def capture_custom_event(
    event_type: str = Query(..., description="Event type"),
    service: str = Query(..., description="Service name"),
    message: str = Query(..., description="Event message"),
    severity: str = Query("info", description="Severity: critical, high, medium, low, info"),
    details: Optional[dict] = None
):
    """
    Capture a custom observability event.
    Use this to log important events from external systems.
    """
    try:
        from permanent_observability_daemon import capture_custom_event as do_capture

        await do_capture(
            event_type=event_type,
            service=service,
            message=message,
            severity=severity,
            details=details or {}
        )

        return {
            "status": "captured",
            "event_type": event_type,
            "service": service,
            "message": message
        }
    except Exception as e:
        logger.error(f"Failed to capture event: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/dashboard")
async def get_visibility_dashboard():
    """
    Complete visibility dashboard - everything in one endpoint.
    Perfect for monitoring tools and dashboards.
    """
    try:
        from permanent_observability_daemon import (
            get_observability_daemon,
            get_system_health_summary
        )

        daemon = get_observability_daemon()
        stats = daemon.get_stats()
        health = await get_system_health_summary()
        recent_events = await daemon.get_recent_events(limit=20)

        # Get recent alerts
        alerts = [e for e in recent_events if e["event_type"] == "alert"]

        return {
            "daemon_status": "running" if stats["running"] else "stopped",
            "overall_health": health["overall"],
            "services": health["services"],
            "health_counts": health["counts"],
            "daemon_stats": {
                "total_checks": stats["total_checks"],
                "total_events": stats["total_events"],
                "alerts_generated": stats["alerts_generated"],
                "events_persisted": stats["events_persisted"],
                "last_check": stats["last_check"]
            },
            "recent_alerts": alerts,
            "recent_events": recent_events[:10],
            "consecutive_failures": stats["consecutive_failures"],
            "timestamp": health["timestamp"]
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard: {e}")
        return {
            "daemon_status": "error",
            "error": str(e)
        }
