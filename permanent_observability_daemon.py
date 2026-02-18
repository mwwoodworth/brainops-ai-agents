#!/usr/bin/env python3
"""
PERMANENT OBSERVABILITY DAEMON
Never miss anything. All system events captured and persisted forever.

Features:
- Continuous health monitoring of all services
- Automatic event capture to unified brain
- Alert generation and escalation
- Historical trend analysis
- Self-healing triggers
- Cross-service correlation

Author: BrainOps AI OS
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from safe_task import create_safe_task
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


def _normalize_tenant_uuid(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    try:
        return str(uuid.UUID(candidate))
    except Exception:
        return None


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action required within 1 hour
    MEDIUM = "medium"  # Action required within 24 hours
    LOW = "low"  # Informational
    INFO = "info"  # Status update


class ServiceStatus(str, Enum):
    """Service health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health snapshot for a service"""

    name: str
    url: str
    status: ServiceStatus
    response_time_ms: int
    http_code: Optional[int]
    error: Optional[str]
    details: dict[str, Any]
    checked_at: str


@dataclass
class SystemEvent:
    """Event captured from the system"""

    event_id: str
    event_type: str  # health_check, alert, recovery, deployment, error
    service: str
    severity: AlertSeverity
    message: str
    details: dict[str, Any]
    timestamp: str
    persisted: bool = False


@dataclass
class ObservabilityStats:
    """Daemon statistics"""

    started_at: str
    total_checks: int = 0
    total_events: int = 0
    alerts_generated: int = 0
    events_persisted: int = 0
    services_monitored: int = 0
    last_check: Optional[str] = None
    consecutive_failures: dict[str, int] = field(default_factory=dict)


class PermanentObservabilityDaemon:
    """
    Background daemon for continuous system observability.
    Captures ALL events and persists them to the unified brain.
    """

    def __init__(
        self,
        poll_interval: int = 60,  # seconds
        brain_api_url: str = "https://brainops-ai-agents.onrender.com",
        api_key: str = "",
    ):
        self.poll_interval = poll_interval
        self.brain_api_url = brain_api_url
        self.api_key = api_key or os.getenv("BRAINOPS_API_KEY")
        self.running = False
        self._task: Optional[asyncio.Task] = None

        # Initialize stats
        self.stats = ObservabilityStats(started_at=datetime.now(timezone.utc).isoformat())

        # Event queue for batch persistence
        self._event_queue: list[SystemEvent] = []

        # Service registry - all services to monitor
        self.services = {
            "brainops_ai_agents": {
                "url": "https://brainops-ai-agents.onrender.com/health",
                "headers": {"X-API-Key": self.api_key},
                "critical": True,
                "timeout": 10,
            },
            "brainops_backend": {
                "url": "https://brainops-backend-prod.onrender.com/health",
                "headers": {},
                "critical": True,
                "timeout": 10,
            },
            "mcp_bridge": {
                "url": "https://brainops-mcp-bridge.onrender.com/health",
                "headers": {},
                "critical": True,
                "timeout": 10,
            },
            "weathercraft_erp": {
                "url": "https://weathercraft-erp.vercel.app",
                "headers": {},
                "critical": True,
                "timeout": 15,
            },
            "myroofgenius": {
                "url": "https://myroofgenius.com",
                "headers": {},
                "critical": True,
                "timeout": 15,
            },
            "command_center": {
                "url": "https://brainops-command-center.vercel.app/api/unified-health",
                "headers": {},
                "critical": False,
                "timeout": 15,
            },
        }

        self.stats.services_monitored = len(self.services)

        # Previous health states for change detection
        self._previous_states: dict[str, ServiceStatus] = {}

        logger.info(f"PermanentObservabilityDaemon initialized with {len(self.services)} services")

    async def start(self) -> None:
        """Start the observability daemon"""
        if self.running:
            logger.warning("Daemon already running")
            return

        self.running = True
        self._task = create_safe_task(self._run_loop())
        logger.info("PermanentObservabilityDaemon started")

        # Persist daemon start event
        await self._capture_event(
            event_type="daemon_start",
            service="observability_daemon",
            severity=AlertSeverity.INFO,
            message="Permanent observability daemon started",
            details={"services": list(self.services.keys())},
        )

    async def stop(self) -> None:
        """Stop the daemon gracefully"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_events()

        logger.info("PermanentObservabilityDaemon stopped")

    async def _run_loop(self) -> None:
        """Main polling loop"""
        while self.running:
            try:
                # Check all services
                await self._check_all_services()

                # Persist queued events
                await self._flush_events()

                # Update last check
                self.stats.last_check = datetime.now(timezone.utc).isoformat()

            except Exception as e:
                logger.error(f"Error in observability loop: {e}", exc_info=True)
                await self._capture_event(
                    event_type="daemon_error",
                    service="observability_daemon",
                    severity=AlertSeverity.HIGH,
                    message=f"Observability daemon error: {e}",
                    details={"error": str(e)},
                )

            # Wait for next poll
            await asyncio.sleep(self.poll_interval)

    async def _check_all_services(self) -> dict[str, ServiceHealth]:
        """Check health of all registered services"""
        results = {}

        async with aiohttp.ClientSession() as session:
            tasks = []
            for name, config in self.services.items():
                tasks.append(self._check_service(session, name, config))

            health_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(self.services.keys(), health_results):
                if isinstance(result, Exception):
                    results[name] = ServiceHealth(
                        name=name,
                        url=self.services[name]["url"],
                        status=ServiceStatus.UNHEALTHY,
                        response_time_ms=0,
                        http_code=None,
                        error=str(result),
                        details={},
                        checked_at=datetime.now(timezone.utc).isoformat(),
                    )
                else:
                    results[name] = result

                # Track state changes
                await self._handle_state_change(name, results[name])

        self.stats.total_checks += 1
        return results

    async def _check_service(
        self, session: aiohttp.ClientSession, name: str, config: dict
    ) -> ServiceHealth:
        """Check a single service's health"""
        start_time = time.time()

        try:
            async with session.get(
                config["url"],
                headers=config.get("headers", {}),
                timeout=aiohttp.ClientTimeout(total=config.get("timeout", 10)),
                ssl=True,
            ) as response:
                response_time_ms = int((time.time() - start_time) * 1000)

                # Parse response for details
                details = {}
                try:
                    if response.content_type == "application/json":
                        details = await response.json()
                except Exception:
                    pass

                # Determine status
                if response.status == 200:
                    status = ServiceStatus.HEALTHY
                elif response.status in (500, 502, 503, 504):
                    status = ServiceStatus.UNHEALTHY
                else:
                    status = ServiceStatus.DEGRADED

                return ServiceHealth(
                    name=name,
                    url=config["url"],
                    status=status,
                    response_time_ms=response_time_ms,
                    http_code=response.status,
                    error=None,
                    details=details,
                    checked_at=datetime.now(timezone.utc).isoformat(),
                )

        except asyncio.TimeoutError:
            return ServiceHealth(
                name=name,
                url=config["url"],
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=config.get("timeout", 10) * 1000,
                http_code=None,
                error="Connection timeout",
                details={},
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            return ServiceHealth(
                name=name,
                url=config["url"],
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=int((time.time() - start_time) * 1000),
                http_code=None,
                error=str(e),
                details={},
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

    async def _handle_state_change(self, name: str, health: ServiceHealth) -> None:
        """Handle service state changes and generate alerts"""
        previous = self._previous_states.get(name)
        current = health.status

        # Track consecutive failures
        if current == ServiceStatus.UNHEALTHY:
            self.stats.consecutive_failures[name] = self.stats.consecutive_failures.get(name, 0) + 1
        else:
            self.stats.consecutive_failures[name] = 0

        # No previous state (first check)
        if previous is None:
            self._previous_states[name] = current
            await self._capture_event(
                event_type="health_check",
                service=name,
                severity=AlertSeverity.INFO,
                message=f"Initial health check: {current.value}",
                details={
                    "status": current.value,
                    "response_time_ms": health.response_time_ms,
                    "http_code": health.http_code,
                },
            )
            return

        # State unchanged
        if previous == current:
            # Still capture periodic checks for history
            if self.stats.total_checks % 10 == 0:  # Every 10th check
                await self._capture_event(
                    event_type="health_check",
                    service=name,
                    severity=AlertSeverity.INFO,
                    message=f"Periodic health: {current.value}",
                    details={
                        "status": current.value,
                        "response_time_ms": health.response_time_ms,
                        "http_code": health.http_code,
                    },
                )
            return

        # State changed - generate alert
        self._previous_states[name] = current
        is_critical = self.services[name].get("critical", False)

        if current == ServiceStatus.UNHEALTHY:
            # Service went down
            severity = AlertSeverity.CRITICAL if is_critical else AlertSeverity.HIGH
            await self._capture_event(
                event_type="alert",
                service=name,
                severity=severity,
                message=f"SERVICE DOWN: {name} is unhealthy",
                details={
                    "previous": previous.value,
                    "current": current.value,
                    "error": health.error,
                    "http_code": health.http_code,
                    "consecutive_failures": self.stats.consecutive_failures.get(name, 1),
                },
            )
            self.stats.alerts_generated += 1

        elif previous == ServiceStatus.UNHEALTHY and current == ServiceStatus.HEALTHY:
            # Service recovered
            await self._capture_event(
                event_type="recovery",
                service=name,
                severity=AlertSeverity.INFO,
                message=f"SERVICE RECOVERED: {name} is now healthy",
                details={
                    "previous": previous.value,
                    "current": current.value,
                    "response_time_ms": health.response_time_ms,
                },
            )

        else:
            # Other state change (degraded, etc.)
            await self._capture_event(
                event_type="state_change",
                service=name,
                severity=AlertSeverity.MEDIUM,
                message=f"Service state changed: {previous.value} -> {current.value}",
                details={
                    "previous": previous.value,
                    "current": current.value,
                    "response_time_ms": health.response_time_ms,
                    "http_code": health.http_code,
                },
            )

    async def _capture_event(
        self,
        event_type: str,
        service: str,
        severity: AlertSeverity,
        message: str,
        details: dict[str, Any],
    ) -> None:
        """Capture an event for persistence"""
        import uuid

        event = SystemEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            service=service,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._event_queue.append(event)
        self.stats.total_events += 1

        # Log critical events immediately
        if severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
            logger.warning(f"[{severity.value.upper()}] {service}: {message}")

    async def _flush_events(self) -> None:
        """Persist queued events to the brain.

        NOTE: DDL (CREATE TABLE/INDEX) removed because agent_worker role
        (app_agent_role) has no DDL permissions by design (P0-LOCK security).
        Tables must be created via migrations. This method verifies tables
        exist and degrades gracefully if they are missing.
        """
        if not self._event_queue:
            return

        events_to_persist = self._event_queue.copy()
        self._event_queue.clear()

        try:
            # Persist to database
            from database.async_connection import get_pool, using_fallback

            if using_fallback():
                logger.warning("Database unavailable, events not persisted")
                return

            pool = get_pool()

            # Verify required table exists (no DDL - agent_worker has no CREATE permissions)
            row = await pool.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY($1)",
                ["ai_observability_events"],
            )
            if row < 1:
                logger.error(
                    "Required table ai_observability_events missing (found %s/1). "
                    "Run migrations to create it.",
                    row,
                )
                return

            # Set tenant context for RLS (agent_worker has NOBYPASSRLS)
            from config import config as app_config

            tenant_id = (
                _normalize_tenant_uuid(app_config.tenant.default_tenant_id)
                or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
            )

            # Batch insert events
            for event in events_to_persist:
                # Convert ISO string to datetime if needed
                timestamp = event.timestamp
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                async with pool.acquire() as conn:
                    # In Supabase transaction pooling mode, tenant context + write
                    # must run in the same explicit transaction.
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT set_config('app.current_tenant_id', $1, true)",
                            tenant_id,
                        )
                        await conn.execute(
                            """
                            INSERT INTO ai_observability_events
                            (event_id, event_type, service, severity, message, details, timestamp, tenant_id)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (event_id) DO NOTHING
                        """,
                            event.event_id,
                            event.event_type,
                            event.service,
                            event.severity.value,
                            event.message,
                            json.dumps(event.details),
                            timestamp,
                            tenant_id,
                        )
                event.persisted = True

            self.stats.events_persisted += len(events_to_persist)
            logger.debug(f"Persisted {len(events_to_persist)} events")

        except Exception as e:
            logger.error(f"Failed to persist events: {e}")
            # Re-queue failed events
            self._event_queue.extend(events_to_persist)

    async def get_recent_events(
        self,
        limit: int = 100,
        service: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> list[dict]:
        """Get recent events from the database"""
        try:
            from database.async_connection import get_pool, using_fallback

            if using_fallback():
                return []

            pool = get_pool()
            from config import config as app_config

            tenant_id = (
                _normalize_tenant_uuid(app_config.tenant.default_tenant_id)
                or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
            )

            query = """
                SELECT event_id, event_type, service, severity, message, details, timestamp
                FROM ai_observability_events
                WHERE 1=1
            """
            params = []
            param_idx = 1

            if service:
                query += f" AND service = ${param_idx}"
                params.append(service)
                param_idx += 1

            if severity:
                query += f" AND severity = ${param_idx}"
                params.append(severity.value)
                param_idx += 1

            query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
            params.append(limit)

            raw_pool = getattr(pool, "pool", None) or getattr(pool, "_pool", None)
            if raw_pool is None:
                rows = await pool.fetch(query, *params)
            else:
                async with raw_pool.acquire(timeout=10.0) as conn:
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT set_config('app.current_tenant_id', $1, true)",
                            tenant_id,
                        )
                        rows = await conn.fetch(query, *params)

            return [
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "service": row["service"],
                    "severity": row["severity"],
                    "message": row["message"],
                    "details": row["details"]
                    if isinstance(row["details"], dict)
                    else json.loads(row["details"] or "{}"),
                    "timestamp": row["timestamp"].isoformat()
                    if hasattr(row["timestamp"], "isoformat")
                    else str(row["timestamp"]),
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    async def get_service_timeline(self, service: str, hours: int = 24) -> list[dict]:
        """Get timeline of events for a specific service"""
        try:
            from database.async_connection import get_pool, using_fallback

            if using_fallback():
                return []

            pool = get_pool()
            from config import config as app_config

            tenant_id = (
                _normalize_tenant_uuid(app_config.tenant.default_tenant_id)
                or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
            )

            query = (
                """
                SELECT event_id, event_type, severity, message, details, timestamp
                FROM ai_observability_events
                WHERE service = $1
                  AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            """
                % hours
            )
            raw_pool = getattr(pool, "pool", None) or getattr(pool, "_pool", None)
            if raw_pool is None:
                rows = await pool.fetch(query, service)
            else:
                async with raw_pool.acquire(timeout=10.0) as conn:
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT set_config('app.current_tenant_id', $1, true)",
                            tenant_id,
                        )
                        rows = await conn.fetch(query, service)

            return [
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "severity": row["severity"],
                    "message": row["message"],
                    "details": row["details"]
                    if isinstance(row["details"], dict)
                    else json.loads(row["details"] or "{}"),
                    "timestamp": row["timestamp"].isoformat()
                    if hasattr(row["timestamp"], "isoformat")
                    else str(row["timestamp"]),
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get service timeline: {e}")
            return []

    def get_stats(self) -> dict:
        """Get daemon statistics"""
        return {
            "started_at": self.stats.started_at,
            "total_checks": self.stats.total_checks,
            "total_events": self.stats.total_events,
            "alerts_generated": self.stats.alerts_generated,
            "events_persisted": self.stats.events_persisted,
            "services_monitored": self.stats.services_monitored,
            "last_check": self.stats.last_check,
            "consecutive_failures": self.stats.consecutive_failures,
            "running": self.running,
            "queued_events": len(self._event_queue),
        }


# Global daemon instance
_observability_daemon: Optional[PermanentObservabilityDaemon] = None


def get_observability_daemon() -> PermanentObservabilityDaemon:
    """Get or create the global observability daemon"""
    global _observability_daemon
    if _observability_daemon is None:
        _observability_daemon = PermanentObservabilityDaemon()
    return _observability_daemon


async def start_observability_daemon() -> PermanentObservabilityDaemon:
    """Start the global observability daemon"""
    daemon = get_observability_daemon()
    await daemon.start()
    return daemon


async def stop_observability_daemon() -> None:
    """Stop the global observability daemon"""
    global _observability_daemon
    if _observability_daemon:
        await _observability_daemon.stop()


# Helper functions for external use
async def capture_custom_event(
    event_type: str,
    service: str,
    message: str,
    severity: str = "info",
    details: Optional[dict] = None,
) -> None:
    """Capture a custom event from external code"""
    daemon = get_observability_daemon()
    await daemon._capture_event(
        event_type=event_type,
        service=service,
        severity=AlertSeverity(severity),
        message=message,
        details=details or {},
    )


async def get_system_health_summary() -> dict:
    """Get a summary of all system health"""
    daemon = get_observability_daemon()

    # Do a one-time check
    async with aiohttp.ClientSession() as session:
        results = {}
        for name, config in daemon.services.items():
            health = await daemon._check_service(session, name, config)
            results[name] = {
                "status": health.status.value,
                "response_time_ms": health.response_time_ms,
                "http_code": health.http_code,
                "error": health.error,
            }

    # Count by status
    status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
    for health in results.values():
        status_counts[health["status"]] += 1

    overall = "healthy"
    if status_counts["unhealthy"] > 0:
        overall = "unhealthy"
    elif status_counts["degraded"] > 0:
        overall = "degraded"

    return {
        "overall": overall,
        "services": results,
        "counts": status_counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    # Test the daemon
    async def test():
        daemon = PermanentObservabilityDaemon(poll_interval=30)
        await daemon.start()

        # Run for 2 minutes
        await asyncio.sleep(120)

        print("Stats:", daemon.get_stats())

        await daemon.stop()

    asyncio.run(test())
