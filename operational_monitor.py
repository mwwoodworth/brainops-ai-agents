#!/usr/bin/env python3
"""Operational monitor for scheduled execution, invariants, and memory growth."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from brain_store_helper import brain_store
from database.async_connection import get_pool, get_tenant_pool, using_fallback
from safe_task import create_safe_task

logger = logging.getLogger("OPERATIONAL_MONITOR")


def _normalize_tenant_id(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    try:
        return str(uuid.UUID(candidate))
    except Exception:
        return None


def _resolve_tenant_id() -> str:
    return (
        _normalize_tenant_id(os.getenv("DEFAULT_TENANT_ID"))
        or _normalize_tenant_id(os.getenv("TENANT_ID"))
        or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
    )


class OperationalMonitor:
    """Periodic operational checks with alerting to Unified Brain."""

    def __init__(self, interval_seconds: int = 300) -> None:
        self.interval_seconds = max(60, int(interval_seconds))
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._started_at = datetime.now(timezone.utc)
        self._startup_grace_seconds = max(
            0, int(os.getenv("OPERATIONAL_MONITOR_STARTUP_GRACE_SECONDS", "3600"))
        )

        self._last_run_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._alerts_emitted = 0
        self._recent_alerts: list[dict[str, Any]] = []

    async def start(self) -> None:
        if self.is_running:
            return
        self._shutdown_event.clear()
        self._task = create_safe_task(self._run_loop(), "operational_monitor_loop")
        self.is_running = True
        logger.info("OperationalMonitor started (interval=%ss)", self.interval_seconds)

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.is_running = False
        logger.info("OperationalMonitor stopped")

    async def _run_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await self.run_checks()
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Operational monitor cycle failed: %s", exc)

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def run_checks(self) -> list[dict[str, Any]]:
        self._last_run_at = datetime.now(timezone.utc).isoformat()

        if using_fallback():
            return []

        try:
            pool = get_tenant_pool(_resolve_tenant_id())
        except Exception:
            pool = get_pool()
        alerts: list[dict[str, Any]] = []

        scheduled_alert = await self._check_scheduled_agents(pool)
        if scheduled_alert:
            alerts.append(scheduled_alert)

        invariants_alert = await self._check_invariant_violations(pool)
        if invariants_alert:
            alerts.append(invariants_alert)

        memory_alert = await self._check_memory_growth(pool)
        if memory_alert:
            alerts.append(memory_alert)

        # Keep recent alerts focused on active issues.
        if not alerts:
            self._recent_alerts = []

        for alert in alerts:
            await self._publish_alert(alert)

        return alerts

    async def _check_scheduled_agents(self, pool) -> Optional[dict[str, Any]]:
        # Fresh deploys can legitimately have sparse execution history.
        uptime_seconds = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        if uptime_seconds < self._startup_grace_seconds:
            return None

        rows = await pool.fetch(
            """
            SELECT
                a.name AS agent_name,
                s.frequency_minutes AS frequency_minutes,
                MAX(e.created_at) AS last_execution
            FROM agent_schedules s
            JOIN ai_agents a ON a.id = s.agent_id
            LEFT JOIN ai_agent_executions e
              ON e.agent_name = a.name
             AND e.created_at > NOW() - make_interval(mins => GREATEST(s.frequency_minutes * 2, 60))
            WHERE s.enabled = TRUE
            GROUP BY a.name, s.frequency_minutes
            HAVING MAX(e.created_at) IS NULL
            ORDER BY s.frequency_minutes ASC, a.name
            LIMIT 25
            """
        )

        if not rows:
            return None

        missing_agents = []
        for row in rows:
            name = str(row.get("agent_name") or "").strip()
            if not name:
                continue
            frequency = row.get("frequency_minutes")
            if frequency:
                missing_agents.append(f"{name} ({frequency}m)")
            else:
                missing_agents.append(name)
        return {
            "type": "scheduled_agents_missing_execution",
            "severity": "critical" if len(missing_agents) >= 3 else "warning",
            "summary": (
                f"{len(missing_agents)} scheduled agents are beyond expected execution window"
            ),
            "details": {
                "missing_agents": missing_agents,
                "window": "per-agent frequency * 2 (minimum 60 minutes)",
            },
        }

    async def _check_invariant_violations(self, pool) -> Optional[dict[str, Any]]:
        unresolved = int(
            await pool.fetchval("SELECT COUNT(*) FROM invariant_violations WHERE resolved = false")
            or 0
        )
        if unresolved == 0:
            return None

        rows = await pool.fetch(
            """
            SELECT check_name, created_at
            FROM invariant_violations
            WHERE resolved = false
            ORDER BY created_at DESC
            LIMIT 10
            """
        )
        checks = [str(row.get("check_name")) for row in rows if row.get("check_name")]

        return {
            "type": "unresolved_invariant_violations",
            "severity": "critical" if unresolved >= 5 else "warning",
            "summary": f"{unresolved} unresolved invariant violations detected",
            "details": {
                "unresolved_count": unresolved,
                "sample_checks": checks,
            },
        }

    async def _check_memory_growth(self, pool) -> Optional[dict[str, Any]]:
        row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                ) AS recent_count,
                COUNT(*) FILTER (
                    WHERE created_at <= NOW() - INTERVAL '1 hour'
                      AND created_at > NOW() - INTERVAL '2 hours'
                ) AS previous_count
            FROM unified_ai_memory
            """
        )

        recent_count = int(row.get("recent_count") or 0) if row else 0
        previous_count = int(row.get("previous_count") or 0) if row else 0

        min_spike = int(os.getenv("OPERATIONAL_MONITOR_MEMORY_SPIKE_MIN", "400"))
        ratio_threshold = float(os.getenv("OPERATIONAL_MONITOR_MEMORY_SPIKE_RATIO", "2.0"))

        abnormal = False
        ratio = float(recent_count) / float(previous_count) if previous_count > 0 else None
        if recent_count >= min_spike and previous_count == 0:
            abnormal = True
        elif recent_count >= min_spike and ratio is not None and ratio >= ratio_threshold:
            abnormal = True

        if not abnormal:
            return None

        return {
            "type": "abnormal_memory_growth",
            "severity": "warning" if recent_count < (min_spike * 2) else "critical",
            "summary": (
                f"Memory growth spike detected: {recent_count} entries in the last hour "
                f"(previous hour: {previous_count})"
            ),
            "details": {
                "recent_count": recent_count,
                "previous_count": previous_count,
                "ratio": round(ratio, 2) if ratio is not None else None,
                "threshold_ratio": ratio_threshold,
                "threshold_min_count": min_spike,
            },
        }

    async def _publish_alert(self, alert: dict[str, Any]) -> None:
        serialized = json.dumps(alert, sort_keys=True, default=str)
        fingerprint = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
        key = f"operational_monitor_{alert.get('type', 'alert')}_{fingerprint}"
        priority = "critical" if str(alert.get("severity", "")).lower() == "critical" else "high"

        stored = await brain_store(
            key=key,
            value={
                "alert": alert,
                "emitted_at": datetime.now(timezone.utc).isoformat(),
                "component": "operational_monitor",
            },
            category="alert",
            priority=priority,
            source="operational_monitor",
            metadata={
                "component": "operational_monitor",
                "alert_type": alert.get("type"),
                "severity": alert.get("severity"),
            },
            ttl_hours=168,
        )

        self._recent_alerts.append(alert)
        self._recent_alerts = self._recent_alerts[-25:]

        if stored:
            self._alerts_emitted += 1
        logger.warning("Operational alert: %s", alert.get("summary", alert.get("type")))

    def get_status(self) -> dict[str, Any]:
        uptime_seconds = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        return {
            "is_running": self.is_running,
            "interval_seconds": self.interval_seconds,
            "uptime_seconds": round(uptime_seconds, 2),
            "startup_grace_seconds": self._startup_grace_seconds,
            "last_run_at": self._last_run_at,
            "alerts_emitted": self._alerts_emitted,
            "recent_alerts": list(self._recent_alerts[-10:]),
            "last_error": self._last_error,
        }


_operational_monitor: Optional[OperationalMonitor] = None
_operational_monitor_lock = threading.Lock()


def get_operational_monitor(interval_seconds: Optional[int] = None) -> OperationalMonitor:
    global _operational_monitor
    if _operational_monitor is None:
        with _operational_monitor_lock:
            if _operational_monitor is None:
                _operational_monitor = OperationalMonitor(interval_seconds or 300)
    elif interval_seconds and _operational_monitor.interval_seconds != interval_seconds:
        _operational_monitor.interval_seconds = max(60, int(interval_seconds))
    return _operational_monitor
