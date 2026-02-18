#!/usr/bin/env python3
"""Nerve Center: lightweight operational health coordinator."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from brain_store_helper import brain_store
from safe_task import create_safe_task

try:
    from system_awareness import get_system_awareness

    SYSTEM_AWARENESS_AVAILABLE = True
except Exception:
    SYSTEM_AWARENESS_AVAILABLE = False
    get_system_awareness = None

logger = logging.getLogger("NERVE_CENTER")


class NerveCenter:
    """Coordinates actionable operational scans and publishes findings to brain."""

    DEVOPS_INTERVAL_SECONDS = 60
    ERROR_INTERVAL_SECONDS = 300
    ACTIONABLE_SEVERITIES = {"warning", "critical"}

    def __init__(self) -> None:
        self.is_online = False
        self.start_time = datetime.now(timezone.utc)
        self.system_awareness = None

        # Compatibility no-ops for modules that still probe old attributes.
        self.alive_core = None
        self.proactive = None

        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self._scan_stats: dict[str, int] = {
            "devops_runs": 0,
            "error_runs": 0,
            "actionable_findings": 0,
            "stored_findings": 0,
        }
        self._last_scan_at: dict[str, Optional[str]] = {"devops": None, "errors": None}
        self._last_findings: list[dict[str, Any]] = []

    @staticmethod
    def _serialize_insight(insight: Any) -> dict[str, Any]:
        if hasattr(insight, "to_dict"):
            try:
                return insight.to_dict()
            except Exception:
                pass

        category = getattr(insight, "category", None)
        category_value = getattr(category, "value", str(category) if category else "unknown")
        return {
            "category": category_value,
            "title": str(getattr(insight, "title", "Unknown finding")),
            "description": str(getattr(insight, "description", "")),
            "severity": str(getattr(insight, "severity", "info")),
            "data": getattr(insight, "data", {}) or {},
            "action_recommended": getattr(insight, "action_recommended", None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _is_actionable(self, insight_payload: dict[str, Any]) -> bool:
        severity = str(insight_payload.get("severity") or "").lower()
        action_recommended = bool(insight_payload.get("action_recommended"))
        return severity in self.ACTIONABLE_SEVERITIES or action_recommended

    async def _store_actionable_finding(self, source_scan: str, insight_payload: dict[str, Any]) -> None:
        fingerprint = hashlib.sha1(
            (
                f"{source_scan}|{insight_payload.get('category')}|"
                f"{insight_payload.get('title')}|{insight_payload.get('severity')}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        key = f"nerve_center_{source_scan}_{fingerprint}"

        priority = "critical" if str(insight_payload.get("severity", "")).lower() == "critical" else "high"
        stored = await brain_store(
            key=key,
            value={
                "source_scan": source_scan,
                "finding": insight_payload,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            category="alert",
            priority=priority,
            source="nerve_center",
            metadata={
                "component": "nerve_center",
                "source_scan": source_scan,
                "severity": insight_payload.get("severity"),
            },
            ttl_hours=72,
        )
        if stored:
            self._scan_stats["stored_findings"] += 1

    async def _persist_actionable_findings(self, source_scan: str, insights: list[Any]) -> None:
        for raw in insights:
            payload = self._serialize_insight(raw)
            if not self._is_actionable(payload):
                continue

            self._scan_stats["actionable_findings"] += 1
            self._last_findings.append(payload)
            self._last_findings = self._last_findings[-25:]
            await self._store_actionable_finding(source_scan, payload)

    async def _run_devops_scan(self) -> None:
        if not self.system_awareness:
            return
        insights = await self.system_awareness.scan_devops_status()
        self._scan_stats["devops_runs"] += 1
        self._last_scan_at["devops"] = datetime.now(timezone.utc).isoformat()
        await self._persist_actionable_findings("devops", insights)

    async def _run_error_scan(self) -> None:
        if not self.system_awareness:
            return
        insights = await self.system_awareness.scan_error_rates()
        self._scan_stats["error_runs"] += 1
        self._last_scan_at["errors"] = datetime.now(timezone.utc).isoformat()
        await self._persist_actionable_findings("errors", insights)

    async def _loop(self, interval_seconds: int, runner) -> None:
        while not self._shutdown_event.is_set():
            try:
                await runner()
            except Exception as exc:
                logger.warning("NerveCenter scan failed: %s", exc)

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def activate(self) -> None:
        """Start periodic operational scans."""
        if self.is_online:
            return

        if not SYSTEM_AWARENESS_AVAILABLE or not get_system_awareness:
            raise RuntimeError("SystemAwareness is unavailable; NerveCenter cannot start")

        self.system_awareness = get_system_awareness()
        self._shutdown_event.clear()
        self._tasks = [
            create_safe_task(
                self._loop(self.DEVOPS_INTERVAL_SECONDS, self._run_devops_scan),
                "nerve_center_devops_scan",
            ),
            create_safe_task(
                self._loop(self.ERROR_INTERVAL_SECONDS, self._run_error_scan),
                "nerve_center_error_scan",
            ),
        ]
        self.is_online = True
        logger.info("NerveCenter activated (devops=60s, errors=300s)")

    async def deactivate(self) -> None:
        """Stop periodic scans."""
        self._shutdown_event.set()
        self.is_online = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("NerveCenter deactivated")

    def get_status(self) -> dict[str, Any]:
        uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "is_online": self.is_online,
            "uptime_seconds": round(uptime_seconds, 1),
            "scan_stats": dict(self._scan_stats),
            "last_scan_at": dict(self._last_scan_at),
            "recent_actionable_findings": list(self._last_findings[-10:]),
            "health": "healthy" if self.is_online else "offline",
        }


_nerve_center: Optional[NerveCenter] = None
_nerve_center_lock = threading.Lock()


def get_nerve_center() -> NerveCenter:
    global _nerve_center
    if _nerve_center is None:
        with _nerve_center_lock:
            if _nerve_center is None:
                _nerve_center = NerveCenter()
    return _nerve_center
