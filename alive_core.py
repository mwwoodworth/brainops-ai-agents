#!/usr/bin/env python3
"""AliveCore: operational system status provider."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from database.async_connection import get_pool, using_fallback

logger = logging.getLogger(__name__)


class ErrorRates(BaseModel):
    """Execution error-rate snapshot for a fixed window."""

    window_minutes: int = Field(default=60)
    total_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    error_rate: float = Field(default=0.0)


class SystemHealth(BaseModel):
    """Operational health payload returned by AliveCore."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float
    active_agents: int
    error_rates: ErrorRates
    status: str


class AliveCore:
    """Provides real-time operational status derived from runtime + database."""

    def __init__(self) -> None:
        self._started_at = datetime.now(timezone.utc)

    async def system_status(self) -> dict[str, Any]:
        uptime_seconds = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        active_agents = 0
        total_executions = 0
        failed_executions = 0
        degraded_reason = None

        try:
            if using_fallback():
                degraded_reason = "database_fallback"
            else:
                pool = get_pool()
                stats = await pool.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM ai_agents WHERE status = 'active') AS active_agents,
                        (SELECT COUNT(*)
                         FROM ai_agent_executions
                         WHERE created_at > NOW() - INTERVAL '1 hour') AS total_executions,
                        (SELECT COUNT(*)
                         FROM ai_agent_executions
                         WHERE status IN ('failed', 'error')
                           AND created_at > NOW() - INTERVAL '1 hour') AS failed_executions
                    """
                )
                if stats:
                    active_agents = int(stats.get("active_agents") or 0)
                    total_executions = int(stats.get("total_executions") or 0)
                    failed_executions = int(stats.get("failed_executions") or 0)
        except Exception as exc:
            degraded_reason = f"status_query_failed:{exc}"
            logger.warning("AliveCore status query failed: %s", exc)

        error_rate = (failed_executions / total_executions) if total_executions > 0 else 0.0

        status = "healthy"
        if degraded_reason:
            status = "degraded"
        elif error_rate >= 0.50:
            status = "critical"
        elif error_rate >= 0.20:
            status = "degraded"

        payload = SystemHealth(
            uptime_seconds=round(uptime_seconds, 1),
            active_agents=active_agents,
            error_rates=ErrorRates(
                total_executions=total_executions,
                failed_executions=failed_executions,
                error_rate=round(error_rate, 4),
            ),
            status=status,
        ).model_dump(mode="json")

        if degraded_reason:
            payload["reason"] = degraded_reason
        return payload

    async def get_status(self) -> dict[str, Any]:
        """Compatibility wrapper for legacy callers."""
        return await self.system_status()


_alive_core: Optional[AliveCore] = None


def get_alive_core() -> AliveCore:
    global _alive_core
    if _alive_core is None:
        _alive_core = AliveCore()
    return _alive_core
