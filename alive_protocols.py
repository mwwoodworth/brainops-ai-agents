"""
Alive Protocols
===============
Background daemons that keep the AI OS in an always-on, autonomous state.

1) Self-Health: Polls /health and logs status to unified_brain.
2) Revenue Drive: Scans for stalled leads and triggers nurture sequences.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import httpx

from safe_task import create_safe_task

from database.async_connection import get_pool
from revenue_pipeline_agents import NurtureExecutorAgentReal
from unified_brain import brain

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_health_url() -> str:
    base = os.getenv("BRAINOPS_API_URL", "http://localhost:10000").rstrip("/")
    return f"{base}/health"


def _resolve_api_key() -> str | None:
    return os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY") or os.getenv("API_KEY")


async def _log_unified_brain_snapshot(
    key: str,
    value: Dict[str, Any],
    category: str,
    priority: int = 2,
) -> None:
    try:
        await brain.store(
            key=key,
            value=value,
            category=category,
            priority=priority,
            source="alive_protocols",
            metadata={"logged_at": datetime.now(timezone.utc).isoformat()},
            tags=[category, "alive"],
        )
    except Exception as exc:
        logger.warning("Failed to log %s snapshot to unified_brain: %s", category, exc)


async def run_self_health_daemon(interval_seconds: int | None = None) -> None:
    """Poll /health and store the latest status snapshot in unified_brain."""
    poll_interval = interval_seconds or int(os.getenv("SELF_HEALTH_INTERVAL_SEC", "60"))
    health_url = _resolve_health_url()
    api_key = _resolve_api_key()

    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    logger.info("🩺 Self-Health daemon started (interval=%ss, url=%s)", poll_interval, health_url)

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            started = datetime.now(timezone.utc)
            snapshot: Dict[str, Any] = {
                "endpoint": health_url,
                "status": "unknown",
                "status_code": None,
                "checked_at": started.isoformat(),
            }
            try:
                response = await client.get(health_url, headers=headers)
                snapshot["status_code"] = response.status_code
                snapshot["status"] = "healthy" if response.is_success else "degraded"
                try:
                    snapshot["payload"] = response.json()
                except Exception:
                    snapshot["payload"] = {"raw": response.text[:2000]}
            except Exception as exc:
                snapshot["status"] = "error"
                snapshot["error"] = str(exc)

            await _log_unified_brain_snapshot(
                key="self_health_snapshot",
                value=snapshot,
                category="self_health",
                priority=1 if snapshot.get("status") == "healthy" else 3,
            )

            await asyncio.sleep(poll_interval)


async def run_revenue_drive_daemon(interval_seconds: int | None = None) -> None:
    """Scan for stalled leads and enqueue nurture sequences."""
    poll_interval = interval_seconds or int(os.getenv("REVENUE_DRIVE_INTERVAL_SEC", "900"))
    stall_days = int(os.getenv("REVENUE_DRIVE_STALL_DAYS", "7"))
    max_batch = int(os.getenv("REVENUE_DRIVE_BATCH", "50"))

    logger.info(
        "💸 Revenue Drive daemon started (interval=%ss, stall_days=%s, batch=%s)",
        poll_interval,
        stall_days,
        max_batch,
    )

    nurture_agent = NurtureExecutorAgentReal()

    while True:
        snapshot: Dict[str, Any] = {
            "status": "unknown",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "stall_days": stall_days,
        }
        try:
            pool = get_pool()
            cutoff = datetime.now(timezone.utc) - timedelta(days=stall_days)

            leads = await pool.fetch(
                """
                SELECT id, contact_email, contact_name, stage, last_contact, updated_at, metadata
                FROM revenue_leads
                WHERE stage NOT IN ('won', 'lost')
                  AND (last_contact IS NULL OR last_contact < $1)
                ORDER BY updated_at NULLS FIRST
                LIMIT $2
                """,
                cutoff,
                max_batch,
            )

            processed = 0
            sequences_created = 0

            for lead in leads:
                lead_id = str(lead["id"])
                exists = await pool.fetchval(
                    """
                    SELECT 1
                    FROM ai_nurture_sequences
                    WHERE configuration->>'lead_id' = $1
                    """,
                    lead_id,
                )
                if exists:
                    continue

                result = await nurture_agent.create_nurture_sequence(
                    lead_id=lead_id,
                    sequence_type="nurture",
                    lead_data=dict(lead),
                )
                processed += 1
                if result.get("status") == "completed":
                    sequences_created += 1
                    await pool.execute(
                        """
                        UPDATE revenue_leads
                        SET stage = 'nurturing',
                            updated_at = NOW(),
                            last_contact = NOW()
                        WHERE id = $1
                        """,
                        lead["id"],
                    )

            snapshot.update(
                {
                    "status": "completed",
                    "leads_considered": len(leads),
                    "leads_processed": processed,
                    "sequences_created": sequences_created,
                }
            )
        except Exception as exc:
            snapshot["status"] = "error"
            snapshot["error"] = str(exc)
            logger.exception("Revenue Drive daemon failed: %s", exc)

        await _log_unified_brain_snapshot(
            key="revenue_drive_snapshot",
            value=snapshot,
            category="revenue_drive",
            priority=2 if snapshot.get("status") == "completed" else 3,
        )

        await asyncio.sleep(poll_interval)


def start_alive_protocols(tenant_id: str = "system") -> None:
    """Fire-and-forget startup for alive protocol daemons."""
    if _env_flag("ENABLE_SELF_HEALTH_DAEMON", True):
        create_safe_task(run_self_health_daemon())
        logger.info("✅ Self-Health daemon scheduled")
    if _env_flag("ENABLE_REVENUE_DRIVE_DAEMON", True):
        create_safe_task(run_revenue_drive_daemon())
        logger.info("✅ Revenue Drive daemon scheduled")