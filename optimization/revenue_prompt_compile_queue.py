from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _default_tenant_id() -> str:
    return (
        os.getenv("DEFAULT_TENANT_ID")
        or os.getenv("TENANT_ID")
        or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
    )


async def enqueue_revenue_prompt_compile_task(
    *,
    pool: Any,
    lead_id: str | None,
    reason: str,
    tenant_id: str | None = None,
    priority: int = 80,
    force: bool = True,
) -> Optional[str]:
    """
    Queue a `revenue_prompt_compile` task for the ai_task_queue consumer.

    Safe defaults:
    - Disabled unless `DSPY_REVENUE_AUTO_RECOMPILE=true` and `ENABLE_DSPY_OPTIMIZATION=true`
    - Best-effort dedupe (skip if a pending/processing compile task exists recently)
    - Never raises (caller should not fail a revenue event because optimization is unavailable)
    """
    if not _env_bool("DSPY_REVENUE_AUTO_RECOMPILE", default=False):
        return None
    if not _env_bool("ENABLE_DSPY_OPTIMIZATION", default=False):
        return None

    tenant_id = (tenant_id or "").strip() or _default_tenant_id()
    dedupe_seconds = max(30, _coerce_int(os.getenv("DSPY_REVENUE_COMPILE_TASK_DEDUPE_SECONDS"), 600))

    try:
        existing = await pool.fetchrow(
            """
            SELECT id
            FROM ai_task_queue
            WHERE task_type = 'revenue_prompt_compile'
              AND status IN ('pending', 'processing')
              AND created_at > NOW() - ($1 * INTERVAL '1 second')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            dedupe_seconds,
        )
        if existing and existing.get("id"):
            return None
    except Exception:
        # Dedupe is best-effort only.
        pass

    payload = {
        "lead_id": lead_id,
        "reason": reason,
        "force": bool(force),
    }

    try:
        row = await pool.fetchrow(
            """
            INSERT INTO ai_task_queue (tenant_id, task_type, payload, priority, status, created_at, updated_at)
            VALUES ($1, 'revenue_prompt_compile', $2::jsonb, $3, 'pending', NOW(), NOW())
            RETURNING id
            """,
            tenant_id or None,
            json.dumps(payload, default=str),
            int(priority),
        )
        return str(row["id"]) if row and row.get("id") else None
    except Exception as exc:
        logger.debug("Failed to queue revenue_prompt_compile task: %s", exc)
        return None

