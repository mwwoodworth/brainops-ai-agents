"""Helpers for writing operational findings to Unified Brain."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from unified_brain import get_brain

logger = logging.getLogger(__name__)


async def brain_store(
    *,
    key: str,
    value: Any,
    category: str = "operational",
    priority: str = "medium",
    source: str = "operational_monitor",
    metadata: Optional[dict[str, Any]] = None,
    ttl_hours: Optional[int] = None,
) -> bool:
    """Best-effort write to Unified Brain for actionable operational context."""
    try:
        await get_brain().store(
            key=key,
            value=value,
            category=category,
            priority=priority,
            source=source,
            metadata={
                "stored_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            },
            ttl_hours=ttl_hours,
        )
        return True
    except Exception as exc:
        logger.warning("Brain store failed for key=%s: %s", key, exc)
        return False
