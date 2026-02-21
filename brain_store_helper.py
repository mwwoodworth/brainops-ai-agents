"""Helpers for writing operational findings to Unified Brain.

CANONICAL MEMORY WRITE PATH
All brain memory writes in brainops-ai-agents MUST go through this module.
See OS_CORE_CERTIFICATION.md for the memory protocol and failure runbooks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from unified_brain import get_brain

logger = logging.getLogger(__name__)

# ── Failure telemetry (visible via get_brain_store_stats) ────────────────────
_store_successes: int = 0
_store_failures: int = 0
_last_failure: Optional[str] = None
_last_failure_at: Optional[str] = None


def get_brain_store_stats() -> dict[str, Any]:
    """Return brain store health counters for the /health endpoint."""
    return {
        "successes": _store_successes,
        "failures": _store_failures,
        "last_failure": _last_failure,
        "last_failure_at": _last_failure_at,
    }


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
    global _store_successes, _store_failures, _last_failure, _last_failure_at
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
        _store_successes += 1
        return True
    except Exception as exc:
        _store_failures += 1
        _last_failure = str(exc)
        _last_failure_at = datetime.now(timezone.utc).isoformat()
        logger.error(
            "MEMORY_LOSS: brain_store failed key=%s failures=%d err=%s",
            key,
            _store_failures,
            exc,
        )
        return False
