#!/usr/bin/env python3
"""
Alive Core (Lightweight)
========================
Minimal, import-safe implementation of the ALIVE subsystem.

Why this exists:
- Some parts of the codebase (e.g. `nerve_center.py`) and tests expect an `alive_core` module.
- The previous heavyweight implementation was archived/superseded.
- This version is deliberately low-risk: no mandatory DB access, no noisy background loops.

It provides:
- `ConsciousnessState`, `ThoughtType`, `VitalSigns`
- `AliveCore` with self-state snapshot persistence hooks
- `get_alive_core()` singleton
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import psutil


class ConsciousnessState(str, Enum):
    AWAKENING = "awakening"
    ALERT = "alert"
    FOCUSED = "focused"
    DREAMING = "dreaming"
    HEALING = "healing"
    EVOLVING = "evolving"
    EMERGENCY = "emergency"


class ThoughtType(str, Enum):
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    DECISION = "decision"
    ACTION = "action"
    LEARNING = "learning"
    PREDICTION = "prediction"
    CONCERN = "concern"
    OPPORTUNITY = "opportunity"


@dataclass
class VitalSigns:
    cpu_percent: float
    memory_percent: float
    active_connections: int
    requests_per_minute: float
    error_rate: float
    response_time_avg: float
    uptime_seconds: float
    consciousness_state: ConsciousnessState
    thought_rate: float
    attention_focus: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": float(self.cpu_percent or 0.0),
            "memory_percent": float(self.memory_percent or 0.0),
            "active_connections": int(self.active_connections or 0),
            "requests_per_minute": float(self.requests_per_minute or 0.0),
            "error_rate": float(self.error_rate or 0.0),
            "response_time_avg": float(self.response_time_avg or 0.0),
            "uptime_seconds": float(self.uptime_seconds or 0.0),
            "consciousness_state": str(self.consciousness_state),
            "thought_rate": float(self.thought_rate or 0.0),
            "attention_focus": str(self.attention_focus or ""),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Memory:
    """
    A tiny memory envelope for self-state snapshots.

    The real system has richer memory types; tests only require `.content`.
    """

    id: str
    content: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class AliveCore:
    """
    Lightweight AliveCore focused on self-state snapshots.

    This class is designed to be safe to import in any environment (dev/test/prod).
    It does not start background tasks by default and does not require DB connectivity.
    """

    def __init__(self) -> None:
        self._tenant_id: Optional[str] = os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID")
        self._self_state_interval: int = int(os.getenv("ALIVE_SELF_STATE_INTERVAL_SEC", "30"))
        self._last_self_state_at: Optional[datetime] = None
        self._last_self_state: Optional[dict[str, Any]] = None

    def _get_memory_manager(self) -> Any | None:
        """
        Return an object with `.store(memory)` if available.
        In production, this can be wired to UnifiedMemoryManager; tests monkeypatch it.
        """
        return None

    def _get_pending_tasks_count(self) -> int:
        return 0

    def _get_active_agents_count(self) -> int:
        return 0

    def _get_last_error(self) -> dict[str, Any] | None:
        return None

    async def _maybe_store_self_state(self, vitals: VitalSigns) -> bool:
        """
        Persist a self-state snapshot to memory (best-effort).

        Returns True if a snapshot was stored.
        """
        if not self._tenant_id:
            return False

        mm = self._get_memory_manager()
        if not mm:
            return False

        now = datetime.now(timezone.utc)
        if self._self_state_interval > 0 and self._last_self_state_at:
            age = (now - self._last_self_state_at).total_seconds()
            if age < float(self._self_state_interval):
                return False

        pending_tasks = int(self._get_pending_tasks_count() or 0)
        active_agents = int(self._get_active_agents_count() or 0)
        last_error = self._get_last_error()

        # Keep it deterministic and small; no secrets, no raw env dumps.
        try:
            vm = psutil.virtual_memory()
            memory_used_bytes = int(getattr(vm, "used", 0) or 0)
        except Exception:
            memory_used_bytes = 0

        content: dict[str, Any] = {
            "tenant_id": self._tenant_id,
            "vitals": vitals.to_dict(),
            "pending_tasks": pending_tasks,
            "active_agents": active_agents,
            "last_error": last_error,
            "memory_used_bytes": memory_used_bytes,
        }

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            metadata={"type": "alive_self_state"},
        )

        # Store via memory manager (sync or async store are both tolerated).
        try:
            store_fn = getattr(mm, "store", None)
            if callable(store_fn):
                result = store_fn(memory)
                if hasattr(result, "__await__"):
                    await result
        except Exception:
            return False

        self._last_self_state_at = now
        self._last_self_state = content
        return True

    def _calculate_health_score(self, vitals: VitalSigns, *, pending_tasks: int = 0) -> int:
        """
        Compute a coarse 0-100 health score.
        """
        score = 100.0

        score -= min(100.0, max(0.0, float(vitals.cpu_percent or 0.0))) * 0.2
        score -= min(100.0, max(0.0, float(vitals.memory_percent or 0.0))) * 0.2

        # error_rate is 0..1; scale to 0..100 and penalize.
        score -= min(100.0, max(0.0, float(vitals.error_rate or 0.0) * 100.0)) * 1.0

        # response_time_avg in seconds; penalize above ~0.1s gradually
        score -= min(100.0, max(0.0, float(vitals.response_time_avg or 0.0) * 10.0)) * 1.0

        score -= min(50.0, max(0.0, float(pending_tasks or 0))) * 0.5

        return int(max(0.0, min(100.0, score)))

    def _infer_mood(self, health_score: int, last_error: dict[str, Any] | None) -> str:
        """
        Map health + errors into a human-readable system mood.
        """
        severity = str((last_error or {}).get("severity") or "").strip().lower()
        if severity in {"critical", "high"}:
            return "distressed"
        if health_score >= 80 and not severity:
            return "healthy"
        if health_score >= 60:
            return "degraded"
        return "unhealthy"


_alive_core: Optional[AliveCore] = None


def get_alive_core() -> AliveCore:
    global _alive_core
    if _alive_core is None:
        _alive_core = AliveCore()
    return _alive_core

