import sys
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alive_core import AliveCore, ConsciousnessState, VitalSigns


class DummyMemoryManager:
    def __init__(self):
        self.stored = []

    def store(self, memory):
        self.stored.append(memory)
        return "memory-id"


@pytest.mark.asyncio
async def test_self_state_snapshot_persisted(monkeypatch):
    core = AliveCore()
    core._tenant_id = "test-tenant"
    core._self_state_interval = 0

    dummy_memory = DummyMemoryManager()

    monkeypatch.setattr(core, "_get_memory_manager", lambda: dummy_memory)
    monkeypatch.setattr(core, "_get_pending_tasks_count", lambda: 4)
    monkeypatch.setattr(core, "_get_active_agents_count", lambda: 2)
    monkeypatch.setattr(core, "_get_last_error", lambda: {"severity": "low"})

    class DummyVM:
        used = 256 * 1024 * 1024

    monkeypatch.setattr(psutil, "virtual_memory", lambda: DummyVM())

    vitals = VitalSigns(
        cpu_percent=10.0,
        memory_percent=20.0,
        active_connections=0,
        requests_per_minute=0,
        error_rate=0.01,
        response_time_avg=0.2,
        uptime_seconds=42.0,
        consciousness_state=ConsciousnessState.ALERT,
        thought_rate=1.0,
        attention_focus="init",
    )

    await core._maybe_store_self_state(vitals)

    assert dummy_memory.stored
    stored = dummy_memory.stored[0]
    assert stored.content["pending_tasks"] == 4
    assert stored.content["active_agents"] == 2
    assert core._last_self_state is not None


def test_health_score_and_mood():
    core = AliveCore()
    vitals = VitalSigns(
        cpu_percent=10.0,
        memory_percent=15.0,
        active_connections=0,
        requests_per_minute=0,
        error_rate=0.0,
        response_time_avg=0.1,
        uptime_seconds=10.0,
        consciousness_state=ConsciousnessState.ALERT,
        thought_rate=5.0,
        attention_focus="normal",
    )

    score = core._calculate_health_score(vitals, pending_tasks=0)
    assert score >= 80
    assert core._infer_mood(score, None) == "healthy"
