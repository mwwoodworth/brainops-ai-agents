import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alive_core as alive_core_module
from alive_core import AliveCore


class DummyPool:
    def __init__(self, stats):
        self._stats = stats

    async def fetchrow(self, _query: str):
        return self._stats


@pytest.mark.asyncio
async def test_system_status_reports_degraded_when_database_fallback(monkeypatch):
    monkeypatch.setattr(alive_core_module, "using_fallback", lambda: True)
    core = AliveCore()

    status = await core.system_status()

    assert status["status"] == "degraded"
    assert status["reason"] == "database_fallback"
    assert status["active_agents"] == 0
    assert status["error_rates"]["error_rate"] == 0.0


@pytest.mark.asyncio
async def test_system_status_uses_database_execution_snapshot(monkeypatch):
    monkeypatch.setattr(alive_core_module, "using_fallback", lambda: False)
    monkeypatch.setattr(
        alive_core_module,
        "get_pool",
        lambda: DummyPool({"active_agents": 4, "total_executions": 20, "failed_executions": 2}),
    )
    core = AliveCore()

    status = await core.system_status()

    assert status["status"] == "healthy"
    assert status["active_agents"] == 4
    assert status["error_rates"]["total_executions"] == 20
    assert status["error_rates"]["failed_executions"] == 2
    assert status["error_rates"]["error_rate"] == 0.1


@pytest.mark.asyncio
async def test_get_status_compat_wrapper(monkeypatch):
    monkeypatch.setattr(alive_core_module, "using_fallback", lambda: False)
    monkeypatch.setattr(
        alive_core_module,
        "get_pool",
        lambda: DummyPool({"active_agents": 1, "total_executions": 2, "failed_executions": 1}),
    )
    core = AliveCore()

    status = await core.get_status()

    assert status["status"] == "critical"
    assert status["error_rates"]["error_rate"] == 0.5
