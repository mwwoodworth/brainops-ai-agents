"""Chaos-style tests for InvariantEngine resilience.

Verifies:
1. Double-fail (both DB stores) logs CRITICAL instead of silently passing.
2. Awareness staleness check detects stale heartbeats.
3. Awareness staleness check is clean when heartbeats are fresh.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import invariant_monitor  # noqa: E402
from invariant_monitor import InvariantEngine  # noqa: E402


def _make_engine():
    """Create an InvariantEngine with a mocked pool."""
    engine = InvariantEngine.__new__(InvariantEngine)
    engine.run_count = 0
    engine.last_run = None
    engine.last_violations = []
    engine.total_violations_detected = 0
    engine.consecutive_clean = 0
    engine.EXPECTED_AGENT_USER = "agent_worker"
    return engine


@pytest.mark.asyncio
async def test_double_fail_logs_critical(caplog):
    """When both invariant_violations INSERT and unified_brain_logs fallback fail,
    INVARIANT_DOUBLE_FAIL is logged at CRITICAL level."""
    engine = _make_engine()

    mock_pool = AsyncMock()
    # Primary INSERT fails
    mock_pool.fetchval = AsyncMock(side_effect=ConnectionError("db down"))
    # Fallback INSERT also fails
    mock_pool.execute = AsyncMock(side_effect=ConnectionError("still down"))

    engine._get_pool = lambda: mock_pool

    result = await engine._persist_violation("test_check", "critical", "test msg", {})
    assert result is None
    assert any("INVARIANT_DOUBLE_FAIL" in r.message for r in caplog.records)
    critical_records = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(critical_records) >= 1


@pytest.mark.asyncio
async def test_primary_fail_fallback_succeeds_no_critical(caplog):
    """When primary INSERT fails but fallback succeeds, no CRITICAL is logged."""
    engine = _make_engine()

    mock_pool = AsyncMock()
    mock_pool.fetchval = AsyncMock(side_effect=ConnectionError("db down"))
    mock_pool.execute = AsyncMock(return_value=None)

    engine._get_pool = lambda: mock_pool

    result = await engine._persist_violation("test_check", "high", "test msg", {})
    assert result is None
    critical_records = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(critical_records) == 0


@pytest.mark.asyncio
async def test_awareness_staleness_detects_stale():
    """When system_awareness_state has old heartbeats, violations are recorded."""
    engine = _make_engine()

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=15)
    mock_row = {"component_id": "brainops_overall", "last_heartbeat": stale_time}

    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=[mock_row])
    # _record will call _persist_violation which needs fetchval
    mock_pool.fetchval = AsyncMock(return_value="fake-id")

    engine._get_pool = lambda: mock_pool

    violations = []
    await engine._check_awareness_staleness(mock_pool, violations)

    assert len(violations) == 1
    assert violations[0]["check"] == "awareness_staleness"
    assert violations[0]["severity"] == "high"
    assert "brainops_overall" in violations[0]["message"]


@pytest.mark.asyncio
async def test_awareness_staleness_fresh_no_violation():
    """When heartbeats are fresh, no violations are generated."""
    engine = _make_engine()

    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=[])  # No stale rows

    violations = []
    await engine._check_awareness_staleness(mock_pool, violations)

    assert len(violations) == 0


@pytest.mark.asyncio
async def test_awareness_staleness_missing_table_no_error():
    """When system_awareness_state table doesn't exist, check is silently skipped."""
    engine = _make_engine()

    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(
        side_effect=Exception('relation "system_awareness_state" does not exist')
    )

    violations = []
    await engine._check_awareness_staleness(mock_pool, violations)

    assert len(violations) == 0
