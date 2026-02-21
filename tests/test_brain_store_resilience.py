"""Chaos-style resilience tests for brain_store_helper.

Verifies that brain store failures are never silent — counters increment,
MEMORY_LOSS prefix appears in logs, and get_brain_store_stats() exposes
the telemetry for health endpoint consumption.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import brain_store_helper  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_counters():
    """Reset module-level counters before each test."""
    brain_store_helper._store_successes = 0
    brain_store_helper._store_failures = 0
    brain_store_helper._last_failure = None
    brain_store_helper._last_failure_at = None


@pytest.mark.asyncio
async def test_failure_increments_counter_and_logs_memory_loss(caplog):
    """When brain store raises, counter increments and MEMORY_LOSS is logged."""
    mock_brain = AsyncMock()
    mock_brain.store = AsyncMock(side_effect=ConnectionError("db gone"))

    with patch("brain_store_helper.get_brain", return_value=mock_brain):
        result = await brain_store_helper.brain_store(key="test_key", value="data")

    assert result is False
    assert brain_store_helper._store_failures == 1
    assert brain_store_helper._last_failure == "db gone"
    assert brain_store_helper._last_failure_at is not None
    assert any("MEMORY_LOSS" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_success_increments_counter():
    """Successful store increments success counter and returns True."""
    mock_brain = AsyncMock()
    mock_brain.store = AsyncMock(return_value=None)

    with patch("brain_store_helper.get_brain", return_value=mock_brain):
        result = await brain_store_helper.brain_store(key="ok_key", value="ok")

    assert result is True
    assert brain_store_helper._store_successes == 1
    assert brain_store_helper._store_failures == 0


@pytest.mark.asyncio
async def test_multiple_failures_accumulate():
    """Repeated failures accumulate in the counter."""
    mock_brain = AsyncMock()
    mock_brain.store = AsyncMock(side_effect=RuntimeError("timeout"))

    with patch("brain_store_helper.get_brain", return_value=mock_brain):
        for _ in range(5):
            await brain_store_helper.brain_store(key="k", value="v")

    assert brain_store_helper._store_failures == 5


def test_stats_dict_shape():
    """get_brain_store_stats() returns expected keys."""
    stats = brain_store_helper.get_brain_store_stats()
    assert "successes" in stats
    assert "failures" in stats
    assert "last_failure" in stats
    assert "last_failure_at" in stats
    assert isinstance(stats["successes"], int)
    assert isinstance(stats["failures"], int)
