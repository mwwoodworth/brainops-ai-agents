"""Chaos-style tests for BaseAgent.log_execution dual-failure detection.

Verifies that when BOTH persistence stores fail, MemoryLossError is raised
(triggering the executor's retry loop) rather than silently losing the result.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base_agent import BaseAgent, MemoryLossError  # noqa: E402


def _make_agent():
    return BaseAgent(name="test_agent", agent_type="test")


def _sample_task():
    return {"action": "test", "tenant_id": "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"}


def _sample_result():
    return {"status": "completed", "output": "ok"}


@pytest.mark.asyncio
async def test_both_stores_fail_raises_memory_loss():
    """When both legacy DB and UnifiedBrain fail, MemoryLossError is raised."""
    agent = _make_agent()

    mock_pool = AsyncMock()
    mock_pool.execute = AsyncMock(side_effect=ConnectionError("db down"))

    mock_brain = MagicMock()
    mock_brain.store = MagicMock(side_effect=ConnectionError("brain down"))

    with (
        patch("base_agent.get_pool", return_value=mock_pool),
        patch("base_agent.UnifiedBrain", return_value=mock_brain),
    ):
        with pytest.raises(MemoryLossError, match="both stores failed"):
            await agent.log_execution(_sample_task(), _sample_result())


@pytest.mark.asyncio
async def test_legacy_fails_brain_succeeds_no_error():
    """When only legacy DB fails but brain succeeds, no error is raised."""
    agent = _make_agent()

    mock_pool = AsyncMock()
    mock_pool.execute = AsyncMock(side_effect=ConnectionError("db down"))

    mock_brain = MagicMock()
    mock_brain.store = MagicMock(return_value=None)

    with (
        patch("base_agent.get_pool", return_value=mock_pool),
        patch("base_agent.UnifiedBrain", return_value=mock_brain),
    ):
        # Should NOT raise
        await agent.log_execution(_sample_task(), _sample_result())


@pytest.mark.asyncio
async def test_brain_fails_legacy_succeeds_no_error():
    """When only brain fails but legacy DB succeeds, no error is raised."""
    agent = _make_agent()

    mock_pool = AsyncMock()
    mock_pool.execute = AsyncMock(return_value=None)

    mock_brain = MagicMock()
    mock_brain.store = MagicMock(side_effect=ConnectionError("brain down"))

    with (
        patch("base_agent.get_pool", return_value=mock_pool),
        patch("base_agent.UnifiedBrain", return_value=mock_brain),
    ):
        # Should NOT raise
        await agent.log_execution(_sample_task(), _sample_result())


@pytest.mark.asyncio
async def test_both_succeed_no_error():
    """When both stores succeed, no error is raised."""
    agent = _make_agent()

    mock_pool = AsyncMock()
    mock_pool.execute = AsyncMock(return_value=None)

    mock_brain = MagicMock()
    mock_brain.store = MagicMock(return_value=None)

    with (
        patch("base_agent.get_pool", return_value=mock_pool),
        patch("base_agent.UnifiedBrain", return_value=mock_brain),
    ):
        await agent.log_execution(_sample_task(), _sample_result())
