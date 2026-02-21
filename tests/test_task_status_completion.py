"""Regression test for _update_task_status completed_at fix.

Verifies that when status is set to completed/failed/cancelled,
the completed_at column is also set.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intelligent_task_orchestrator import IntelligentTaskOrchestrator  # noqa: E402


def _make_orchestrator():
    """Create an IntelligentTaskOrchestrator without full init."""
    orch = IntelligentTaskOrchestrator.__new__(IntelligentTaskOrchestrator)
    orch.running_tasks = {}
    return orch


class TestUpdateTaskStatus:
    """Verify _update_task_status sets completed_at for terminal states."""

    @pytest.mark.asyncio
    async def test_completed_sets_completed_at(self):
        """When status='completed', SQL should SET completed_at = NOW()."""
        orch = _make_orchestrator()

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("intelligent_task_orchestrator.get_db_connection", return_value=mock_conn):
            await orch._update_task_status("fake-id", "completed")

        # Verify the SQL includes completed_at
        call_args = mock_cur.execute.call_args
        sql = call_args[0][0]
        assert "completed_at" in sql

    @pytest.mark.asyncio
    async def test_failed_sets_completed_at(self):
        """When status='failed', SQL should SET completed_at = NOW()."""
        orch = _make_orchestrator()

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("intelligent_task_orchestrator.get_db_connection", return_value=mock_conn):
            await orch._update_task_status("fake-id", "failed")

        sql = mock_cur.execute.call_args[0][0]
        assert "completed_at" in sql

    @pytest.mark.asyncio
    async def test_in_progress_does_not_set_completed_at(self):
        """When status='in_progress', SQL should NOT include completed_at."""
        orch = _make_orchestrator()

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("intelligent_task_orchestrator.get_db_connection", return_value=mock_conn):
            await orch._update_task_status("fake-id", "in_progress")

        sql = mock_cur.execute.call_args[0][0]
        assert "completed_at" not in sql
