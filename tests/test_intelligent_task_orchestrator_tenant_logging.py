from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import intelligent_task_orchestrator as ito


class _FakeCursor:
    def __init__(self) -> None:
        self.query = ""
        self.params = ()

    def execute(self, query, params) -> None:
        self.query = query
        self.params = params

    def close(self) -> None:
        return None


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_store_execution_history_includes_tenant_id(monkeypatch):
    cursor = _FakeCursor()
    conn = _FakeConn(cursor)

    @contextmanager
    def _fake_get_db_connection():
        yield conn

    monkeypatch.setattr(ito, "get_db_connection", _fake_get_db_connection)

    orchestrator = ito.IntelligentTaskOrchestrator.__new__(ito.IntelligentTaskOrchestrator)
    now = datetime.now(timezone.utc)
    task = SimpleNamespace(
        id="task-1",
        assigned_agent="agent-x",
        started_at=now,
        completed_at=now,
        retry_count=1,
        risk_assessment={"risk": 0.1},
        confidence_score=0.9,
        human_escalation_required=False,
        escalation_reason=None,
    )

    await orchestrator._store_execution_history(task, "completed", {"ok": True})

    assert "tenant_id" in cursor.query
    assert cursor.params[-1] == ito._resolve_valid_tenant_id()
