from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live_memory_brain import LiveMemoryBrain, MemoryNode, MemoryType


class _DummyCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, query: str, params: tuple) -> None:
        self.executed.append((query, params))

    def close(self) -> None:
        return None


class _DummyConn:
    def __init__(self) -> None:
        self.cursor_obj = _DummyCursor()
        self.committed = False

    def cursor(self) -> _DummyCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True


class _ConnCtx:
    def __init__(self, conn: _DummyConn) -> None:
        self._conn = conn

    def __enter__(self) -> _DummyConn:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Do not suppress exceptions.
        return False


class _DummySyncPool:
    def __init__(self) -> None:
        self.conn = _DummyConn()

    def get_connection(self) -> _ConnCtx:
        return _ConnCtx(self.conn)


@pytest.mark.asyncio
async def test_persist_memory_coerces_array_payloads() -> None:
    """Regression: avoid inserting JSON array strings (\"[]\") into TEXT[] columns."""
    brain = LiveMemoryBrain()
    brain._shared_sync_pool = _DummySyncPool()  # type: ignore[attr-defined]

    mem = MemoryNode(
        id="00000000-0000-0000-0000-000000000000",
        content="test",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        confidence=1.0,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0,
        provenance={},
        connections=set(),
        temporal_context={},
        predictions="[]",
        contradictions="[]",
        crystallization_count=0,
    )

    await brain._persist_memory(mem)  # type: ignore[attr-defined]

    executed = brain._shared_sync_pool.conn.cursor_obj.executed  # type: ignore[attr-defined]
    assert executed, "Expected _persist_memory to execute an INSERT"
    _query, params = executed[0]

    # Parameter ordering comes from live_memory_brain.py INSERT statement
    connections = params[9]
    predictions = params[11]
    contradictions = params[12]
    tenant_id = params[14]

    assert isinstance(connections, list)
    assert isinstance(predictions, list)
    assert isinstance(contradictions, list)
    assert predictions == []
    assert contradictions == []
    assert str(uuid.UUID(str(tenant_id))) == str(tenant_id)
