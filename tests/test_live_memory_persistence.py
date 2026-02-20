import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live_memory_brain import LiveMemoryBrain, MemoryNode, MemoryType


@dataclass
class _FakeCursor:
    last_sql: str | None = None
    last_params: tuple | None = None

    def execute(self, sql: str, params: tuple) -> None:  # noqa: A003 - matches psycopg cursor API
        self.last_sql = sql
        self.last_params = params

    def close(self) -> None:
        return


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        return


class _FakeConnCM:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def __enter__(self) -> _FakeConn:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSyncPool:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def get_connection(self) -> _FakeConnCM:
        return _FakeConnCM(_FakeConn(self._cursor))


@pytest.mark.asyncio
async def test_persist_memory_coerces_predictions_to_text_array():
    """
    Regression test:
    - `live_brain_memories.predictions` is `text[]` in prod.
    - Passing JSON string \"[]\" causes: malformed array literal: \"[]\"
    """
    cursor = _FakeCursor()
    brain = LiveMemoryBrain.__new__(LiveMemoryBrain)
    brain._shared_sync_pool = _FakeSyncPool(cursor)

    now = datetime.now(timezone.utc)
    memory = MemoryNode(
        id="00000000-0000-0000-0000-000000000001",
        content={"hello": "world"},
        memory_type=MemoryType.EPISODIC,
        importance=0.5,
        confidence=0.5,
        created_at=now,
        last_accessed=now,
        access_count=0,
        provenance={"source": "test"},
        connections=set(),
        temporal_context={},
        predictions=[{"a": 1}],
        contradictions=[],
        crystallization_count=0,
    )

    await brain._persist_memory(memory)

    assert cursor.last_params is not None

    # Param order matches the INSERT in `_persist_memory`.
    predictions_param = cursor.last_params[11]
    contradictions_param = cursor.last_params[12]
    connections_param = cursor.last_params[9]
    tenant_param = cursor.last_params[14]

    assert isinstance(predictions_param, list)
    assert predictions_param and isinstance(predictions_param[0], str)

    assert isinstance(contradictions_param, list)
    assert contradictions_param == []

    assert isinstance(connections_param, list)
    assert connections_param == []
    assert str(uuid.UUID(str(tenant_param))) == str(tenant_param)


@pytest.mark.asyncio
async def test_persist_memory_handles_json_string_predictions():
    cursor = _FakeCursor()
    brain = LiveMemoryBrain.__new__(LiveMemoryBrain)
    brain._shared_sync_pool = _FakeSyncPool(cursor)

    now = datetime.now(timezone.utc)
    memory = MemoryNode(
        id="00000000-0000-0000-0000-000000000002",
        content="test",
        memory_type=MemoryType.EPISODIC,
        importance=0.5,
        confidence=0.5,
        created_at=now,
        last_accessed=now,
        access_count=0,
        provenance={},
        connections=set(),
        temporal_context={},
        predictions="[]",
        contradictions="[]",
        crystallization_count=0,
    )

    await brain._persist_memory(memory)

    assert cursor.last_params is not None
    predictions_param = cursor.last_params[11]
    contradictions_param = cursor.last_params[12]
    tenant_param = cursor.last_params[14]

    assert predictions_param == []
    assert contradictions_param == []
    assert str(uuid.UUID(str(tenant_param))) == str(tenant_param)
