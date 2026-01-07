import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent_executor as executor_mod
from agent_executor import AgentExecutor


class DummyCore:
    def __init__(self, reasoning_json, critique_json):
        self._reasoning_json = reasoning_json
        self._critique_json = critique_json

    async def reason(self, *args, **kwargs):
        return {"reasoning": json.dumps(self._reasoning_json)}

    async def generate(self, *args, **kwargs):
        return json.dumps(self._critique_json)

    def _safe_json(self, text):
        return json.loads(text)


@pytest.mark.asyncio
async def test_reasoning_guard_blocks_on_low_confidence(monkeypatch):
    dummy_core = DummyCore({"confidence": 40, "recommendation": "ask_human"}, {"severity": "low", "block": False})

    monkeypatch.setattr(executor_mod, "ai_core", dummy_core)
    monkeypatch.setattr(executor_mod, "ENABLE_REASONING_GUARD", True)
    monkeypatch.setattr(executor_mod, "REASONING_GUARD_CONFIDENCE_THRESHOLD", 0.7)

    executor = AgentExecutor.__new__(AgentExecutor)

    async def _noop(*args, **kwargs):
        return None

    executor._store_reasoning_audit = _noop

    result = await executor._run_reasoning_guard("DeployAgent", {"action": "deploy"})
    assert result["block"] is not None


@pytest.mark.asyncio
async def test_reasoning_guard_allows_on_high_confidence(monkeypatch):
    dummy_core = DummyCore({"confidence": 92, "recommendation": "act"}, {"severity": "low", "block": False})

    monkeypatch.setattr(executor_mod, "ai_core", dummy_core)
    monkeypatch.setattr(executor_mod, "ENABLE_REASONING_GUARD", True)
    monkeypatch.setattr(executor_mod, "REASONING_GUARD_CONFIDENCE_THRESHOLD", 0.7)

    executor = AgentExecutor.__new__(AgentExecutor)

    async def _noop(*args, **kwargs):
        return None

    executor._store_reasoning_audit = _noop

    result = await executor._run_reasoning_guard("DeployAgent", {"action": "deploy"})
    assert result["block"] is None
