import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marketplace.follow_up_engine import FollowUpEngine
from marketplace import follow_up_engine as follow_up_module


class DummyFollowUpSystem:
    async def _analyze_followup_strategy(self, *args, **kwargs):
        return {"timing": {"initial_delay": 1, "retry_interval": 24}, "channels": ["email"]}

    async def _generate_touchpoints(self, *args, **kwargs):
        return [{"step": 1, "delay_hours": 1, "channel": "email"}]

    async def create_followup_sequence(self, *args, **kwargs):
        return "seq-123"


@pytest.mark.asyncio
async def test_follow_up_engine_requires_followup_type(monkeypatch):
    async def _subscribed(*_args, **_kwargs):
        return True

    async def _record_usage(*_args, **_kwargs):
        return True

    monkeypatch.setattr(follow_up_module.UsageMetering, "check_subscription", _subscribed)
    monkeypatch.setattr(follow_up_module.UsageMetering, "record_usage", _record_usage)

    engine = FollowUpEngine("tenant-1")

    with pytest.raises(ValueError, match="followup_type"):
        await engine.generate_sequence({"entity_id": "cust-1"})


@pytest.mark.asyncio
async def test_follow_up_engine_generates_sequence(monkeypatch):
    async def _subscribed(*_args, **_kwargs):
        return True

    async def _record_usage(*_args, **_kwargs):
        return True

    monkeypatch.setattr(follow_up_module.UsageMetering, "check_subscription", _subscribed)
    monkeypatch.setattr(follow_up_module.UsageMetering, "record_usage", _record_usage)
    monkeypatch.setattr(follow_up_module, "IntelligentFollowUpSystem", DummyFollowUpSystem)

    engine = FollowUpEngine("tenant-2")

    result = await engine.generate_sequence({
        "followup_type": "lead_inquiry",
        "entity_id": "cust-2",
        "entity_type": "customer",
    })

    assert result["sequence_id"] == "seq-123"
    assert result["followup_type"] == "lead_inquiry"
    assert result["touchpoints"]
