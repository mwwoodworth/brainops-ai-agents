import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bleeding_edge_ooda as ooda  # type: ignore


@pytest.mark.asyncio
async def test_speculate_actions_requires_executor_in_production(monkeypatch):
    monkeypatch.setattr(ooda, "ENVIRONMENT", "production")
    monkeypatch.setattr(ooda, "ALLOW_OODA_SPECULATION_MOCK", False)

    controller = ooda.BleedingEdgeOODAController("test-tenant")

    with pytest.raises(RuntimeError):
        await controller.speculate_actions(
            [{"type": "query", "probability": 0.9}],
            {},
        )


@pytest.mark.asyncio
async def test_speculate_actions_allows_mock_in_dev_when_enabled(monkeypatch):
    monkeypatch.setattr(ooda, "ENVIRONMENT", "development")
    monkeypatch.setattr(ooda, "ALLOW_OODA_SPECULATION_MOCK", True)

    controller = ooda.BleedingEdgeOODAController("test-tenant")
    result = await controller.speculate_actions(
        [{"type": "query", "probability": 0.95, "params": {"q": "status"}}],
        {},
    )

    assert "query" in result
    assert result["query"]["status"] == "speculated"
