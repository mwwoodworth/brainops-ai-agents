import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictive_market_intelligence import PredictiveMarketIntelligence


@pytest.mark.asyncio
async def test_news_signals_requires_api_key(monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setenv("NEWS_API_QUERY", "roofing")

    engine = PredictiveMarketIntelligence()

    with pytest.raises(RuntimeError, match="NEWS_API_KEY"):
        await engine._fetch_news_signals()


@pytest.mark.asyncio
async def test_news_signals_requires_query(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.delenv("NEWS_API_QUERY", raising=False)

    engine = PredictiveMarketIntelligence()

    with pytest.raises(RuntimeError, match="NEWS_API_QUERY"):
        await engine._fetch_news_signals()
