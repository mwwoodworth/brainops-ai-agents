import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import revenue_complete  # noqa: E402


class FakePool:
    def __init__(self, *, fail_email_query: bool = False):
        self.fail_email_query = fail_email_query

    async def fetchrow(self, query: str, *_params):
        if "FROM gumroad_sales" in query:
            return {
                "total_sales": 4,
                "total_revenue": 1200,
                "unique_customers": 3,
            }
        if "FROM stripe_events" in query:
            return {
                "total_charges": 2,
                "total_cents": 4500,
            }
        if "FROM revenue_leads" in query:
            return {
                "total_leads": 10,
                "new_leads": 4,
                "contacted": 3,
                "qualified": 2,
                "won": 1,
                "won_value": 900,
            }
        if "FROM ai_email_queue" in query:
            if self.fail_email_query:
                raise RuntimeError("UndefinedTableError: relation ai_email_queue does not exist")
            return {
                "total_sent": 8,
                "delivered": 7,
                "opened": 5,
            }
        if "FROM agent_executions" in query:
            return {
                "total_executions": 99,
                "unique_agents": 12,
            }
        if "FROM api_usage" in query:
            return {
                "total_calls": 456,
                "unique_keys": 6,
                "total_revenue_cents": 0,
            }
        raise AssertionError(f"Unexpected query: {query}")


@pytest.mark.asyncio
async def test_revenue_complete_dashboard_returns_healthy_when_all_queries_succeed(monkeypatch):
    async def _fake_get_pool():
        return FakePool()

    monkeypatch.setattr(revenue_complete, "get_pool", _fake_get_pool)

    payload = await revenue_complete.revenue_dashboard(days=14)

    assert payload["period_days"] == 14
    assert payload["health"]["status"] == "healthy"
    assert payload["health"]["all_streams_active"] is True
    assert payload["health"]["missing_streams"] == []
    assert payload["revenue_breakdown"]["gumroad"]["sales"] == 4
    assert payload["agent_activity"]["executions"] == 99


@pytest.mark.asyncio
async def test_revenue_complete_dashboard_degrades_gracefully_on_missing_table(monkeypatch):
    async def _fake_get_pool():
        return FakePool(fail_email_query=True)

    monkeypatch.setattr(revenue_complete, "get_pool", _fake_get_pool)

    payload = await revenue_complete.revenue_dashboard(days=30)

    assert payload["health"]["status"] == "degraded"
    assert payload["health"]["all_streams_active"] is False
    assert "email_campaigns" in payload["health"]["missing_streams"]
    assert payload["email_campaigns"]["sent"] == 0
    assert any(warning.startswith("email_campaigns:") for warning in payload["health"]["warnings"])
