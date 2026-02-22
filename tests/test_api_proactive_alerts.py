"""
Tests for api/proactive_alerts.py — Proactive alert generation and recommendations.
"""
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.proactive_alerts as alerts_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(alerts_api.router)

    monkeypatch.setattr(alerts_api, "VALID_API_KEYS", {"test-key"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alerts_require_auth(client):
    resp = await client.get("/proactive/alerts")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_alerts_invalid_key(client):
    resp = await client.get(
        "/proactive/alerts",
        headers={"X-API-Key": "bad"},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestAlertPriority:
    def test_values(self):
        assert alerts_api.AlertPriority.CRITICAL.value == "critical"
        assert alerts_api.AlertPriority.HIGH.value == "high"
        assert alerts_api.AlertPriority.MEDIUM.value == "medium"
        assert alerts_api.AlertPriority.LOW.value == "low"


class TestAlertType:
    def test_values(self):
        assert alerts_api.AlertType.OPPORTUNITY.value == "opportunity"
        assert alerts_api.AlertType.OPTIMIZATION.value == "optimization"
        assert alerts_api.AlertType.ANOMALY.value == "anomaly"
        assert alerts_api.AlertType.RECOMMENDATION.value == "recommendation"
        assert alerts_api.AlertType.WARNING.value == "warning"


# ---------------------------------------------------------------------------
# ProactiveAlert dataclass
# ---------------------------------------------------------------------------


class TestProactiveAlert:
    def test_creation(self):
        alert = alerts_api.ProactiveAlert(
            alert_type=alerts_api.AlertType.OPPORTUNITY,
            title="New revenue opportunity",
            recommendation="Pursue lead XYZ",
            action="send_outreach",
            priority=alerts_api.AlertPriority.HIGH,
            source="lead_analysis",
        )
        assert alert.alert_type == alerts_api.AlertType.OPPORTUNITY
        assert alert.priority == alerts_api.AlertPriority.HIGH
        assert alert.title == "New revenue opportunity"

    def test_all_priority_levels(self):
        for priority in alerts_api.AlertPriority:
            alert = alerts_api.ProactiveAlert(
                alert_type=alerts_api.AlertType.WARNING,
                title="Test",
                recommendation="Test rec",
                action="test_action",
                priority=priority,
                source="test",
            )
            assert alert.priority == priority
