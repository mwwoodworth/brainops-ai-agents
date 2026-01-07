import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as ai_app  # type: ignore
import api.revenue as revenue_api
from database.async_connection import DatabaseUnavailableError


def _authorize():
    ai_app.config.security.valid_api_keys = {"test-key"}
    ai_app.config.security.auth_configured = True
    revenue_api.VALID_API_KEYS.clear()
    revenue_api.VALID_API_KEYS.add("test-key")
    return {"X-API-Key": "test-key"}


def test_discover_leads_surfaces_db_unavailable(monkeypatch):
    async def _raise(*_args, **_kwargs):
        raise DatabaseUnavailableError("db down")

    monkeypatch.setattr(revenue_api, "generate_realistic_leads", _raise)
    monkeypatch.setattr(revenue_api, "get_pool", lambda: object())

    client = TestClient(ai_app.app)
    response = client.post(
        "/api/v1/revenue/discover-leads",
        json={"industry": "roofing", "location": "USA", "limit": 1},
        headers=_authorize(),
    )

    assert response.status_code == 503
