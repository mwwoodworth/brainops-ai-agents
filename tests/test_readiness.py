import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as ai_app  # type: ignore


@pytest.mark.asyncio
async def test_ready_fails_when_db_unavailable(monkeypatch):
    class DummyPool:
        async def test_connection(self):
            return False

    monkeypatch.setattr(ai_app, "get_pool", lambda: DummyPool())

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_capabilities_requires_key(monkeypatch):
    monkeypatch.setattr(ai_app.config.security, "valid_api_keys", {"test-key"})
    monkeypatch.setattr(ai_app.config.security, "auth_configured", True)

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/capabilities", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == ai_app.config.service_name
