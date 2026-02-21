import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as ai_app  # type: ignore
import api.agents as agents_api  # type: ignore
from database.async_connection import DatabaseUnavailableError


def _authorize():
    ai_app.config.security.valid_api_keys = {"test-key"}
    ai_app.config.security.auth_configured = True
    return {"X-API-Key": "test-key"}


@pytest.mark.asyncio
async def test_agents_returns_503_when_db_unavailable(monkeypatch):
    def _raise():
        raise DatabaseUnavailableError("db down")

    monkeypatch.setattr(ai_app, "get_pool", _raise)
    monkeypatch.setattr(agents_api, "get_pool", _raise)

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/agents", headers=_authorize())

    assert response.status_code == 503
    assert "db down" in response.json()["detail"]


@pytest.mark.asyncio
async def test_executions_returns_503_on_query_failure(monkeypatch):
    class DummyPool:
        async def fetch(self, *args, **kwargs):
            raise RuntimeError("query failed")

    monkeypatch.setattr(ai_app, "get_pool", lambda: DummyPool())

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/executions", headers=_authorize())

    assert response.status_code == 503
