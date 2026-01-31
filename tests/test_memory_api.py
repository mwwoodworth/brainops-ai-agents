import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as ai_app  # type: ignore
import api.memory as memory_api  # type: ignore


def _authorize():
    ai_app.config.security.valid_api_keys = {"test-key"}
    ai_app.config.security.auth_configured = True
    return {"X-API-Key": "test-key"}


@pytest.mark.asyncio
async def test_memory_status_returns_503_when_fallback(monkeypatch):
    monkeypatch.setattr(memory_api, "using_fallback", lambda: True)

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/memory/status", headers=_authorize())

    assert response.status_code == 503
    assert "in-memory fallback" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_memory_store_returns_503_when_fallback(monkeypatch):
    monkeypatch.setattr(memory_api, "using_fallback", lambda: True)

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/memory/store",
            headers=_authorize(),
            json={"content": "hello", "memory_type": "semantic"},
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_semantic_search_requires_embedding(monkeypatch):
    monkeypatch.setattr(memory_api, "using_fallback", lambda: False)

    async def _none_embedding(_: str):
        return None

    monkeypatch.setattr(memory_api, "generate_embedding", _none_embedding)
    monkeypatch.setattr(memory_api, "get_pool", lambda: object())

    transport = httpx.ASGITransport(app=ai_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/memory/search",
            headers=_authorize(),
            params={"query": "roof", "use_semantic": "true"},
        )

    assert response.status_code == 503
    assert "semantic search unavailable" in response.json()["detail"].lower()
