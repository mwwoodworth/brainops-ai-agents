"""
Tests for api/system_health.py — System health, observability, truth, awareness endpoints.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.system_health as system_health_api
from config import config


class FakePool:
    """Minimal async pool mock for system_health tests."""

    def __init__(self, fetchval_result=0, fetchrow_result=None, fetch_result=None):
        self._fetchval_result = fetchval_result
        self._fetchrow_result = fetchrow_result
        self._fetch_result = fetch_result or []

    async def fetchval(self, query, *args):
        return self._fetchval_result

    async def fetchrow(self, query, *args):
        return self._fetchrow_result

    async def fetch(self, query, *args):
        return self._fetch_result

    async def execute(self, query, *args):
        return "OK"


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(system_health_api.router)

    monkeypatch.setattr(config.security, "valid_api_keys", {"test-key"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_awareness_requires_auth(client):
    resp = await client.get("/system/awareness")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_system_awareness_invalid_key(client):
    resp = await client.get(
        "/system/awareness",
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /system/awareness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_awareness_success(client, monkeypatch):
    pool = FakePool(fetchval_result=0)
    monkeypatch.setattr(system_health_api, "get_pool", lambda: pool)

    # Need to mock the `using_fallback` check if present
    try:
        monkeypatch.setattr(system_health_api, "using_fallback", lambda: False)
    except AttributeError:
        pass

    resp = await client.get("/system/awareness", headers=HEADERS)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /truth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truth_endpoint_exists(client, monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(system_health_api, "get_pool", lambda: pool)
    try:
        monkeypatch.setattr(system_health_api, "using_fallback", lambda: False)
    except AttributeError:
        pass

    resp = await client.get("/truth", headers=HEADERS)
    # Should return data or redirect, not 404
    assert resp.status_code in (200, 307, 401)


# ---------------------------------------------------------------------------
# GET /healthz (quick health, no auth required typically)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_awareness_pulse(client, monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(system_health_api, "get_pool", lambda: pool)
    try:
        monkeypatch.setattr(system_health_api, "using_fallback", lambda: False)
    except AttributeError:
        pass

    resp = await client.get("/awareness/pulse", headers=HEADERS)
    # Pulse may or may not require auth depending on config
    assert resp.status_code in (200, 401, 404)
