"""
Tests for api/daily_briefing.py — Daily AI briefing endpoints.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.daily_briefing as briefing_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(briefing_api.router)

    monkeypatch.setattr(briefing_api, "VALID_API_KEYS", {"test-key"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# Authentication — use /briefing/ (trailing slash matches router prefix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_briefing_requires_auth(client):
    resp = await client.get("/briefing/")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_briefing_invalid_key(client):
    resp = await client.get(
        "/briefing/",
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /briefing/
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_briefing_db_unavailable(client, monkeypatch):
    """When DB is in fallback mode, should return 503."""

    def fail_pool():
        raise HTTPException(status_code=503, detail="Database unavailable")

    monkeypatch.setattr(briefing_api, "_get_pool", fail_pool)

    resp = await client.get("/briefing/", headers=HEADERS)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /briefing/stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_db_unavailable(client, monkeypatch):
    def fail_pool():
        raise HTTPException(status_code=503, detail="Database unavailable")

    monkeypatch.setattr(briefing_api, "_get_pool", fail_pool)

    resp = await client.get("/briefing/stats", headers=HEADERS)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /briefing/health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client, monkeypatch):
    resp = await client.get("/briefing/health", headers=HEADERS)
    # Health endpoint should not require DB
    assert resp.status_code == 200
