"""
Tests for api/db_intelligence.py — Database Intelligence API.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.db_intelligence as db_intel_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(db_intel_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /db-intelligence/ — root
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_available(client, monkeypatch):
    monkeypatch.setattr(db_intel_api, "DB_INTELLIGENCE_AVAILABLE", True)

    resp = await client.get("/db-intelligence/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True


@pytest.mark.asyncio
async def test_root_unavailable(client, monkeypatch):
    monkeypatch.setattr(db_intel_api, "DB_INTELLIGENCE_AVAILABLE", False)

    resp = await client.get("/db-intelligence/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


# ---------------------------------------------------------------------------
# GET /db-intelligence/health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_unavailable(client, monkeypatch):
    monkeypatch.setattr(db_intel_api, "DB_INTELLIGENCE_AVAILABLE", False)

    resp = await client.get("/db-intelligence/health")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_health_available(client, monkeypatch):
    monkeypatch.setattr(db_intel_api, "DB_INTELLIGENCE_AVAILABLE", True)

    async def mock_health():
        return {
            "healthy": True,
            "connection_count": 12,
            "slow_queries": 0,
        }

    monkeypatch.setattr(db_intel_api, "get_db_health", mock_health)

    resp = await client.get("/db-intelligence/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /db-intelligence/optimize-query — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optimize_query_validation(client):
    resp = await client.post(
        "/db-intelligence/optimize-query",
        json={},  # Missing 'query'
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_optimize_query_unavailable(client, monkeypatch):
    monkeypatch.setattr(db_intel_api, "DB_INTELLIGENCE_AVAILABLE", False)

    resp = await client.post(
        "/db-intelligence/optimize-query",
        json={"query": "SELECT * FROM customers"},
    )
    assert resp.status_code == 503
