"""
Tests for api/knowledge.py — Knowledge Base API Router.
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

import api.knowledge as knowledge_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(knowledge_api.router)

    monkeypatch.setattr(knowledge_api, "VALID_API_KEYS", {"test-key"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# GET /api/v1/knowledge-base/health — public, no auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_no_auth_required(client, monkeypatch):
    """Health check is public — no API key needed."""
    monkeypatch.setattr(knowledge_api, "KNOWLEDGE_BASE_AVAILABLE", True)
    resp = await client.get("/api/v1/knowledge-base/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"


@pytest.mark.asyncio
async def test_health_unavailable(client, monkeypatch):
    monkeypatch.setattr(knowledge_api, "KNOWLEDGE_BASE_AVAILABLE", False)
    resp = await client.get("/api/v1/knowledge-base/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unavailable"


# ---------------------------------------------------------------------------
# GET /api/v1/knowledge-base/entries — requires auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entries_requires_auth(client):
    resp = await client.get("/api/v1/knowledge-base/entries")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_entries_invalid_key(client):
    resp = await client.get(
        "/api/v1/knowledge-base/entries",
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_entry_validation_title_too_short(client, monkeypatch):
    monkeypatch.setattr(knowledge_api, "KNOWLEDGE_BASE_AVAILABLE", True)

    resp = await client.post(
        "/api/v1/knowledge-base/entries",
        headers=HEADERS,
        json={
            "title": "ab",  # min_length=3
            "content": "A " * 10,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_entry_validation_content_too_short(client, monkeypatch):
    monkeypatch.setattr(knowledge_api, "KNOWLEDGE_BASE_AVAILABLE", True)

    resp = await client.post(
        "/api/v1/knowledge-base/entries",
        headers=HEADERS,
        json={
            "title": "Valid title",
            "content": "short",  # min_length=10
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_validation_query_too_short(client, monkeypatch):
    monkeypatch.setattr(knowledge_api, "KNOWLEDGE_BASE_AVAILABLE", True)

    resp = await client.post(
        "/api/v1/knowledge-base/search",
        headers=HEADERS,
        json={"query": "x"},  # min_length=2
    )
    assert resp.status_code == 422
