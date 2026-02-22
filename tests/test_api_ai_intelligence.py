"""
Tests for api/ai_intelligence.py — True AI Intelligence API.
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

import api.ai_intelligence as ai_intel_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(ai_intel_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /intelligence/ — root
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_available(client, monkeypatch):
    monkeypatch.setattr(ai_intel_api, "AI_INTELLIGENCE_AVAILABLE", True)

    resp = await client.get("/intelligence/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert "anthropic" in data["models"]


@pytest.mark.asyncio
async def test_root_unavailable(client, monkeypatch):
    monkeypatch.setattr(ai_intel_api, "AI_INTELLIGENCE_AVAILABLE", False)

    resp = await client.get("/intelligence/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


# ---------------------------------------------------------------------------
# POST /intelligence/analyze — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_validation_missing_issue(client):
    resp = await client.post(
        "/intelligence/analyze",
        json={},  # Missing 'issue'
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(ai_intel_api, "AI_INTELLIGENCE_AVAILABLE", False)

    resp = await client.post(
        "/intelligence/analyze",
        json={"issue": "High memory usage on agents service"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_analyze_with_context(client, monkeypatch):
    monkeypatch.setattr(ai_intel_api, "AI_INTELLIGENCE_AVAILABLE", True)

    async def mock_analyze(issue, context=None, depth="standard"):
        return {
            "analysis": "Memory pressure from embedding cache",
            "confidence": 0.85,
            "recommendations": ["Reduce cache size", "Add memory limits"],
        }

    monkeypatch.setattr(ai_intel_api, "analyze_with_ai", mock_analyze)

    resp = await client.post(
        "/intelligence/analyze",
        json={
            "issue": "High memory on agents",
            "context": {"service": "brainops-agents"},
            "depth": "deep",
        },
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /intelligence/fix-plan — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fix_plan_validation_missing_issues(client):
    resp = await client.post(
        "/intelligence/fix-plan",
        json={},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_fix_plan_unavailable(client, monkeypatch):
    monkeypatch.setattr(ai_intel_api, "AI_INTELLIGENCE_AVAILABLE", False)

    resp = await client.post(
        "/intelligence/fix-plan",
        json={"issues": ["slow queries", "high error rate"]},
    )
    assert resp.status_code == 503
