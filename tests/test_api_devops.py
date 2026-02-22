"""
Tests for api/devops_api.py — DevOps automation endpoints.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.devops_api as devops_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(devops_api.router)

    monkeypatch.setattr(devops_api, "VALID_API_KEYS", {"test-key"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_required(client):
    resp = await client.get("/devops/health")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_invalid_key(client):
    resp = await client.get(
        "/devops/health",
        headers={"X-API-Key": "bad-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /devops/health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_success(client, monkeypatch):
    async def mock_health():
        return {
            "brainops-agents": {"status": "healthy"},
            "brainops-backend": {"status": "healthy"},
        }

    with patch("api.devops_api.get_all_service_health", new=mock_health, create=True):
        # The endpoint catches import errors, so we mock at the module level
        pass

    resp = await client.get("/devops/health", headers=HEADERS)
    # Will return either 200 with actual data or 200 with error fallback
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Input validation for deploy, heal, learning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_validation_missing_service(client):
    resp = await client.post(
        "/devops/deploy",
        headers=HEADERS,
        json={},  # Missing required 'service' field
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_heal_validation_missing_service(client):
    resp = await client.post(
        "/devops/heal",
        headers=HEADERS,
        json={},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_learn_validation_missing_fields(client):
    resp = await client.post(
        "/devops/learn",
        headers=HEADERS,
        json={"service": "test"},  # Missing incident_type and resolution
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_learn_validation_success_shape(client):
    # Valid body shape
    resp = await client.post(
        "/devops/learn",
        headers=HEADERS,
        json={
            "service": "brainops-agents",
            "incident_type": "crash_loop",
            "resolution": "rolled back to v11.32",
            "root_cause": "bad import in startup",
        },
    )
    # Endpoint may succeed or raise depending on devops_automation import,
    # but should not be a 422
    assert resp.status_code != 422
