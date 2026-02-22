"""
Tests for api/infrastructure.py — Render service scaling and restart endpoints.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.infrastructure as infra_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(infra_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# POST /infrastructure/scale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scale_missing_render_key(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", None)
    resp = await client.post(
        "/infrastructure/scale",
        json={"service_id": "srv-123", "num_instances": 2},
    )
    assert resp.status_code == 503
    assert "RENDER_API_KEY" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_scale_success(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", "rnd_fake")

    mock_response = SimpleNamespace(
        status_code=200,
        json=lambda: {"numInstances": 2},
        text="OK",
    )

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("api.infrastructure.httpx.AsyncClient", return_value=mock_client_instance):
        resp = await client.post(
            "/infrastructure/scale",
            json={"service_id": "srv-123", "num_instances": 2},
        )

    assert resp.status_code == 200
    assert resp.json()["numInstances"] == 2


@pytest.mark.asyncio
async def test_scale_render_error_propagates(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", "rnd_fake")

    mock_response = AsyncMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("api.infrastructure.httpx.AsyncClient", return_value=mock_client_instance):
        resp = await client.post(
            "/infrastructure/scale",
            json={"service_id": "srv-123", "num_instances": 1},
        )

    assert resp.status_code == 500  # Exception caught by outer handler


@pytest.mark.asyncio
async def test_scale_validation_error(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", "rnd_fake")
    resp = await client.post(
        "/infrastructure/scale",
        json={"service_id": "srv-123"},  # Missing num_instances
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /infrastructure/restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_missing_render_key(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", None)
    resp = await client.post(
        "/infrastructure/restart",
        json={"service_id": "srv-123"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_restart_success(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", "rnd_fake")

    mock_response = AsyncMock()
    mock_response.status_code = 202

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("api.infrastructure.httpx.AsyncClient", return_value=mock_client_instance):
        resp = await client.post(
            "/infrastructure/restart",
            json={"service_id": "srv-123"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "restart_initiated"
    assert data["service_id"] == "srv-123"


@pytest.mark.asyncio
async def test_restart_validation_error(client, monkeypatch):
    monkeypatch.setattr(infra_api, "RENDER_API_KEY", "rnd_fake")
    resp = await client.post(
        "/infrastructure/restart",
        json={},  # Missing service_id
    )
    assert resp.status_code == 422
