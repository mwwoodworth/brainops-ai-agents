"""
Tests for api/ai_observability_api.py — AI Observability metrics and events.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.ai_observability_api as obs_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(obs_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/v1/ai/metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_unavailable(client, monkeypatch):
    monkeypatch.setattr(obs_api, "_observability", None)
    monkeypatch.setattr(obs_api, "get_observability", lambda: None)

    resp = await client.get("/api/v1/ai/metrics")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_metrics_success(client, monkeypatch):
    fake_obs = SimpleNamespace(
        get_prometheus_metrics=MagicMock(return_value="# HELP brainops_up\nbrainops_up 1\n")
    )
    monkeypatch.setattr(obs_api, "_observability", fake_obs)
    monkeypatch.setattr(obs_api, "get_observability", lambda: fake_obs)

    resp = await client.get("/api/v1/ai/metrics")
    assert resp.status_code == 200
    assert "brainops_up" in resp.text


# ---------------------------------------------------------------------------
# GET /api/v1/ai/metrics/json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_json_unavailable(client, monkeypatch):
    monkeypatch.setattr(obs_api, "_observability", None)
    monkeypatch.setattr(obs_api, "get_observability", lambda: None)

    resp = await client.get("/api/v1/ai/metrics/json")
    assert resp.status_code == 200  # Returns error dict, not 503
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_metrics_json_success(client, monkeypatch):
    fake_metrics = SimpleNamespace(
        get_all_metrics=MagicMock(
            return_value={
                "agent_executions_total": 42,
                "memory_entries": 1500,
            }
        )
    )
    fake_obs = SimpleNamespace(metrics=fake_metrics)
    monkeypatch.setattr(obs_api, "_observability", fake_obs)
    monkeypatch.setattr(obs_api, "get_observability", lambda: fake_obs)

    resp = await client.get("/api/v1/ai/metrics/json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_executions_total"] == 42
