"""
Tests for api/self_healing.py — Self-Healing system API endpoints.
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

import api.self_healing as self_healing_api


def _make_fake_engine(initialized=True, incidents=None):
    engine = SimpleNamespace(
        _initialized=initialized,
        incidents=incidents or {},
        detect_anomaly=AsyncMock(
            return_value={
                "anomaly_detected": True,
                "severity": "medium",
                "incident_id": "inc-001",
            }
        ),
        get_all_incidents=MagicMock(return_value=[]),
        initialize=AsyncMock(),
    )
    return engine


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(self_healing_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/v1/self-healing/status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_success(client, monkeypatch):
    engine = _make_fake_engine()
    monkeypatch.setattr(self_healing_api, "_engine", engine)

    resp = await client.get("/api/v1/self-healing/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "capabilities" in data
    assert "anomaly_detection" in data["capabilities"]


@pytest.mark.asyncio
async def test_status_engine_unavailable(client, monkeypatch):
    monkeypatch.setattr(self_healing_api, "_engine", None)

    def fail_engine():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Self-Healing Engine not available")

    monkeypatch.setattr(self_healing_api, "_get_engine", fail_engine)

    resp = await client.get("/api/v1/self-healing/status")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_status_reports_active_incidents(client, monkeypatch):
    engine = _make_fake_engine(incidents={"inc-1": {}, "inc-2": {}})
    monkeypatch.setattr(self_healing_api, "_engine", engine)

    resp = await client.get("/api/v1/self-healing/status")
    assert resp.status_code == 200
    assert resp.json()["active_incidents"] == 2


# ---------------------------------------------------------------------------
# POST /api/v1/self-healing/detect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_anomaly_success(client, monkeypatch):
    engine = _make_fake_engine()
    monkeypatch.setattr(self_healing_api, "_engine", engine)

    resp = await client.post(
        "/api/v1/self-healing/detect",
        json={
            "system_id": "brainops-agents",
            "metrics": {"cpu": 92.0, "memory": 85.0, "error_rate": 0.15},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly_detected"] is True
    assert data["incident_id"] == "inc-001"


@pytest.mark.asyncio
async def test_detect_anomaly_with_context(client, monkeypatch):
    engine = _make_fake_engine()
    monkeypatch.setattr(self_healing_api, "_engine", engine)

    resp = await client.post(
        "/api/v1/self-healing/detect",
        json={
            "system_id": "db-primary",
            "metrics": {"connections": 150.0},
            "context": {"region": "us-east-2"},
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_detect_anomaly_validation_error(client, monkeypatch):
    engine = _make_fake_engine()
    monkeypatch.setattr(self_healing_api, "_engine", engine)

    resp = await client.post(
        "/api/v1/self-healing/detect",
        json={"metrics": {"cpu": 90.0}},  # Missing system_id
    )
    assert resp.status_code == 422
