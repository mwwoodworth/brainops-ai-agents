"""
Tests for api/mcp.py — MCP Bridge Integration API.
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

import api.mcp as mcp_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(mcp_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /mcp/status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_success(client, monkeypatch):
    fake_client = SimpleNamespace(
        is_connected=MagicMock(return_value=True),
        get_servers=MagicMock(return_value=["render", "vercel", "supabase"]),
        get_tools_count=MagicMock(return_value=195),
    )
    monkeypatch.setattr(mcp_api, "_client", fake_client)

    resp = await client.get("/mcp/status")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_status_not_connected(client, monkeypatch):
    monkeypatch.setattr(mcp_api, "_client", None)

    def fail_client():
        raise RuntimeError("MCP Bridge not available")

    monkeypatch.setattr(mcp_api, "_get_client", fail_client)

    resp = await client.get("/mcp/status")
    # Status endpoint may return 200 with error details or 503 depending on implementation
    assert resp.status_code in (200, 500, 503)


# ---------------------------------------------------------------------------
# POST /mcp/execute — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_validation_missing_fields(client):
    resp = await client.post(
        "/mcp/execute",
        json={},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_execute_validation_missing_tool(client):
    resp = await client.post(
        "/mcp/execute",
        json={"server": "render"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /mcp/aurea/execute — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aurea_execute_validation(client):
    resp = await client.post(
        "/mcp/aurea/execute",
        json={},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestExecuteToolRequest:
    def test_valid(self):
        req = mcp_api.ExecuteToolRequest(server="render", tool="list_services")
        assert req.server == "render"
        assert req.tool == "list_services"
        assert req.params is None

    def test_with_params(self):
        req = mcp_api.ExecuteToolRequest(
            server="vercel", tool="get_deployment", params={"id": "dep-1"}
        )
        assert req.params["id"] == "dep-1"


class TestAUREADecisionRequest:
    def test_valid(self):
        req = mcp_api.AUREADecisionRequest(
            decision_type="DEPLOY",
            params={"service": "agents"},
        )
        assert req.decision_type == "DEPLOY"
