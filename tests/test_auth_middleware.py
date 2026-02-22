"""
Tests for authentication middleware and API key verification across the app.

Uses the shared conftest.py fixtures: client, auth_headers, configure_test_security, patch_pool.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as ai_app


# ---------------------------------------------------------------------------
# API Key enforcement
# ---------------------------------------------------------------------------


async def test_missing_api_key_returns_403(client):
    """Endpoints that require auth should reject requests without X-API-Key."""
    resp = await client.get("/agents/status")
    assert resp.status_code == 403
    data = resp.json()
    assert "authentication" in data["detail"].lower() or "api key" in data["detail"].lower()


async def test_invalid_api_key_returns_403(client):
    """Wrong API key should be rejected."""
    resp = await client.get("/agents/status", headers={"X-API-Key": "completely-wrong"})
    assert resp.status_code == 403


async def test_valid_api_key_passes_auth(client, auth_headers, patch_pool, monkeypatch):
    """Valid API key should pass authentication check."""
    monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)

    def fetch_handler(query, *args):
        if "FROM ai_agents" in query:
            return []
        return []

    patch_pool.fetch_handler = fetch_handler

    resp = await client.get("/agents/status", headers=auth_headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------


async def test_healthz_no_auth_required(client):
    """Quick health check should not require authentication."""
    resp = await client.get("/healthz")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limit category middleware
# ---------------------------------------------------------------------------


async def test_rate_limit_allows_requests(client, auth_headers, patch_pool, monkeypatch):
    """Rate limiter is disabled in test mode; requests should pass."""
    monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)
    patch_pool.fetch_handler = lambda q, *a: []

    resp = await client.get("/agents/status", headers=auth_headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error response format
# ---------------------------------------------------------------------------


async def test_error_response_has_detail(client):
    """Error responses should include a 'detail' field."""
    resp = await client.get("/agents/status")  # No auth
    assert resp.status_code == 403
    data = resp.json()
    assert "detail" in data


async def test_404_for_nonexistent_route(client, auth_headers):
    """Non-existent routes should return 404."""
    resp = await client.get("/this/route/does/not/exist", headers=auth_headers)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Content-Type handling
# ---------------------------------------------------------------------------


async def test_json_content_type_accepted(client, auth_headers, patch_pool, monkeypatch):
    """JSON request bodies should be properly parsed."""
    monkeypatch.setattr(ai_app, "AGENTS_AVAILABLE", True)

    class FakeExecutor:
        async def execute(self, agent_name, task):
            return {"status": "completed"}

    monkeypatch.setattr(ai_app, "AGENT_EXECUTOR", FakeExecutor())

    patch_pool.fetchrow_handler = lambda query, *a: (
        {"id": "test-id", "name": "test-agent", "type": "test"} if "FROM agents" in query else None
    )

    resp = await client.post(
        "/agents/execute",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={"agent_type": "test", "task": "do something", "parameters": {}},
    )
    assert resp.status_code == 200


async def test_wrong_content_type_returns_422(client, auth_headers):
    """Non-JSON content type for JSON-expecting endpoint should fail."""
    resp = await client.post(
        "/agents/execute",
        headers={**auth_headers, "Content-Type": "text/plain"},
        content="not json",
    )
    assert resp.status_code == 422
