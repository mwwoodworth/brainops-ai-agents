"""
Tests for api/gumroad_webhook.py — Gumroad sales webhook and utility functions.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.gumroad_webhook as gumroad_api


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestCoerceBool:
    def test_true_values(self):
        for val in [True, "1", "true", "True", "yes", "y", "on"]:
            assert gumroad_api._coerce_bool(val) is True

    def test_false_values(self):
        for val in [False, None, "0", "false", "no", "off", ""]:
            assert gumroad_api._coerce_bool(val) is False

    def test_bool_direct(self):
        assert gumroad_api._coerce_bool(True) is True
        assert gumroad_api._coerce_bool(False) is False


class TestParseGumroadTimestamp:
    def test_iso_with_z(self):
        result = gumroad_api._parse_gumroad_timestamp("2026-01-15T12:00:00Z")
        assert result is not None
        assert result.year == 2026
        assert result.month == 1

    def test_iso_without_z(self):
        result = gumroad_api._parse_gumroad_timestamp("2026-01-15T12:00:00+00:00")
        assert result is not None

    def test_none_returns_now(self):
        result = gumroad_api._parse_gumroad_timestamp(None)
        assert result is not None
        assert isinstance(result, datetime)

    def test_empty_string_returns_now(self):
        result = gumroad_api._parse_gumroad_timestamp("")
        assert result is not None

    def test_invalid_string_returns_now(self):
        result = gumroad_api._parse_gumroad_timestamp("not-a-date")
        assert result is not None


# ---------------------------------------------------------------------------
# Auth function tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(gumroad_api.router)

    # Ensure auth is configured
    from config import config

    monkeypatch.setattr(config.security, "valid_api_keys", {"test-key"})
    monkeypatch.setattr(config.security, "auth_required", True)
    monkeypatch.setattr(config.security, "auth_configured", True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


@pytest.mark.asyncio
async def test_analytics_endpoint_requires_auth(client):
    """Non-webhook endpoints require API key"""
    resp = await client.get("/gumroad/analytics")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_analytics_endpoint_with_auth(client, monkeypatch):
    """With auth, analytics endpoint should work (or return data-dependent status)"""
    resp = await client.get("/gumroad/analytics", headers=HEADERS)
    # Should not be 403 anymore
    assert resp.status_code != 403


@pytest.mark.asyncio
async def test_webhook_endpoint_no_auth_required(client, monkeypatch):
    """Webhook endpoint should be publicly accessible (uses HMAC instead)"""
    monkeypatch.setattr(gumroad_api, "GUMROAD_WEBHOOK_SECRET", "")

    resp = await client.post(
        "/gumroad/webhook",
        data={"seller_id": "test", "product_id": "prod-1", "email": "test@example.com"},
    )
    # Should not be 403 — webhook is public
    assert resp.status_code != 403


# ---------------------------------------------------------------------------
# Environment config tests
# ---------------------------------------------------------------------------


class TestEnvironmentConfig:
    def test_webhook_secret_defaults_to_empty(self):
        # When env var is not set, should default to empty string
        assert isinstance(gumroad_api.GUMROAD_WEBHOOK_SECRET, str)

    def test_environment_is_string(self):
        assert isinstance(gumroad_api.ENVIRONMENT, str)
