"""
Tests for api/email_capture.py — Email capture and lead generation.
"""
import sys
from pathlib import Path
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.email_capture as email_api


# ---------------------------------------------------------------------------
# _resolve_tenant_uuid tests
# ---------------------------------------------------------------------------


class TestResolveTenantUUID:
    def test_valid_uuid_from_env(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")
        result = email_api._resolve_tenant_uuid(None)
        assert result == "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"

    def test_invalid_uuid_returns_none(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TENANT_ID", "not-a-uuid")
        monkeypatch.delenv("TENANT_ID", raising=False)
        monkeypatch.delenv("APP_DEFAULT_TENANT_ID", raising=False)
        result = email_api._resolve_tenant_uuid(None)
        assert result is None

    def test_empty_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)
        monkeypatch.delenv("TENANT_ID", raising=False)
        monkeypatch.delenv("APP_DEFAULT_TENANT_ID", raising=False)
        result = email_api._resolve_tenant_uuid(None)
        assert result is None


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestEmailCaptureRequest:
    def test_valid_email(self):
        req = email_api.EmailCaptureRequest(email="test@example.com")
        assert req.email == "test@example.com"
        assert req.source == "landing_page"

    def test_invalid_email(self):
        with pytest.raises(Exception):
            email_api.EmailCaptureRequest(email="not-an-email")

    def test_with_utm(self):
        req = email_api.EmailCaptureRequest(
            email="lead@test.com",
            utm_source="google",
            utm_medium="cpc",
            utm_campaign="roofing",
        )
        assert req.utm_source == "google"

    def test_with_name(self):
        req = email_api.EmailCaptureRequest(
            email="lead@test.com",
            name="Jordan Smith",
        )
        assert req.name == "Jordan Smith"


class TestEmailCaptureResponse:
    def test_success(self):
        resp = email_api.EmailCaptureResponse(success=True, message="Captured")
        assert resp.success is True
        assert resp.message == "Captured"

    def test_failure(self):
        resp = email_api.EmailCaptureResponse(success=False, message="Duplicate")
        assert resp.success is False


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    app.include_router(email_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_capture_validation_invalid_email(client):
    resp = await client.post(
        "/email/capture",
        json={"email": "not-valid"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_capture_validation_missing_email(client):
    resp = await client.post(
        "/email/capture",
        json={},
    )
    assert resp.status_code == 422
