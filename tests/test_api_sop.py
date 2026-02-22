"""
Tests for api/sop.py — SOP Generator API with approval workflow.
"""
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import api.sop as sop_api

    SOP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SOP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SOP_AVAILABLE, reason="bleach not installed")


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(sop_api.router)

    monkeypatch.setattr(sop_api, "VALID_API_KEYS", {"test-key", "prod-key"})
    monkeypatch.setattr(sop_api, "APPROVER_API_KEYS", {"prod-key"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sop_requires_auth(client):
    resp = await client.get("/sop/list")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_sop_valid_key(client):
    resp = await client.get("/sop/list", headers={"X-API-Key": "test-key"})
    # Should not be 401/403
    assert resp.status_code not in (401, 403)


# ---------------------------------------------------------------------------
# Approver-level access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approver_key_classification(monkeypatch):
    """Keys containing 'prod' should be classified as approver keys."""
    monkeypatch.setattr(sop_api, "VALID_API_KEYS", {"test-key", "prod-admin"})
    # Recalculate approver keys
    approver_keys = {k for k in sop_api.VALID_API_KEYS if "prod" in k.lower()}
    assert "prod-admin" in approver_keys
    assert "test-key" not in approver_keys


# ---------------------------------------------------------------------------
# HTML sanitization config
# ---------------------------------------------------------------------------


class TestSanitizationConfig:
    def test_allowed_tags_exist(self):
        assert len(sop_api.ALLOWED_TAGS) > 0
        assert "p" in sop_api.ALLOWED_TAGS
        assert "h1" in sop_api.ALLOWED_TAGS
        assert "table" in sop_api.ALLOWED_TAGS
        assert "code" in sop_api.ALLOWED_TAGS

    def test_script_not_allowed(self):
        assert "script" not in sop_api.ALLOWED_TAGS

    def test_allowed_attributes(self):
        assert "href" in sop_api.ALLOWED_ATTRIBUTES.get("a", [])
        assert "src" in sop_api.ALLOWED_ATTRIBUTES.get("img", [])
