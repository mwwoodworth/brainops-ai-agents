"""
Tests for api/voice.py — Voice & Communications API (ElevenLabs, Twilio).
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

import api.voice as voice_api
from config import config


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(voice_api.router)

    # Configure test auth
    monkeypatch.setattr(config.security, "valid_api_keys", {"test-key"})
    monkeypatch.setattr(config.security, "auth_required", True)
    monkeypatch.setattr(config.security, "auth_configured", True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# POST /api/v1/voice/speak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speak_no_auth(client):
    resp = await client.post(
        "/api/v1/voice/speak",
        json={"text": "Hello world"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_speak_elevenlabs_not_configured(client, monkeypatch):
    monkeypatch.setattr(voice_api, "ELEVENLABS_API_KEY", None)
    resp = await client.post(
        "/api/v1/voice/speak",
        json={"text": "Hello world"},
        headers=HEADERS,
    )
    assert resp.status_code == 503
    assert "ElevenLabs" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_speak_validation_missing_text(client, monkeypatch):
    monkeypatch.setattr(voice_api, "ELEVENLABS_API_KEY", "fake-key")
    resp = await client.post(
        "/api/v1/voice/speak",
        json={},
        headers=HEADERS,
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Auth edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speak_invalid_api_key(client):
    resp = await client.post(
        "/api/v1/voice/speak",
        json={"text": "Hello"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401
