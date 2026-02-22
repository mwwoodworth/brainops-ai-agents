"""
Tests for api/learning.py — Learning Feedback Loop API.
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

import api.learning as learning_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(learning_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestApprovalRequest:
    def test_defaults(self):
        req = learning_api.ApprovalRequest()
        assert req.approver == "human"
        assert req.notes is None

    def test_custom(self):
        req = learning_api.ApprovalRequest(approver="admin", notes="looks good")
        assert req.approver == "admin"


class TestRejectionRequest:
    def test_reason_required(self):
        with pytest.raises(Exception):
            learning_api.RejectionRequest()  # Missing 'reason'

    def test_valid(self):
        req = learning_api.RejectionRequest(reason="too risky")
        assert req.reason == "too risky"


class TestRunCycleRequest:
    def test_defaults(self):
        req = learning_api.RunCycleRequest()
        assert req.analysis_window_hours == 24

    def test_custom(self):
        req = learning_api.RunCycleRequest(analysis_window_hours=48)
        assert req.analysis_window_hours == 48


# ---------------------------------------------------------------------------
# GET /api/learning/pending-proposals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_proposals_success(client, monkeypatch):
    fake_loop = SimpleNamespace(
        get_pending_proposals=AsyncMock(
            return_value=[
                {"id": "p-1", "status": "proposed", "description": "Optimize query"},
            ]
        ),
    )
    monkeypatch.setattr(learning_api, "_feedback_loop", fake_loop)

    async def get_loop():
        return fake_loop

    monkeypatch.setattr(learning_api, "get_loop", get_loop)

    resp = await client.get("/api/learning/pending-proposals")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pending_proposals_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(learning_api, "_feedback_loop", None)

    async def fail_loop():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    monkeypatch.setattr(learning_api, "get_loop", fail_loop)

    resp = await client.get("/api/learning/pending-proposals")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/learning/status — queries DB directly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_success(client, monkeypatch):
    """Status endpoint queries the database directly."""
    from enum import Enum

    class FakeRiskLevel(Enum):
        LOW = "low"

    class FakeImprovementType(Enum):
        AGENT_CONFIG = "agent_config"

    class FakePool:
        async def fetchrow(self, q, *a):
            if "ai_improvement_proposals" in q:
                return {
                    "total_proposals": 10,
                    "pending": 3,
                    "approved": 5,
                    "implementing": 0,
                    "queued_for_self_build": 0,
                    "completed": 2,
                    "self_build_completed": 0,
                    "pr_opened": 0,
                    "rejected": 0,
                    "auto_approved_count": 1,
                    "latest_proposal": None,
                }
            if "ai_learning_patterns" in q:
                return {
                    "total_patterns": 5,
                    "pattern_types": 3,
                    "latest_pattern": None,
                }
            if "ai_learning_insights" in q:
                return None
            return None

        async def fetch(self, q, *a):
            return []

    fake_loop = SimpleNamespace(
        analysis_window_hours=24,
        min_pattern_confidence=0.7,
        min_occurrence_count=3,
        auto_approve_risk_levels=[FakeRiskLevel.LOW],
        auto_approve_types=[FakeImprovementType.AGENT_CONFIG],
    )
    monkeypatch.setattr(learning_api, "_feedback_loop", fake_loop)

    async def get_loop_mock():
        return fake_loop

    monkeypatch.setattr(learning_api, "get_loop", get_loop_mock)

    with patch("database.async_connection.get_pool", return_value=FakePool()):
        resp = await client.get("/api/learning/status")
    assert resp.status_code == 200
