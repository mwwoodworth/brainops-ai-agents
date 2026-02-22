"""
Tests for api/workflows.py — Advanced Workflow Engine API.
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

import api.workflows as workflows_api


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestWorkflowStartRequest:
    def test_valid_minimal(self):
        req = workflows_api.WorkflowStartRequest(workflow_type="test")
        assert req.workflow_type == "test"
        assert req.initial_state == {}
        assert req.timeout_seconds == 300

    def test_valid_full(self):
        req = workflows_api.WorkflowStartRequest(
            workflow_type="customer_onboarding",
            initial_state={"customer_id": "cust-1"},
            tenant_id="t-1",
            timeout_seconds=600,
        )
        assert req.tenant_id == "t-1"
        assert req.timeout_seconds == 600

    def test_timeout_min_boundary(self):
        req = workflows_api.WorkflowStartRequest(
            workflow_type="test",
            timeout_seconds=10,
        )
        assert req.timeout_seconds == 10

    def test_timeout_max_boundary(self):
        req = workflows_api.WorkflowStartRequest(
            workflow_type="test",
            timeout_seconds=3600,
        )
        assert req.timeout_seconds == 3600

    def test_timeout_below_min_raises(self):
        with pytest.raises(Exception):
            workflows_api.WorkflowStartRequest(
                workflow_type="test",
                timeout_seconds=5,  # Below 10
            )

    def test_timeout_above_max_raises(self):
        with pytest.raises(Exception):
            workflows_api.WorkflowStartRequest(
                workflow_type="test",
                timeout_seconds=7200,  # Above 3600
            )


class TestWorkflowResumeRequest:
    def test_valid(self):
        req = workflows_api.WorkflowResumeRequest(workflow_id="wf-123")
        assert req.workflow_id == "wf-123"
        assert req.additional_state is None

    def test_with_additional_state(self):
        req = workflows_api.WorkflowResumeRequest(
            workflow_id="wf-123",
            additional_state={"decision": "approve"},
        )
        assert req.additional_state["decision"] == "approve"


class TestApprovalSubmitRequest:
    def test_valid(self):
        req = workflows_api.ApprovalSubmitRequest(
            request_id="req-1",
            response="approve",
            responded_by="admin",
        )
        assert req.request_id == "req-1"


# ---------------------------------------------------------------------------
# POST /workflows/start — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_validation_missing_type(client):
    resp = await client.post(
        "/workflows/start",
        json={},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_start_validation_invalid_timeout(client):
    resp = await client.post(
        "/workflows/start",
        json={"workflow_type": "test", "timeout_seconds": 1},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /workflows/resume — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_validation_missing_id(client):
    resp = await client.post(
        "/workflows/resume",
        json={},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /workflows/approve — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_validation_missing_fields(client):
    resp = await client.post(
        "/workflows/approvals/submit",
        json={"request_id": "r-1"},
    )
    assert resp.status_code == 422
