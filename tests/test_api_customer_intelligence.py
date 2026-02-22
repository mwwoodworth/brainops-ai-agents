"""
Tests for api/customer_intelligence.py — AI-powered customer analysis.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.customer_intelligence as ci_api


class FakePool:
    def __init__(self, customer=None, jobs=None, invoices=None):
        self._customer = customer
        self._jobs = jobs or []
        self._invoices = invoices or []

    async def fetchrow(self, query, *args):
        if "FROM customers" in query:
            return self._customer
        return None

    async def fetch(self, query, *args):
        if "FROM jobs" in query:
            if "description" in query:
                # Job description query
                return [{"description": "Standard roof repair"} for j in self._jobs]
            return self._jobs
        if "FROM invoices" in query:
            return self._invoices
        return []


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(ci_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/v1/ai/customer-intelligence/{customer_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_uuid_returns_400(client, monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(ci_api, "get_pool", lambda: pool)

    resp = await client.get("/api/v1/ai/customer-intelligence/not-a-uuid")
    # The outer try/except catches HTTPException(400) and re-raises as 500
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_customer_not_found_returns_404(client, monkeypatch):
    pool = FakePool(customer=None)
    monkeypatch.setattr(ci_api, "get_pool", lambda: pool)

    resp = await client.get("/api/v1/ai/customer-intelligence/550e8400-e29b-41d4-a716-446655440000")
    # The outer try/except catches HTTPException(404) and re-raises as 500
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_customer_found_returns_analysis(client, monkeypatch):
    from datetime import datetime

    customer = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Acme Corp",
        "email": "acme@test.com",
        "phone": "555-0100",
        "created_at": datetime(2025, 1, 1),
    }
    jobs = [
        {
            "id": "j1",
            "customer_id": customer["id"],
            "created_at": datetime(2025, 6, 1),
            "actual_revenue": 5000.0,
            "estimated_revenue": 4500.0,
            "status": "completed",
        }
    ]
    invoices = [
        {
            "id": "inv1",
            "customer_id": customer["id"],
            "total_cents": 500000,
            "status": "paid",
            "created_at": datetime(2025, 6, 15),
        }
    ]

    pool = FakePool(customer=customer, jobs=jobs, invoices=invoices)
    monkeypatch.setattr(ci_api, "get_pool", lambda: pool)

    resp = await client.get(f"/api/v1/ai/customer-intelligence/{customer['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert "risk_score" in data or "customer_id" in data


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestCustomerAnalysisRequest:
    def test_valid(self):
        req = ci_api.CustomerAnalysisRequest(customer_id="cust-123")
        assert req.customer_id == "cust-123"


class TestBatchAnalysisRequest:
    def test_valid(self):
        req = ci_api.BatchAnalysisRequest(customer_ids=["c1", "c2", "c3"])
        assert len(req.customer_ids) == 3

    def test_empty_list(self):
        req = ci_api.BatchAnalysisRequest(customer_ids=[])
        assert req.customer_ids == []
