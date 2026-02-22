"""
Tests for api/content_revenue.py, api/payments.py, and api/sync.py
— Content orchestration, payment capture, and memory sync endpoints.
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

import api.content_revenue as content_api
import api.payments as payments_api
import api.sync as sync_api


# ===========================================================================
# api/content_revenue.py
# ===========================================================================


@pytest_asyncio.fixture
async def content_client(monkeypatch):
    app = FastAPI()
    app.include_router(content_api.router)

    # Mock pool
    class FakePool:
        async def fetch(self, q, *a):
            return []

        async def fetchrow(self, q, *a):
            return None

        async def fetchval(self, q, *a):
            return 0

        async def execute(self, q, *a):
            return "OK"

    monkeypatch.setattr(content_api, "get_pool", lambda: FakePool())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestContentRevenue:
    @pytest.mark.asyncio
    async def test_logs_recent(self, content_client):
        resp = await content_client.get("/logs/recent")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_content_orchestrator_unavailable(self, content_client, monkeypatch):
        monkeypatch.setattr(content_api, "CONTENT_ORCHESTRATOR_AVAILABLE", False)

        resp = await content_client.post(
            "/content/generate",
            json={"content_type": "blog_post", "topic": "AI in roofing"},
        )
        assert resp.status_code in (404, 422, 503)

    @pytest.mark.asyncio
    async def test_log_buffer_exists(self):
        assert hasattr(content_api, "LOG_BUFFER")
        assert hasattr(content_api.LOG_BUFFER, "maxlen")
        assert content_api.LOG_BUFFER.maxlen == 500


# ===========================================================================
# api/payments.py — Pydantic models
# ===========================================================================


class TestPaymentsModels:
    def test_mark_paid_request(self):
        req = payments_api.MarkPaidRequest(verified_by="admin")
        assert req.payment_method == "manual"
        assert req.payment_reference is None
        assert req.verified_by == "admin"

    def test_capture_payment_request_defaults(self):
        req = payments_api.CapturePaymentRequest()
        assert req.amount is None
        assert req.payment_method == "manual"
        assert req.verified_by == "system"

    def test_capture_payment_request_custom(self):
        req = payments_api.CapturePaymentRequest(
            amount=1500.50,
            payment_method="stripe",
            payment_reference="pi_12345",
            verified_by="admin",
        )
        assert req.amount == 1500.50
        assert req.payment_method == "stripe"

    def test_payment_plan_request(self):
        req = payments_api.PaymentPlanRequest(
            installment_count=3,
            interval_days=30,
        )
        assert req.installment_count == 3
        assert req.interval_days == 30
        assert req.configured_by == "system"

    def test_payment_plan_request_custom(self):
        req = payments_api.PaymentPlanRequest(
            installment_count=6,
            interval_days=14,
            min_installment_amount=500.0,
            configured_by="admin",
        )
        assert req.min_installment_amount == 500.0

    def test_payments_contact_email(self):
        assert isinstance(payments_api.PAYMENTS_CONTACT_EMAIL, str)
        assert len(payments_api.PAYMENTS_CONTACT_EMAIL) > 0


# ===========================================================================
# api/sync.py — Pydantic models and migration status
# ===========================================================================


class TestSyncModels:
    def test_migration_request_defaults(self):
        req = sync_api.MigrationRequest()
        assert "memories" in req.tables
        assert req.batch_size == 100
        assert req.generate_embeddings is True
        assert req.dry_run is False

    def test_migration_request_custom(self):
        req = sync_api.MigrationRequest(
            tables=["memories"],
            batch_size=50,
            generate_embeddings=False,
            dry_run=True,
            limit_per_table=1000,
        )
        assert req.tables == ["memories"]
        assert req.batch_size == 50
        assert req.dry_run is True

    def test_migration_request_batch_size_validation(self):
        with pytest.raises(Exception):
            sync_api.MigrationRequest(batch_size=5)  # Below min 10

        with pytest.raises(Exception):
            sync_api.MigrationRequest(batch_size=5000)  # Above max 1000

    def test_migration_status_response_model(self):
        resp = sync_api.MigrationStatus(
            running=False,
            current_table=None,
            progress={},
            errors=[],
            started_at=None,
            completed_at=None,
        )
        assert resp.running is False

    def test_migration_status_initial_state(self):
        status = sync_api._migration_status
        assert status["running"] is False
        assert status["current_table"] is None
        assert status["errors"] == []

    def test_default_tenant_id(self):
        assert isinstance(sync_api.DEFAULT_TENANT_ID, str)
        assert len(sync_api.DEFAULT_TENANT_ID) > 0

    def test_batch_size_constant(self):
        assert sync_api.BATCH_SIZE == 100

    def test_embedding_dimension(self):
        assert sync_api.EMBEDDING_DIMENSION == 1536
