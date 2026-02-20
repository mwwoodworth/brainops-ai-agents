import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revenue_automation_engine import (  # noqa: E402
    Industry,
    Lead,
    LeadSource,
    LeadStatus,
    RevenueAutomationEngine,
    RevenueTransaction,
)


def _lead(lead_id: str) -> Lead:
    now = datetime.utcnow().isoformat()
    return Lead(
        lead_id=lead_id,
        email=f"{lead_id}@example.com",
        phone=None,
        name=f"Lead {lead_id}",
        company="Example Co",
        industry=Industry.GENERIC,
        source=LeadSource.WEBSITE,
        status=LeadStatus.NEW,
        score=50,
        estimated_value=Decimal("1000"),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_get_lead_uses_bounded_cache(monkeypatch):
    engine = RevenueAutomationEngine()
    engine._lead_cache_limit = 2

    async def _fake_fetch(lead_id: str):
        return _lead(lead_id)

    monkeypatch.setattr(engine, "_fetch_lead_from_db", _fake_fetch)

    await engine.get_lead("lead-1")
    await engine.get_lead("lead-2")
    await engine.get_lead("lead-3")

    assert list(engine.leads.keys()) == ["lead-2", "lead-3"]


@pytest.mark.asyncio
async def test_load_from_db_only_loads_aggregates_and_sequences(monkeypatch):
    engine = RevenueAutomationEngine()
    engine._db_url = "postgres://example"

    class FakeConnection:
        async def fetchrow(self, query: str):
            if "FROM revenue_leads" in query:
                return {"pipeline_value": Decimal("12000")}
            if "FROM revenue_transactions" in query:
                return {
                    "total_revenue": Decimal("45000"),
                    "monthly_revenue": Decimal("9000"),
                }
            return {}

        async def fetch(self, _query: str):
            return [
                {
                    "sequence_id": "seq-db-1",
                    "name": "DB Sequence",
                    "industry": "generic",
                    "trigger": "new_lead",
                    "steps": [],
                    "active": True,
                    "success_rate": 0.21,
                    "total_sent": 10,
                    "conversions": 2,
                }
            ]

        async def close(self):
            return None

    async def _fake_connect(_url: str):
        return FakeConnection()

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(connect=_fake_connect))

    await engine._load_from_db()

    assert len(engine.leads) == 0
    assert len(engine.transactions) == 0
    assert engine.pipeline_value == Decimal("12000")
    assert engine.total_revenue == Decimal("45000")
    assert engine.monthly_revenue == Decimal("9000")
    assert "seq-db-1" in engine.sequences


@pytest.mark.asyncio
async def test_process_payment_webhook_fetches_transaction_from_db_when_cache_miss(monkeypatch):
    engine = RevenueAutomationEngine()
    tx = RevenueTransaction(
        transaction_id="tx-123",
        lead_id="lead-123",
        amount=Decimal("2500"),
        currency="USD",
        status="pending",
        payment_method="stripe",
        processor_id=None,
        created_at=datetime.utcnow().isoformat(),
        completed_at=None,
        industry=Industry.GENERIC,
        product_service="roof repair",
    )

    async def _fake_fetch_transaction(_tx_id: str):
        return tx

    async def _fake_get_lead(_lead_id: str):
        return None

    async def _fake_persist_transaction(_tx: RevenueTransaction):
        return None

    monkeypatch.setattr(engine, "_fetch_transaction_from_db", _fake_fetch_transaction)
    monkeypatch.setattr(engine, "get_lead", _fake_get_lead)
    monkeypatch.setattr(engine, "_persist_transaction", _fake_persist_transaction)

    result = await engine.process_payment_webhook(
        {
            "type": "payment_intent.succeeded",
            "id": "pi_123",
            "metadata": {"transaction_id": "tx-123"},
        }
    )

    assert result["status"] == "success"
    assert "tx-123" in engine.transactions
    assert engine.transactions["tx-123"].status == "completed"

