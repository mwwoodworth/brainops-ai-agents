import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_activation_system import BusinessEventType
from drives.revenue_drive import RevenueDrive


class DummyActivationSystem:
    def __init__(self):
        self.calls = []

    async def handle_business_event(self, event_type, event_data):
        self.calls.append((event_type, event_data))
        return {"success": True}


@pytest.mark.asyncio
async def test_revenue_drive_triggers_events(monkeypatch):
    activation = DummyActivationSystem()
    drive = RevenueDrive(tenant_id="test-tenant", activation_system=activation)
    drive.dry_run = True

    monkeypatch.setattr(drive, "_fetch_stale_leads", lambda: [{"entity_id": "lead-1"}])
    monkeypatch.setattr(drive, "_fetch_overdue_invoices", lambda: [{"entity_id": "inv-1"}])
    monkeypatch.setattr(drive, "_fetch_upsell_candidates", lambda: [{"entity_id": "lead-2"}])
    monkeypatch.setattr(drive, "_recent_task_exists", lambda *_: False)

    result = await drive.run_async()

    assert result["stale_leads"] == 1
    assert result["overdue_invoices"] == 1
    assert result["upsell_candidates"] == 1
    assert len(activation.calls) == 3

    event_types = [call[0] for call in activation.calls]
    assert BusinessEventType.REVENUE_OPPORTUNITY in event_types
    assert BusinessEventType.INVOICE_OVERDUE in event_types
