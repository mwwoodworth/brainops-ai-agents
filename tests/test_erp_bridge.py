
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI

from erp_event_bridge import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)

@patch("erp_event_bridge.CustomerSuccessAgent")
@patch("erp_event_bridge.get_revenue_system")
@patch("erp_event_bridge.get_intelligent_followup_system")
def test_handle_erp_event(mock_followup, mock_revenue, mock_csa):
    # Mock systems
    mock_csa_instance = MagicMock()
    mock_csa_instance.generate_onboarding_plan = AsyncMock()
    mock_csa.return_value = mock_csa_instance

    mock_revenue_instance = MagicMock()
    mock_revenue.return_value = mock_revenue_instance

    mock_followup_instance = MagicMock()
    mock_followup_instance.create_followup_sequence = AsyncMock()
    mock_followup.return_value = mock_followup_instance

    # Test NEW_CUSTOMER
    payload = {
        "id": "evt_123",
        "type": "NEW_CUSTOMER",
        "payload": {"id": "cust_123", "name": "Test Customer"},
        "tenant_id": "tenant_123",
        "timestamp": datetime.utcnow().isoformat()
    }

    response = client.post("/events/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    assert data["event_id"] == "evt_123"

    # Verify CustomerSuccessAgent was initialized and called
    mock_csa.assert_called_with("tenant_123")

    # Test JOB_COMPLETED
    payload_job = {
        "id": "evt_456",
        "type": "JOB_COMPLETED",
        "payload": {"id": "job_123", "customer_id": "cust_123", "job_type": "repair"},
        "tenant_id": "tenant_123",
        "timestamp": datetime.utcnow().isoformat()
    }

    response = client.post("/events/webhook", json=payload_job)
    assert response.status_code == 200

    # Verify Followup System called
    mock_followup.assert_called()
