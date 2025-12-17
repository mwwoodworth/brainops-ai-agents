
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erp_event_bridge import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

@patch("erp_event_bridge.CustomerSuccessAgent")
@patch("erp_event_bridge.get_revenue_system")
@patch("erp_event_bridge.get_intelligent_followup_system")
def test_handle_erp_event(mock_followup, mock_revenue, mock_csa):
    # Mock systems
    mock_csa_instance = MagicMock()
    mock_csa.return_value = mock_csa_instance
    
    mock_revenue_instance = MagicMock()
    mock_revenue.return_value = mock_revenue_instance
    
    mock_followup_instance = MagicMock()
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
    assert response.json() == {"status": "processed", "event_id": "evt_123"}
    
    # Verify CustomerSuccessAgent was initialized and called
    mock_csa.assert_called_with("tenant_123")
    # asyncio.create_task is hard to mock directly without loop, but we can check if it attempted to call the method
    # Since we are not running async loop in TestClient in this simple way, the task creation might happen but not execution if not awaited.
    # But for unit test of the route logic, this confirms the path was taken.

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
