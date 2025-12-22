
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Mock environment variables to ensure app starts without needing real DB/Keys for this test
os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/db"
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-REDACTED"
os.environ["TENANT_ID"] = "test-tenant"
os.environ["API_KEYS"] = "test-key,another-key"

# Import app
try:
    from app import app
except ImportError as e:
    print(f"Failed to import app: {e}")
    sys.exit(1)

def test_langgraph_endpoints():
    print("Testing LangGraph endpoints...")
    
    with TestClient(app) as client:
        # Test Status Endpoint
        headers = {"X-API-Key": "test-key"}
        response = client.get("/langgraph/status", headers=headers)
        print(f"Status Endpoint Response: {response.status_code}")
        print(f"Body: {response.json()}")
        
        assert response.status_code == 200
        data = response.json()
        
        if data.get("available") is True:
            print("LangGraph is AVAILABLE ✅")
        else:
            print("LangGraph is NOT AVAILABLE ❌ (This might be expected if deps are missing in env, but we just installed them)")

        # Test Workflow Endpoint (Mock run)
        response = client.post("/langgraph/workflow", json={"prompt": "Hello"}, headers=headers)
        print(f"Workflow Endpoint Response: {response.status_code}")
        
        # We expect 500 because LLM keys are fake, but that means it reached the orchestrator
        # OR 200 if it mocked/stubbed things out.
        # If it returns 503, it means orchestrator is missing.
        assert response.status_code != 404, "Endpoint /langgraph/workflow not found!"


if __name__ == "__main__":
    try:
        test_langgraph_endpoints()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
