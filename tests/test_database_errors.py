import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as ai_app  # type: ignore
from database.async_connection import DatabaseUnavailableError


def _authorize():
    ai_app.config.security.valid_api_keys = {"test-key"}
    ai_app.config.security.auth_configured = True
    return {"X-API-Key": "test-key"}


def test_agents_returns_503_when_db_unavailable(monkeypatch):
    def _raise():
        raise DatabaseUnavailableError("db down")

    monkeypatch.setattr(ai_app, "get_pool", _raise)

    client = TestClient(ai_app.app)
    response = client.get("/agents", headers=_authorize())

    assert response.status_code == 503
    assert "db down" in response.json()["detail"]


def test_executions_returns_503_on_query_failure(monkeypatch):
    class DummyPool:
        async def fetch(self, *args, **kwargs):
            raise RuntimeError("query failed")

    monkeypatch.setattr(ai_app, "get_pool", lambda: DummyPool())

    client = TestClient(ai_app.app)
    response = client.get("/executions", headers=_authorize())

    assert response.status_code == 503
