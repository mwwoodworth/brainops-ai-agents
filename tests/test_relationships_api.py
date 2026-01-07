import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.relationships as relationships_api


class DummyPool:
    async def fetch(self, query, *args):
        if "FROM brain_relationships" in query:
            return [
                {"entity_a": "a", "entity_b": "b", "relationship_type": "owns"},
                {"entity_a": "b", "entity_b": "c", "relationship_type": "related_to"},
            ]
        if "FROM brain_entities" in query:
            return [
                {"id": "a", "name": "Customer A", "entity_type": "customer"},
                {"id": "b", "name": "Job B", "entity_type": "job"},
                {"id": "c", "name": "Invoice C", "entity_type": "invoice"},
            ]
        return []


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(relationships_api.router)

    monkeypatch.setattr(relationships_api, "VALID_API_KEYS", {"test-key"})
    monkeypatch.setattr(relationships_api, "using_fallback", lambda: False)
    monkeypatch.setattr(relationships_api, "get_pool", lambda: DummyPool())

    return TestClient(app)


def test_relationship_path(client):
    response = client.post(
        "/relationships/path",
        json={"source_id": "a", "target_id": "c", "max_depth": 4},
        headers={"X-API-Key": "test-key", "X-Tenant-ID": "tenant"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["depth"] == 2
    assert payload["path"]
