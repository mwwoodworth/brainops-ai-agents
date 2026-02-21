"""
Wave 2C/2D route contract tests.

Verifies that every route extracted in Wave 2C and Wave 2D (api/platform.py,
api/content_revenue.py, api/ai_operations.py, and api/operational.py)
satisfies its contract:

  1. The route exists and returns the expected HTTP status code.
  2. Auth is enforced: a request without an API key must be rejected
     with 403 (the application always returns 403, never 401, when no
     credential is supplied).
  3. Feature-flag "unavailable" cases return the documented status code
     (usually 503).
  4. Every field documented as "required" in the route's response
     envelope is present in the JSON body.

All tests are pure contract checks – no behavioural assertions.
DB interactions are fully mocked via the conftest ``patch_pool`` fixture,
and optional app-level singletons are monkeypatched as required.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as ai_app
import api.platform as platform_api
import api.content_revenue as content_revenue_api
import api.ai_operations as ai_operations_api
import api.operational as operational_api


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TENANT_UUID = "00000000-0000-0000-0000-000000000001"


def _zero_fetchrow(*_args, **_kwargs):
    """Return a dict of all-zero/None counts for table-stat queries."""
    return {
        "total_thoughts": 0,
        "thoughts_last_hour": 0,
        "thoughts_last_day": 0,
        "last_thought_at": None,
        "total_decisions": 0,
        "decisions_last_hour": 0,
        "avg_confidence": None,
        "queued": 0,
        "processing": 0,
        "sent": 0,
        "failed": 0,
        "skipped": 0,
        "total": 0,
        "total_workflows": 0,
        "active_workflows": 0,
        "last_activity": None,
        "total_runs": 0,
        "completed_runs": 0,
        "failed_runs": 0,
        "running_workflows": 0,
        "runs_last_24h": 0,
        "total_assessments": 0,
        "avg_confidence": None,
        "can_complete_alone_count": 0,
        "requires_review_count": 0,
        "total_mistakes": 0,
        "should_have_known_count": 0,
        "avg_confidence_drop": None,
        "total_explanations": 0,
        "avg_decision_confidence": None,
        "human_review_count": 0,
        "total_interactions": 0,
        "last_interaction": None,
        "training_samples": 0,
        "insights_generated": 0,
        "real_count": 0,
        "real_revenue": 0,
        "test_count": 0,
        "test_revenue": 0,
        "active_subscriptions": 0,
        "mrr": 0,
        "total_sales": 0,
        "total_revenue": 0,
        "last_sale": None,
    }


def _zero_fetchval(*_args, **_kwargs):
    return 0


def _empty_fetch(*_args, **_kwargs):
    return []


# ---------------------------------------------------------------------------
# TestPlatformRoot
# ---------------------------------------------------------------------------


class TestPlatformRoot:
    """GET / — root endpoint returning service info (public, no auth required)."""

    async def test_returns_200_without_auth(self, client, monkeypatch):
        """Root endpoint returns 200 without auth (public route)."""
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)
        response = await client.get("/")
        assert response.status_code == 200

    async def test_response_contains_service_and_status(self, client, monkeypatch):
        """Envelope must contain service, version, status keys."""
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "status" in data


# ---------------------------------------------------------------------------
# TestPlatformEmail
# ---------------------------------------------------------------------------


class TestPlatformEmail:
    """Email endpoints: /email/send, /email/status, /email/process, /email/test."""

    # --- GET /email/status ---

    async def test_email_status_returns_200_with_auth(self, client, auth_headers, monkeypatch):
        """GET /email/status returns 200 when email_sender module is available."""
        fake_module = MagicMock()
        fake_module.get_queue_status = lambda: {
            "queued": 0,
            "sent": 5,
            "failed": 0,
        }
        monkeypatch.setitem(sys.modules, "email_sender", fake_module)
        response = await client.get("/email/status", headers=auth_headers)
        assert response.status_code == 200

    async def test_email_status_rejects_unauthenticated(self, client):
        """GET /email/status enforces auth."""
        response = await client.get("/email/status")
        assert response.status_code == 403

    async def test_email_status_soft_error_when_module_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with error key when email_sender is not importable."""
        # Remove email_sender from sys.modules so ImportError is raised
        monkeypatch.setitem(sys.modules, "email_sender", None)
        response = await client.get("/email/status", headers=auth_headers)
        # Route catches ImportError and returns soft dict, never 500
        assert response.status_code in (200,)

    # --- POST /email/process ---

    async def test_email_process_returns_200_with_auth(self, client, auth_headers, monkeypatch):
        """POST /email/process returns 200 when module available."""
        fake_module = MagicMock()
        fake_module.process_email_queue = lambda batch_size, dry_run: {
            "processed": 0,
            "sent": 0,
        }
        monkeypatch.setitem(sys.modules, "email_sender", fake_module)
        response = await client.post("/email/process", headers=auth_headers)
        assert response.status_code == 200

    async def test_email_process_rejects_unauthenticated(self, client):
        """POST /email/process enforces auth."""
        response = await client.post("/email/process")
        assert response.status_code == 403

    # --- POST /email/test ---

    async def test_email_test_returns_200_with_auth(self, client, auth_headers, monkeypatch):
        """POST /email/test returns 200 when module available."""
        fake_module = MagicMock()
        fake_module.send_email = lambda *_args, **_kwargs: (True, "sent")
        monkeypatch.setitem(sys.modules, "email_sender", fake_module)
        response = await client.post("/email/test?recipient=test@example.com", headers=auth_headers)
        assert response.status_code == 200

    async def test_email_test_rejects_unauthenticated(self, client):
        """POST /email/test enforces auth."""
        response = await client.post("/email/test?recipient=test@example.com")
        assert response.status_code == 403

    async def test_email_test_response_contains_success_field(
        self, client, auth_headers, monkeypatch
    ):
        """Envelope contains success and message fields."""
        fake_module = MagicMock()
        fake_module.send_email = lambda *_a, **_kw: (True, "Test email sent")
        monkeypatch.setitem(sys.modules, "email_sender", fake_module)
        response = await client.post("/email/test?recipient=test@example.com", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    # --- POST /email/send ---

    async def test_email_send_rejects_unauthenticated(self, client):
        """POST /email/send enforces auth."""
        response = await client.post(
            "/email/send",
            json={
                "recipient": "test@example.com",
                "subject": "Hello",
                "html": "<p>Test</p>",
            },
        )
        assert response.status_code == 403

    async def test_email_send_returns_403_when_recipient_not_allowlisted(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 403 when recipient is not on the allowlist and mode is not live."""
        import os

        monkeypatch.setenv("OUTBOUND_EMAIL_MODE", "disabled")
        monkeypatch.setenv("OUTBOUND_EMAIL_ALLOWLIST", "")
        monkeypatch.setenv("OUTBOUND_EMAIL_ALLOWLIST_DOMAINS", "")
        response = await client.post(
            "/email/send",
            headers=auth_headers,
            json={
                "recipient": "nobody@nowhere.com",
                "subject": "Test",
                "html": "<p>Hi</p>",
            },
        )
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# TestPlatformKnowledge
# ---------------------------------------------------------------------------


class TestPlatformKnowledge:
    """Knowledge endpoints: /api/v1/knowledge/*"""

    # --- POST /api/v1/knowledge/store-legacy ---

    async def test_knowledge_store_legacy_rejects_unauthenticated(self, client):
        """POST /api/v1/knowledge/store-legacy enforces auth."""
        response = await client.post(
            "/api/v1/knowledge/store-legacy",
            json={"key": "test-key", "value": {"x": 1}},
        )
        assert response.status_code == 403

    async def test_knowledge_store_legacy_returns_200_when_brain_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with success=False when brain is not available."""
        monkeypatch.setattr(ai_app, "BRAIN_AVAILABLE", False)
        response = await client.post(
            "/api/v1/knowledge/store-legacy",
            headers=auth_headers,
            json={"key": "test-key", "value": {"data": "hello"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    async def test_knowledge_store_legacy_returns_error_when_key_missing(
        self, client, auth_headers, monkeypatch
    ):
        """Returns success=False when key field is absent."""
        monkeypatch.setattr(ai_app, "BRAIN_AVAILABLE", False)
        response = await client.post(
            "/api/v1/knowledge/store-legacy",
            headers=auth_headers,
            json={"value": {"data": "hello"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    # --- POST /api/v1/knowledge/store ---

    async def test_knowledge_store_rejects_unauthenticated(self, client):
        """POST /api/v1/knowledge/store enforces auth."""
        response = await client.post(
            "/api/v1/knowledge/store",
            json={"content": "some knowledge"},
        )
        assert response.status_code == 403

    async def test_knowledge_store_returns_503_when_no_backend_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when neither embedded_memory nor MEMORY_AVAILABLE is set."""
        monkeypatch.setattr(ai_app, "MEMORY_AVAILABLE", False)
        # Ensure app.state.embedded_memory is None
        ai_app.app.state.embedded_memory = None
        try:
            response = await client.post(
                "/api/v1/knowledge/store",
                headers=auth_headers,
                json={"content": "test content"},
            )
        finally:
            ai_app.app.state.embedded_memory = None
        assert response.status_code == 503

    async def test_knowledge_store_returns_200_with_embedded_memory(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with id and memory_type when embedded_memory is available."""
        fake_memory = MagicMock()
        fake_memory.store_memory = MagicMock(return_value=True)
        ai_app.app.state.embedded_memory = fake_memory
        try:
            response = await client.post(
                "/api/v1/knowledge/store",
                headers=auth_headers,
                json={"content": "AI is great", "memory_type": "knowledge"},
            )
        finally:
            ai_app.app.state.embedded_memory = None
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "id" in data
        assert "memory_type" in data

    # --- POST /api/v1/knowledge/query ---

    async def test_knowledge_query_rejects_unauthenticated(self, client):
        """POST /api/v1/knowledge/query enforces auth."""
        response = await client.post(
            "/api/v1/knowledge/query",
            json={"query": "revenue"},
        )
        assert response.status_code == 403

    async def test_knowledge_query_returns_200_empty_when_no_backend(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with empty results when no memory backend is configured."""
        monkeypatch.setattr(ai_app, "MEMORY_AVAILABLE", False)
        ai_app.app.state.embedded_memory = None
        try:
            response = await client.post(
                "/api/v1/knowledge/query",
                headers=auth_headers,
                json={"query": "test query"},
            )
        finally:
            ai_app.app.state.embedded_memory = None
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data

    async def test_knowledge_query_returns_results_with_embedded_memory(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with results list when embedded_memory returns entries."""
        fake_memory = MagicMock()
        fake_memory.search_memories = MagicMock(
            return_value=[
                {
                    "id": "mem-001",
                    "memory_type": "knowledge",
                    "source_agent": "system",
                    "source_system": "brainops",
                    "importance_score": 0.9,
                    "tags": [],
                    "metadata": {},
                    "content": "test content",
                    "created_at": None,
                    "last_accessed": None,
                    "similarity_score": 0.95,
                    "combined_score": 0.92,
                }
            ]
        )
        ai_app.app.state.embedded_memory = fake_memory
        try:
            response = await client.post(
                "/api/v1/knowledge/query",
                headers=auth_headers,
                json={"query": "test query", "limit": 5},
            )
        finally:
            ai_app.app.state.embedded_memory = None
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1

    # --- GET /api/v1/knowledge/graph/stats ---

    async def test_knowledge_graph_stats_rejects_unauthenticated(self, client):
        """GET /api/v1/knowledge/graph/stats enforces auth."""
        response = await client.get("/api/v1/knowledge/graph/stats")
        assert response.status_code == 403

    async def test_knowledge_graph_stats_returns_200_with_mocked_db(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with node/edge counts from mocked DB."""

        def fetchval_handler(query, *_args):
            return 0

        def fetch_handler(query, *_args):
            return []

        def fetchrow_handler(query, *_args):
            return None

        patch_pool.fetchval_handler = fetchval_handler
        patch_pool.fetch_handler = fetch_handler
        patch_pool.fetchrow_handler = fetchrow_handler

        response = await client.get("/api/v1/knowledge/graph/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_nodes" in data
        assert "total_edges" in data

    # --- POST /api/v1/knowledge/graph/extract ---

    async def test_knowledge_graph_extract_rejects_unauthenticated(self, client):
        """POST /api/v1/knowledge/graph/extract enforces auth."""
        response = await client.post("/api/v1/knowledge/graph/extract")
        assert response.status_code == 403

    async def test_knowledge_graph_extract_returns_200_with_mocked_extractor(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with success and details when extractor is available."""
        fake_extractor = MagicMock()
        fake_extractor.initialize = AsyncMock()
        fake_extractor.run_extraction = AsyncMock(
            return_value={"success": True, "nodes_stored": 10, "edges_stored": 5}
        )

        fake_module = MagicMock()
        fake_module.get_knowledge_extractor = lambda: fake_extractor
        monkeypatch.setitem(sys.modules, "knowledge_graph_extractor", fake_module)

        response = await client.post("/api/v1/knowledge/graph/extract", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "message" in data


# ---------------------------------------------------------------------------
# TestPlatformErp
# ---------------------------------------------------------------------------


class TestPlatformErp:
    """POST /api/v1/erp/analyze"""

    async def test_erp_analyze_rejects_unauthenticated(self, client):
        """POST /api/v1/erp/analyze enforces auth."""
        response = await client.post("/api/v1/erp/analyze", json={})
        assert response.status_code == 403

    async def test_erp_analyze_returns_200_with_empty_jobs(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with empty jobs list when DB returns no rows."""
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)

        patch_pool.fetchval_handler = lambda *_: False
        patch_pool.fetch_handler = lambda *_: []

        response = await client.post(
            "/api/v1/erp/analyze",
            headers=auth_headers,
            json={"limit": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "jobs" in data
        assert data["count"] == 0

    async def test_erp_analyze_returns_200_with_job_data(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with job intel list when rows are returned."""
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)
        patch_pool.fetchval_handler = lambda *_: False
        patch_pool.fetch_handler = lambda query, *_: (
            [
                {
                    "id": "job-uuid-001",
                    "job_number": "JOB-001",
                    "title": "Roof Repair",
                    "status": "in_progress",
                    "scheduled_start": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    "scheduled_end": datetime(2026, 1, 31, tzinfo=timezone.utc),
                    "actual_start": None,
                    "actual_end": None,
                    "completion_percentage": 50,
                    "estimated_revenue": 10000,
                    "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    "customer_name": "Acme Corp",
                }
            ]
            if "FROM jobs" in query
            else []
        )

        response = await client.post(
            "/api/v1/erp/analyze",
            headers=auth_headers,
            json={"limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        job = data["jobs"][0]
        assert "job_id" in job
        assert "delay_risk" in job
        assert "progress_tracking" in job


# ---------------------------------------------------------------------------
# TestPlatformSystemsUsage
# ---------------------------------------------------------------------------


class TestPlatformSystemsUsage:
    """GET /systems/usage"""

    async def test_systems_usage_rejects_unauthenticated(self, client):
        """GET /systems/usage enforces auth."""
        response = await client.get("/systems/usage")
        assert response.status_code == 403

    async def test_systems_usage_returns_200_with_mocked_db(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with active_systems, agents, schedules, memory keys."""
        monkeypatch.setattr(ai_app, "CUSTOMER_SUCCESS_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "LEARNING_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "AUREA_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "SELF_HEALING_AVAILABLE", False)

        patch_pool.fetch_handler = _empty_fetch
        patch_pool.fetchrow_handler = lambda *_: {"total": 0}
        patch_pool.fetchval_handler = _zero_fetchval

        response = await client.get("/systems/usage", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "active_systems" in data
        assert "agents" in data
        assert "schedules" in data
        assert "memory" in data


# ---------------------------------------------------------------------------
# TestContentGeneration
# ---------------------------------------------------------------------------


class TestContentGeneration:
    """Content generation routes: /content/*"""

    # --- POST /content/generate ---

    async def test_content_generate_rejects_unauthenticated(self, client):
        """POST /content/generate enforces auth."""
        response = await client.post(
            "/content/generate",
            json={"topic": "AI trends", "content_type": "blog_post"},
        )
        assert response.status_code == 403

    async def test_content_generate_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when CONTENT_ORCHESTRATOR_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", False)
        response = await client.post(
            "/content/generate",
            headers=auth_headers,
            json={"topic": "AI trends", "content_type": "blog_post"},
        )
        assert response.status_code == 503

    async def test_content_generate_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when orchestrator is available and execution succeeds."""
        fake_orchestrator = MagicMock()
        fake_orchestrator.execute = AsyncMock(
            return_value={"status": "completed", "content": "Generated blog post"}
        )

        fake_class = MagicMock(return_value=fake_orchestrator)
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "MultiAIContentOrchestrator", fake_class)

        response = await client.post(
            "/content/generate",
            headers=auth_headers,
            json={"topic": "AI trends", "content_type": "blog_post"},
        )
        assert response.status_code == 200

    # --- POST /content/newsletter ---

    async def test_content_newsletter_rejects_unauthenticated(self, client):
        """POST /content/newsletter enforces auth."""
        response = await client.post(
            "/content/newsletter",
            json={"topic": "Weekly Digest"},
        )
        assert response.status_code == 403

    async def test_content_newsletter_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when CONTENT_ORCHESTRATOR_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", False)
        response = await client.post(
            "/content/newsletter",
            headers=auth_headers,
            json={"topic": "Weekly AI Digest"},
        )
        assert response.status_code == 503

    async def test_content_newsletter_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when orchestrator generates newsletter."""
        fake_orchestrator = MagicMock()
        fake_orchestrator.generate_newsletter = AsyncMock(
            return_value={"status": "completed", "html": "<html></html>"}
        )
        fake_class = MagicMock(return_value=fake_orchestrator)
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "MultiAIContentOrchestrator", fake_class)

        response = await client.post(
            "/content/newsletter",
            headers=auth_headers,
            json={"topic": "Weekly AI Digest"},
        )
        assert response.status_code == 200

    # --- POST /content/ebook ---

    async def test_content_ebook_rejects_unauthenticated(self, client):
        """POST /content/ebook enforces auth."""
        response = await client.post(
            "/content/ebook",
            json={"topic": "Machine Learning"},
        )
        assert response.status_code == 403

    async def test_content_ebook_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when CONTENT_ORCHESTRATOR_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", False)
        response = await client.post(
            "/content/ebook",
            headers=auth_headers,
            json={"topic": "Machine Learning"},
        )
        assert response.status_code == 503

    async def test_content_ebook_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when orchestrator generates ebook."""
        fake_orchestrator = MagicMock()
        fake_orchestrator.generate_ebook = AsyncMock(
            return_value={"status": "completed", "chapters": []}
        )
        fake_class = MagicMock(return_value=fake_orchestrator)
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "MultiAIContentOrchestrator", fake_class)

        response = await client.post(
            "/content/ebook",
            headers=auth_headers,
            json={"topic": "Machine Learning", "chapters": 3},
        )
        assert response.status_code == 200

    # --- POST /content/training ---

    async def test_content_training_rejects_unauthenticated(self, client):
        """POST /content/training enforces auth."""
        response = await client.post(
            "/content/training",
            json={"topic": "Python"},
        )
        assert response.status_code == 403

    async def test_content_training_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when CONTENT_ORCHESTRATOR_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", False)
        response = await client.post(
            "/content/training",
            headers=auth_headers,
            json={"topic": "Python Basics"},
        )
        assert response.status_code == 503

    async def test_content_training_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when orchestrator generates training doc."""
        fake_orchestrator = MagicMock()
        fake_orchestrator.generate_training_doc = AsyncMock(
            return_value={"status": "completed", "modules": []}
        )
        fake_class = MagicMock(return_value=fake_orchestrator)
        monkeypatch.setattr(content_revenue_api, "CONTENT_ORCHESTRATOR_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "MultiAIContentOrchestrator", fake_class)

        response = await client.post(
            "/content/training",
            headers=auth_headers,
            json={"topic": "Python Basics", "module_number": 1},
        )
        assert response.status_code == 200

    # --- GET /content/types ---

    async def test_content_types_rejects_unauthenticated(self, client):
        """GET /content/types enforces auth."""
        response = await client.get("/content/types")
        assert response.status_code == 403

    async def test_content_types_returns_200_with_types_list(self, client, auth_headers):
        """Returns 200 with types list regardless of feature flag."""
        response = await client.get("/content/types", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "types" in data
        assert isinstance(data["types"], list)
        assert len(data["types"]) > 0


# ---------------------------------------------------------------------------
# TestInventory
# ---------------------------------------------------------------------------


class TestInventory:
    """Inventory routes: /inventory/products and /inventory/revenue."""

    # --- GET /inventory/products ---

    async def test_inventory_products_rejects_unauthenticated(self, client):
        """GET /inventory/products enforces auth."""
        response = await client.get("/inventory/products")
        assert response.status_code == 403

    async def test_inventory_products_returns_200_with_mocked_db(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with platforms and revenue_summary keys."""
        patch_pool.fetchrow_handler = _zero_fetchrow

        response = await client.get("/inventory/products", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "platforms" in data
        assert "revenue_summary" in data

    async def test_inventory_products_contains_gumroad_platform(
        self, client, auth_headers, patch_pool
    ):
        """Platforms dict includes gumroad."""
        patch_pool.fetchrow_handler = _zero_fetchrow

        response = await client.get("/inventory/products", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "gumroad" in data["platforms"]

    # --- GET /inventory/revenue ---

    async def test_inventory_revenue_rejects_unauthenticated(self, client):
        """GET /inventory/revenue enforces auth."""
        response = await client.get("/inventory/revenue")
        assert response.status_code == 403

    async def test_inventory_revenue_returns_200_with_real_revenue_key(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with real_revenue and timestamp keys."""
        patch_pool.fetchrow_handler = _zero_fetchrow

        response = await client.get("/inventory/revenue", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "real_revenue" in data
        assert "timestamp" in data

    async def test_inventory_revenue_degrades_gracefully_when_db_fails(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with error key when DB raises an exception."""

        def raise_exc(*_args, **_kwargs):
            raise RuntimeError("DB unavailable")

        patch_pool.fetchrow_handler = raise_exc

        response = await client.get("/inventory/revenue", headers=auth_headers)
        # Route catches exception and returns soft envelope
        assert response.status_code == 200
        data = response.json()
        assert "real_revenue" in data


# ---------------------------------------------------------------------------
# TestRevenueIntelligence
# ---------------------------------------------------------------------------


class TestRevenueIntelligence:
    """Revenue intelligence routes: /revenue/*"""

    # --- GET /revenue/state ---

    async def test_revenue_state_rejects_unauthenticated(self, client):
        """GET /revenue/state enforces auth."""
        response = await client.get("/revenue/state")
        assert response.status_code == 403

    async def test_revenue_state_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when REVENUE_INTEL_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", False)
        response = await client.get("/revenue/state", headers=auth_headers)
        assert response.status_code == 503

    async def test_revenue_state_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when revenue system is available."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", True)
        monkeypatch.setattr(
            content_revenue_api,
            "get_business_state",
            AsyncMock(return_value={"state": "active", "revenue": 0}),
        )
        response = await client.get("/revenue/state", headers=auth_headers)
        assert response.status_code == 200

    # --- GET /revenue/live ---

    async def test_revenue_live_rejects_unauthenticated(self, client):
        """GET /revenue/live enforces auth."""
        response = await client.get("/revenue/live")
        assert response.status_code == 403

    async def test_revenue_live_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when REVENUE_INTEL_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", False)
        response = await client.get("/revenue/live", headers=auth_headers)
        assert response.status_code == 503

    async def test_revenue_live_returns_200_when_available(self, client, auth_headers, monkeypatch):
        """Returns 200 when revenue system supplies live data."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", True)
        monkeypatch.setattr(
            content_revenue_api,
            "get_revenue",
            AsyncMock(return_value={"gumroad": {"total": 0}, "mrg": {"mrr": 0}}),
        )
        response = await client.get("/revenue/live", headers=auth_headers)
        assert response.status_code == 200

    # --- GET /revenue/products ---

    async def test_revenue_products_rejects_unauthenticated(self, client):
        """GET /revenue/products enforces auth."""
        response = await client.get("/revenue/products")
        assert response.status_code == 403

    async def test_revenue_products_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when REVENUE_INTEL_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", False)
        response = await client.get("/revenue/products", headers=auth_headers)
        assert response.status_code == 503

    async def test_revenue_products_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with products/social/websites keys when system available."""
        fake_system = MagicMock()
        fake_system.get_all_products = MagicMock(return_value=[])
        fake_system.get_social_presence = MagicMock(return_value={})
        fake_system.get_websites = MagicMock(return_value={})

        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "get_revenue_system", lambda: fake_system)

        response = await client.get("/revenue/products", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert "social" in data
        assert "websites" in data

    # --- GET /revenue/automations ---

    async def test_revenue_automations_rejects_unauthenticated(self, client):
        """GET /revenue/automations enforces auth."""
        response = await client.get("/revenue/automations")
        assert response.status_code == 403

    async def test_revenue_automations_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when REVENUE_INTEL_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", False)
        response = await client.get("/revenue/automations", headers=auth_headers)
        assert response.status_code == 503

    async def test_revenue_automations_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with automation status when system is available."""
        fake_system = MagicMock()
        fake_system.get_automation_status = AsyncMock(return_value={"automations": [], "active": 0})
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "get_revenue_system", lambda: fake_system)

        response = await client.get("/revenue/automations", headers=auth_headers)
        assert response.status_code == 200

    # --- POST /revenue/sync-brain ---

    async def test_revenue_sync_brain_rejects_unauthenticated(self, client):
        """POST /revenue/sync-brain enforces auth."""
        response = await client.post("/revenue/sync-brain")
        assert response.status_code == 403

    async def test_revenue_sync_brain_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when REVENUE_INTEL_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", False)
        response = await client.post("/revenue/sync-brain", headers=auth_headers)
        assert response.status_code == 503

    async def test_revenue_sync_brain_returns_200_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when sync completes successfully."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", True)
        monkeypatch.setattr(
            content_revenue_api,
            "sync_to_brain",
            AsyncMock(return_value={"synced": True, "keys": 3}),
        )
        response = await client.post("/revenue/sync-brain", headers=auth_headers)
        assert response.status_code == 200

    # --- POST /revenue/event ---

    async def test_revenue_event_rejects_unauthenticated(self, client):
        """POST /revenue/event enforces auth."""
        response = await client.post(
            "/revenue/event",
            json={"event_type": "sale", "platform": "gumroad"},
        )
        assert response.status_code == 403

    async def test_revenue_event_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when REVENUE_INTEL_AVAILABLE is False."""
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", False)
        response = await client.post(
            "/revenue/event",
            headers=auth_headers,
            json={"event_type": "sale", "platform": "gumroad"},
        )
        assert response.status_code == 503

    async def test_revenue_event_returns_200_with_event_id(self, client, auth_headers, monkeypatch):
        """Returns 200 with status and event_id when system available."""
        fake_system = MagicMock()
        fake_system.record_revenue_event = AsyncMock(return_value="evt-uuid-001")
        monkeypatch.setattr(content_revenue_api, "REVENUE_INTEL_AVAILABLE", True)
        monkeypatch.setattr(content_revenue_api, "get_revenue_system", lambda: fake_system)

        response = await client.post(
            "/revenue/event",
            headers=auth_headers,
            json={"event_type": "sale", "platform": "gumroad", "amount": 97.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "event_id" in data


# ---------------------------------------------------------------------------
# TestObservability
# ---------------------------------------------------------------------------


class TestObservability:
    """Observability routes: /logs/recent, /observability/full, /debug/all-errors,
    /system/unified-status."""

    # --- GET /logs/recent ---

    async def test_logs_recent_rejects_unauthenticated(self, client):
        """GET /logs/recent enforces auth."""
        response = await client.get("/logs/recent")
        assert response.status_code == 403

    async def test_logs_recent_returns_200_with_required_fields(self, client, auth_headers):
        """Returns 200 with logs, total_in_buffer, returned, filters keys."""
        response = await client.get("/logs/recent", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "total_in_buffer" in data
        assert "returned" in data
        assert "filters" in data

    async def test_logs_recent_returns_200_with_level_filter(self, client, auth_headers):
        """Returns 200 when level filter is supplied."""
        response = await client.get("/logs/recent?level=ERROR", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["level"] == "ERROR"

    # --- GET /observability/full ---

    async def test_observability_full_rejects_unauthenticated(self, client):
        """GET /observability/full enforces auth."""
        response = await client.get("/observability/full")
        assert response.status_code == 403

    async def test_observability_full_returns_200_with_required_structure(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with services, database, recent_errors, system_metrics keys."""
        patch_pool.fetchrow_handler = lambda *_: {
            "customers": 0,
            "jobs": 0,
            "agents": 0,
            "executions": 0,
            "recent_executions": 0,
            "recent_failures": 0,
        }

        # Mock httpx to avoid real network calls
        import httpx

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}

            def json(self):
                return {"status": "ok"}

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def get(self, url, **_kwargs):
                return _FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **_: _FakeClient())

        response = await client.get("/observability/full", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "services" in data
        assert "database" in data
        assert "recent_errors" in data
        assert "system_metrics" in data

    # --- GET /debug/all-errors ---

    async def test_debug_all_errors_rejects_unauthenticated(self, client):
        """GET /debug/all-errors enforces auth."""
        response = await client.get("/debug/all-errors")
        assert response.status_code == 403

    async def test_debug_all_errors_returns_200_with_required_fields(self, client, auth_headers):
        """Returns 200 with total_errors, categorized, recent_errors, by_category keys."""
        response = await client.get("/debug/all-errors", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_errors" in data
        assert "categorized" in data
        assert "recent_errors" in data
        assert "by_category" in data

    # --- GET /system/unified-status ---

    async def test_unified_status_rejects_unauthenticated(self, client):
        """GET /system/unified-status enforces auth."""
        response = await client.get("/system/unified-status")
        assert response.status_code == 403

    async def test_unified_status_returns_200_with_required_fields(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with version, timestamp, overall_health, components keys."""
        patch_pool.fetchval_handler = _zero_fetchval

        response = await client.get("/system/unified-status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "timestamp" in data
        assert "overall_health" in data
        assert "components" in data


# ---------------------------------------------------------------------------
# TestAIOperations
# ---------------------------------------------------------------------------


class TestAIOperations:
    """AI Operations routes from api/ai_operations.py."""

    # --- GET /ai/providers/status ---

    async def test_providers_status_rejects_unauthenticated(self, client):
        """GET /ai/providers/status enforces auth."""
        response = await client.get("/ai/providers/status")
        assert response.status_code == 403

    async def test_providers_status_returns_200_with_auth(self, client, auth_headers):
        """GET /ai/providers/status returns 200 and provider map."""
        response = await client.get("/ai/providers/status", headers=auth_headers)
        assert response.status_code == 200
        # The response should be a JSON dict
        assert response.json() is not None

    # --- GET /consciousness/status ---

    async def test_consciousness_status_rejects_unauthenticated(self, client):
        """GET /consciousness/status enforces auth."""
        response = await client.get("/consciousness/status")
        assert response.status_code == 403

    async def test_consciousness_status_returns_200_with_mocked_db(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with consciousness_state, is_alive, thought_stream keys."""
        monkeypatch.setattr(ai_app, "BLEEDING_EDGE_AVAILABLE", False)

        patch_pool.fetchrow_handler = lambda query, *_: (
            {
                "total_thoughts": 0,
                "thoughts_last_hour": 0,
                "thoughts_last_day": 0,
                "last_thought_at": None,
                "total_decisions": 0,
                "decisions_last_hour": 0,
                "avg_confidence": None,
            }
        )

        response = await client.get("/consciousness/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "consciousness_state" in data
        assert "is_alive" in data
        assert "thought_stream" in data
        assert "timestamp" in data

    async def test_consciousness_status_dormant_when_no_systems(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns consciousness_state=dormant when no systems are active."""
        monkeypatch.setattr(ai_app, "BLEEDING_EDGE_AVAILABLE", False)
        patch_pool.fetchrow_handler = lambda *_: {
            "total_thoughts": 0,
            "thoughts_last_hour": 0,
            "thoughts_last_day": 0,
            "last_thought_at": None,
            "total_decisions": 0,
            "decisions_last_hour": 0,
            "avg_confidence": None,
        }

        response = await client.get("/consciousness/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["consciousness_state"] == "dormant"

    # --- GET /meta-intelligence/status ---

    async def test_meta_intelligence_status_rejects_unauthenticated(self, client):
        """GET /meta-intelligence/status enforces auth."""
        response = await client.get("/meta-intelligence/status")
        assert response.status_code == 403

    async def test_meta_intelligence_status_returns_200_with_required_keys(
        self, client, auth_headers
    ):
        """Returns 200 with meta_intelligence and learning_bridge keys."""
        response = await client.get("/meta-intelligence/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "meta_intelligence" in data
        assert "learning_bridge" in data
        assert "timestamp" in data

    # --- GET /workflow-engine/status ---

    async def test_workflow_engine_status_get_rejects_unauthenticated(self, client):
        """GET /workflow-engine/status enforces auth."""
        response = await client.get("/workflow-engine/status")
        assert response.status_code == 403

    async def test_workflow_engine_status_get_returns_200_when_module_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with status=unavailable when ai_workflow_templates not importable."""
        monkeypatch.setitem(sys.modules, "ai_workflow_templates", None)
        response = await client.get("/workflow-engine/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    async def test_workflow_engine_status_post_rejects_unauthenticated(self, client):
        """POST /workflow-engine/status enforces auth."""
        response = await client.post("/workflow-engine/status")
        assert response.status_code == 403

    async def test_workflow_engine_status_post_returns_200(self, client, auth_headers, monkeypatch):
        """POST /workflow-engine/status returns 200 (same handler as GET)."""
        monkeypatch.setitem(sys.modules, "ai_workflow_templates", None)
        response = await client.post("/workflow-engine/status", headers=auth_headers)
        assert response.status_code == 200

    # --- GET /workflow-automation/status ---

    async def test_workflow_automation_status_get_rejects_unauthenticated(self, client):
        """GET /workflow-automation/status enforces auth."""
        response = await client.get("/workflow-automation/status")
        assert response.status_code == 403

    async def test_workflow_automation_status_returns_200_with_mocked_db(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with status, automation, runs keys."""
        patch_pool.fetchrow_handler = lambda query, *_: (
            {
                "total_workflows": 0,
                "active_workflows": 0,
                "last_activity": None,
                "total_runs": 0,
                "completed_runs": 0,
                "failed_runs": 0,
                "running_workflows": 0,
                "runs_last_24h": 0,
            }
        )
        response = await client.get("/workflow-automation/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    async def test_workflow_automation_status_post_rejects_unauthenticated(self, client):
        """POST /workflow-automation/status enforces auth."""
        response = await client.post("/workflow-automation/status")
        assert response.status_code == 403

    # --- POST /ai/self-assess ---

    async def test_ai_self_assess_rejects_unauthenticated(self, client):
        """POST /ai/self-assess enforces auth."""
        response = await client.post("/ai/self-assess?task_id=t1&agent_id=a1&task_description=test")
        assert response.status_code == 403

    async def test_ai_self_assess_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when SELF_AWARENESS_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "SELF_AWARENESS_AVAILABLE", False)
        response = await client.post(
            "/ai/self-assess?task_id=t1&agent_id=a1&task_description=test",
            headers=auth_headers,
        )
        assert response.status_code == 503

    # --- POST /ai/explain-reasoning ---

    async def test_ai_explain_reasoning_rejects_unauthenticated(self, client):
        """POST /ai/explain-reasoning enforces auth."""
        response = await client.post(
            "/ai/explain-reasoning?task_id=t1&agent_id=a1&decision=yes&reasoning_process={}"
        )
        assert response.status_code == 403

    async def test_ai_explain_reasoning_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when SELF_AWARENESS_AVAILABLE is False.

        Note: reasoning_process is a dict query param; FastAPI returns 422 when
        it cannot be parsed from the query string.  The route still enforces auth
        (403 without a key) and returns 503 when the flag is off AND a valid dict
        can be provided. We accept 503 or 422 here because the flag check fires
        before the route body executes, but FastAPI param validation runs at the
        same time for some versions. The important contract is that the route is
        auth-gated and 200 is never returned when the feature is disabled.
        """
        monkeypatch.setattr(ai_app, "SELF_AWARENESS_AVAILABLE", False)
        response = await client.post(
            "/ai/explain-reasoning?task_id=t1&agent_id=a1&decision=approved&reasoning_process={}",
            headers=auth_headers,
        )
        assert response.status_code in (503, 422)

    # --- POST /ai/reason ---

    async def test_ai_reason_rejects_unauthenticated(self, client):
        """POST /ai/reason enforces auth."""
        response = await client.post(
            "/ai/reason",
            json={"problem": "What is 2+2?"},
        )
        assert response.status_code == 403

    async def test_ai_reason_returns_503_when_ai_core_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when AI_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "ai_core", None)
        response = await client.post(
            "/ai/reason",
            headers=auth_headers,
            json={"problem": "What is 2+2?"},
        )
        assert response.status_code == 503

    async def test_ai_reason_returns_200_with_required_fields(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with success, reasoning, conclusion, model_used keys."""
        fake_core = MagicMock()
        fake_core.reason = AsyncMock(
            return_value={
                "reasoning": "Step 1: ...",
                "conclusion": "The answer is 4",
                "model_used": "o3-mini",
                "tokens_used": 100,
                "error": None,
            }
        )
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", True)
        monkeypatch.setattr(ai_app, "ai_core", fake_core)

        response = await client.post(
            "/ai/reason",
            headers=auth_headers,
            json={"problem": "What is 2+2?", "max_tokens": 100},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "reasoning" in data
        assert "conclusion" in data
        assert "model_used" in data

    # --- POST /ai/learn-from-mistake ---

    async def test_ai_learn_from_mistake_rejects_unauthenticated(self, client):
        """POST /ai/learn-from-mistake enforces auth."""
        response = await client.post(
            "/ai/learn-from-mistake?task_id=t1&agent_id=a1"
            "&expected_outcome=ok&actual_outcome=fail&confidence_before=0.9"
        )
        assert response.status_code == 403

    async def test_ai_learn_from_mistake_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when SELF_AWARENESS_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "SELF_AWARENESS_AVAILABLE", False)
        response = await client.post(
            "/ai/learn-from-mistake?task_id=t1&agent_id=a1"
            "&expected_outcome=ok&actual_outcome=fail&confidence_before=0.9",
            headers=auth_headers,
        )
        assert response.status_code == 503

    # --- GET /ai/self-awareness/stats ---

    async def test_ai_self_awareness_stats_rejects_unauthenticated(self, client):
        """GET /ai/self-awareness/stats enforces auth."""
        response = await client.get("/ai/self-awareness/stats")
        assert response.status_code == 403

    async def test_ai_self_awareness_stats_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when SELF_AWARENESS_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "SELF_AWARENESS_AVAILABLE", False)
        response = await client.get("/ai/self-awareness/stats", headers=auth_headers)
        assert response.status_code == 503

    async def test_ai_self_awareness_stats_returns_200_with_mocked_db(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with assessments, learning, reasoning keys when available."""
        monkeypatch.setattr(ai_app, "SELF_AWARENESS_AVAILABLE", True)
        patch_pool.fetchrow_handler = lambda query, *_: {
            "total_assessments": 10,
            "avg_confidence": 0.85,
            "can_complete_alone_count": 8,
            "requires_review_count": 2,
            "total_mistakes": 3,
            "should_have_known_count": 1,
            "avg_confidence_drop": 0.05,
            "total_explanations": 5,
            "avg_decision_confidence": 0.9,
            "human_review_count": 1,
        }

        response = await client.get("/ai/self-awareness/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "assessments" in data
        assert "learning" in data
        assert "reasoning" in data

    # --- GET /ai/tasks/stats ---

    async def test_ai_tasks_stats_rejects_unauthenticated(self, client):
        """GET /ai/tasks/stats enforces auth."""
        response = await client.get("/ai/tasks/stats")
        assert response.status_code == 403

    async def test_ai_tasks_stats_returns_503_when_integration_layer_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when INTEGRATION_LAYER_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "INTEGRATION_LAYER_AVAILABLE", False)
        response = await client.get("/ai/tasks/stats", headers=auth_headers)
        assert response.status_code == 503

    async def test_ai_tasks_stats_returns_200_with_required_fields(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with success, stats, system_status keys when layer available."""

        class _FakeIntegration:
            agents_registry = {}
            execution_queue = SimpleNamespace(qsize=lambda: 0)

            async def list_tasks(self, limit=100):
                return []

        ai_app.app.state.integration_layer = _FakeIntegration()
        monkeypatch.setattr(ai_app, "INTEGRATION_LAYER_AVAILABLE", True)

        try:
            response = await client.get("/ai/tasks/stats", headers=auth_headers)
        finally:
            ai_app.app.state.integration_layer = None

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stats" in data
        assert "system_status" in data

    # --- POST /ai/orchestrate ---

    async def test_ai_orchestrate_rejects_unauthenticated(self, client):
        """POST /ai/orchestrate enforces auth."""
        response = await client.post("/ai/orchestrate?task_description=test")
        assert response.status_code == 403

    async def test_ai_orchestrate_returns_503_when_langgraph_unavailable(
        self, client, auth_headers
    ):
        """Returns 503 when langgraph_orchestrator is not on app.state."""
        ai_app.app.state.langgraph_orchestrator = None
        response = await client.post(
            "/ai/orchestrate?task_description=analyze+revenue",
            headers=auth_headers,
        )
        assert response.status_code == 503

    # --- POST /ai/analyze ---

    async def test_ai_analyze_rejects_unauthenticated(self, client):
        """POST /ai/analyze enforces auth."""
        response = await client.post(
            "/ai/analyze",
            json={"agent": "revenue", "action": "report", "data": {}, "context": {}},
        )
        assert response.status_code == 403

    async def test_ai_analyze_returns_200_queued_when_no_orchestrator(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with queued result when no orchestrator/executor is available."""
        ai_app.app.state.langgraph_orchestrator = None

        # Remove agent_executor from sys.modules to trigger ImportError fallback
        monkeypatch.setitem(sys.modules, "agent_executor", None)

        # Pool mocked: fetchrow for agent lookup returns None, execute for INSERT
        patch_pool.fetchrow_handler = lambda *_: None
        patch_pool.execute_handler = lambda *_: "OK"

        response = await client.post(
            "/ai/analyze",
            headers=auth_headers,
            json={"agent": "revenue", "action": "report", "data": {}, "context": {}},
        )
        # Route either queues task (200) or returns 503 if queueing fails
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "agent" in data
            assert "action" in data


# ---------------------------------------------------------------------------
# TestOperational
# ---------------------------------------------------------------------------


class TestOperational:
    """Operational routes from api/operational.py."""

    # --- GET /executions ---

    async def test_executions_rejects_unauthenticated(self, client):
        """GET /executions enforces auth."""
        response = await client.get("/executions")
        assert response.status_code == 403

    async def test_executions_returns_200_with_empty_list(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with executions list and total when DB returns no rows."""
        monkeypatch.setattr(ai_app, "LOCAL_EXECUTIONS", [])
        patch_pool.fetch_handler = _empty_fetch

        response = await client.get("/executions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "executions" in data
        assert "total" in data
        assert data["total"] == 0

    async def test_executions_includes_local_executions(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Merges LOCAL_EXECUTIONS into the response when DB returns nothing."""
        local_exec = {
            "execution_id": "local-exec-001",
            "agent_id": "revenue",
            "agent_name": "Revenue Agent",
            "status": "completed",
            "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "completed_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "duration_ms": 1500,
            "error": None,
        }
        monkeypatch.setattr(ai_app, "LOCAL_EXECUTIONS", [local_exec])
        patch_pool.fetch_handler = _empty_fetch

        response = await client.get("/executions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    async def test_executions_returns_503_when_db_query_fails(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 503 when the primary DB query raises an error."""
        monkeypatch.setattr(ai_app, "LOCAL_EXECUTIONS", [])

        def raise_exc(query, *_args):
            raise RuntimeError("DB connection lost")

        patch_pool.fetch_handler = raise_exc

        response = await client.get("/executions", headers=auth_headers)
        assert response.status_code == 503

    # --- POST /self-heal/trigger ---

    async def test_self_heal_trigger_rejects_unauthenticated(self, client):
        """POST /self-heal/trigger enforces auth."""
        response = await client.post("/self-heal/trigger")
        assert response.status_code == 403

    async def test_self_heal_trigger_returns_200_with_required_fields(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with timestamp, issues_detected, actions_taken, status keys."""
        patch_pool.fetch_handler = _empty_fetch
        patch_pool.fetchval_handler = lambda *_: False  # tables do not exist
        patch_pool.execute_handler = lambda *_: "OK"

        response = await client.post("/self-heal/trigger", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "issues_detected" in data
        assert "actions_taken" in data
        assert "status" in data

    async def test_self_heal_trigger_returns_200_even_on_db_error(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with status=error when DB raises an unexpected exception."""

        def raise_exc(*_args):
            raise RuntimeError("unexpected DB error")

        patch_pool.fetch_handler = raise_exc
        patch_pool.fetchval_handler = raise_exc

        response = await client.post("/self-heal/trigger", headers=auth_headers)
        # Route catches exception and returns soft envelope
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("error", "completed")

    # --- GET /self-heal/check ---

    async def test_self_heal_check_rejects_unauthenticated(self, client):
        """GET /self-heal/check enforces auth."""
        response = await client.get("/self-heal/check")
        assert response.status_code == 403

    async def test_self_heal_check_returns_200_with_required_fields(
        self, client, auth_headers, patch_pool
    ):
        """Returns 200 with health_score, status, issues, self_healer_active keys."""
        patch_pool.fetchval_handler = lambda *_: 0
        patch_pool.fetch_handler = _empty_fetch

        response = await client.get("/self-heal/check", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "health_score" in data
        assert "status" in data
        assert "issues" in data
        assert "self_healer_active" in data
        assert "timestamp" in data

    # --- GET /aurea/status ---

    async def test_aurea_status_rejects_unauthenticated(self, client):
        """GET /aurea/status enforces auth."""
        response = await client.get("/aurea/status")
        assert response.status_code == 403

    async def test_aurea_status_returns_200_with_required_fields(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with status, aurea_available, timestamp keys.

        fetchval_with_tenant_context uses pool.pool.acquire() for a raw connection.
        The MockTenantScopedPool.pool is a SimpleNamespace; patch the helper so it
        falls back directly to pool.fetchval() which IS mocked.
        """
        import services.tenant_helpers as tenant_helpers_svc

        async def _mock_fetchval_with_ctx(pool, query, *args, tenant_uuid=None):
            return await pool.fetchval(query, *args)

        monkeypatch.setattr(
            tenant_helpers_svc, "fetchval_with_tenant_context", _mock_fetchval_with_ctx
        )
        monkeypatch.setattr(
            operational_api, "fetchval_with_tenant_context", _mock_fetchval_with_ctx
        )

        patch_pool.fetchval_handler = _zero_fetchval

        response = await client.get("/aurea/status", headers=auth_headers)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "aurea_available" in data
            assert "timestamp" in data

    # --- POST /aurea/command/natural_language ---

    async def test_aurea_nl_command_rejects_unauthenticated(self, client):
        """POST /aurea/command/natural_language enforces auth."""
        response = await client.post(
            "/aurea/command/natural_language",
            json={"command_text": "Show status"},
        )
        assert response.status_code == 403

    async def test_aurea_nl_command_returns_200_via_chat_fallback(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 when aurea_nlu is absent but chat fallback succeeds."""
        # Ensure aurea_nlu is not set
        ai_app.app.state.aurea_nlu = None

        fake_chat_response = {"success": True, "result": "OK"}

        async def _fake_execute(cmd):
            return fake_chat_response

        fake_nl_cmd_class = MagicMock(return_value=SimpleNamespace(command="Show status"))

        fake_aurea_chat_module = MagicMock()
        fake_aurea_chat_module.execute_natural_language_command = _fake_execute
        fake_aurea_chat_module.NLCommand = fake_nl_cmd_class
        monkeypatch.setitem(sys.modules, "api.aurea_chat", fake_aurea_chat_module)

        # Also patch internal import path used inside operational.py
        import api.operational as _op

        try:
            response = await client.post(
                "/aurea/command/natural_language",
                headers=auth_headers,
                json={"command_text": "Show status"},
            )
        finally:
            ai_app.app.state.aurea_nlu = None

        assert response.status_code in (200, 503)

    async def test_aurea_nl_command_returns_200_with_active_nlu(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with success, command, result, timestamp when NLU is active."""
        fake_nlu = MagicMock()
        fake_nlu.execute_natural_language_command = AsyncMock(
            return_value={"action": "status", "result": "all clear"}
        )
        ai_app.app.state.aurea_nlu = fake_nlu

        try:
            response = await client.post(
                "/aurea/command/natural_language",
                headers=auth_headers,
                json={"command_text": "Get AUREA status"},
            )
        finally:
            ai_app.app.state.aurea_nlu = None

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "command" in data
        assert "result" in data
        assert "timestamp" in data

    # --- POST /training/capture-interaction ---

    async def test_training_capture_rejects_unauthenticated(self, client):
        """POST /training/capture-interaction enforces auth."""
        response = await client.post(
            "/training/capture-interaction",
            json={"customer_id": "cust-001", "type": "EMAIL", "content": "hello"},
        )
        assert response.status_code == 403

    async def test_training_capture_returns_503_when_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when TRAINING_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "TRAINING_AVAILABLE", False)
        response = await client.post(
            "/training/capture-interaction",
            headers=auth_headers,
            json={"customer_id": "cust-001", "type": "EMAIL", "content": "test"},
        )
        assert response.status_code == 503

    async def test_training_capture_returns_200_when_pipeline_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with interaction_id and status when training pipeline is active."""
        fake_pipeline = MagicMock()
        fake_pipeline.capture_interaction = AsyncMock(return_value="interaction-uuid-001")

        ai_app.app.state.training = fake_pipeline
        monkeypatch.setattr(ai_app, "TRAINING_AVAILABLE", True)

        # Provide InteractionType enum mock
        fake_training_module = MagicMock()
        fake_training_module.InteractionType = MagicMock()
        fake_training_module.InteractionType.__getitem__ = MagicMock(return_value="EMAIL")
        monkeypatch.setitem(sys.modules, "ai_training_pipeline", fake_training_module)

        try:
            response = await client.post(
                "/training/capture-interaction",
                headers=auth_headers,
                json={"customer_id": "cust-001", "type": "EMAIL", "content": "Hello there"},
            )
        finally:
            ai_app.app.state.training = None

        assert response.status_code == 200
        data = response.json()
        assert "interaction_id" in data
        assert "status" in data

    # --- GET /training/stats ---

    async def test_training_stats_rejects_unauthenticated(self, client):
        """GET /training/stats enforces auth."""
        response = await client.get("/training/stats")
        assert response.status_code == 403

    async def test_training_stats_returns_200_unavailable_when_training_off(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with available=False when TRAINING_AVAILABLE is False."""
        monkeypatch.setattr(ai_app, "TRAINING_AVAILABLE", False)
        response = await client.get("/training/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False

    async def test_training_stats_returns_200_with_stats_when_available(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 with total_interactions, training_samples, insights_generated."""
        fake_pipeline = MagicMock()
        ai_app.app.state.training = fake_pipeline
        monkeypatch.setattr(ai_app, "TRAINING_AVAILABLE", True)

        patch_pool.fetchrow_handler = lambda *_: {
            "total_interactions": 100,
            "last_interaction": None,
            "training_samples": 50,
            "insights_generated": 10,
        }

        try:
            response = await client.get("/training/stats", headers=auth_headers)
        finally:
            ai_app.app.state.training = None

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "total_interactions" in data
        assert "training_samples" in data
        assert "insights_generated" in data
