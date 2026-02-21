"""
Wave 2B route contract tests.

Verifies that every route extracted in Wave 2B (api/scheduler.py and
api/agents.py) satisfies its contract:

  1. The route exists and returns the expected HTTP status code.
  2. Auth is enforced: a request without an API key must be rejected
     with 403 (the application always returns 403, never 401, when no
     credential is supplied).
  3. Every field documented as "required" in the route's response
     envelope is present in the JSON body.

All tests are pure contract checks - no behavioral assertions.
DB interactions are fully mocked via the conftest ``patch_pool``
fixture, and optional app-level singletons (scheduler, health
monitor, resolver, etc.) are monkeypatched as required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as ai_app
import api.agents as agents_api
import api.scheduler as scheduler_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AGENT_UUID = "2c6b12f7-90a9-4d0f-b7f0-0f3de0e9332a"
_AGENT_ROW = {
    "id": _AGENT_UUID,
    "name": "Revenue Agent",
    "type": "revenue",
    "category": "analytics",
    "enabled": True,
    "description": "Revenue optimisation agent",
    "capabilities": [],
    "configuration": {},
    "schedule_hours": [],
    "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "status": "active",
    "total_executions": 0,
    "last_active": None,
}


def _make_agent_fetchrow(agent_row: dict | None = None):
    """Return a fetchrow_handler that serves the given agent row for agents queries."""
    row = agent_row if agent_row is not None else _AGENT_ROW

    def handler(query, *_args):
        if "FROM agents" in query or "FROM ai_agents" in query:
            return row
        return None

    return handler


def _make_execution_fetch_handler():
    """Return a fetch_handler that returns empty execution history lists."""

    def handler(query, *_args):
        return []

    return handler


# ---------------------------------------------------------------------------
# Scheduler route contracts
# ---------------------------------------------------------------------------


class TestSchedulerEmailStats:
    """GET /email/scheduler-stats"""

    async def test_returns_200_with_auth(self, client, auth_headers, patch_pool, monkeypatch):
        """Route returns 200 when authenticated."""
        patch_pool.fetchrow_handler = lambda *_: {
            "queued": 0,
            "processing": 0,
            "sent": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
        }

        class _FakeDaemon:
            def get_stats(self):
                return {"running": True, "poll_count": 5}

        monkeypatch.setattr(
            scheduler_api,
            "_scheduler_available",
            lambda: False,
        )
        import sys

        fake_module = MagicMock()
        fake_module.get_email_scheduler = lambda: _FakeDaemon()
        monkeypatch.setitem(sys.modules, "email_scheduler_daemon", fake_module)

        response = await client.get("/email/scheduler-stats", headers=auth_headers)
        # The route catches ImportError and returns a soft error dict; if the
        # module IS importable we get the full envelope.  Either way the route
        # must exist (no 404/405) and be accessible.
        assert response.status_code in (200,)

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth - no API key returns 403."""
        response = await client.get("/email/scheduler-stats")
        assert response.status_code == 403

    async def test_response_contains_timestamp_on_success(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """When daemon is importable, envelope must contain daemon_stats, queue_counts, timestamp."""
        patch_pool.fetchrow_handler = lambda *_: {
            "queued": 1,
            "processing": 0,
            "sent": 10,
            "failed": 0,
            "skipped": 0,
            "total": 11,
        }

        class _FakeDaemon:
            def get_stats(self):
                return {"running": True, "poll_count": 5, "sent_count": 10}

        import sys

        fake_module = MagicMock()
        fake_module.get_email_scheduler = lambda: _FakeDaemon()
        monkeypatch.setitem(sys.modules, "email_scheduler_daemon", fake_module)

        response = await client.get("/email/scheduler-stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        # If the daemon module loaded, all three top-level keys must be present.
        if "error" not in data:
            assert "daemon_stats" in data
            assert "queue_counts" in data
            assert "timestamp" in data


class TestSchedulerStatus:
    """GET /scheduler/status"""

    async def test_returns_200_with_auth_when_scheduler_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with enabled=False when scheduler is not running."""
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)
        response = await client.get("/scheduler/status", headers=auth_headers)
        assert response.status_code == 200

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.get("/scheduler/status")
        assert response.status_code == 403

    async def test_response_contains_enabled_and_timestamp(self, client, auth_headers, monkeypatch):
        """Envelope must contain at minimum: enabled, timestamp."""
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)
        response = await client.get("/scheduler/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "timestamp" in data


class TestSchedulerRestartStuck:
    """POST /scheduler/restart-stuck"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/scheduler/restart-stuck")
        assert response.status_code == 403

    async def test_returns_200_with_required_fields(self, client, auth_headers, monkeypatch):
        """When resolver is available, envelope contains success, items_fixed, action, details, timestamp."""
        fake_result = SimpleNamespace(
            success=True,
            items_fixed=0,
            action=SimpleNamespace(value="fix_stuck"),
            details={"fixed": []},
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        class _FakeResolver:
            async def fix_stuck_agents(self):
                return fake_result

        import sys

        fake_module = MagicMock()
        fake_module.get_resolver = lambda: _FakeResolver()
        monkeypatch.setitem(sys.modules, "autonomous_issue_resolver", fake_module)

        response = await client.post("/scheduler/restart-stuck", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "items_fixed" in data
        assert "action" in data
        assert "details" in data
        assert "timestamp" in data

    async def test_returns_500_when_resolver_raises(self, client, auth_headers, monkeypatch):
        """Propagates 500 if resolver throws."""
        import sys

        fake_module = MagicMock()
        fake_module.get_resolver = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        # Make import succeed but get_resolver raise
        resolver_obj = MagicMock()
        resolver_obj.fix_stuck_agents = AsyncMock(side_effect=RuntimeError("resolver failed"))
        fake_module2 = MagicMock()
        fake_module2.get_resolver = lambda: resolver_obj
        monkeypatch.setitem(sys.modules, "autonomous_issue_resolver", fake_module2)

        response = await client.post("/scheduler/restart-stuck", headers=auth_headers)
        assert response.status_code == 500


class TestSchedulerActivateAll:
    """POST /scheduler/activate-all"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/scheduler/activate-all")
        assert response.status_code == 403

    async def test_returns_503_when_scheduler_unavailable(self, client, auth_headers, monkeypatch):
        """Returns 503 when app scheduler is not running."""
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)
        response = await client.post("/scheduler/activate-all", headers=auth_headers)
        assert response.status_code == 503

    async def test_returns_200_with_required_fields_when_scheduler_available(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains success, new_schedules, already_scheduled, total_agents."""
        # Provide a fake scheduler on app.state
        fake_scheduler = MagicMock()
        fake_scheduler.add_schedule = MagicMock()
        ai_app.app.state.scheduler = fake_scheduler
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", True)

        # fetch_active_agents and fetch_scheduled_agent_ids both hit pool.fetch
        def fetch_handler(query, *_args):
            if "FROM ai_agents" in query:
                return [
                    {
                        "id": _AGENT_UUID,
                        "name": "Revenue Agent",
                        "type": "revenue",
                        "category": "analytics",
                    }
                ]
            if "FROM agent_schedules" in query:
                return []
            return []

        patch_pool.fetch_handler = fetch_handler

        try:
            response = await client.post("/scheduler/activate-all", headers=auth_headers)
        finally:
            # Clean up state to avoid bleed into other tests
            ai_app.app.state.scheduler = None

        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "new_schedules" in data
        assert "already_scheduled" in data
        assert "total_agents" in data


class TestAgentsSchedule:
    """POST /agents/schedule"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/agents/schedule", json={"agent_id": "some-agent-id"})
        assert response.status_code == 403

    async def test_returns_200_with_required_fields_on_success(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains success, action, schedule_id."""
        schedule_id = str(uuid.uuid4())

        def fetchrow_handler(query, *args):
            if "SELECT id, name FROM agents" in query:
                return {"id": _AGENT_UUID, "name": "Revenue Agent"}
            if "SELECT id FROM agent_schedules" in query:
                return None  # No existing schedule → insert path
            return None

        patch_pool.fetchrow_handler = fetchrow_handler
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)

        response = await client.post(
            "/agents/schedule",
            headers=auth_headers,
            json={"agent_id": _AGENT_UUID, "frequency_minutes": 60},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "action" in data
        assert "schedule_id" in data

    async def test_returns_404_when_agent_not_found(self, client, auth_headers, patch_pool):
        """Returns 404 when the requested agent does not exist in the database."""
        patch_pool.fetchrow_handler = lambda *_: None

        response = await client.post(
            "/agents/schedule",
            headers=auth_headers,
            json={"agent_id": "nonexistent-agent"},
        )
        assert response.status_code == 404

    async def test_returns_422_when_agent_id_missing(self, client, auth_headers):
        """FastAPI validation rejects missing required body field."""
        response = await client.post(
            "/agents/schedule",
            headers=auth_headers,
            json={"frequency_minutes": 30},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Agent route contracts
# ---------------------------------------------------------------------------


class TestAgentsProductRun:
    """POST /agents/product/run"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/agents/product/run", json={"concept": "test"})
        assert response.status_code == 403

    async def test_returns_503_when_product_agent_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when the LangGraph Product Agent is not installed."""
        monkeypatch.setattr(ai_app, "PRODUCT_AGENT_AVAILABLE", False)
        response = await client.post(
            "/agents/product/run", headers=auth_headers, json={"concept": "new SaaS tool"}
        )
        assert response.status_code == 503

    async def test_returns_200_with_required_fields_when_available(
        self, client, auth_headers, monkeypatch
    ):
        """Envelope contains status, result, trace when product agent runs successfully."""
        mock_msg = MagicMock()
        mock_msg.content = "Generated product spec"
        fake_graph = MagicMock()
        fake_graph.invoke = MagicMock(return_value={"messages": [mock_msg, mock_msg]})
        monkeypatch.setattr(ai_app, "PRODUCT_AGENT_AVAILABLE", True)
        monkeypatch.setattr(ai_app, "product_agent_graph", fake_graph)

        response = await client.post(
            "/agents/product/run",
            headers=auth_headers,
            json={"concept": "AI invoice scanner"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "result" in data
        assert "trace" in data


class TestGetAgents:
    """GET /agents"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.get("/agents")
        assert response.status_code == 403

    async def test_returns_200_with_required_fields(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains agents list, total, page, page_size."""
        patch_pool.fetchval_handler = lambda *_: 1
        patch_pool.fetch_handler = lambda query, *_args: (
            [_AGENT_ROW] if "FROM agents a" in query else []
        )

        response = await client.get("/agents", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    async def test_returns_empty_agents_list_when_db_is_empty(
        self, client, auth_headers, patch_pool
    ):
        """Handles empty database gracefully."""
        patch_pool.fetchval_handler = lambda *_: 0
        patch_pool.fetch_handler = lambda *_: []

        response = await client.get("/agents", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["agents"] == []


class TestExecuteAgentById:
    """POST /agents/{agent_id}/execute"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post(f"/agents/{_AGENT_UUID}/execute", json={})
        assert response.status_code == 403

    async def test_returns_200_with_required_fields_on_success(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains agent_id, execution_id, status."""
        patch_pool.fetchrow_handler = _make_agent_fetchrow()

        monkeypatch.setattr(ai_app, "AGENTS_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)

        response = await client.post(
            f"/agents/{_AGENT_UUID}/execute",
            headers=auth_headers,
            json={"task": "run analysis"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "execution_id" in data
        assert "status" in data

    async def test_returns_404_when_agent_not_found(self, client, auth_headers, patch_pool):
        """Returns 404 for an unknown agent_id."""
        patch_pool.fetchrow_handler = lambda *_: None
        response = await client.post(
            "/agents/nonexistent-agent/execute",
            headers=auth_headers,
            json={"task": "run"},
        )
        assert response.status_code == 404


class TestGetAgentsStatus:
    """GET /agents/status"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.get("/agents/status")
        assert response.status_code == 403

    async def test_returns_200_with_required_fields(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains total_agents, agents list, timestamp."""
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)

        patch_pool.fetch_handler = lambda query, *_: (
            [
                {
                    "id": _AGENT_UUID,
                    "name": "Revenue Agent",
                    "type": "revenue",
                    "status": "active",
                    "last_active": None,
                    "total_executions": 0,
                    "scheduled": False,
                    "frequency_minutes": None,
                    "last_execution": None,
                    "next_execution": None,
                }
            ]
            if "FROM ai_agents" in query
            else []
        )

        response = await client.get("/agents/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_agents" in data
        assert "agents" in data
        assert "timestamp" in data


class TestGetAgentById:
    """GET /agents/{agent_id}"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.get(f"/agents/{_AGENT_UUID}")
        assert response.status_code == 403

    async def test_returns_200_with_agent_model_fields(self, client, auth_headers, patch_pool):
        """Returns a valid Agent model with id, name, category, enabled."""
        patch_pool.fetchrow_handler = _make_agent_fetchrow()

        response = await client.get(f"/agents/{_AGENT_UUID}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "category" in data
        assert "enabled" in data

    async def test_returns_404_for_unknown_agent(self, client, auth_headers, patch_pool):
        """Returns 404 when the agent is not in the database."""
        patch_pool.fetchrow_handler = lambda *_: None
        response = await client.get("/agents/missing-agent", headers=auth_headers)
        assert response.status_code == 404


class TestGetAgentHistory:
    """GET /agents/{agent_id}/history"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.get(f"/agents/{_AGENT_UUID}/history")
        assert response.status_code == 403

    async def test_returns_200_with_required_fields(self, client, auth_headers, patch_pool):
        """Envelope contains agent_id, agent_name, history, count."""
        patch_pool.fetchrow_handler = lambda query, *_: (
            {"id": _AGENT_UUID, "name": "Revenue Agent"} if "FROM agents" in query else None
        )
        patch_pool.fetch_handler = _make_execution_fetch_handler()

        response = await client.get(f"/agents/{_AGENT_UUID}/history?limit=10", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "agent_name" in data
        assert "history" in data
        assert "count" in data

    async def test_returns_404_for_unknown_agent(self, client, auth_headers, patch_pool):
        """Returns 404 for unknown agent."""
        patch_pool.fetchrow_handler = lambda *_: None
        response = await client.get("/agents/unknown/history", headers=auth_headers)
        assert response.status_code == 404


class TestExecuteScheduledAgents:
    """POST /execute"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/execute", json={})
        assert response.status_code == 403

    async def test_returns_200_with_required_fields_when_scheduler_disabled(
        self, client, auth_headers, monkeypatch
    ):
        """Returns soft 200 envelope when scheduler is disabled."""
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", False)

        response = await client.post("/execute", headers=auth_headers, json={})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    async def test_returns_200_with_full_envelope_when_scheduler_enabled(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains status, executed, results, timestamp when scheduler runs."""
        fake_scheduler = MagicMock()
        ai_app.app.state.scheduler = fake_scheduler
        monkeypatch.setattr(ai_app, "SCHEDULER_AVAILABLE", True)
        monkeypatch.setattr(ai_app, "AGENTS_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "AGENT_EXECUTOR", None)

        # No scheduled agents found for current hour
        patch_pool.fetch_handler = lambda *_: []

        try:
            response = await client.post("/execute", headers=auth_headers, json={})
        finally:
            ai_app.app.state.scheduler = None

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "executed" in data
        assert "results" in data
        assert "timestamp" in data


class TestAgentsHealthCheck:
    """POST /agents/health/check"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/agents/health/check")
        assert response.status_code == 403

    async def test_returns_503_when_health_monitor_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when health monitor module is not installed."""
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)
        response = await client.post("/agents/health/check", headers=auth_headers)
        assert response.status_code == 503

    async def test_returns_200_when_health_monitor_available(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 200 with health check result when monitor is available."""
        fake_monitor = MagicMock()
        fake_monitor.check_all_agents_health = MagicMock(
            return_value={"total_agents": 3, "healthy": 3, "degraded": 0, "critical": 0}
        )
        import sys

        fake_module = MagicMock()
        fake_module.get_health_monitor = lambda: fake_monitor
        monkeypatch.setitem(sys.modules, "agent_health_monitor", fake_module)
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", True)

        response = await client.post("/agents/health/check", headers=auth_headers)
        assert response.status_code == 200


class TestAgentRestart:
    """POST /agents/{agent_id}/restart"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post(f"/agents/{_AGENT_UUID}/restart")
        assert response.status_code == 403

    async def test_returns_503_when_health_monitor_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when health monitor is not installed."""
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)
        response = await client.post(f"/agents/{_AGENT_UUID}/restart", headers=auth_headers)
        assert response.status_code == 503

    async def test_returns_200_on_successful_restart(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 200 restart result when monitor and agent are present."""
        patch_pool.fetchrow_handler = lambda query, *_: (
            {"name": "Revenue Agent"} if "FROM ai_agents" in query else None
        )
        fake_monitor = MagicMock()
        fake_monitor.restart_failed_agent = MagicMock(
            return_value={"success": True, "message": "Restarted"}
        )
        import sys

        fake_module = MagicMock()
        fake_module.get_health_monitor = lambda: fake_monitor
        monkeypatch.setitem(sys.modules, "agent_health_monitor", fake_module)
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", True)

        response = await client.post(f"/agents/{_AGENT_UUID}/restart", headers=auth_headers)
        assert response.status_code == 200

    async def test_returns_404_when_agent_not_found(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Returns 404 when the agent is absent from ai_agents."""
        patch_pool.fetchrow_handler = lambda *_: None
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", True)
        import sys

        fake_module = MagicMock()
        monkeypatch.setitem(sys.modules, "agent_health_monitor", fake_module)

        response = await client.post("/agents/missing-id/restart", headers=auth_headers)
        assert response.status_code == 404


class TestAgentsHealthAutoRestart:
    """POST /agents/health/auto-restart"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/agents/health/auto-restart")
        assert response.status_code == 403

    async def test_returns_503_when_health_monitor_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when health monitor is not installed."""
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)
        response = await client.post("/agents/health/auto-restart", headers=auth_headers)
        assert response.status_code == 503

    async def test_returns_200_when_monitor_available(self, client, auth_headers, monkeypatch):
        """Returns 200 auto-restart result when monitor runs successfully."""
        fake_monitor = MagicMock()
        fake_monitor.auto_restart_critical_agents = MagicMock(
            return_value={"restarted": [], "skipped": []}
        )
        import sys

        fake_module = MagicMock()
        fake_module.get_health_monitor = lambda: fake_monitor
        monkeypatch.setitem(sys.modules, "agent_health_monitor", fake_module)
        monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", True)

        response = await client.post("/agents/health/auto-restart", headers=auth_headers)
        assert response.status_code == 200


class TestAgentsExecuteGeneric:
    """POST /agents/execute"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post(
            "/agents/execute", json={"agent_type": "revenue", "task": "run"}
        )
        assert response.status_code == 403

    async def test_returns_200_with_required_fields_on_success(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains success, execution_id, agent_id, agent_name."""

        class _FakeExecutor:
            async def execute(self, agent_name, task):
                return {"status": "completed", "output": "done"}

        monkeypatch.setattr(ai_app, "AGENTS_AVAILABLE", True)
        monkeypatch.setattr(ai_app, "AGENT_EXECUTOR", _FakeExecutor())

        patch_pool.fetchrow_handler = lambda query, *_: (
            {"id": _AGENT_UUID, "name": "Revenue Agent", "type": "revenue"}
            if "FROM agents" in query
            else None
        )

        response = await client.post(
            "/agents/execute",
            headers=auth_headers,
            json={"agent_type": "revenue", "task": "analyse pipeline", "parameters": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "execution_id" in data
        assert "agent_id" in data
        assert "agent_name" in data

    async def test_returns_422_when_parameters_is_wrong_type(self, client, auth_headers):
        """FastAPI validation rejects non-dict parameters value."""
        response = await client.post(
            "/agents/execute",
            headers=auth_headers,
            json={"agent_type": "revenue", "task": "x", "parameters": ["invalid"]},
        )
        assert response.status_code == 422


class TestExecuteAiTask:
    """POST /ai/tasks/execute/{task_id}"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/ai/tasks/execute/task-123")
        assert response.status_code == 403

    async def test_returns_503_when_integration_layer_unavailable(
        self, client, auth_headers, monkeypatch
    ):
        """Returns 503 when the integration layer is not initialised."""
        monkeypatch.setattr(ai_app, "INTEGRATION_LAYER_AVAILABLE", False)
        response = await client.post("/ai/tasks/execute/task-123", headers=auth_headers)
        assert response.status_code == 503

    async def test_returns_200_with_required_fields_on_success(
        self, client, auth_headers, monkeypatch
    ):
        """Envelope contains success, message, task_id."""
        task_id = "task-abc-001"
        fake_task = {"id": task_id, "status": "pending"}

        class _FakeIntegration:
            async def get_task_status(self, tid):
                return fake_task if tid == task_id else None

            async def _execute_task(self, task):
                pass

        ai_app.app.state.integration_layer = _FakeIntegration()
        monkeypatch.setattr(ai_app, "INTEGRATION_LAYER_AVAILABLE", True)

        try:
            response = await client.post(f"/ai/tasks/execute/{task_id}", headers=auth_headers)
        finally:
            ai_app.app.state.integration_layer = None

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert data["task_id"] == task_id

    async def test_returns_404_when_task_not_found(self, client, auth_headers, monkeypatch):
        """Returns 404 when the task_id is unknown."""

        class _FakeIntegration:
            async def get_task_status(self, tid):
                return None

        ai_app.app.state.integration_layer = _FakeIntegration()
        monkeypatch.setattr(ai_app, "INTEGRATION_LAYER_AVAILABLE", True)

        try:
            response = await client.post("/ai/tasks/execute/nonexistent-task", headers=auth_headers)
        finally:
            ai_app.app.state.integration_layer = None

        assert response.status_code == 404


class TestApiV1AgentsExecute:
    """POST /api/v1/agents/execute"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post(
            "/api/v1/agents/execute", json={"agent_id": _AGENT_UUID, "payload": {}}
        )
        assert response.status_code == 403

    async def test_returns_400_when_agent_id_missing(self, client, auth_headers):
        """Returns 400 when neither agent_id nor id is provided."""
        response = await client.post(
            "/api/v1/agents/execute",
            headers=auth_headers,
            json={"payload": {}},
        )
        assert response.status_code == 400

    async def test_delegates_to_execute_agent_and_returns_execution_fields(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Delegates to /agents/{agent_id}/execute and returns agent_id, execution_id, status."""
        patch_pool.fetchrow_handler = _make_agent_fetchrow()
        monkeypatch.setattr(ai_app, "AGENTS_AVAILABLE", False)
        monkeypatch.setattr(ai_app, "AI_AVAILABLE", False)

        response = await client.post(
            "/api/v1/agents/execute",
            headers=auth_headers,
            json={"agent_id": _AGENT_UUID, "payload": {"task": "audit"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "execution_id" in data
        assert "status" in data


class TestApiV1AgentsActivate:
    """POST /api/v1/agents/activate"""

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post(
            "/api/v1/agents/activate", json={"agent_id": _AGENT_UUID, "enabled": True}
        )
        assert response.status_code == 403

    async def test_returns_400_when_neither_id_nor_name_provided(self, client, auth_headers):
        """Returns 400 when the request body lacks agent_id and agent_name."""
        response = await client.post(
            "/api/v1/agents/activate",
            headers=auth_headers,
            json={"enabled": True},
        )
        assert response.status_code == 400

    async def test_returns_200_with_success_and_agent_fields(
        self, client, auth_headers, patch_pool
    ):
        """Envelope contains success and agent sub-object with id, name, category, enabled."""
        patch_pool.fetchrow_handler = lambda query, *_: (
            {
                "id": _AGENT_UUID,
                "name": "Revenue Agent",
                "category": "analytics",
                "enabled": True,
            }
            if "UPDATE agents" in query
            else None
        )

        response = await client.post(
            "/api/v1/agents/activate",
            headers=auth_headers,
            json={"agent_id": _AGENT_UUID, "enabled": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        agent = data["agent"]
        assert "id" in agent
        assert "name" in agent
        assert "category" in agent
        assert "enabled" in agent

    async def test_returns_404_when_agent_not_found(self, client, auth_headers, patch_pool):
        """Returns 404 when no matching row is found in agents table."""
        patch_pool.fetchrow_handler = lambda *_: None

        response = await client.post(
            "/api/v1/agents/activate",
            headers=auth_headers,
            json={"agent_id": "nonexistent-id", "enabled": True},
        )
        assert response.status_code == 404


class TestApiV1AureaExecuteEvent:
    """POST /api/v1/aurea/execute-event"""

    _VALID_PAYLOAD = {
        "event_id": "evt-001",
        "topic": "revenue.opportunity",
        "source": "event_bus",
        "payload": {"amount": 5000},
        "target_agent": {
            "name": "Revenue Agent",
            "role": "revenue",
            "capabilities": ["analyse"],
        },
    }

    async def test_rejects_unauthenticated_requests(self, client):
        """Route enforces auth."""
        response = await client.post("/api/v1/aurea/execute-event", json=self._VALID_PAYLOAD)
        assert response.status_code == 403

    async def test_returns_422_when_required_fields_missing(self, client, auth_headers):
        """FastAPI validates required Pydantic fields in the request body."""
        response = await client.post(
            "/api/v1/aurea/execute-event",
            headers=auth_headers,
            json={"event_id": "evt-001"},  # missing topic, source, payload, target_agent
        )
        assert response.status_code == 422

    async def test_returns_200_with_required_fields_on_success(
        self, client, auth_headers, patch_pool, monkeypatch
    ):
        """Envelope contains success, event_id, agent, topic, result."""
        patch_pool.fetchrow_handler = lambda query, *_: (
            {
                "id": _AGENT_UUID,
                "name": "Revenue Agent",
                "category": "analytics",
                "enabled": True,
            }
            if "FROM agents" in query
            else None
        )

        response = await client.post(
            "/api/v1/aurea/execute-event",
            headers=auth_headers,
            json=self._VALID_PAYLOAD,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "event_id" in data
        assert "agent" in data
        assert "topic" in data
        assert "result" in data

    async def test_returns_404_when_target_agent_not_found(self, client, auth_headers, patch_pool):
        """Returns 404 when the named agent is absent or disabled."""
        patch_pool.fetchrow_handler = lambda *_: None

        response = await client.post(
            "/api/v1/aurea/execute-event",
            headers=auth_headers,
            json=self._VALID_PAYLOAD,
        )
        assert response.status_code == 404
