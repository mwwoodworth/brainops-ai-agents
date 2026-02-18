from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import api.brain as brain_api
import api.campaigns as campaigns_api
import api.revenue_automation as revenue_automation_api
import api.taskmate as taskmate_api
import app as ai_app
import optimization.revenue_prompt_optimizer as optimizer_mod
import unified_memory_manager as unified_memory_manager


class _FakeLead:
    def __init__(self, lead_id: str, name: str):
        self.lead_id = lead_id
        self.email = f"{name.lower()}@example.com"
        self.name = name
        self.company = f"{name} Co"
        self.industry = SimpleNamespace(value="roofing")
        self.source = SimpleNamespace(value="api")
        self.status = SimpleNamespace(value="new")
        self.score = 0.8
        self.estimated_value = 12000.0
        self.created_at = "2026-02-18T00:00:00Z"
        self.updated_at = "2026-02-18T00:00:00Z"
        self.contacted_at = None
        self.converted_at = None
        self.custom_fields = {}
        self.automation_history = []
        self.notes = None


class _FakeRevenueEngine:
    def __init__(self):
        self.leads = {
            "lead-1": _FakeLead("lead-1", "Ava"),
            "lead-2": _FakeLead("lead-2", "Blake"),
            "lead-3": _FakeLead("lead-3", "Casey"),
        }

    async def capture_lead(self, **kwargs):
        return {"ok": True, "lead_id": "lead-new", "input": kwargs}

    def get_revenue_metrics(self):
        return {
            "total_revenue": 25000.0,
            "monthly_revenue": 5000.0,
            "pipeline_value": 80000.0,
            "total_leads": 3,
            "qualified_leads": 2,
            "won_leads": 1,
            "conversion_rate": 0.33,
            "revenue_by_industry": {"roofing": 25000.0},
            "revenue_by_source": {"api": 25000.0},
        }

    def get_pipeline_dashboard(self):
        return {
            "leads_by_stage": {"new": 2, "qualified": 1},
            "value_by_stage": {"new": 30000.0, "qualified": 50000.0},
            "pipeline_value": 80000.0,
        }


def _taskmate_ready(monkeypatch):
    monkeypatch.setattr(taskmate_api, "using_fallback", lambda: False)
    monkeypatch.setattr(taskmate_api, "verify_tables_async", AsyncMock(return_value=True))


def _mock_campaign(campaign_id: str = "roofing-campaign"):
    return SimpleNamespace(
        id=campaign_id,
        name="Roofing Campaign",
        is_active=True,
        templates=[
            SimpleNamespace(step=1, delay_days=0, subject="Intro", call_to_action="Reply now"),
        ],
        handoff_partner=None,
    )


async def test_authentication_required_for_secured_endpoint(client):
    response = await client.get("/agents/status")
    assert response.status_code == 403
    assert "authentication" in response.json()["detail"].lower()


async def test_health_endpoint_returns_full_payload_fields(client, auth_headers, patch_pool, monkeypatch):
    monkeypatch.setattr(ai_app, "EMBEDDED_MEMORY_AVAILABLE", False)
    monkeypatch.setattr(ai_app, "SERVICE_CIRCUIT_BREAKERS_AVAILABLE", False)

    response = await client.get("/health?force_refresh=true", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    required_fields = {
        "status",
        "version",
        "build",
        "database",
        "db_pool",
        "active_systems",
        "system_count",
        "embedded_memory_active",
        "capabilities",
        "circuit_breakers",
        "config",
        "missing_systems",
    }
    assert required_fields.issubset(data.keys())


async def test_healthz_quick_health_check(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "build" in data


async def test_brain_store_success(client, auth_headers, sample_brain_entry, monkeypatch):
    fake_brain = SimpleNamespace(store=AsyncMock(return_value="brain-entry-1"))
    monkeypatch.setattr(brain_api, "BRAIN_AVAILABLE", True)
    monkeypatch.setattr(brain_api, "brain", fake_brain)

    response = await client.post("/brain/store", headers=auth_headers, json=sample_brain_entry)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "stored"
    assert data["key"] == sample_brain_entry["key"]


async def test_brain_store_validation_error_returns_422(client, auth_headers):
    response = await client.post("/brain/store", headers=auth_headers, json={"value": "missing key"})
    assert response.status_code == 422


async def test_brain_recall_success(client, auth_headers, monkeypatch):
    class FakeMemoryManager:
        def __init__(self):
            self.tenant_id = "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
            self.last_call = None

        def recall(self, query, tenant_id=None, context=None, limit=10, memory_type=None):
            self.last_call = {
                "query": query,
                "tenant_id": tenant_id,
                "context": context,
                "limit": limit,
                "memory_type": memory_type,
            }
            return [{"id": "mem-1", "content": "roof inspection", "embedding": [0.1, 0.2]}]

    fake_memory = FakeMemoryManager()
    monkeypatch.setattr(unified_memory_manager, "get_memory_manager", lambda: fake_memory)

    response = await client.post(
        "/brain/recall",
        headers=auth_headers,
        json={"query": "roof inspection", "limit": 5, "memory_type": "semantic"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["count"] == 1
    assert "embedding" not in data["results"][0]
    assert fake_memory.last_call["tenant_id"] == "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"


async def test_brain_recall_validation_error_returns_422(client, auth_headers):
    response = await client.post("/brain/recall", headers=auth_headers, json={"limit": 2})
    assert response.status_code == 422


async def test_brain_critical_success(client, auth_headers, monkeypatch):
    fake_brain = SimpleNamespace(
        get_all_critical=AsyncMock(
            return_value=[{"key": "system:prod", "value": "up", "metadata": {"scope": "global"}}]
        )
    )
    monkeypatch.setattr(brain_api, "BRAIN_AVAILABLE", True)
    monkeypatch.setattr(brain_api, "brain", fake_brain)

    response = await client.get("/brain/critical", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["count"] == 1
    assert data["critical_items"][0]["key"] == "system:prod"


async def test_brain_get_specific_key_success(client, auth_headers, monkeypatch):
    fake_brain = SimpleNamespace(
        get=AsyncMock(return_value={"key": "release:current", "value": {"version": "v1"}})
    )
    monkeypatch.setattr(brain_api, "BRAIN_AVAILABLE", True)
    monkeypatch.setattr(brain_api, "brain", fake_brain)

    response = await client.get("/brain/get/release:current", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["key"] == "release:current"


async def test_brain_get_specific_key_not_found_returns_404(client, auth_headers, monkeypatch):
    fake_brain = SimpleNamespace(get=AsyncMock(return_value=None))
    monkeypatch.setattr(brain_api, "BRAIN_AVAILABLE", True)
    monkeypatch.setattr(brain_api, "brain", fake_brain)

    response = await client.get("/brain/get/missing-key", headers=auth_headers)
    assert response.status_code == 404


async def test_agents_status_returns_registry_snapshot(client, auth_headers, patch_pool, monkeypatch):
    monkeypatch.setattr(ai_app, "HEALTH_MONITOR_AVAILABLE", False)

    def fetch_handler(query, *_args):
        if "FROM ai_agents a" in query:
            return [
                {
                    "id": "agent-1",
                    "name": "Revenue Agent",
                    "type": "revenue",
                    "status": "active",
                    "last_active": None,
                    "total_executions": 3,
                    "scheduled": True,
                    "frequency_minutes": 30,
                    "last_execution": None,
                    "next_execution": None,
                }
            ]
        return []

    patch_pool.fetch_handler = fetch_handler

    response = await client.get("/agents/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_agents"] == 1
    assert data["agents"][0]["name"] == "Revenue Agent"


async def test_agents_execute_success(client, auth_headers, patch_pool, monkeypatch):
    class FakeExecutor:
        async def execute(self, agent_name, task):
            return {"status": "completed", "agent_name": agent_name, "task": task["task"]}

    monkeypatch.setattr(ai_app, "AGENTS_AVAILABLE", True)
    monkeypatch.setattr(ai_app, "AGENT_EXECUTOR", FakeExecutor())

    patch_pool.fetchrow_handler = lambda query, *_args: (
        {"id": "2c6b12f7-90a9-4d0f-b7f0-0f3de0e9332a", "name": "Revenue Agent", "type": "revenue"}
        if "FROM agents" in query
        else None
    )

    response = await client.post(
        "/agents/execute",
        headers=auth_headers,
        json={"agent_type": "revenue", "task": "follow up lead", "parameters": {"priority": "high"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["status"] == "completed"


async def test_agents_execute_validation_error_returns_422(client, auth_headers):
    response = await client.post(
        "/agents/execute",
        headers=auth_headers,
        json={"agent_type": "revenue", "task": "x", "parameters": []},
    )
    assert response.status_code == 422


async def test_agents_history_success(client, auth_headers, patch_pool):
    patch_pool.fetchrow_handler = lambda query, *_args: (
        {"id": "2c6b12f7-90a9-4d0f-b7f0-0f3de0e9332a", "name": "Revenue Agent"}
        if "FROM agents" in query
        else None
    )
    patch_pool.fetch_handler = lambda query, *_args: (
        [
            {
                "id": "exec-1",
                "status": "completed",
                "task_type": "revenue",
                "input_data": {"task": "follow up"},
                "output_data": {"ok": True},
                "error_message": None,
                "execution_time_ms": 150,
                "created_at": datetime(2026, 2, 18, tzinfo=timezone.utc),
            }
        ]
        if "FROM ai_agent_executions" in query
        else []
    )

    response = await client.get("/agents/Revenue Agent/history?limit=1", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["agent_name"] == "Revenue Agent"
    assert data["count"] == 1
    assert data["history"][0]["execution_id"] == "exec-1"


async def test_agents_history_not_found_returns_404(client, auth_headers, patch_pool):
    patch_pool.fetchrow_handler = lambda *_args: None

    response = await client.get("/agents/missing-agent/history", headers=auth_headers)
    assert response.status_code == 404


async def test_taskmate_list_tasks_respects_tenant_isolation(
    client, auth_headers, patch_pool, tenant_ids, monkeypatch
):
    _taskmate_ready(monkeypatch)

    def fetch_handler(_query, *args):
        tenant_id = args[0]
        if tenant_id == tenant_ids["tenant_a"]:
            return [{"id": "1", "task_id": "P7-TRUTH-001", "title": "Tenant A Task"}]
        if tenant_id == tenant_ids["tenant_b"]:
            return [{"id": "2", "task_id": "P7-TRUTH-002", "title": "Tenant B Task"}]
        return []

    patch_pool.fetch_handler = fetch_handler

    response_a = await client.get(
        "/taskmate/tasks",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
    )
    response_b = await client.get(
        "/taskmate/tasks",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_b"]},
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.json()["tasks"][0]["task_id"] == "P7-TRUTH-001"
    assert response_b.json()["tasks"][0]["task_id"] == "P7-TRUTH-002"


async def test_taskmate_create_task_success(
    client, auth_headers, patch_pool, sample_task, tenant_ids, monkeypatch
):
    _taskmate_ready(monkeypatch)
    patch_pool.fetchrow_handler = lambda query, *_args: (
        {"id": "task-row-1", "task_id": "P7-TRUTH-001", "status": "open"}
        if "INSERT INTO taskmate_tasks" in query
        else None
    )

    response = await client.post(
        "/taskmate/tasks",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
        json=sample_task,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["task_id"] == "P7-TRUTH-001"


async def test_taskmate_create_task_conflict_returns_409(
    client, auth_headers, patch_pool, sample_task, tenant_ids, monkeypatch
):
    _taskmate_ready(monkeypatch)
    patch_pool.fetchrow_handler = lambda *_args: None

    response = await client.post(
        "/taskmate/tasks",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
        json=sample_task,
    )
    assert response.status_code == 409


async def test_taskmate_update_task_success_string_task_id(
    client, auth_headers, patch_pool, tenant_ids, monkeypatch
):
    _taskmate_ready(monkeypatch)
    patch_pool.fetchrow_handler = lambda query, *_args: (
        {
            "id": "task-row-1",
            "task_id": "P7-TRUTH-001",
            "status": "closed",
            "updated_at": datetime(2026, 2, 18, tzinfo=timezone.utc),
        }
        if "UPDATE taskmate_tasks" in query
        else None
    )

    response = await client.patch(
        "/taskmate/tasks/P7-TRUTH-001",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
        json={"status": "closed", "evidence": "Verified"},
    )
    assert response.status_code == 200
    assert response.json()["task_id"] == "P7-TRUTH-001"


async def test_taskmate_update_task_validation_error_returns_422(client, auth_headers, tenant_ids):
    response = await client.patch(
        "/taskmate/tasks/P7-TRUTH-001",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
        json=[],
    )
    assert response.status_code == 422


async def test_taskmate_delete_task_success(client, auth_headers, patch_pool, tenant_ids, monkeypatch):
    _taskmate_ready(monkeypatch)
    patch_pool.fetchrow_handler = lambda query, *_args: (
        {"id": "task-row-1", "task_id": "P7-TRUTH-001"}
        if "DELETE FROM taskmate_tasks" in query
        else None
    )

    response = await client.delete(
        "/taskmate/tasks/P7-TRUTH-001",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
    )
    assert response.status_code == 200
    assert response.json()["deleted"] is True


async def test_taskmate_delete_task_not_found_returns_404(
    client, auth_headers, patch_pool, tenant_ids, monkeypatch
):
    _taskmate_ready(monkeypatch)
    patch_pool.fetchrow_handler = lambda *_args: None

    response = await client.delete(
        "/taskmate/tasks/P7-TRUTH-404",
        headers={**auth_headers, "X-Tenant-ID": tenant_ids["tenant_a"]},
    )
    assert response.status_code == 404


async def test_revenue_pipeline_success(client, auth_headers, monkeypatch):
    fake_engine = _FakeRevenueEngine()
    monkeypatch.setattr(revenue_automation_api, "_get_engine", AsyncMock(return_value=fake_engine))

    response = await client.get("/revenue/pipeline", headers=auth_headers)
    assert response.status_code == 200
    assert "leads_by_stage" in response.json()


async def test_revenue_metrics_success(client, auth_headers, monkeypatch):
    fake_engine = _FakeRevenueEngine()
    monkeypatch.setattr(revenue_automation_api, "_get_engine", AsyncMock(return_value=fake_engine))

    response = await client.get("/revenue/metrics", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["total_revenue"] == 25000.0


async def test_revenue_create_lead_success(client, auth_headers, monkeypatch):
    fake_engine = _FakeRevenueEngine()
    monkeypatch.setattr(revenue_automation_api, "_get_engine", AsyncMock(return_value=fake_engine))

    response = await client.post(
        "/revenue/leads",
        headers=auth_headers,
        json={
            "email": "lead@example.com",
            "name": "Lead Name",
            "industry": "roofing",
            "source": "api",
            "company": "Roof Co",
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


async def test_revenue_create_lead_validation_error_returns_422(client, auth_headers):
    response = await client.post(
        "/revenue/leads",
        headers=auth_headers,
        json={"email": "not-an-email", "name": "Lead", "industry": "roofing", "source": "api"},
    )
    assert response.status_code == 422


async def test_revenue_prompt_optimizer_status_success(client, auth_headers, monkeypatch):
    class FakeOptimizer:
        def status(self):
            return {"enabled": True, "compiled": True}

    monkeypatch.setattr(optimizer_mod, "get_revenue_prompt_optimizer", lambda: FakeOptimizer())

    response = await client.get("/revenue/prompt-optimizer/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"]["enabled"] is True


async def test_revenue_leads_list_pagination_success(client, auth_headers, monkeypatch):
    fake_engine = _FakeRevenueEngine()
    monkeypatch.setattr(revenue_automation_api, "_get_engine", AsyncMock(return_value=fake_engine))

    response = await client.get("/revenue/leads?limit=2", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["leads"]) == 2


async def test_revenue_leads_list_invalid_pagination_returns_422(client, auth_headers):
    response = await client.get("/revenue/leads?limit=1001", headers=auth_headers)
    assert response.status_code == 422


async def test_campaigns_enroll_success(client, auth_headers, patch_pool, monkeypatch):
    campaign = _mock_campaign()
    lead_id = str(uuid.uuid4())

    monkeypatch.setattr(campaigns_api, "get_campaign", lambda cid: campaign if cid == campaign.id else None)
    monkeypatch.setattr(
        campaigns_api,
        "personalize_template",
        lambda template, _lead, _campaign: (template.subject, "<p>Body</p>"),
    )
    monkeypatch.setattr(campaigns_api, "notify_lead_to_partner", AsyncMock(return_value=None))

    patch_pool.fetchrow_handler = lambda query, *args: (
        {
            "id": args[0],
            "company_name": "Acme Roofing",
            "contact_name": "Jordan",
            "email": "jordan@example.com",
            "metadata": {},
            "stage": "new",
            "source": "api",
        }
        if "SELECT * FROM revenue_leads WHERE id = $1" in query
        else None
    )

    response = await client.post(
        "/campaigns/enroll",
        headers=auth_headers,
        json={"campaign_id": campaign.id, "lead_id": lead_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["campaign_id"] == campaign.id


async def test_campaigns_enroll_not_found_returns_404(client, auth_headers, monkeypatch):
    monkeypatch.setattr(campaigns_api, "get_campaign", lambda _cid: None)

    response = await client.post(
        "/campaigns/enroll",
        headers=auth_headers,
        json={"campaign_id": "missing-campaign", "lead_id": str(uuid.uuid4())},
    )
    assert response.status_code == 404


async def test_campaigns_batch_enroll_success(client, auth_headers, patch_pool, monkeypatch):
    campaign = _mock_campaign("roofing-batch")
    lead_a = uuid.uuid4()
    lead_b = uuid.uuid4()

    monkeypatch.setattr(campaigns_api, "get_campaign", lambda cid: campaign if cid == campaign.id else None)
    monkeypatch.setattr(
        campaigns_api,
        "personalize_template",
        lambda template, _lead, _campaign: (template.subject, "<p>Body</p>"),
    )
    monkeypatch.setattr(campaigns_api, "notify_lead_to_partner", AsyncMock(return_value=None))

    patch_pool.fetch_handler = lambda query, *_args: (
        [{"id": lead_a}, {"id": lead_b}] if "SELECT id FROM revenue_leads" in query else []
    )
    patch_pool.fetchrow_handler = lambda query, *args: (
        {
            "id": args[0],
            "company_name": "Acme Roofing",
            "contact_name": "Jordan",
            "email": "jordan@example.com",
            "metadata": {},
            "stage": "new",
            "source": "api",
        }
        if "SELECT * FROM revenue_leads WHERE id = $1" in query
        else None
    )

    response = await client.post(
        "/campaigns/batch-enroll",
        headers=auth_headers,
        json={"campaign_id": campaign.id, "limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["campaign_id"] == campaign.id
    assert data["enrolled"] == 2


async def test_campaigns_batch_enroll_validation_error_returns_422(client, auth_headers):
    response = await client.post(
        "/campaigns/batch-enroll",
        headers=auth_headers,
        json={"campaign_id": "roofing", "limit": 999},
    )
    assert response.status_code == 422
