# Phase 2 Wave 2B — Change Log

**Date:** 2026-02-21
**Type:** Architecture refactor — bounded-context extraction from app.py
**app.py delta:** 7,674 → 6,250 lines (-1,424 lines)
**Cumulative Phase 2 delta:** 9,116 → 6,250 lines (-2,866 lines)
**Quality gates:** py_compile PASS, pytest 362 passed / 2 skipped / 0 failed
**Contract changes:** None (same paths, same methods, same auth on all routes)

---

## Files Created (NEW)

### api/scheduler.py — 263 lines

Extracted bounded context: scheduler and scheduling management.

Changes:
- Created APIRouter with tag "scheduler".
- Moved `email_scheduler_stats` handler (GET /email/scheduler-stats) with daemon stats + queue counts (line 76).
- Moved `get_scheduler_status` handler (GET /scheduler/status) with scheduler diagnostics (line 99).
- Moved `restart_stuck_executions` handler (POST /scheduler/restart-stuck) delegating to resolver (line 149).
- Moved `activate_all_agents_scheduler` handler (POST /scheduler/activate-all) with bulk scheduling (line 178).
- Moved `schedule_agent` handler (POST /agents/schedule) with schedule CRUD (line 242).
- Lazy imports from app module for SCHEDULER_AVAILABLE and app.state.scheduler.
- Imports: database.async_connection, services.scheduler_queries.

### api/agents.py — 1,253 lines

Extracted bounded context: agent CRUD, execution, dispatch, and health monitoring.

Changes:
- Created APIRouter with tag "agents".
- Defined Pydantic models: ProductRequest, AgentExecuteRequest, AgentActivateRequest, AUREAEventRequest.
- Moved `run_product_agent` handler (POST /agents/product/run) with LangGraph integration (line 155).
- Moved `get_agents` handler (GET /agents) with cache, pagination, execution stats (line 172).
- Moved `execute_agent` handler (POST /agents/{agent_id}/execute) with full execution lifecycle (line 325).
- Moved `get_all_agents_status` handler (GET /agents/status) with health monitor fallback (line 497).
- Moved `get_agent` handler (GET /agents/{agent_id}) with row_to_agent conversion (line 562).
- Moved `get_agent_history` handler (GET /agents/{agent_id}/history) with execution history (line 579).
- Moved `execute_scheduled_agents` handler (POST /execute) with cron-based scheduling (line 640).
- Moved `check_agents_health` handler (POST /agents/health/check) (line 817).
- Moved `restart_agent` handler (POST /agents/{agent_id}/restart) (line 833).
- Moved `auto_restart_critical_agents` handler (POST /agents/health/auto-restart) (line 860).
- Moved `execute_agent_generic` handler (POST /agents/execute) with type-based dispatch (line 878).
- Moved `execute_ai_task` handler (POST /ai/tasks/execute/{task_id}) with integration layer (line 1054).
- Moved `api_v1_agents_execute` handler (POST /api/v1/agents/execute) delegating to execute_agent (line 1080).
- Moved `api_v1_agents_activate` handler (POST /api/v1/agents/activate) with enable/disable (line 1114).
- Moved `execute_aurea_event` handler (POST /api/v1/aurea/execute-event) with AUREA dispatch (line 1157).
- Lazy imports from app module for feature flags, executor, limiter, local_executions.
- Imports: config, database.async_connection, models.agent, observability.TTLCache, services.agent_helpers.

### services/scheduler_queries.py — 111 lines

Shared helper: scheduler-related database queries.

Changes:
- Created `fetch_email_queue_counts() -> dict` — email queue status counts (line 18).
- Created `fetch_active_agents(pool) -> list` — active agents from ai_agents (line 33).
- Created `fetch_scheduled_agent_ids(pool) -> set[str]` — enabled schedule agent IDs (line 40).
- Created `insert_agent_schedule(pool, agent_id, frequency_minutes)` — new schedule row (line 47).
- Created `upsert_agent_schedule(pool, agent_id, frequency_minutes, enabled, schedule_id) -> dict|None` — create or update schedule (line 60).

### services/agent_helpers.py — 97 lines

Shared helper: agent data conversion utilities.

Changes:
- Created `_parse_capabilities(raw) -> list[dict]` — normalize capabilities payload (line 14).
- Created `_parse_configuration(raw) -> dict` — normalize configuration payload (line 61).
- Created `row_to_agent(row) -> Agent` — convert DB row to Agent model (line 77).
- All functions were previously inline in app.py.

### tests/test_wave2b_contracts.py — 1,134 lines

Contract tests for all Wave 2B extracted routes.

Changes:
- 64 test cases across 20 test classes.
- Covers all 20 Wave 2B routes.
- Tests: auth enforcement (403 without key), status codes, required response fields.
- Tests: error paths (404 for missing agents, 503 for unavailable services).

---

## Files Modified (EXISTING)

### app.py — 7,674 → 6,250 lines (-1,424 lines)

Changes:

**Imports added (lines ~275-276):**
```python
from api.agents import router as agents_router  # WAVE 2B: extracted agent CRUD/execution/dispatch
from api.scheduler import router as scheduler_router  # WAVE 2B: extracted scheduler/scheduling
```

**Router registrations added (lines ~2343-2348):**
```python
app.include_router(
    agents_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2B: /agents/*, /execute, /api/v1/agents/*, /api/v1/aurea/execute-event
app.include_router(
    scheduler_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2B: /scheduler/*, /agents/schedule, /email/scheduler-stats
```

**Inline route functions removed — Extraction 1 (Scheduler):**
- `email_scheduler_stats` (GET /email/scheduler-stats)
- `get_scheduler_status` (GET /scheduler/status)
- `restart_stuck_executions` (POST /scheduler/restart-stuck)
- `activate_all_agents_scheduler` (POST /scheduler/activate-all)
- `schedule_agent` (POST /agents/schedule)
- Total: 5 inline functions removed (~261 lines)

**Inline route functions removed — Extraction 2 (Agents):**
- `run_product_agent` (POST /agents/product/run)
- `get_agents` (GET /agents)
- `execute_agent` (POST /agents/{agent_id}/execute)
- `get_all_agents_status` (GET /agents/status)
- `get_agent` (GET /agents/{agent_id})
- `get_agent_history` (GET /agents/{agent_id}/history)
- `execute_scheduled_agents` (POST /execute)
- `check_agents_health` (POST /agents/health/check)
- `restart_agent` (POST /agents/{agent_id}/restart)
- `auto_restart_critical_agents` (POST /agents/health/auto-restart)
- `execute_agent_generic` (POST /agents/execute)
- `execute_ai_task` (POST /ai/tasks/execute/{task_id})
- `api_v1_agents_execute` (POST /api/v1/agents/execute)
- `api_v1_agents_activate` (POST /api/v1/agents/activate)
- `execute_aurea_event` (POST /api/v1/aurea/execute-event)
- Total: 15 inline functions removed (~1,125 lines)

**Pydantic models removed (moved to api/agents.py):**
- `ProductRequest`
- `AgentExecuteRequest`
- `AgentActivateRequest`
- `AUREAEventRequest`
- Total: 4 classes removed (~23 lines)

**Product agent import block simplified:**
- Removed module-level `HumanMessage = None` fallback (moved to lazy import in api/agents.py).

### tests/conftest.py

Changes:
- Added imports: `api.agents`, `api.scheduler`, `services.scheduler_queries`.
- Extended `patch_pool` fixture with 3 additional monkeypatches:
  - `agents_api.get_pool`
  - `scheduler_api.get_pool`
  - `scheduler_queries_svc.get_pool`

### tests/test_database_errors.py

Changes:
- Added import: `api.agents as agents_api`.
- Extended `test_agents_returns_503_when_db_unavailable` to also patch `agents_api.get_pool`.

---

## What Was NOT Changed

- No route paths, HTTP methods, or response shapes were modified.
- No authentication models were changed (same keys, same dependencies).
- No database schema changes.
- No changes to any other api/*.py file.
- No changes to Docker configuration, requirements, or deployment files.
- No Render environment variable changes required.

---

## Quality Gate Results

| Gate | Command | Result |
|------|---------|--------|
| Syntax check | `python3 -m py_compile` (all files) | PASS |
| Test suite | `pytest` | 362 passed, 2 skipped, 0 failed |
| Docker build | `docker build` v11.35.0 | PASS |
| Docker push | `docker push` latest + v11.35.0 | PASS |
| Render deploy | API trigger | dep-d6d25sstgctc73esqbdg |
| Route contract | 64 contract tests | ALL PASS |
