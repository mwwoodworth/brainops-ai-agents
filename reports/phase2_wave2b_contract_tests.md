# Phase 2 Wave 2B — Contract Tests Report

**Date:** 2026-02-21
**File:** tests/test_wave2b_contracts.py
**Total tests:** 64 across 20 test classes
**Result:** ALL PASS

---

## Test Coverage Matrix

### Scheduler Routes (5 routes, 16 tests)

| Route | Method | Auth 403 | Happy 200 | Error paths | Required fields |
|-------|--------|----------|-----------|-------------|-----------------|
| /email/scheduler-stats | GET | YES | YES | ImportError fallback | daemon_stats, queue_counts, timestamp |
| /scheduler/status | GET | YES | YES | Scheduler disabled | enabled, timestamp |
| /scheduler/restart-stuck | POST | YES | YES | Resolver error 500 | success, items_fixed, action, details, timestamp |
| /scheduler/activate-all | POST | YES | YES | 503 scheduler off | success, new_schedules, already_scheduled, total_agents |
| /agents/schedule | POST | YES | YES | 404 agent, 422 body | success, action, schedule_id |

### Agent Routes (15 routes, 48 tests)

| Route | Method | Auth 403 | Happy 200 | Error paths | Required fields |
|-------|--------|----------|-----------|-------------|-----------------|
| /agents/product/run | POST | YES | YES | 503 unavailable | status, result, trace |
| /agents | GET | YES | YES | Empty DB | agents, total, page, page_size |
| /agents/{id}/execute | POST | YES | YES | 404 agent | agent_id, execution_id, status |
| /agents/status | GET | YES | YES | — | total_agents, agents, timestamp |
| /agents/{id} | GET | YES | YES | 404 agent | id, name, enabled, category |
| /agents/{id}/history | GET | YES | YES | 404 agent | agent_id, agent_name, history, count |
| /execute | POST | YES | YES | Scheduler disabled | status, executed, results, timestamp |
| /agents/health/check | POST | YES | YES | 503 unavailable | health result dict |
| /agents/{id}/restart | POST | YES | YES | 503, 404 | success result |
| /agents/health/auto-restart | POST | YES | YES | 503 unavailable | restart result dict |
| /agents/execute | POST | YES | YES | 422 bad params | success, execution_id, agent_id, agent_name |
| /ai/tasks/execute/{id} | POST | YES | YES | 503, 404 | success, message, task_id |
| /api/v1/agents/execute | POST | YES | YES | 400 missing id | Delegates to /agents/{id}/execute |
| /api/v1/agents/activate | POST | YES | YES | 400, 404 | success, agent.id, agent.name |
| /api/v1/aurea/execute-event | POST | YES | YES | 422, 404 | success, event_id, agent, topic, result |

---

## Test Patterns

### Auth Enforcement
Every route is tested with a request that omits the `X-API-Key` header, asserting HTTP 403.

### Required Fields
Happy-path tests assert the presence of documented response fields using `assert field in data`.

### Error Paths
- 404: Missing agent/task ID
- 503: Unavailable service (scheduler, health monitor, integration layer)
- 422: Malformed request body
- 400: Missing required fields
- 500: Internal errors (resolver failures)

### Mock Strategy
- `patch_pool` fixture patches `get_pool` across all relevant modules.
- `monkeypatch.setitem(sys.modules, ...)` stubs optional modules (email_scheduler_daemon, autonomous_issue_resolver, agent_health_monitor).
- `app.state` attributes set directly with cleanup in `finally` blocks.

---

## Pre-existing Tests (298)

All 298 pre-existing tests continue to pass without modification.
2 tests continue to be skipped (pre-existing skip markers).
