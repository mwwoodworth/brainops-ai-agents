# Phase 2 Wave 2A — Extraction Map

**Date:** 2026-02-21
**Scope:** app.py bounded-context extractions (Wave 2A)
**Net reduction:** 9,116 → 7,674 lines (-1,442 lines from app.py)

---

## Summary

| Extraction | Bounded Context | Lines Removed from app.py | Destination |
|------------|-----------------|---------------------------|-------------|
| 1 | Health / Readiness / Status | ~706 | api/health.py (NEW, 518 lines) + shared helpers |
| 2 | Memory / Brain extensions | ~462 | api/memory.py (+200 lines), api/brain.py (+141 lines) |
| 3 | Observability / Metrics | ~511 | api/system_health.py (NEW, 532 lines) |

---

## Extraction 1: Health / Readiness / Status

**Source:** app.py inline route functions (approx. lines 3204–3910 pre-extraction)
**Primary target:** `api/health.py` — NEW file, 518 lines

### Routes extracted

| Route | Handler function | Destination |
|-------|-----------------|-------------|
| GET /health | `health_check` | api/health.py:73 |
| GET /healthz | `healthz` | api/health.py:292 |
| GET /ready | `readiness_check` | api/health.py:309 |
| GET /alive | `alive_status` | api/health.py:337 |
| GET /capabilities | `capabilities` | api/health.py:387 |
| GET /diagnostics | `diagnostics` | api/health.py:414 |
| GET /system/alerts | `get_system_alerts` | api/health.py:488 |

### Helpers extracted / created alongside

| Name | Type | Destination |
|------|------|-------------|
| `pool_roundtrip_healthy` | async function | services/db_health.py:16 |
| `attempt_db_pool_init_once` | async function | services/db_health.py:34 |
| `_require_diagnostics_key` | auth helper | api/health.py:43 |
| `collect_active_systems` | imported from shared | services/system_status.py:14 |

### Test fixture update
- File: `tests/conftest.py`
- Change: `patch_pool` fixture extended to patch `api.health` and `services.db_health` module paths in addition to existing patches.

---

## Extraction 2: Memory / Brain Extensions

**Source:** app.py inline route functions (dead-code duplicates and new routes)
**Targets:** existing `api/memory.py` and `api/brain.py` (both already registered routers)

### Routes added to api/memory.py (+200 lines, now 1,174 lines total)

| Route | Handler function | Line |
|-------|-----------------|------|
| GET /memory/unified-search | `unified_search` | api/memory.py:984 |
| POST /memory/backfill-embeddings | `backfill_embeddings` | api/memory.py:1039 |
| POST /memory/force-sync | `force_sync_embedded_memory` | api/memory.py:1143 |

### Routes added to api/brain.py (+141 lines, now 871 lines total)

| Route | Handler function | Line |
|-------|-----------------|------|
| GET /brain/decide | `brain_decide` | api/brain.py:736 |
| POST /brain/learn | `brain_learn` | api/brain.py:811 |

### Dead-code routes removed from app.py

The following 3 inline routes were removed because they were already served by the
`api/memory.py` router registered earlier in the include order (first-match wins):

| Route | Reason for removal |
|-------|--------------------|
| POST /memory/store | Duplicated by api/memory.py router (registered first) |
| GET /memory/search | Duplicated by api/memory.py router (registered first) |
| GET /memory/stats | Duplicated by api/memory.py router (registered first) |

**Total app.py reduction:** 8 inline functions removed (~462 lines)

---

## Extraction 3: Observability / Metrics

**Source:** app.py inline route functions (approx. lines 3209–3766 pre-extraction)
**Primary target:** `api/system_health.py` — NEW file, 532 lines

### Routes extracted

| Route | Handler function | Destination |
|-------|-----------------|-------------|
| GET /system/awareness | `system_awareness` | api/system_health.py:42 |
| GET /alive/thoughts | `get_recent_thoughts` | api/system_health.py:229 |
| GET /awareness | `get_awareness_status` | api/system_health.py:243 |
| GET /awareness/report | `get_full_awareness_report` | api/system_health.py:259 |
| GET /awareness/pulse | `get_system_pulse` | api/system_health.py:275 |
| GET /truth | `get_truth` | api/system_health.py:298 |
| GET /truth/quick | `get_truth_quick` | api/system_health.py:319 |
| POST /api/v1/telemetry/events | `receive_telemetry_events` | api/system_health.py:342 |
| GET /observability/metrics | `observability_metrics` | api/system_health.py:392 |
| GET /debug/database | `debug_database` | api/system_health.py:427 |
| GET /debug/aurea | `debug_aurea` | api/system_health.py:484 |
| GET /debug/scheduler | `debug_scheduler` | api/system_health.py:506 |

**Total app.py reduction:** 12 inline functions removed (~511 lines)

---

## Shared Services Created

**Package init:** `services/__init__.py` — marks services/ as a Python package

### services/db_health.py (60 lines — NEW)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `pool_roundtrip_healthy` | `async (pool, timeout=4.0) -> bool` | Issue a SELECT 1 against the pool with a timeout; returns False on any error |
| `attempt_db_pool_init_once` | `async (app_state, context, timeout=5.0) -> bool` | One-shot pool warm-up with idempotency guard on app_state |

### services/system_status.py (260 lines — NEW)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `collect_active_systems` | `(app_state) -> list[str]` | Walk app.state attributes to enumerate running subsystem names |
| `scheduler_snapshot` | `(app_state) -> dict` | Serialize scheduler job list and run counts |
| `aurea_status` | `(app_state) -> dict` | Return AUREA orchestrator health summary |
| `self_healing_status` | `(app_state) -> dict` | Return self-healing agent status and last action |
| `memory_stats_snapshot` | `async (pool) -> dict` | Query unified_ai_memory for row counts and type distribution |
| `get_agent_usage` | `async (pool) -> dict` | Query unified_brain_logs for recent agent execution summary |
| `get_schedule_usage` | `async (pool) -> dict` | Query pg_cron / schedule metadata for active job counts |

---

## Router Registration in app.py

```python
# app.py lines 2332-2337
app.include_router(
    health_router
)  # WAVE 2A: /health, /healthz, /ready, /alive, /capabilities, /diagnostics, /system/alerts
app.include_router(
    system_health_router
)  # WAVE 2A: /system/awareness, /awareness/*, /truth/*, /debug/*, etc.
```

Both health routers are registered **without** `SECURED_DEPENDENCIES` because they implement
their own tiered auth (public minimal response vs. authenticated full response, or
`_require_diagnostics_key` for sensitive endpoints).

---

## File Size Reference (post-extraction)

| File | Lines | Status |
|------|-------|--------|
| app.py | 7,674 | Modified (was 9,116) |
| api/health.py | 518 | NEW |
| api/system_health.py | 532 | NEW |
| api/memory.py | 1,174 | Extended (+200) |
| api/brain.py | 871 | Extended (+141) |
| services/__init__.py | ~5 | NEW |
| services/db_health.py | 60 | NEW |
| services/system_status.py | 260 | NEW |
