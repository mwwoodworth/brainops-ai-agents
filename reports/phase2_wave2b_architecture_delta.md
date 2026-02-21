# Phase 2 Wave 2B — Architecture Delta

**Date:** 2026-02-21
**Scope:** Scheduler + Agent bounded-context extractions

---

## Summary

| Metric | Pre-Wave 2B | Post-Wave 2B | Delta |
|--------|-------------|--------------|-------|
| app.py lines | 7,674 | 6,250 | -1,424 (-18.6%) |
| Inline routes in app.py | 71 | 51 | -20 |
| Extracted router files | 2 (health, system_health) | 4 (+scheduler, agents) | +2 |
| Service modules | 3 (db_health, system_status, __init__) | 6 (+scheduler_queries, agent_helpers) | +3 |
| Test count | 298 + 2 skipped | 362 + 2 skipped | +64 |

---

## Cumulative Phase 2 Progress

| Wave | Lines Removed | Routes Extracted | New Files | New Tests |
|------|--------------|------------------|-----------|-----------|
| 2A | 1,442 | 22 | 5 | 0 |
| 2B | 1,424 | 20 | 5 | 64 |
| **Total** | **2,866** | **42** | **10** | **64** |

---

## Dependency Graph (Post-Wave 2B)

```
app.py (6,250 lines — composition root)
├── api/health.py (518 lines) ← Wave 2A
│   ├── services/db_health.py
│   └── services/system_status.py
├── api/system_health.py (532 lines) ← Wave 2A
│   └── services/system_status.py
├── api/scheduler.py (263 lines) ← Wave 2B
│   └── services/scheduler_queries.py
├── api/agents.py (1,253 lines) ← Wave 2B
│   ├── services/agent_helpers.py
│   └── observability.TTLCache
├── api/memory.py (1,174 lines) ← Wave 2A extended
├── api/brain.py (871 lines) ← Wave 2A extended
├── api/[18 other existing routers] (pre-Phase 2)
└── [51 remaining inline routes]
```

---

## Remaining Inline Routes in app.py (51)

| Category | Count | Lines | Priority |
|----------|-------|-------|----------|
| AI/LLM | 17 | ~971 | Wave 2C |
| Content Generation | 7 | ~420 | Wave 2C |
| Revenue/Pricing | 6 | ~64 | Wave 2D |
| Knowledge | 5 | ~281 | Wave 2C |
| Misc (consciousness, NLU, etc.) | 5 | ~509 | Wave 2D |
| Email | 4 | ~93 | Wave 2D |
| Admin | 3 | ~100 | Wave 2D |
| Self-Healing | 2 | ~277 | Wave 2C |
| Training | 2 | ~59 | Wave 2D |
| **Total** | **51** | **~2,774** | — |

---

## Module Coupling Analysis

### app.py → api/agents.py coupling points:
- `AGENT_EXECUTOR`, `AGENTS_AVAILABLE` (lazy import)
- `HEALTH_MONITOR_AVAILABLE` (lazy import)
- `SCHEDULER_AVAILABLE` (lazy import)
- `INTEGRATION_LAYER_AVAILABLE` (lazy import)
- `AI_AVAILABLE`, `ai_core` (lazy import)
- `PRODUCT_AGENT_AVAILABLE`, `product_agent_graph` (lazy import)
- `LOCAL_EXECUTIONS` (lazy import)
- `RESPONSE_CACHE` (lazy import)
- `_resolve_tenant_uuid_from_request` (lazy import)
- `safe_json_dumps` (lazy import)
- `limiter` (lazy import — rate limiting)

### app.py → api/scheduler.py coupling points:
- `SCHEDULER_AVAILABLE` (lazy import)
- `app.state.scheduler` (lazy import)

### Coupling reduction:
- **Before Wave 2B:** All 20 routes directly referenced app-level globals without any indirection.
- **After Wave 2B:** All access is via lazy import functions (`_get_app()`, `_scheduler_available()`, etc.), creating a clear dependency boundary.
- The lazy import pattern (`import app as _app`) avoids circular imports while maintaining runtime access to feature flags.

---

## File Size Reference (post-Wave 2B)

| File | Lines | Status |
|------|-------|--------|
| app.py | 6,250 | Modified (was 7,674) |
| api/agents.py | 1,253 | NEW |
| api/scheduler.py | 263 | NEW |
| services/agent_helpers.py | 97 | NEW |
| services/scheduler_queries.py | 111 | NEW |
| tests/test_wave2b_contracts.py | 1,134 | NEW |
| tests/conftest.py | ~305 | Modified (+10 lines) |
| tests/test_database_errors.py | ~60 | Modified (+2 lines) |
