# Phase 2 Wave 2A — Route Contract Matrix

**Date:** 2026-02-21
**Purpose:** Verify every route extracted in Wave 2A retains an identical public contract
(method, path, auth requirement) after relocation.

The contract is defined as: HTTP method + path + authentication model.
No path, method, or authentication change was permitted during extraction.

---

## Verification Status

All 22 routes: CONTRACT UNCHANGED.
pytest result: 298 passed, 2 skipped, 0 failed.
py_compile: All files pass.

---

## Extraction 1: Health / Readiness / Status Routes

Router registered at: `app.include_router(health_router)` (no prefix, no SECURED_DEPENDENCIES)

| Method | Path | app.py (pre-extraction) | New location | Function name | Auth model | Contract unchanged? |
|--------|------|------------------------|--------------|---------------|------------|---------------------|
| GET | /health | ~3204 | api/health.py:73 | `health_check` | Public (unauthenticated = minimal response; X-API-Key = full diagnostics) | YES |
| GET | /healthz | ~3700 | api/health.py:292 | `healthz` | Public (no auth check) | YES |
| GET | /ready | ~3750 | api/health.py:309 | `readiness_check` | Public (no auth check) | YES |
| GET | /alive | ~3800 | api/health.py:337 | `alive_status` | Public (no auth check) | YES |
| GET | /capabilities | ~3850 | api/health.py:387 | `capabilities` | `_require_diagnostics_key` (403 if key missing/invalid) | YES |
| GET | /diagnostics | ~3870 | api/health.py:414 | `diagnostics` | `_require_diagnostics_key` (403 if key missing/invalid) | YES |
| GET | /system/alerts | ~3900 | api/health.py:488 | `get_system_alerts` | Public (no auth check, read-only alerts) | YES |

**Auth model note — `_require_diagnostics_key`:** checks `X-API-Key` header against
`config.security.valid_api_keys` or `MASTER_API_KEY`; raises HTTP 403 on failure.
This is identical to the inline behavior that existed in app.py.

---

## Extraction 2: Memory / Brain Extension Routes

### Memory routes (router prefix: /memory)

Router registered at: `app.include_router(memory_router, dependencies=SECURED_DEPENDENCIES)`

| Method | Path | app.py (pre-extraction) | New location | Function name | Auth model | Contract unchanged? |
|--------|------|------------------------|--------------|---------------|------------|---------------------|
| GET | /memory/unified-search | inline (dead-code, shadowed) | api/memory.py:984 | `unified_search` | verify_api_key (via SECURED_DEPENDENCIES) | YES |
| POST | /memory/backfill-embeddings | inline (dead-code, shadowed) | api/memory.py:1039 | `backfill_embeddings` | verify_api_key (via SECURED_DEPENDENCIES) | YES |
| POST | /memory/force-sync | inline (dead-code, shadowed) | api/memory.py:1143 | `force_sync_embedded_memory` | verify_api_key (via SECURED_DEPENDENCIES) | YES |

**Dead-code clarification:** Three app.py inline routes (POST /memory/store, GET /memory/search,
GET /memory/stats) were registered in app.py *after* the `api/memory.py` router was included.
FastAPI's first-match routing meant the api/memory.py versions were always served.
The dead-code inline functions were removed; no client-visible behavior changed.

### Brain routes (router prefix: /brain)

Router registered at: `app.include_router(brain_router, dependencies=SECURED_DEPENDENCIES)`

| Method | Path | app.py (pre-extraction) | New location | Function name | Auth model | Contract unchanged? |
|--------|------|------------------------|--------------|---------------|------------|---------------------|
| GET | /brain/decide | inline | api/brain.py:736 | `brain_decide` | verify_api_key (via SECURED_DEPENDENCIES) | YES |
| POST | /brain/learn | inline | api/brain.py:811 | `brain_learn` | verify_api_key (via SECURED_DEPENDENCIES) | YES |

---

## Extraction 3: Observability / Metrics Routes

Router registered at: `app.include_router(system_health_router)` (no prefix, no SECURED_DEPENDENCIES — internal auth per endpoint via `Depends(verify_api_key)`)

| Method | Path | app.py (pre-extraction) | New location | Function name | Auth model | Contract unchanged? |
|--------|------|------------------------|--------------|---------------|------------|---------------------|
| GET | /system/awareness | ~3209 | api/system_health.py:42 | `system_awareness` | `verify_api_key` via route Depends | YES |
| GET | /alive/thoughts | ~3500 | api/system_health.py:229 | `get_recent_thoughts` | `verify_api_key` via route Depends | YES |
| GET | /awareness | ~3530 | api/system_health.py:243 | `get_awareness_status` | `verify_api_key` via route Depends | YES |
| GET | /awareness/report | ~3560 | api/system_health.py:259 | `get_full_awareness_report` | `verify_api_key` via route Depends | YES |
| GET | /awareness/pulse | ~3590 | api/system_health.py:275 | `get_system_pulse` | `verify_api_key` via route Depends | YES |
| GET | /truth | ~3620 | api/system_health.py:298 | `get_truth` | `verify_api_key` via route Depends | YES |
| GET | /truth/quick | ~3650 | api/system_health.py:319 | `get_truth_quick` | `verify_api_key` via route Depends | YES |
| POST | /api/v1/telemetry/events | ~3680 | api/system_health.py:342 | `receive_telemetry_events` | `verify_api_key` via route Depends | YES |
| GET | /observability/metrics | ~3710 | api/system_health.py:392 | `observability_metrics` | `verify_api_key` via route Depends | YES |
| GET | /debug/database | ~3730 | api/system_health.py:427 | `debug_database` | `verify_api_key` via route Depends | YES |
| GET | /debug/aurea | ~3750 | api/system_health.py:484 | `debug_aurea` | `verify_api_key` via route Depends | YES |
| GET | /debug/scheduler | ~3766 | api/system_health.py:506 | `debug_scheduler` | `verify_api_key` via route Depends | YES |

---

## Auth Model Legend

| Symbol | Meaning |
|--------|---------|
| Public | No authentication required; endpoint returns data to any caller |
| Public (tiered) | Unauthenticated callers receive a reduced payload; valid X-API-Key callers receive full payload |
| `_require_diagnostics_key` | Custom auth helper in api/health.py; raises HTTP 403 if X-API-Key is missing or invalid |
| `verify_api_key` via SECURED_DEPENDENCIES | APIKeyHeader dependency injected at router registration by app.py; raises HTTP 403 on failure |
| `verify_api_key` via route Depends | Same verify_api_key function declared directly on each route's `dependencies=` list |

---

## Routes NOT Touched in Wave 2A

All other routes already residing in their own `api/*.py` router files were not
modified. This matrix covers only the 22 routes extracted or relocated during Wave 2A.
