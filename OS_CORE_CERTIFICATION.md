# OS Core Certification

**Date**: 2026-02-21
**Scope**: Silent failure elimination across 3 services
**Status**: CERTIFIED — 7/7 failure modes fixed, 19/19 tests passing

---

## Failure Detection Matrix

| # | Failure Mode | File | Fix | Test | Log Pattern |
|---|-------------|------|-----|------|-------------|
| 1 | brain_store returns False silently | `brain_store_helper.py` | Failure counter + `logger.error` | `test_brain_store_resilience.py` (4 tests) | `MEMORY_LOSS:` |
| 2 | BaseAgent dual persistence loss | `base_agent.py` | `MemoryLossError` on both-fail | `test_base_agent_memory_loss.py` (4 tests) | `EXECUTION_LOST:` |
| 3 | Invariant violation double-fail | `invariant_monitor.py:112` | `logger.critical` replaces `pass` | `test_invariant_resilience.py` (2 tests) | `INVARIANT_DOUBLE_FAIL:` |
| 4 | Awareness daemon staleness | `invariant_monitor.py` (new check #15) | Query heartbeat freshness | `test_invariant_resilience.py` (3 tests) | `awareness_staleness` |
| 5 | MCP Bridge SQLite stack overflow | `server.js:recordExecution` | `_retryDepth` guard | `record-execution.test.js` (2 tests) | `SQLite recordExecution failed after retry` |
| 6 | MCP Bridge PG logging permanent disable | `server.js:recordExecution` | 5-min cooldown + table probe | `record-execution.test.js` (4 tests) | `PG logging re-enabled after cooldown recovery` |
| 7 | Backend brain_store silent debug | `core/brain_store.py` | `logger.warning` + counter | Compile-verified | `BRAIN_STORE_FAIL:` |

---

## Memory Protocol

### Canonical Store

**Table**: `unified_ai_memory` (122,112+ rows)

### Write Paths

| Service | Function | Transport |
|---------|----------|-----------|
| brainops-ai-agents | `brain_store_helper.brain_store()` | Direct asyncpg via `unified_brain.py` |
| myroofgenius-backend | `core.brain_store.dispatch_brain_store()` | HTTP POST to agents `/brain/store` |

### Rules

1. All agent-side memory writes go through `brain_store_helper.brain_store()`
2. Backend writes go through `dispatch_brain_store()` → HTTP API → agents service
3. Both paths now have failure counters exposed via health endpoints
4. Both paths log at WARNING or ERROR level on failure (never DEBUG or silent)

---

## Runbooks

### MEMORY_LOSS — Brain Store Failures

**Symptoms**: `MEMORY_LOSS:` entries in Render logs for brainops-ai-agents

**Steps**:
1. Check health endpoint: `curl -s $AGENTS_URL/health -H "X-API-Key: $KEY" | jq '.brain_store_stats'`
2. If `failures > 0`, check DB connectivity
3. Verify asyncpg pool is healthy: check `pool_health` in health response
4. If DB is down, the self-healing recovery system should trigger — check `self_healing` subsystem status

### EXECUTION_LOST — Agent Dual Persistence Failure

**Symptoms**: `EXECUTION_LOST:` entries at CRITICAL level in Render logs

**Steps**:
1. This triggers the executor's 3-attempt retry loop automatically
2. If persists, check both DB pool health AND UnifiedBrain connectivity
3. The agent execution will still complete — only the persistence failed
4. Search for the exec ID in logs to find the execution result data

### INVARIANT_DOUBLE_FAIL — Both DB Stores Down

**Symptoms**: `INVARIANT_DOUBLE_FAIL:` entries at CRITICAL level

**Steps**:
1. This means both `invariant_violations` INSERT and `unified_brain_logs` fallback failed
2. Check PostgreSQL connectivity immediately
3. The invariant violation data is in the CRITICAL log line — it's not lost, just not in DB
4. Once DB recovers, the next invariant cycle will re-detect any ongoing issues

### Awareness Staleness

**Symptoms**: `awareness_staleness` violations in invariant engine output

**Steps**:
1. Check if the awareness daemon is running: `pgrep -f ai-awareness-daemon`
2. If not running, restart: `python3 ~/dev/_scripts/ai-awareness-daemon.py &`
3. Check `system_awareness_state` table: `SELECT component_id, last_heartbeat FROM system_awareness_state ORDER BY last_heartbeat`
4. Once daemon restarts, heartbeats will refresh within 60 seconds

### MCP SQLite Crash

**Symptoms**: MCP Bridge process restarts, `SQLite recordExecution failed after retry` in logs

**Steps**:
1. Check Render dashboard for restart count
2. Verify `/mnt/extra` disk is mounted and writable
3. The depth guard prevents stack overflow — the bridge stays running
4. SQLite recording is non-critical; tool execution still works without it

### MCP PG Logging Down

**Symptoms**: Health endpoint shows `memorySystem.durable: false`

**Steps**:
1. Check if `mcp_execution_logs` table exists in Supabase
2. If missing, run migration: `psql $DATABASE_URL < migrations/001_create_mcp_execution_logs.sql`
3. After migration, PG logging self-heals within 5 minutes (cooldown recovery)
4. Verify: `curl -s $MCP_URL/health | jq '.memorySystem'`

---

## Test Suite

```bash
# brainops-ai-agents (13 tests)
cd ~/dev/brainops-ai-agents
python3 -m pytest tests/test_brain_store_resilience.py tests/test_base_agent_memory_loss.py tests/test_invariant_resilience.py -v

# mcp-bridge (6 tests)
cd ~/dev/mcp-bridge
npx jest tests/record-execution.test.js --verbose
```

---

## Files Modified

| Repo | File | Change |
|------|------|--------|
| brainops-ai-agents | `brain_store_helper.py` | Failure counters, MEMORY_LOSS logging, `get_brain_store_stats()` |
| brainops-ai-agents | `base_agent.py` | `MemoryLossError`, dual-fail detection, EXECUTION_LOST logging |
| brainops-ai-agents | `invariant_monitor.py` | INVARIANT_DOUBLE_FAIL logging, awareness staleness check (#15) |
| brainops-ai-agents | `tests/test_brain_store_resilience.py` | 4 chaos tests |
| brainops-ai-agents | `tests/test_base_agent_memory_loss.py` | 4 chaos tests |
| brainops-ai-agents | `tests/test_invariant_resilience.py` | 5 chaos tests |
| brainops-ai-agents | `OS_CORE_CERTIFICATION.md` | This document |
| mcp-bridge | `server.js` | SQLite depth guard, PG logging cooldown recovery |
| mcp-bridge | `tests/record-execution.test.js` | 6 resilience tests |
| myroofgenius-backend | `core/brain_store.py` | Failure counters, BRAIN_STORE_FAIL logging, `get_brain_store_health()` |
