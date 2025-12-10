# MCP Bridge Usage Analysis

## Bridge Inventory
- Config: `mcp-bridge/mcp-config.json` (13 servers, 358 tools). Servers and tool counts: openai (7), anthropic (3), gemini (3), vercel (34), render (39), supabase (40), playwright (60), github (50), python_executor (8), ollama (2), docker (53), stripe (55), ai-cli (4).
- Service defaults to `http://localhost:3100`; Render deployment documented at https://brainops-mcp-bridge.onrender.com (`mcp-bridge/README.md`).

## Where the Bridge Is Referenced
- Monitoring/ops: `ai-awareness-daemon.py:109-137` pings `https://brainops-mcp-bridge.onrender.com/health` then `http://localhost:3100/health`; `mcp-bridge/health-monitor.sh` curls the local `/health`; `scripts/devops-doctor.sh:165` checks the local port; systemd/PM2 units start the local service (`systemd/brainops-mcp-bridge.service`, `ecosystem.config.js`).
- Brain CLI: `brain-cli/lib/ai_orchestrator.py` fetches `/mcp/tools/definitions` and POSTs `/mcp/execute` against `http://localhost:3100`; `brain-cli/commands/mcp.py` and `brain-cli/commands/verify.py` hit the same base for list/health/tools.
- Command Center: `brainops-command-center/lib/mcp-task-bridge.ts` and `app/api/devops/mcp/route.ts` proxy to `MCP_BRIDGE_URL` (default `http://localhost:3100`); health check in `app/api/unified-health/route.ts` and status metadata in `app/api/context/unified/route.ts` also point to localhost. No UI/components call these APIs.
- ERP/MRG: Weathercraft ERP only whitelists the Render URL in CSP (`weathercraft-erp/src/middleware.ts:153`); no code paths call the bridge. No references in `myroofgenius-*`.
- AI Agents service: `brainops-ai-agents` has zero `mcp-bridge` or Render URL mentions.

## Usage Assessment
- Only automated traffic observed is health checks (ai-awareness daemon, health-monitor, devops doctor). The active code paths that could execute tools default to localhost, not the Render URL, and Vercel/env files do not set `MCP_BRIDGE_URL`.
- Command Center front-end does not reference `/api/devops/mcp` or other MCP endpoints; `lib/mcp-task-bridge.ts` maps to servers (`filesystem`, `memory`, `postgres`, `git`, `fetch`) that do not exist in the bridge config, so executions would fail if triggered.
- Brain CLI orchestrator could invoke tools manually (agents allow vercel/render/docker/github/supabase/stripe/openai via `config/agents.json`), but `/home/matt-woodworth/logs/mcp-bridge-error.log` is empty and `/home/matt-woodworth/logs/mcp-bridge-out.log` shows only startup/shutdown banners—no evidence of tool calls.
- No ERP/MRG code or deployed AI agent endpoints hit `brainops-mcp-bridge.onrender.com`; the Render domain appears only in documentation and CSP allowlisting.

## Tools Available vs. Actually Used
- Available: 358 tools across the 13 servers listed above.
- Actually used: none observed in code or logs; no recorded executions, and the only runtime HTTP requests found are health checks.

## Recommendation
- The Render-hosted bridge appears unused. If cost/maintenance matters, it is safe to pause or delete the Render service while keeping the local bridge for development.
- If remote access is desired later, set `MCP_BRIDGE_URL=https://brainops-mcp-bridge.onrender.com` in Command Center/CLI deployments and align task mappings with the actual servers in `mcp-bridge/mcp-config.json`.
