# BrainOps DevOps Comprehensive Audit
Date: 2025-01-09  
Scope: Static audit of BrainOps systems under `/home/matt-woodworth/dev`. Live calls to Render API and service health endpoints were **not run** because outbound network access is restricted in this environment.

## 1) Infrastructure Health
- Render configuration found in `brainops-ai-agents/render.yaml`; health check path is `/health` and autoDeploy is enabled, but live status was not verified (network blocked).  
- Monitoring scripts exist (`brainops-ai-agents/monitor_deployment.py`, `/home/matt-woodworth/dev/monitor-deployment.sh`) but they require external HTTP access and are not wired into alerting.  
- Supabase project config present in `supabase/config.toml` (project_id `dev`, API enabled, max_rows=1000). No evidence of automated drift detection between local config and production.  
- Deployment tooling for Render (`SET_RENDER_ENV_VARS.sh`) depends on manual API key export; no Terraform/IaC coverage identified.

## 2) Security Posture
- **Auth fail-closed (brainops-ai-agents):** `brainops-ai-agents/config.py` defaults `AUTH_REQUIRED=True` and raises at startup if no API keys are provided; test key is only auto-added when `DEV_MODE` or non-production envs are used. Ensure `ENVIRONMENT=production` and `ALLOW_TEST_KEY` stays false in prod.  
- **MCP Bridge API key middleware:** `mcp-bridge/server.js` terminates if `MCP_API_KEY` is missing and rejects requests without matching `X-API-Key`. CORS is wide-open; no rate limiting/auditing.  
- **Hardcoded secrets in repo:**  
  - `brainops-ai-agents/render.yaml` includes production API keys (`API_KEYS`, `AGENTS_API_KEY`, `BRAINOPS_API_KEY`) committed to the repo.  
  - `brainops/README.md` and multiple checklists contain Supabase hostname, username, and password (`PGPASSWORD=$DB_PASSWORD`).  
- **Plaintext credentials in workspace:** Root-level files (`BrainOps.env`, `AI_ORCHESTRATION_CREDENTIALS.env`, `google-*-service-account.json`, etc.) hold secrets unencrypted; `.env` files exist in `brainops`, `brainops-ai-agents`, `myroofgenius-backend`, and repo root (gitignored but still sensitive).  
- **Secret scanning CI:** Only `brainops-command-center` runs Gitleaks; other repos lack automated secret scanning.  
- **High-privilege surfaces:** `mcp-bridge` executes arbitrary MCP server commands; compromise of `MCP_API_KEY` would permit code execution and optional Supabase logging. No secondary checks or IP allowlists observed.

## 3) CI/CD & Monitoring
- GitHub Actions coverage:  
  - `brainops`: only `.github/workflows/system-registry.yml` (registry sync; no tests/build).  
  - `brainops-command-center`: `.github/workflows/secrets-scan.yml` (Gitleaks only).  
  - `brainops-ai-agents`, `mcp-bridge`, `brainops-gumroad`, `mcp-servers`: **no CI workflows**.  
  - `BrainOps-Weather`: multiple workflows for PR/push/labels/stale, but not tied to core BrainOps stack.  
- No observed automated deployment gates, unit/integ tests, linting, or container scanning for critical services.  
- Monitoring/alerting: manual scripts only; no uptime alerts, log aggregation, or metrics dashboards referenced.

## 4) Scalability
- **Database indexes:** Supabase migrations define numerous indexes on tasks/events/memories/scheduling tables (e.g., GIN on tags, composite task indexes). Actual usage/duplication can’t be confirmed offline.  
- **Connection pooling:** Supabase `db.pooler` is **disabled** in `supabase/config.toml`. The AI Agents app uses an `asyncpg` pool capped at min=1/max=3 (`brainops-ai-agents/database/async_connection.py`) to stay under Supabase limits; without a DB-side pooler, bursts may still exhaust connections.  
- **API limits:** Supabase API max_rows=1000; no pagination safeguards noted in app code review (not fully assessed).  
- **Caching/edge:** Command Center (Next.js) likely on Vercel, but no CDN/cache rules documented for API responses.

## Risk Matrix
| Risk | Severity | Evidence | Impact | Recommendation |
| --- | --- | --- | --- | --- |
| Production API keys committed to VCS | High | `brainops-ai-agents/render.yaml` | Credential leak; API abuse | Rotate keys immediately, remove from git history, move to Render/Vault secrets |
| Supabase prod password in docs | High | `brainops/README.md`, deployment guides | DB compromise if exposed | Rotate DB creds, scrub docs, replace with env placeholders |
| Plaintext credential files in workspace | High | `BrainOps.env`, `AI_ORCHESTRATION_CREDENTIALS.env`, `google-*-service-account.json` | Lateral movement, supply-chain risk | Encrypt or relocate to secrets manager; restrict filesystem access |
| No CI for main services | Medium | Missing workflows in AI Agents, MCP Bridge, Gumroad, MCP Servers | Undetected regressions, insecure builds | Add build/test/lint/security workflows and protected branches |
| Lack of monitoring/alerts | Medium | Only manual scripts; no uptime/metrics | Outages undetected; slow incident response | Integrate uptime checks, logging, metrics with alerting (e.g., Render/Supabase/Vercel + Pager/SMS) |
| DB pooler disabled | Medium | `supabase/config.toml` (`db.pooler.enabled=false`) | Connection exhaustion under load | Enable pgbouncer/pgbouncer-equivalent; tune pool sizes |
| Auth test-key fallback outside prod | Low | `brainops-ai-agents/config.py` adds default test key when not prod/DEV_MODE | Accidental open access in non-prod | Lock env to production in prod slots; disable `ALLOW_TEST_KEY` |

## Prioritized Action Items
1) Rotate and invalidate exposed Render/Supabase credentials; purge from git history and config files.  
2) Centralize secrets (Render/Vercel project secrets, Supabase vault, or HashiCorp Vault); remove plaintext `.env`/JSON creds from repo root and docs.  
3) Stand up CI for `brainops-ai-agents` and `mcp-bridge`: lint, tests, Docker build, SAST/secret scan, artifact publishing.  
4) Enable monitoring: uptime checks for `https://brainops-ai-agents.onrender.com/health`, Render deploy webhooks to alerts, Supabase metrics, and log shipping.  
5) Enable Supabase connection pooling; revisit app pool sizes after DB-side pooling to maximize concurrency.  
6) Run live DB health/index-usage checks from a network-permitted host using the provided SQL in `brainops-ai-agents/DATABASE_HEALTH_REPORT.md`; drop unused indexes and add FK/supporting indexes as needed.  
7) Harden MCP Bridge: tighten CORS, add rate limiting/audit logs, and restrict source IPs if possible.

## Architecture Diagram
```mermaid
graph TD
  U[Users/Admins] --> CC[Command Center (Next.js/Vercel)]
  CC -->|Supabase auth/realtime| DB[(Supabase Postgres)]
  CC --> AIA[AI Agents Service (FastAPI on Render)]
  AIA --> DB
  AIA --> MCP[MCP Bridge (Express)]
  MCP --> MCPS[MCP Servers + tools]
  AIA --> Gum[Gumroad/External APIs]
  AIA --> Wx[Weather/ERP endpoints]
```

## Notes on Constraints
- Render API check (`https://api.render.com/v1/services?limit=10`) and live health ping (`https://brainops-ai-agents.onrender.com/health`) were **not executed** due to restricted network access. Run these from a permitted host to confirm real-time status.
