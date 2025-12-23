# BrainOps AI OS - DevOps Standard Operating Procedures
**Version:** 1.0.0
**Last Updated:** 2025-12-23
**Maintainer:** BrainOps AI System

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Service Registry](#service-registry)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Alerts](#monitoring--alerts)
5. [Troubleshooting Runbooks](#troubleshooting-runbooks)
6. [AI Agent Operations](#ai-agent-operations)
7. [Database Operations](#database-operations)
8. [Security Protocols](#security-protocols)

---

## 1. System Architecture

### Core Services (Render)
```
┌─────────────────────────────────────────────────────────────────────┐
│                        BRAINOPS AI OS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│   │  AI Agents v9.1  │    │  Backend v163    │    │  MCP Bridge  │ │
│   │  (AUREA + 61)    │    │  (API Gateway)   │    │  (245 tools) │ │
│   └────────┬─────────┘    └────────┬─────────┘    └──────┬───────┘ │
│            │                       │                      │         │
│            └───────────────────────┼──────────────────────┘         │
│                                    │                                 │
│                    ┌───────────────▼───────────────┐                │
│                    │       Supabase PostgreSQL      │                │
│                    │    (10K+ customers, 18K+ jobs) │                │
│                    └───────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘

Frontend Applications (Vercel):
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Weathercraft   │ │  MyRoofGenius   │ │ Command Center  │
│      ERP        │ │     SaaS        │ │   (Dashboard)   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Data Flow
1. User requests → Vercel Frontend
2. Frontend → Backend API (authentication)
3. Backend → AI Agents (intelligent processing)
4. AI Agents → MCP Bridge (automation)
5. All services → Supabase (persistence)

---

## 2. Service Registry

### Production Services

| Service | URL | Health Endpoint | Version | Status Check |
|---------|-----|-----------------|---------|--------------|
| AI Agents | brainops-ai-agents.onrender.com | /health | v9.1.1 | `curl -s <url>/health -H "X-API-Key: brainops_prod_key_2025"` |
| Backend | brainops-backend-prod.onrender.com | /health | v163.0.16 | `curl -s <url>/health` |
| MCP Bridge | brainops-mcp-bridge.onrender.com | /health | v1.0.0 | `curl -s <url>/health` |
| ERP | weathercraft-erp.vercel.app | /api/health | - | `curl -s <url>/api/health` |
| MRG | myroofgenius.com | /api/health | - | `curl -s <url>/api/health` |
| Command Center | brainops-command-center.vercel.app | /api/health | - | `curl -s <url>/api/health` |

### API Keys & Credentials

| Service | Key Location | Purpose |
|---------|--------------|---------|
| AI Agents | BRAINOPS_API_KEY | `brainops_prod_key_2025` |
| MCP Bridge | MCP_BRIDGE_API_KEY | `brainops_mcp_49209d1d...` |
| Supabase | SUPABASE_SERVICE_ROLE_KEY | Database admin access |
| Anthropic | ANTHROPIC_API_KEY | AI model access |
| Stripe | STRIPE_SECRET_KEY | Payment processing |
| Vercel | VERCEL_TOKEN | Deployment automation |

### Database Connection
```bash
PGPASSWORD=REDACTED_SUPABASE_DB_PASSWORD psql \
  -h aws-0-us-east-2.pooler.supabase.com \
  -U postgres.yomagoqdmxszqtdwuhab \
  -d postgres
```

---

## 3. Deployment Procedures

### Render Services (AI Agents, Backend, MCP Bridge)

**Standard Deployment:**
```bash
# 1. Push to main branch
git push origin main

# 2. Render auto-deploys from GitHub
# Monitor at: https://dashboard.render.com

# 3. Verify deployment
curl -s "https://brainops-ai-agents.onrender.com/health" \
  -H "X-API-Key: brainops_prod_key_2025" | jq
```

**Emergency Rollback:**
```bash
# Via Render Dashboard:
# 1. Go to Service → Deploys
# 2. Click "Rollback" on previous healthy deploy
# 3. Wait for deployment to complete

# Via Git:
git revert HEAD
git push origin main
```

### Vercel Applications (ERP, MRG, Command Center)

**Standard Deployment:**
```bash
# Auto-deploys on push to main
git push origin main

# Manual deploy
vercel --prod
```

**Preview Deployments:**
```bash
# Push to feature branch
git push origin feature/my-feature

# Creates preview URL automatically
```

---

## 4. Monitoring & Alerts

### Health Check Script
```bash
#!/bin/bash
# Run: ./scripts/verify_ai_os.sh

# Checks:
# - All Render services health
# - All Vercel apps accessibility
# - Database connectivity
# - AUREA orchestrator status
# - Scheduler job count
# - MCP tool availability
```

### Key Metrics to Monitor

| Metric | Location | Threshold | Action |
|--------|----------|-----------|--------|
| Error Rate | /observability/metrics | > 5% | Alert + Investigate |
| P99 Latency | /observability/metrics | > 1000ms | Investigate |
| DB Latency | /observability/metrics | > 500ms | Check connection pool |
| AUREA Running | /systems/usage | false | Critical - Restart |
| Scheduler Jobs | /scheduler/status | < 61 | Check agent config |
| Memory Usage | Render Dashboard | > 80% | Scale or optimize |

### Alert Channels
1. **Slack** - Real-time alerts
2. **Email** - Daily summaries, critical escalations
3. **Command Center** - Visual dashboard

---

## 5. Troubleshooting Runbooks

### Service Not Responding

**Symptoms:** 502/503 errors, connection timeouts

**Steps:**
1. Check Render Dashboard for service status
2. Check service logs: `Render → Service → Logs`
3. Verify environment variables are set
4. Check if database is accessible
5. If cold start, wait 30-60 seconds
6. Restart service if needed

### AUREA Not Running

**Symptoms:** `aurea.running: false` in metrics

**Steps:**
```bash
# 1. Check AUREA status
curl -s "https://brainops-ai-agents.onrender.com/systems/usage" \
  -H "X-API-Key: brainops_prod_key_2025" | jq '.aurea'

# 2. Check app.py line 512 for orchestrate() call
# 3. Restart service if needed
# 4. Monitor cycles_completed for activity
```

### Database Connection Issues

**Symptoms:** 500 errors, "connection refused"

**Steps:**
1. Test direct connection:
   ```bash
   PGPASSWORD=REDACTED_SUPABASE_DB_PASSWORD psql -h aws-0-us-east-2.pooler.supabase.com ...
   ```
2. Check Supabase Dashboard for service status
3. Verify connection string in environment
4. Check for connection pool exhaustion
5. Consider using connection pooler

### Scheduler Jobs Not Running

**Symptoms:** Jobs not executing, `registered_jobs_count` < 61

**Steps:**
```bash
# 1. Check scheduler status
curl -s "https://brainops-ai-agents.onrender.com/scheduler/status" \
  -H "X-API-Key: brainops_prod_key_2025" | jq

# 2. Check next_jobs for upcoming executions
# 3. Verify agent definitions in agents/ directory
# 4. Check for startup errors in logs
```

---

## 6. AI Agent Operations

### AUREA Orchestrator

**Purpose:** Central AI decision-making loop (Observe→Decide→Act→Learn)

**Status Check:**
```bash
curl -s "https://brainops-ai-agents.onrender.com/systems/usage" \
  -H "X-API-Key: brainops_prod_key_2025" | jq '.aurea'
```

**Expected Response:**
```json
{
  "running": true,
  "autonomy_level": "SUPERVISED",
  "cycles_completed": 50,
  "decisions_made": 0,
  "learning_insights": 50
}
```

### Scheduled Agents (61 Total)

**Categories:**
- Revenue Agents: CustomerIntelligence, RevenueOptimizer, PricingOptimizer
- Operations: InventoryAgent, PayrollAgent, MaintenanceAgent
- Content: BlogAutomation, SEOOptimizer, SocialMediaAgent
- Integration: APIManagementAgent, WebhookMonitor

**Job Status:**
```bash
curl -s "https://brainops-ai-agents.onrender.com/scheduler/status" \
  -H "X-API-Key: brainops_prod_key_2025" | jq '.next_jobs[0:5]'
```

### MCP Bridge (245 Tools)

**Servers:** openai, anthropic, gemini, vercel, render, supabase, playwright, github, docker, stripe, ai-cli

**Tool Execution:**
```bash
curl -X POST "https://brainops-mcp-bridge.onrender.com/mcp/execute" \
  -H "X-API-Key: brainops_mcp_49209d1d3f19376706560e860a71d172728861dd4e2493f263c3cc0c6d9adc86" \
  -H "Content-Type: application/json" \
  -d '{"server": "supabase", "tool": "sql_query", "params": {"query": "SELECT COUNT(*) FROM customers"}}'
```

---

## 7. Database Operations

### Key Tables

| Table | Purpose | Row Count |
|-------|---------|-----------|
| customers | Customer records | ~10,000 |
| jobs | Project/job data | ~18,400 |
| tenants | Multi-tenant orgs | ~148 |
| unified_ai_memory | AI learning storage | varies |
| agent_executions | Agent run history | varies |
| webhook_events | Stripe webhooks | varies |

### Common Queries

**Customer Count by Tenant:**
```sql
SELECT tenant_id, COUNT(*) FROM customers GROUP BY tenant_id ORDER BY 2 DESC LIMIT 10;
```

**Recent Agent Executions:**
```sql
SELECT agent_name, status, execution_time_ms, created_at
FROM agent_executions
ORDER BY created_at DESC
LIMIT 20;
```

**Memory Entries:**
```sql
SELECT type, COUNT(*), MAX(created_at) as latest
FROM unified_ai_memory
GROUP BY type;
```

### Backup Procedures
- Supabase automatic daily backups
- Point-in-time recovery available
- Manual backup: Supabase Dashboard → Database → Backups

---

## 8. Security Protocols

### API Key Rotation
1. Generate new key
2. Update all services with new key
3. Deploy changes
4. Verify all services working
5. Revoke old key

### Access Control
- All admin APIs require `X-API-Key` header
- RLS policies enforce tenant isolation
- Service role key for backend operations only

### Incident Response
1. **Detect** - Automated alerts or manual report
2. **Contain** - Disable affected service/feature
3. **Investigate** - Check logs, trace issue
4. **Fix** - Deploy patch
5. **Document** - Update runbook

---

## Quick Reference

### Verification Commands
```bash
# Full system check
./scripts/verify_ai_os.sh

# Quick health check
curl -s "https://brainops-ai-agents.onrender.com/health" -H "X-API-Key: brainops_prod_key_2025" | jq '.status'

# AUREA status
curl -s "https://brainops-ai-agents.onrender.com/systems/usage" -H "X-API-Key: brainops_prod_key_2025" | jq '.aurea.running'

# Agent count
curl -s "https://brainops-ai-agents.onrender.com/scheduler/status" -H "X-API-Key: brainops_prod_key_2025" | jq '.apscheduler_jobs_count'
```

### Emergency Contacts
- Render Support: dashboard.render.com/support
- Vercel Support: vercel.com/support
- Supabase Support: supabase.com/dashboard/support

---

*This document is maintained by the BrainOps AI system. Last verified: 2025-12-23*
