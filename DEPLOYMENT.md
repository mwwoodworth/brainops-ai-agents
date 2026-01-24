# BrainOps AI Agents - Deployment SOP

## CURRENT SETUP: Docker Hub + Render API

**Render pulls from Docker Hub image, deploy triggered via API.**

| Item | Value |
|------|-------|
| Docker Hub | `docker.io/mwwoodworth/brainops-ai-agents:latest` |
| Render Service ID | `srv-d413iu75r7bs738btc10` |
| Production URL | https://brainops-ai-agents.onrender.com |
| DB Port | 6543 (Transaction mode) |

## Credentials Location

| Credential | Location |
|------------|----------|
| RENDER_API_KEY | `$RENDER_API_KEY` env var, stored **encrypted** in `brainops_credentials` |
| Service IDs | Stored in `unified_brain` table, key: `render_service_ids` |
| Deployment SOP | Stored in `unified_brain` table, key: `ai_agents_deployment_sop` |

**Note:** Command Center credential decrypt requires `ENCRYPTION_KEY` in Vercel env (production).

---

## Automated Deployment (Use This)

```bash
# Full deploy with API trigger
./deploy.sh

# Or manually:
VERSION=$(grep "^VERSION" app.py | cut -d'"' -f2)
docker build -t mwwoodworth/brainops-ai-agents:latest -t mwwoodworth/brainops-ai-agents:v$VERSION .
docker push mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:v$VERSION
git push

# Trigger deploy via Render API
curl -X POST "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json"

# Verify
curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'
```

---

## Render API Operations

```bash
# List services
curl -s "https://api.render.com/v1/services" -H "Authorization: Bearer $RENDER_API_KEY" | jq '.[].service | {name, id}'

# Get deploy status
curl -s "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys?limit=3" \
  -H "Authorization: Bearer $RENDER_API_KEY" | jq '.[].deploy | {id, status, createdAt}'

# Update env var
curl -X PUT "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/env-vars/DB_PORT" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"value": "6543"}'

# Restart service
curl -X POST "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/restart" \
  -H "Authorization: Bearer $RENDER_API_KEY"
```

---

## Database Configuration

| Setting | Value |
|---------|-------|
| Host | aws-0-us-east-2.pooler.supabase.com |
| Port | 6543 (Transaction mode - higher connection limits) |
| User | postgres.yomagoqdmxszqtdwuhab |
| Database | postgres |

**Why Port 6543?**
- Session mode (5432): Limited connections, causes `MaxClientsInSessionMode` errors
- Transaction mode (6543): Higher connection limits, better for multi-service environments

---

## Rollback

```bash
# Retag old version as latest
docker pull mwwoodworth/brainops-ai-agents:v9.67.0  # or desired version
docker tag mwwoodworth/brainops-ai-agents:v9.67.0 mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:latest

# Trigger redeploy
curl -X POST "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY"
```

---

## Monitoring

```bash
# Health check
curl -s https://brainops-ai-agents.onrender.com/health | jq '{version, database, build}'

# Brain endpoint
curl -s https://brainops-ai-agents.onrender.com/brain/critical -H "X-API-Key: <YOUR_BRAINOPS_API_KEY>" | jq 'length'

# Agents status
curl -s https://brainops-ai-agents.onrender.com/agents -H "X-API-Key: <YOUR_BRAINOPS_API_KEY>" | jq '.agents | length'
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 9.68.0 | 2025-12-31 | Switch to Transaction mode (port 6543) |
| 9.67.0 | 2025-12-31 | Disable InMemory fallback in production |
| 9.66.0 | 2025-12-31 | Security hardening, SQL injection fixes |

---

## Stored in Brain

All deployment configuration is stored in the database for persistence:
- `brainops_credentials.render_api_key` - API key
- `unified_brain.render_service_ids` - Service IDs
- `unified_brain.ai_agents_deployment_sop` - This SOP
