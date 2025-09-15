# Deploy BrainOps AI Agents to Render

## Quick Deploy Instructions

### 1. Create Render Background Worker

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Background Worker"**
3. Connect GitHub repository: `mwwoodworth/brainops-ai-agents`
4. Name: `brainops-ai-agents`

### 2. Configure Service

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python main.py
```

### 3. Add Environment Variables

Click "Environment" and add these variables:

```
DB_HOST=aws-0-us-east-2.pooler.supabase.com
DB_NAME=postgres
DB_USER=postgres.yomagoqdmxszqtdwuhab
DB_PASSWORD=REDACTED_SUPABASE_DB_PASSWORD
DB_PORT=5432
SYSTEM_USER_ID=44491c1c-0e28-4aa1-ad33-552d1386769c
```

### 4. Select Instance Type

- **Starter**: $7/month (Recommended for 6 agents)
- **Standard**: $25/month (For expansion to 20+ agents)
- **Pro**: $85/month (For 50+ agents with high volume)

### 5. Deploy

Click **"Create Background Worker"** to deploy.

## What This Deploys

### 6 Production Agents Running 24/7:

1. **EstimationAgent** - Universal estimation with multiple pricing tiers
2. **IntelligentScheduler** - Optimized scheduling with crew management
3. **RevenueOptimizer** - Identifies revenue opportunities
4. **WorkflowAutomation** - Automates all business workflows
5. **CustomerIntelligence** - Advanced customer scoring
6. **SystemMonitor** - Health monitoring and alerting

### Features:

- **Multi-System Support**: Works with WeatherCraft, MyRoofGenius, and future systems
- **Connection Pooling**: Handles high load efficiently
- **Auto-Recovery**: Restarts failed agents automatically
- **Priority System**: Critical agents get resources first
- **Comprehensive Logging**: Full visibility into operations
- **Universal Capabilities**: Agents can serve multiple frontends

## Monitoring

Once deployed, monitor at:
- **Render Dashboard**: https://dashboard.render.com
- **Logs**: Available in Render dashboard
- **Database**: Check `ai_agents` table for status

```sql
SELECT name, last_active, total_executions, status
FROM ai_agents
WHERE is_active = true
ORDER BY last_active DESC;
```

## Verify Deployment

After deployment, verify agents are running:

```sql
-- Check agent activity (should see updates every 60 seconds)
SELECT name,
       last_active,
       EXTRACT(EPOCH FROM (NOW() - last_active)) as seconds_since_active,
       total_executions,
       status
FROM ai_agents
WHERE is_active = true
ORDER BY last_active DESC;
```

## Architecture Benefits

This deployment provides:

1. **Centralized**: One agent system for all AI operations
2. **Scalable**: Add more agents without touching frontends
3. **Reusable**: Same agents serve multiple applications
4. **Maintainable**: Update in one place, affects all systems
5. **Cost-Effective**: One Render service instead of multiple
6. **Reliable**: Production-grade with monitoring and recovery

## Integration

Your frontends (WeatherCraft ERP, MyRoofGenius, etc.) can now:

1. Query agent status from the database
2. View agent-generated data (estimates, schedules, etc.)
3. Trigger workflows that agents will process
4. Get real-time insights from agent analysis

This is the production-ready, permanent solution for all BrainOps AI operations.