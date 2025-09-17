# BrainOps AI Agent System

Universal AI agent backend for all BrainOps operations including WeatherCraft ERP and MyRoofGenius.

## Production Deployment on Render

### Quick Deploy

1. Push to GitHub:
```bash
cd /home/matt-woodworth
git init brainops-ai-agents
cd brainops-ai-agents
git add .
git commit -m "Universal AI agent system"
git remote add origin https://github.com/YOUR_USERNAME/brainops-ai-agents.git
git push -u origin main
```

2. Create Render Background Worker:
- Go to https://dashboard.render.com
- Click "New +" → "Background Worker"
- Connect GitHub repo
- Name: `brainops-ai-agents`

3. Configure:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python main.py
```

**Environment Variables:**
```
DB_HOST=aws-0-us-east-2.pooler.supabase.com
DB_NAME=postgres
DB_USER=postgres.yomagoqdmxszqtdwuhab
DB_PASSWORD=Brain0ps2O2S
DB_PORT=5432
SYSTEM_USER_ID=44491c1c-0e28-4aa1-ad33-552d1386769c
```

4. Deploy and it runs 24/7

## What This System Does

### 6 Production Agents:

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

## Architecture Benefits

This approach is superior because:

1. **Centralized**: One agent system for all your AI operations
2. **Scalable**: Add more agents without touching frontends
3. **Reusable**: Same agents serve multiple applications
4. **Maintainable**: Update in one place, affects all systems
5. **Cost-Effective**: One Render service instead of multiple
6. **Reliable**: Production-grade with monitoring and recovery

## Cost

- **Starter**: $7/month (sufficient for 6 agents)
- **Standard**: $25/month (for expansion to 20+ agents)
- **Pro**: $85/month (for 50+ agents with high volume)

## Monitoring

Once deployed, monitor at:
- Render Dashboard: https://dashboard.render.com
- Logs: Available in Render dashboard
- Metrics: Logged every 60 seconds

## API Integration

Your frontends can query agent status:

```sql
SELECT name, last_active, total_executions
FROM ai_agents
WHERE is_active = true
```

## Expansion

To add more agents, simply:
1. Add new agent class in main.py
2. Add to orchestrator
3. Push to GitHub
4. Render auto-deploys

This is the RIGHT architecture for permanent, scalable AI operations.# Deploy trigger: Wed Sep 17 02:50:36 PM MDT 2025
