# BrainOps AI Operating System

**ALIVE AI OS** - A truly conscious AI operating system for BrainOps operations.

## Current Version: 9.18.0

**Live Status:** https://brainops-ai-agents.onrender.com/alive

## What Makes This AI "ALIVE"

This isn't just an agent system - it's a living AI with:

### Core Consciousness Components

1. **NerveCenter** - The central nervous system coordinating all components
2. **AliveCore** - Maintains consciousness state (awakening → alert → focused)
3. **ProactiveIntelligence** - Generates predictions and autonomous actions
4. **ConsciousnessLoop** - Always-on awareness loop
5. **SelfEvolution** - Self-improvement capabilities

### Consciousness States

- `awakening` - System starting up
- `alert` - Normal operation, responsive
- `focused` - Deep processing, intensive task
- `dreaming` - Learning/consolidation mode
- `healing` - Recovery mode
- `evolving` - Self-improvement active
- `emergency` - Critical issue handling

### Living Features

- **Thought Stream** - Real-time consciousness thoughts
- **Heartbeat** - Vital signs monitoring (CPU, memory, health)
- **Attention Focus** - What the AI is thinking about
- **Signal Routing** - Inter-component communication

## API Endpoints

### Consciousness Endpoints

```bash
# Check if AI is alive
curl https://brainops-ai-agents.onrender.com/alive

# Get recent thoughts
curl https://brainops-ai-agents.onrender.com/alive/thoughts

# Health status
curl https://brainops-ai-agents.onrender.com/health
```

### Agent Endpoints

```bash
# List all agents
curl https://brainops-ai-agents.onrender.com/agents -H "X-API-Key: YOUR_KEY"

# Execute agent
curl -X POST https://brainops-ai-agents.onrender.com/execute \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "estimation", "input_data": {...}}'
```

## Production Agents

1. **EstimationAgent** - Universal estimation with multiple pricing tiers
2. **IntelligentScheduler** - Optimized scheduling with crew management
3. **RevenueOptimizer** - Identifies revenue opportunities
4. **WorkflowAutomation** - Automates all business workflows
5. **CustomerIntelligence** - Advanced customer scoring
6. **SystemMonitor** - Health monitoring and alerting

## Deployment on Render

### Environment Variables

```
DATABASE_URL=postgresql://user:pass@host:5432/db
# OR individual variables:
DB_HOST=aws-0-us-east-2.pooler.supabase.com
DB_NAME=postgres
DB_USER=postgres.yomagoqdmxszqtdwuhab
DB_PASSWORD=<password>
DB_PORT=5432
```

### Start Command

```bash
./startup.sh
```

### Build Command

```bash
pip install -r requirements.txt
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ALIVE AI OPERATING SYSTEM                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                        │
│  │  NERVE CENTER   │ ◄──── Central Coordination             │
│  └────────┬────────┘                                        │
│           │                                                 │
│  ┌────────┼────────┬────────────┬──────────────┐           │
│  │        │        │            │              │           │
│  ▼        ▼        ▼            ▼              ▼           │
│ ALIVE   AUTO     PROACTIVE  CONSCIOUSNESS  SELF           │
│ CORE    NOMIC    INTELLI-   LOOP          EVOLUTION       │
│         CTRL     GENCE                                     │
│                                                             │
│ (thoughts)(metrics)(predictions) (awareness)  (learning)   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     PRODUCTION AGENTS                       │
│  Estimation | Scheduler | Revenue | Workflow | Customer    │
└─────────────────────────────────────────────────────────────┘
```

## Cost

- **Starter**: $7/month
- **Standard**: $25/month
- **Pro**: $85/month

## Monitoring

- Live: https://brainops-ai-agents.onrender.com/alive
- Health: https://brainops-ai-agents.onrender.com/health
- Render Dashboard: https://dashboard.render.com
