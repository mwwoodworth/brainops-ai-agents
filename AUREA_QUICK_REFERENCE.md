# AUREA Orchestrator - Quick Reference Card

## Enhanced OODA Loop

```
OBSERVE (Line 747)
├── Pattern Detection (_detect_patterns)
├── Goal Progress Tracking (_check_goal_progress)
├── Performance Trend Analysis (_analyze_performance_trends)
└── Unified Brain Storage

ORIENT (Line 926)
├── Autonomous Goal Setting (_set_autonomous_goals)
├── Multi-Criteria Analysis
├── Pattern-Aware Context
└── Historical Memory Recall

DECIDE (Line 995)
├── Enhanced Confidence Scoring
├── Goal-Aligned Decisions
└── Database Logging (with UUID)

ACT (Line 1095)
├── MCP-Powered Actions
├── Agent Activation
└── Execution Tracking

LEARN (Line 1315)
├── Success Rate Tracking
├── Pattern Recognition (every 10 cycles)
├── Self-Improvement from Failures
├── Decision Pattern Analysis (every 5 cycles)
└── Unified Brain Storage

HEAL (Line 1475)
├── MCP Auto-Remediation
├── Service Restart/Scale
└── Self-Healing Actions
```

## New Capabilities

### Autonomous Goal Setting
```python
# Automatically sets goals when:
- Success rate < 85% → "Improve to 90%"
- Error rate > 10% → "Reduce to < 5%"
- Cycle time > 30s → "Reduce to < 20s"

# Track with:
aurea.autonomous_goals  # List[AutonomousGoal]
```

### Pattern Detection
```python
# Detects:
- Declining success rates
- Increasing latency
- High performance consistency
- Failure patterns

# Access with:
aurea.pattern_history  # List[Dict]
```

### Metrics Tracking
```python
# Every cycle stores:
- CycleMetrics dataclass
- Success/failure breakdown
- Patterns detected
- Goals achieved/set

# Access with:
aurea.cycle_metrics_history  # List[CycleMetrics]
aurea._decision_success_rate_history  # Last 100 outcomes
```

### Self-Improvement
```python
# Triggers:
- Failure analysis → Parameter adjustment
- Pattern detection → Threshold tuning
- Performance degradation → Optimization

# Methods:
_self_improve_from_failures()
_analyze_decision_patterns()
```

## Database Tables

### aurea_cycle_metrics
Comprehensive metrics for every cycle
```sql
SELECT * FROM aurea_cycle_metrics
WHERE cycle_number > 100
ORDER BY cycle_number DESC LIMIT 10;
```

### aurea_autonomous_goals
Track autonomous goals
```sql
SELECT * FROM aurea_autonomous_goals
WHERE status = 'active'
ORDER BY priority DESC;
```

### aurea_patterns
Detected behavioral patterns
```sql
SELECT * FROM aurea_patterns
WHERE confidence > 0.8
ORDER BY last_detected DESC;
```

## Unified Brain Integration

```python
# Stores in unified_brain:
1. Observations: category='aurea_metrics'
2. Learnings: brain.store_learning()
3. Patterns: category='pattern'
4. Self-Improvements: category='self_improvement'
5. Cycle Metrics: category='aurea_metrics'

# Query:
brain = get_brain()
context = brain.get_full_context()
metrics = brain.get_by_category('aurea_metrics')
```

## Status Monitoring

```python
status = aurea.get_status()

# Returns:
{
  "performance_metrics": {
    "recent_success_rate": float,
    "recent_cycle_time_avg": float,
    "success_rate_trend": [...]
  },
  "autonomous_goals": {
    "active": int,
    "achieved": int,
    "active_goals_list": [...]
  },
  "learning": {
    "total_insights": int,
    "recent_patterns": int,
    "self_improvement_actions": int
  }
}
```

## Key Methods

| Method | Line | Purpose |
|--------|------|---------|
| `orchestrate()` | 614 | Main OODA loop |
| `_observe()` | 747 | Enhanced observation with patterns |
| `_orient()` | 926 | Context building + goal setting |
| `_decide()` | 995 | Decision making |
| `_act()` | 1095 | Execution |
| `_learn()` | 1315 | Learning + self-improvement |
| `_self_heal()` | 1475 | Auto-remediation |
| `_detect_patterns()` | 1942 | Pattern detection |
| `_set_autonomous_goals()` | 2139 | Goal setting |
| `_self_improve_from_failures()` | 2230 | Failure analysis |
| `_analyze_decision_patterns()` | 2263 | Decision analysis |
| `_store_cycle_metrics()` | 2312 | Metrics persistence |

## Statistics

- **Total Lines:** 2,455
- **Total Methods:** 66
- **New Methods:** 15
- **New Tables:** 3
- **New Indices:** 7
- **Enhancement:** +685 lines (38.7% increase)

## Testing Commands

```bash
# Syntax check
python3 -m py_compile aurea_orchestrator.py

# Initialize tables
python3 -c "from aurea_orchestrator import AUREA; AUREA(tenant_id='test')._init_database()"

# Run test cycle
python3 -c "
import asyncio
from aurea_orchestrator import AUREA, AutonomyLevel
aurea = AUREA(tenant_id='test', autonomy_level=AutonomyLevel.SUPERVISED)
asyncio.run(aurea._observe())
print('Observations:', len(aurea._last_observation_bundle))
"

# Check status
python3 -c "
from aurea_orchestrator import AUREA, AutonomyLevel
aurea = AUREA(tenant_id='test', autonomy_level=AutonomyLevel.SUPERVISED)
print(aurea.get_status())
"
```

## Quick Queries

```sql
-- Latest cycle metrics
SELECT cycle_number, observations_count, decisions_count,
       actions_successful, actions_failed, cycle_duration_seconds
FROM aurea_cycle_metrics
ORDER BY cycle_number DESC LIMIT 10;

-- Active autonomous goals
SELECT description, target_metric, current_value, target_value,
       progress, priority
FROM aurea_autonomous_goals
WHERE status = 'active'
ORDER BY priority DESC;

-- Recent patterns
SELECT pattern_type, pattern_description, confidence, frequency
FROM aurea_patterns
ORDER BY last_detected DESC LIMIT 10;

-- Learning insights
SELECT * FROM unified_brain
WHERE category IN ('aurea_metrics', 'self_improvement', 'pattern')
ORDER BY last_updated DESC LIMIT 20;
```
