# Self-Healing System - Quick Reference Card

## Import

```python
from self_healing_recovery import get_self_healing_recovery
healer = get_self_healing_recovery()
```

## Core Functions

### 1. Monitor Health

```python
result = healer.monitor_proactive_health('component_name', {
    'cpu_usage': 85.0,
    'memory_usage': 88.0,
    'error_rate': 0.08,
    'latency_ms': 3500.0
})

# Returns: health_score, trend, failure_prediction, warnings
```

### 2. Automatic Rollback

```python
result = await healer.rollback_component('component_name', 'deployment')
# Types: 'previous_state', 'config', 'deployment'
```

### 3. Restart Service

```python
result = await healer.restart_service_via_render('srv-xyz123', 'component_name')
# Requires: RENDER_API_KEY environment variable
```

### 4. Database Recovery

```python
result = healer.recover_database_connection()
# Auto-retries with exponential backoff
```

### 5. Memory Leak Cleanup

```python
result = healer.detect_and_cleanup_memory_leaks('component_name')
# Returns: leak_detected, memory_freed_mb, objects_collected
```

## Health Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| CPU | >70% | >90% |
| Memory | >75% | >90% |
| Error Rate | >5% | >10% |
| Latency | >2000ms | >5000ms |

## Failure Prediction Levels

| Probability | Time to Failure | Action |
|-------------|-----------------|--------|
| >50% | < 1 hour | Critical - Restart/Rollback |
| 30-50% | 1-6 hours | High - Monitor closely |
| 10-30% | 6-24 hours | Medium - Schedule maintenance |
| <10% | >24 hours | Low - Normal operation |

## Return Values

### Health Check
```python
{
    'health_score': 50.0,           # 0-100
    'trend': 'declining',           # improving/stable/declining
    'warnings': ['...'],            # List of warning messages
    'failure_prediction': {
        'probability': 0.55,        # 0.0-1.0
        'time_to_failure': '< 1 hour',
        'reasons': ['Memory leak detected']
    }
}
```

### Rollback
```python
{
    'success': True,
    'component': 'api-gateway',
    'rollback_type': 'deployment',
    'from_state': {...},
    'to_state': {...}
}
```

### Service Restart
```python
{
    'success': True,
    'service_id': 'srv-xyz123',
    'component': 'api-gateway',
    'status_code': 200,
    'message': 'Service restart initiated'
}
```

### Memory Cleanup
```python
{
    'success': True,
    'leak_detected': True,
    'memory_freed_mb': 125.5,
    'objects_collected': 8432,
    'current_memory_mb': 324.2
}
```

## Environment Variables

```bash
# Required
export DB_PASSWORD="${DB_PASSWORD}"

# Optional
export RENDER_API_KEY="${RENDER_API_KEY}"
export HEAL_LATENCY_THRESH="5000"
export HEAL_ERROR_THRESH="0.05"
export HEAL_MEMORY_THRESH="90"
export HEAL_CPU_THRESH="90"
```

## Unified Brain Logging

All actions automatically logged to `unified_brain` table:

```sql
SELECT * FROM unified_brain
WHERE agent_name LIKE '%self_healing%'
ORDER BY executed_at DESC
LIMIT 10;
```

## Action Types Logged

- `proactive_health_check`
- `service_restart`
- `automatic_rollback`
- `db_connection_recovered`
- `memory_leak_cleanup`
- `reconciliation_cycle`
- `remediation_approval`

## Common Patterns

### Continuous Monitoring
```python
while True:
    metrics = get_current_metrics()
    result = healer.monitor_proactive_health('my_service', metrics)

    if result['failure_prediction']['probability'] > 0.5:
        await healer.restart_service_via_render('srv-xyz', 'my_service')

    await asyncio.sleep(60)
```

### Error Recovery Pipeline
```python
try:
    # Your operation
    result = await my_operation()
except DatabaseError:
    healer.recover_database_connection()
    result = await my_operation()  # Retry
```

### Scheduled Leak Checks
```python
import schedule

def check_leaks():
    for component in ['api', 'worker', 'cache']:
        healer.detect_and_cleanup_memory_leaks(component)

schedule.every(15).minutes.do(check_leaks)
```

## Testing

```bash
python3 test_self_healing_upgrade.py
# Expected: 7/7 tests PASSED
```

## Analytics Queries

### Top Actions (24h)
```sql
SELECT action_type, COUNT(*)
FROM unified_brain
WHERE agent_name LIKE '%self_healing%'
  AND executed_at > NOW() - INTERVAL '24 hours'
GROUP BY action_type;
```

### Component Health
```sql
SELECT metadata->>'component', COUNT(*) as interventions
FROM unified_brain
WHERE agent_name LIKE '%self_healing%'
GROUP BY metadata->>'component'
ORDER BY interventions DESC;
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Failed to log to unified_brain" | Check database connection |
| Render API fails | Verify RENDER_API_KEY |
| Memory leak not detected | Need 10+ data points |
| Rollback fails | Check component state exists |

## Best Practices

1. ✓ Monitor health every 60 seconds
2. ✓ Set RENDER_API_KEY for auto-restart
3. ✓ Check failure predictions regularly
4. ✓ Review unified_brain logs daily
5. ✓ Adjust thresholds based on your workload

## Support

- **Documentation:** `SELF_HEALING_UPGRADE_DOCS.md`
- **Tests:** `test_self_healing_upgrade.py`
- **Summary:** `UPGRADE_SUMMARY.md`
- **Database:** Query `unified_brain` table
