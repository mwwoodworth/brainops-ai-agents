# Digital Twin System - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Initialize the System
```python
from digital_twin_system import digital_twin_engine, SystemMetrics, SystemType

# Initialize (only needed once)
await digital_twin_engine.initialize()
```

### 2. Create Your First Twin
```python
twin = await digital_twin_engine.create_twin(
    source_system="my_production_api",
    system_type=SystemType.API_GATEWAY,
    initial_state={"version": "1.0.0", "region": "us-east-1"},
    sync_frequency_seconds=60
)

print(f"Twin created: {twin.twin_id}")
```

### 3. Sync with Real Metrics
```python
# Collect metrics from your system
metrics = SystemMetrics(
    cpu_usage=65.0,
    memory_usage=72.0,
    request_latency_ms=145.0,
    error_rate=0.005,
    throughput_rps=1200,
    active_connections=350
)

# Sync to digital twin
result = await digital_twin_engine.sync_twin(twin.twin_id, metrics)

print(f"Health Score: {result['health_score']}")
print(f"Predictions: {len(result['state_predictions'])}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Predict Future Resource Needs

```python
# Sync metrics over time (e.g., every minute)
for _ in range(20):  # Need at least 10 for predictions
    metrics = get_current_metrics()  # Your monitoring system
    result = await digital_twin_engine.sync_twin(twin.twin_id, metrics)
    await asyncio.sleep(60)

# View predictions
for prediction in result['state_predictions']:
    print(f"At {prediction['prediction_time']}:")
    print(f"  CPU: {prediction['predicted_metrics']['cpu_usage']:.1f}%")
    print(f"  Confidence: {prediction['confidence']:.0%}")
    print(f"  Risks: {prediction['risk_factors']}")
```

**Output Example:**
```
At 2025-12-24T15:30:00Z:
  CPU: 87.5%
  Confidence: 85%
  Risks: ['CPU may reach 95% in 30 minutes']
```

---

### Use Case 2: Detect Anomalies

```python
# Set expected baseline
await digital_twin_engine.set_expected_state(
    twin.twin_id,
    expected_state={
        "cpu_usage": 50.0,
        "memory_usage": 60.0,
        "latency_ms": 120.0
    }
)

# Sync new metrics
result = await digital_twin_engine.sync_twin(twin.twin_id, metrics)

# Check for divergence
if result['divergence_alerts']:
    for alert in result['divergence_alerts']:
        print(f"⚠️  {alert['component']}: {alert['severity']}")
        print(f"   Expected: {alert['expected_value']}")
        print(f"   Actual: {alert['actual_value']}")
        print(f"   Action: {alert['recommended_correction']}")
```

**Output Example:**
```
⚠️  cpu: high
   Expected: 50.0
   Actual: 90.0
   Action: Scale horizontally or investigate CPU-intensive processes
```

---

### Use Case 3: Test Before Deploying

```python
# Simulate a traffic spike
result = await digital_twin_engine.simulate_scenario(
    twin.twin_id,
    scenario={
        "type": "traffic_spike",
        "traffic_multiplier": 10,
        "duration_minutes": 60
    }
)

if result['will_likely_fail']:
    print(f"❌ System will fail with {result['failure_probability']:.0%} probability")
    print(f"Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
else:
    print(f"✅ System can handle the load")
```

**Output Example:**
```
❌ System will fail with 85% probability
Recommendations:
  - Pre-scale before expected traffic
  - Add 18 additional instances for safety margin
```

---

### Use Case 4: Rollback After Issues

```python
# Something went wrong, rollback to last known good state
twin = digital_twin_engine.twins[twin_id]

# Find the snapshot before the issue
for snapshot in twin.state_history:
    if snapshot.health_score > 95.0:
        # Rollback to this state
        result = await digital_twin_engine.rollback_to_snapshot(
            twin.twin_id,
            snapshot.snapshot_id
        )
        print(f"✅ Rolled back to {snapshot.timestamp}")
        break
```

---

## 🔗 Integration Examples

### With State Sync
```python
from realtime_state_sync import get_state_sync

state_sync = get_state_sync()
state_sync.enable_digital_twin_integration(digital_twin_engine)

# Now components automatically sync to twins
state_sync.register_component(component)  # Auto-creates twin
```

### With Realtime Monitor
```python
from realtime_monitor import get_realtime_monitor

monitor = get_realtime_monitor()
monitor.enable_digital_twin_integration(digital_twin_engine)
await monitor.start()

# Subscribe to twin health alerts
def on_health_alert(event):
    print(f"Health alert: {event.data['message']}")

monitor.subscribe(
    client_id="my_app",
    subscription_type=SubscriptionType.ALERTS,
    callback=on_health_alert
)
```

---

## 📊 Monitoring Dashboard

### Get Twin Status
```python
status = digital_twin_engine.get_twin_status(twin.twin_id)

print(f"Twin: {status['twin_id']}")
print(f"Source: {status['source_system']}")
print(f"Health: {status['health_score']}/100")
print(f"Drift: {status['drift_detected']}")
print(f"Active Predictions: {status['active_predictions']}")
print(f"Metrics Collected: {status['metrics_count']}")
```

### Get All Twins
```python
all_twins = digital_twin_engine.list_twins()

for twin in all_twins:
    print(f"{twin['source_system']}: {twin['health_score']}/100")
```

---

## ⚙️ Configuration

### Enable Auto-Correction
```python
twin.auto_correction_enabled = True  # Default

# Disable for critical systems
twin.auto_correction_enabled = False
```

### Adjust Sync Frequency
```python
twin.sync_frequency_seconds = 30  # Sync every 30 seconds
twin.sync_frequency_seconds = 300  # Sync every 5 minutes
```

### Custom Divergence Thresholds
```python
from digital_twin_system import DivergenceDetector

detector = DivergenceDetector()
alerts = detector.detect_divergence(
    twin,
    metrics,
    thresholds={
        "cpu": 10.0,      # More sensitive (10% vs 20% default)
        "memory": 5.0,    # Very sensitive
        "latency": 100.0  # Less sensitive
    }
)
```

---

## 🛡️ Safety Features

### Sync Loop Prevention
The system automatically prevents sync loops:
```python
# This is safe - won't create infinite loops
result = await digital_twin_engine.sync_twin(twin_id, metrics)

# Internal lock prevents re-entry
if self._sync_in_progress.get(twin_id):
    return {"error": "Sync already in progress"}
```

### Safe Rollback
```python
# Automatic backup before rollback
await rollback_to_snapshot(twin_id, snapshot_id)
# Creates "pre_rollback_to_{snapshot_id}" snapshot first
```

### Validation Before Auto-Correction
```python
# Only safe corrections are applied
if alert.auto_correct_eligible and twin.auto_correction_enabled:
    # Apply correction
else:
    # Log and notify for manual review
```

---

## 🐛 Troubleshooting

### No Predictions Generated
**Cause:** Not enough historical data
**Solution:** Need at least 10 metric syncs
```python
# Check metrics count
status = digital_twin_engine.get_twin_status(twin_id)
if status['metrics_count'] < 10:
    print("Collecting more data for predictions...")
```

### Divergence Not Detected
**Cause:** Expected state not set
**Solution:** Set baseline
```python
await digital_twin_engine.set_expected_state(twin_id, {...})
```

### Sync Errors
**Cause:** Database connection issues
**Solution:** Check DATABASE_URL
```python
import os
print(f"Database: {os.getenv('DATABASE_URL', 'Not configured')}")
```

---

## 📚 API Reference

### Core Methods

| Method | Purpose | Required Data |
|--------|---------|---------------|
| `create_twin()` | Create new twin | source_system, system_type, initial_state |
| `sync_twin()` | Update with metrics | twin_id, SystemMetrics |
| `simulate_scenario()` | Run what-if test | twin_id, scenario config |
| `rollback_to_snapshot()` | Restore previous state | twin_id, snapshot_id |
| `set_expected_state()` | Set baseline | twin_id, expected_state dict |
| `get_twin_status()` | Get current status | twin_id |
| `list_twins()` | Get all twins | None |

### Data Classes

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `SystemMetrics` | Current metrics | cpu_usage, memory_usage, latency, error_rate |
| `StatePrediction` | Future prediction | prediction_time, predicted_metrics, confidence |
| `DivergenceAlert` | Anomaly alert | component, severity, recommended_correction |
| `StateHistoryEntry` | State snapshot | snapshot_id, timestamp, state_snapshot |

---

## 💡 Best Practices

### 1. Regular Syncing
```python
# Sync every 1-5 minutes for best predictions
while True:
    metrics = collect_metrics()
    await digital_twin_engine.sync_twin(twin_id, metrics)
    await asyncio.sleep(60)  # 1 minute
```

### 2. Set Realistic Expectations
```python
# Use your actual production baseline
expected_state = {
    "cpu_usage": actual_average_cpu,
    "memory_usage": actual_average_memory,
    # ...
}
```

### 3. Review Auto-Corrections
```python
result = await sync_twin(twin_id, metrics)
if result['corrections_applied']:
    for correction in result['corrections_applied']:
        log.info(f"Auto-corrected: {correction}")
```

### 4. Use Simulations Before Changes
```python
# Before deploying a new version
result = await simulate_scenario(twin_id, {
    "type": "load_test",
    "concurrent_users": expected_users
})

if not result['will_likely_fail']:
    deploy_new_version()
```

---

## 🎓 Learning Path

1. **Day 1:** Create a twin and sync metrics
2. **Day 2:** Set expected state and review divergence alerts
3. **Day 3:** Run simulations and analyze predictions
4. **Day 4:** Enable integrations (state sync, monitoring)
5. **Day 5:** Test rollback and auto-correction features

---

## 📞 Support

### Documentation
- Full details: `/home/matt-woodworth/dev/brainops-ai-agents/DIGITAL_TWIN_ENHANCEMENTS.md`
- Code: `/home/matt-woodworth/dev/brainops-ai-agents/digital_twin_system.py`

### Getting Help
- Check logs for detailed error messages
- Review test suite for usage examples
- All operations log at INFO level or higher

---

**Version:** v9.10.0
**Last Updated:** 2025-12-24
**Status:** Production Ready
