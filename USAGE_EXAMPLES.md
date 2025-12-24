# Orchestration System v2.0 - Usage Examples

## Quick Start

### Initialize Enhanced Features

```python
import asyncio
from orchestrator import orchestrator
from unified_system_integration import get_unified_integration, initialize_all_systems
from autonomous_system_orchestrator import system_orchestrator

async def initialize_all():
    """Initialize all orchestration systems"""
    # Initialize main orchestrator
    await orchestrator.initialize_enhanced_features()

    # Initialize autonomous system orchestrator
    await system_orchestrator.initialize()

    # Initialize unified integration
    await initialize_all_systems()

    print("All systems initialized!")

# Run initialization
asyncio.run(initialize_all())
```

---

## Event-Driven Communication

### Publishing Events

```python
from autonomous_system_orchestrator import EventBus, EventType, SystemEvent

# Get event bus
event_bus = orchestrator.event_bus

# Publish a health change event
await event_bus.publish(SystemEvent(
    event_type=EventType.SYSTEM_HEALTH_CHANGED,
    source="my_service",
    data={
        "service_name": "my_service",
        "health_score": 95.0,
        "status": "healthy"
    },
    priority=3
))

# Publish a deployment event
await event_bus.publish(SystemEvent(
    event_type=EventType.DEPLOYMENT_STARTED,
    source="deployment_agent",
    data={
        "deployment_id": "deploy_123",
        "version": "v2.0.0"
    },
    priority=2
))
```

### Subscribing to Events

```python
# Define event handler
async def handle_health_change(event):
    print(f"Health changed: {event.data['service_name']} → {event.data['health_score']}")

    # Take action based on health
    if event.data['health_score'] < 50:
        print("CRITICAL: Service health below 50%!")
        # Trigger remediation...

# Subscribe to health change events
event_bus.subscribe(EventType.SYSTEM_HEALTH_CHANGED, handle_health_change)

# Subscribe to deployment events
async def handle_deployment(event):
    print(f"Deployment {event.data['deployment_id']} started")

event_bus.subscribe(EventType.DEPLOYMENT_STARTED, handle_deployment)
```

### Getting Recent Events

```python
# Get all recent events
recent_events = event_bus.get_recent_events(limit=50)

# Get specific event type
health_events = event_bus.get_recent_events(
    event_type=EventType.SYSTEM_HEALTH_CHANGED,
    limit=20
)

for event in health_events:
    print(f"{event.timestamp}: {event.source} → {event.data}")
```

---

## Message Queue Operations

### Enqueuing Tasks

```python
from autonomous_system_orchestrator import MessageQueue, Task

# Get message queue
message_queue = orchestrator.message_queue

# Create high-priority task
task = Task(
    task_id="task_001",
    task_type="deployment",
    agent_name="DeploymentAgent",
    data={"version": "v2.0.0", "service": "backend"},
    priority=2  # High priority
)

# Enqueue task
await message_queue.enqueue(task)

# Create low-priority task
analytics_task = Task(
    task_id="task_002",
    task_type="analytics",
    agent_name="AnalyticsAgent",
    data={"report": "daily"},
    priority=7  # Low priority
)

await message_queue.enqueue(analytics_task)
```

### Queue Statistics

```python
# Get queue stats
stats = message_queue.get_queue_stats()

print(f"Queue size: {stats['queue_size']}")
print(f"Active tasks: {stats['active_tasks']}")
print(f"Completed: {stats['completed_tasks']}")
print(f"Workers: {stats['workers']}")
```

---

## Circuit Breaker Protection

### Using Circuit Breakers

```python
from autonomous_system_orchestrator import CircuitBreaker

# Get circuit breaker for a service
circuit = orchestrator.circuit_breakers.get("backend")

# Check if can execute
if circuit.can_execute():
    try:
        # Make external call
        response = await client.get("https://backend/api/health")

        # Record success
        circuit.record_success()

    except Exception as e:
        # Record failure
        circuit.record_failure()
        print(f"Call failed: {e}")
else:
    print("Circuit breaker is OPEN, skipping call")
```

### Circuit Breaker Statistics

```python
# Get circuit stats
stats = circuit.get_stats()

print(f"Circuit: {stats['name']}")
print(f"State: {stats['state']}")
print(f"Total calls: {stats['total_calls']}")
print(f"Failures: {stats['total_failures']}")
print(f"Failure rate: {stats['failure_rate']:.2f}%")
print(f"Last failure: {stats['last_failure']}")
```

### Custom Circuit Breaker

```python
# Create custom circuit breaker
custom_circuit = CircuitBreaker(
    name="my_service_circuit",
    failure_threshold=10,  # Open after 10 failures
    success_threshold=3,   # Close after 3 successes
    timeout=120            # Wait 2 minutes before retry
)

# Use it
if custom_circuit.can_execute():
    try:
        result = await my_service.call()
        custom_circuit.record_success()
    except Exception:
        custom_circuit.record_failure()
```

---

## Load Balancing

### Selecting Instances

```python
from autonomous_system_orchestrator import LoadBalancer, LoadBalancingStrategy, AgentInstance

# Get load balancer
load_balancer = orchestrator.load_balancer

# Select instance for task
instance = load_balancer.select_instance("health_check")

if instance:
    print(f"Selected instance: {instance.instance_id}")
    print(f"Current load: {instance.current_load}/{instance.max_capacity}")

    # Update load before execution
    load_balancer.update_load(instance.instance_id, +1)

    # Execute task...

    # Update load after execution
    load_balancer.update_load(instance.instance_id, -1)
else:
    print("No available instances")
```

### Registering Custom Instances

```python
# Create custom agent instance
instance = AgentInstance(
    instance_id="my_agent_1",
    agent_name="my_agent",
    current_load=0,
    max_capacity=10,
    weight=2,  # Higher weight = more likely to be selected
    healthy=True
)

# Register instance
load_balancer.register_instance(instance)
```

### Load Balancer Statistics

```python
# Get load balancer stats
stats = load_balancer.get_stats()

for agent_name, agent_stats in stats.items():
    print(f"\n{agent_name}:")
    print(f"  Instances: {agent_stats['total_instances']}")
    print(f"  Healthy: {agent_stats['healthy_instances']}")
    print(f"  Capacity: {agent_stats['current_load']}/{agent_stats['total_capacity']}")
    print(f"  Utilization: {agent_stats['utilization']:.1f}%")
```

---

## Priority-Based Routing

### Execute Task with Priority

```python
# Execute high-priority task
result = await system_orchestrator.execute_task_with_priority(
    task_type="critical_alert",
    agent_name="AlertAgent",
    data={"alert": "Database down!"},
    priority=1  # Highest priority
)

# Execute normal priority task (priority auto-determined)
result = await system_orchestrator.execute_task_with_priority(
    task_type="health_check",
    agent_name="HealthAgent",
    data={"systems": ["backend", "frontend"]}
    # priority=3 automatically assigned
)
```

### Custom Priority Mapping

```python
# Add custom priority mappings
orchestrator.priority_routes["custom_operation"] = 2

# Now custom_operation will have priority 2
result = await orchestrator.execute_workflow("custom_operation", {})
```

---

## Health Aggregation

### Updating Health

```python
from autonomous_system_orchestrator import HealthAggregator

# Get health aggregator
health_aggregator = orchestrator.health_aggregator

# Update system health
health_aggregator.update_system_health("backend", {
    "health_score": 95.0,
    "status": "healthy",
    "response_time": 45,
    "error_rate": 0.1
})

# Update module health
health_aggregator.update_module_health("database_module", 88.5)
health_aggregator.update_module_health("cache_module", 100.0)
```

### Getting Aggregated Health

```python
# Get overall health
health = health_aggregator.get_aggregated_health()

print(f"Overall Health: {health['overall_health']:.1f}%")
print(f"Status: {health['status']}")
print(f"Systems: {health['systems_count']}")
print(f"Modules: {health['modules_count']}")
print(f"System Health: {health['avg_system_health']:.1f}%")
print(f"Module Health: {health['avg_module_health']:.1f}%")

# Breakdown
breakdown = health['breakdown']
print(f"\nHealthy: {breakdown['healthy']}")
print(f"Degraded: {breakdown['degraded']}")
print(f"Critical: {breakdown['critical']}")
print(f"Offline: {breakdown['offline']}")
```

### Finding Unhealthy Systems

```python
# Get systems with health < 80%
unhealthy = health_aggregator.get_unhealthy_systems(threshold=80.0)

for system in unhealthy:
    print(f"{system['system_id']}: {system['health_score']:.1f}% - {system['status']}")
    # Trigger remediation if needed
```

---

## Complete Orchestration Example

### Full Workflow with All Features

```python
import asyncio
from orchestrator import orchestrator
from autonomous_system_orchestrator import EventType, SystemEvent

async def execute_deployment_workflow():
    """
    Complete deployment workflow using all orchestration features
    """

    # 1. Initialize
    await orchestrator.initialize_enhanced_features()

    # 2. Check circuit breaker
    backend_circuit = orchestrator.circuit_breakers.get("backend")
    if not backend_circuit.can_execute():
        print("Backend circuit is OPEN, aborting deployment")
        return

    # 3. Select instance via load balancer
    instance = orchestrator.load_balancer.select_instance("deployment")
    if not instance:
        print("No available deployment instances")
        return

    print(f"Using instance: {instance.instance_id}")

    # 4. Publish deployment started event
    await orchestrator.event_bus.publish(SystemEvent(
        event_type=EventType.DEPLOYMENT_STARTED,
        source="deployment_workflow",
        data={"deployment_id": "deploy_123", "version": "v2.0.0"},
        priority=2
    ))

    # 5. Update instance load
    orchestrator.load_balancer.update_load(instance.instance_id, +1)

    try:
        # 6. Execute deployment via priority routing
        result = await orchestrator.execute_workflow(
            workflow_type="deploy_update",
            params={
                "service": "backend",
                "version": "v2.0.0",
                "rollback_enabled": True
            }
        )

        # 7. Record success
        backend_circuit.record_success()

        # 8. Update health
        orchestrator.health_aggregator.update_system_health("backend", {
            "health_score": 100.0,
            "status": "healthy"
        })

        # 9. Publish deployment completed event
        await orchestrator.event_bus.publish(SystemEvent(
            event_type=EventType.DEPLOYMENT_COMPLETED,
            source="deployment_workflow",
            data={"deployment_id": "deploy_123", "status": "success"},
            priority=2
        ))

        print("Deployment successful!")

    except Exception as e:
        # 10. Record failure
        backend_circuit.record_failure()

        # 11. Update health
        orchestrator.health_aggregator.update_system_health("backend", {
            "health_score": 50.0,
            "status": "critical"
        })

        # 12. Publish failed event
        await orchestrator.event_bus.publish(SystemEvent(
            event_type=EventType.DEPLOYMENT_COMPLETED,
            source="deployment_workflow",
            data={"deployment_id": "deploy_123", "status": "failed", "error": str(e)},
            priority=1  # Higher priority for failures
        ))

        print(f"Deployment failed: {e}")

    finally:
        # 13. Update instance load
        orchestrator.load_balancer.update_load(instance.instance_id, -1)

# Run workflow
asyncio.run(execute_deployment_workflow())
```

---

## Monitoring Dashboard

### Get Complete System Stats

```python
async def get_system_dashboard():
    """Get comprehensive system statistics"""

    # Orchestrator stats
    stats = orchestrator.get_orchestrator_stats()

    print("=== ORCHESTRATOR DASHBOARD ===\n")

    # Services
    print(f"Services: {', '.join(stats['services'])}")
    print(f"Enhanced Features: {'ENABLED' if stats['enhanced_features_enabled'] else 'DISABLED'}\n")

    # Circuit Breakers
    print("Circuit Breakers:")
    for name, cb_stats in stats.get('circuit_breakers', {}).items():
        print(f"  {name}:")
        print(f"    State: {cb_stats['state']}")
        print(f"    Failure Rate: {cb_stats['failure_rate']:.1f}%")

    # Message Queue
    queue_stats = stats.get('message_queue', {})
    print(f"\nMessage Queue:")
    print(f"  Queue Size: {queue_stats.get('queue_size', 0)}")
    print(f"  Active: {queue_stats.get('active_tasks', 0)}")
    print(f"  Completed: {queue_stats.get('completed_tasks', 0)}")

    # Load Balancer
    lb_stats = stats.get('load_balancer', {})
    print(f"\nLoad Balancer:")
    for agent, agent_stats in lb_stats.items():
        print(f"  {agent}: {agent_stats['utilization']:.1f}% utilization")

    # Health
    health = stats.get('health_aggregation', {})
    print(f"\nSystem Health:")
    print(f"  Overall: {health.get('overall_health', 0):.1f}%")
    print(f"  Status: {health.get('status', 'unknown')}")
    print(f"  Systems: {health.get('systems_count', 0)}")

    # Recent Events
    events = stats.get('recent_events', [])
    print(f"\nRecent Events ({len(events)}):")
    for event in events[-5:]:
        print(f"  [{event['priority']}] {event['type']}: {event['source']}")

asyncio.run(get_system_dashboard())
```

---

## Integration with Existing Code

### Using the Unified Integration Decorator

```python
from unified_system_integration import with_unified_integration

@with_unified_integration("ProposalAgent", "generate")
async def generate_proposal(task_data):
    """
    Generate proposal with automatic:
    - Pre-execution setup (context enrichment, priority)
    - Event publishing
    - Circuit breaker protection
    - Post-execution cleanup
    - Error handling
    """
    # Your existing code
    proposal = create_proposal(task_data)
    return {"proposal": proposal, "status": "success"}

# Use it
result = await generate_proposal({
    "customer_id": "cust_123",
    "project_type": "residential"
})
```

### Manual Integration

```python
from unified_system_integration import get_unified_integration

async def my_agent_task(task_data):
    integration = get_unified_integration()

    # Pre-execution
    ctx = await integration.pre_execution(
        agent_name="MyAgent",
        task_type="custom_task",
        task_data=task_data
    )

    try:
        # Your task logic
        result = await execute_my_task(task_data)

        # Post-execution (success)
        await integration.post_execution(ctx, result, success=True)

        return result

    except Exception as e:
        # Error handling
        error_info = await integration.on_error(ctx, e)

        # Post-execution (failure)
        await integration.post_execution(ctx, error_info, success=False)

        raise
```

---

## Advanced Patterns

### Multi-System Orchestration

```python
async def orchestrate_multi_system_update():
    """Update multiple systems with proper coordination"""

    systems = ["backend", "frontend", "database"]

    # Parallel health checks
    health_checks = []
    for system in systems:
        task = system_orchestrator.execute_task_with_priority(
            task_type="health_check",
            agent_name="HealthAgent",
            data={"system": system},
            priority=3
        )
        health_checks.append(task)

    results = await asyncio.gather(*health_checks)

    # Sequential deployments (high to low priority)
    for system in systems:
        await system_orchestrator.execute_task_with_priority(
            task_type="deployment",
            agent_name="DeploymentAgent",
            data={"system": system, "version": "v2.0.0"},
            priority=2
        )

        # Wait between deployments
        await asyncio.sleep(30)
```

### Event-Driven Remediation

```python
# Setup remediation handler
async def auto_remediate(event):
    """Automatically remediate unhealthy systems"""

    if event.data.get('health_score', 100) < 50:
        system_id = event.data.get('system_id')

        print(f"Auto-remediating {system_id}")

        # Restart system
        await system_orchestrator.execute_task_with_priority(
            task_type="restart",
            agent_name="SystemAgent",
            data={"system_id": system_id},
            priority=1  # Critical priority
        )

# Subscribe to health changes
orchestrator.event_bus.subscribe(
    EventType.SYSTEM_HEALTH_CHANGED,
    auto_remediate
)
```

---

## Best Practices

### 1. Always Initialize Enhanced Features

```python
# At application startup
await orchestrator.initialize_enhanced_features()
await system_orchestrator.initialize()
await initialize_all_systems()
```

### 2. Use Appropriate Priorities

```python
# Critical operations: 1-2
await execute_task(priority=1)  # Alerts, failures

# Important operations: 2-3
await execute_task(priority=2)  # Deployments, payments

# Normal operations: 5
await execute_task(priority=5)  # Monitoring

# Background operations: 7-10
await execute_task(priority=10)  # Analytics, cleanup
```

### 3. Always Protect External Calls

```python
circuit = orchestrator.circuit_breakers.get("service_name")

if circuit.can_execute():
    try:
        result = await external_call()
        circuit.record_success()
    except Exception:
        circuit.record_failure()
```

### 4. Update Health Regularly

```python
# After every significant operation
health_aggregator.update_system_health(system_id, {
    "health_score": calculate_health(),
    "status": "healthy" if health > 80 else "degraded"
})
```

### 5. Use Events for Decoupling

```python
# Instead of tight coupling
# await notify_all_systems()

# Use events
await event_bus.publish(SystemEvent(
    event_type=EventType.SYSTEM_HEALTH_CHANGED,
    source="my_system",
    data={"status": "updated"}
))
```

---

**For complete API reference, see ORCHESTRATION_ENHANCEMENTS.md**
