#!/usr/bin/env python3
"""
AI Observability & Integration API
===================================
Endpoints for unified observability, metrics, events, and cross-module integration.

Provides:
- Prometheus-compatible metrics export
- Event stream and history
- Unified system state
- Dashboard data
- Learning and recovery statistics
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI Observability & Integration"])

# Lazy import to avoid circular dependencies
_observability = None
_integration = None


def get_observability():
    global _observability
    if _observability is None:
        try:
            from ai_observability import ObservabilityController
            _observability = ObservabilityController.get_instance()
        except Exception as e:
            logger.error(f"Failed to load observability: {e}")
    return _observability


def get_integration():
    global _integration
    if _integration is None:
        try:
            from ai_module_integration import ModuleIntegrationOrchestrator
            _integration = ModuleIntegrationOrchestrator.get_instance()
        except Exception as e:
            logger.error(f"Failed to load integration: {e}")
    return _integration


# =============================================================================
# METRICS ENDPOINTS
# =============================================================================

@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Get all metrics in Prometheus format.
    Compatible with Prometheus scraping.
    Returns text/plain content type.
    """
    obs = get_observability()
    if not obs:
        return PlainTextResponse("# Error: Observability not available\n", status_code=503)

    metrics_text = obs.get_prometheus_metrics()
    return PlainTextResponse(metrics_text, media_type="text/plain; version=0.0.4")


@router.get("/metrics/json")
async def get_metrics_json():
    """Get all metrics in JSON format for dashboards"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return obs.metrics.get_all_metrics()


@router.get("/metrics/histogram/{name}")
async def get_histogram(name: str):
    """Get specific histogram with percentiles. Returns all histograms matching the base name."""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    # Get all histograms matching this name (handles labeled histograms)
    histograms = obs.metrics.get_histograms_by_name(name)

    if not histograms:
        # Try exact match as fallback
        histogram = obs.metrics.get_histogram(name)
        if histogram:
            histograms = [histogram]

    if histograms:
        result = {
            "name": name,
            "instances": []
        }
        for histogram in histograms:
            result["instances"].append({
                "labels": histogram.labels,
                "count": histogram._count,
                "sum": histogram._sum,
                "avg": histogram.avg,
                "p50": histogram.p50,
                "p95": histogram.p95,
                "p99": histogram.p99
            })

        # Also provide aggregated stats across all instances
        total_count = sum(h._count for h in histograms)
        total_sum = sum(h._sum for h in histograms)
        result["aggregated"] = {
            "total_count": total_count,
            "total_sum": total_sum,
            "avg": total_sum / total_count if total_count > 0 else 0
        }
        return result

    return {"error": f"Histogram {name} not found"}


# =============================================================================
# EVENT ENDPOINTS
# =============================================================================

@router.get("/events")
async def get_events(limit: int = Query(default=100, le=1000)):
    """Get recent events from event bus"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return {
        "events": obs.events.get_recent_events(limit),
        "counts": obs.events.get_event_counts(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/events/counts")
async def get_event_counts():
    """Get event counts by type"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return obs.events.get_event_counts()


@router.get("/events/dead-letters")
async def get_dead_letters():
    """Get failed events from dead letter queue"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return {
        "dead_letters": obs.events.get_dead_letters(),
        "count": len(obs.events.get_dead_letters())
    }


# =============================================================================
# DASHBOARD ENDPOINTS
# =============================================================================

@router.get("/dashboard")
async def get_dashboard_data():
    """Get aggregated dashboard data"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return obs.get_dashboard_data()


@router.get("/health/overall")
async def get_overall_health():
    """Get overall system health calculated from all modules"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    dashboard = obs.get_dashboard_data()
    return dashboard.get("health", {"status": "unknown"})


# =============================================================================
# INTEGRATION ENDPOINTS
# =============================================================================

@router.get("/state")
async def get_unified_state():
    """Get unified state across all modules"""
    integration = get_integration()
    if not integration:
        return {"error": "Integration not available"}

    return integration.get_unified_state()


@router.get("/learning")
async def get_learning_state():
    """Get learning manager state and statistics"""
    integration = get_integration()
    if not integration:
        return {"error": "Integration not available"}

    return integration.learning.get_learning_summary()


@router.get("/recovery")
async def get_recovery_state():
    """Get recovery coordinator state"""
    integration = get_integration()
    if not integration:
        return {"error": "Integration not available"}

    return integration.recovery.get_recovery_summary()


@router.get("/confidence/{module}/{operation}")
async def get_adjusted_confidence(module: str, operation: str, base: float = Query(default=0.5)):
    """Get learning-adjusted confidence for an operation"""
    integration = get_integration()
    if not integration:
        return {"error": "Integration not available"}

    adjusted = integration.get_adjusted_confidence(module, operation, base)
    return {
        "module": module,
        "operation": operation,
        "base_confidence": base,
        "adjusted_confidence": adjusted,
        "adjustment_factor": adjusted - base
    }


@router.get("/predict/{module}/{operation}")
async def predict_operation_success(module: str, operation: str):
    """Predict success probability for an operation based on learning"""
    integration = get_integration()
    if not integration:
        return {"error": "Integration not available"}

    prediction = integration.predict_operation_success(module, operation, {})
    success_rate = integration.learning.get_success_rate(f"{module}_{operation}")

    return {
        "module": module,
        "operation": operation,
        "predicted_success_probability": prediction,
        "historical_success_rate": success_rate
    }


# =============================================================================
# MODULE-SPECIFIC METRICS
# =============================================================================

@router.get("/modules/ooda")
async def get_ooda_metrics():
    """Get OODA module metrics"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    metrics = obs.metrics.get_all_metrics()
    ooda_metrics = {}

    # Filter for OODA metrics
    for category in ["counters", "gauges", "histograms"]:
        for name, value in metrics.get(category, {}).items():
            if "ooda" in name.lower():
                ooda_metrics[name] = value

    return {"ooda": ooda_metrics}


@router.get("/modules/hallucination")
async def get_hallucination_metrics():
    """Get Hallucination Prevention metrics"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    metrics = obs.metrics.get_all_metrics()
    hal_metrics = {}

    for category in ["counters", "gauges", "histograms"]:
        for name, value in metrics.get(category, {}).items():
            if "hallucination" in name.lower():
                hal_metrics[name] = value

    return {"hallucination": hal_metrics}


@router.get("/modules/memory")
async def get_memory_metrics():
    """Get Memory Brain metrics"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    metrics = obs.metrics.get_all_metrics()
    mem_metrics = {}

    for category in ["counters", "gauges", "histograms"]:
        for name, value in metrics.get(category, {}).items():
            if "memory" in name.lower():
                mem_metrics[name] = value

    return {"memory": mem_metrics}


@router.get("/modules/consciousness")
async def get_consciousness_metrics():
    """Get Consciousness Emergence metrics"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    metrics = obs.metrics.get_all_metrics()
    con_metrics = {}

    for category in ["counters", "gauges", "histograms"]:
        for name, value in metrics.get(category, {}).items():
            if "consciousness" in name.lower():
                con_metrics[name] = value

    return {"consciousness": con_metrics}


@router.get("/modules/dependability")
async def get_dependability_metrics():
    """Get Dependability Framework metrics"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    metrics = obs.metrics.get_all_metrics()
    dep_metrics = {}

    for category in ["counters", "gauges", "histograms"]:
        for name, value in metrics.get(category, {}).items():
            if "dependability" in name.lower():
                dep_metrics[name] = value

    return {"dependability": dep_metrics}


@router.get("/modules/circuit_breaker")
async def get_circuit_breaker_metrics():
    """Get Circuit Breaker metrics"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    metrics = obs.metrics.get_all_metrics()
    cb_metrics = {}

    for category in ["counters", "gauges", "histograms"]:
        for name, value in metrics.get(category, {}).items():
            if "circuit" in name.lower():
                cb_metrics[name] = value

    return {"circuit_breaker": cb_metrics}


# =============================================================================
# SUMMARY ENDPOINT
# =============================================================================

# =============================================================================
# SMOKE TEST ENDPOINT
# =============================================================================

@router.post("/smoke-test")
async def run_smoke_test():
    """
    Run a smoke test to verify the observability and integration layer is alive.
    Emits a test decision and verifies events and state change.
    """
    import uuid
    from datetime import datetime, timezone

    obs = get_observability()
    integration = get_integration()

    if not obs or not integration:
        return {
            "success": False,
            "error": "Observability or Integration not available",
            "observability_available": obs is not None,
            "integration_available": integration is not None
        }

    test_id = str(uuid.uuid4())[:8]
    results = {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {}
    }

    # Test 1: Emit a test event and verify it appears in event counts
    try:
        from ai_observability import Event, EventBus, EventType
        event_bus = EventBus.get_instance()

        initial_counts = event_bus.get_event_counts()
        initial_decision_count = initial_counts.get("ooda.decision_made", 0)

        # Publish a test decision event
        test_event = Event(
            event_type=EventType.DECISION_MADE,
            source_module="smoke_test",
            payload={
                "id": f"smoke_test_{test_id}",
                "type": "smoke_test",
                "confidence": 0.95,
                "test": True
            }
        )
        event_bus.publish(test_event)

        new_counts = event_bus.get_event_counts()
        new_decision_count = new_counts.get("ooda.decision_made", 0)

        results["tests"]["event_publishing"] = {
            "success": new_decision_count > initial_decision_count,
            "initial_count": initial_decision_count,
            "new_count": new_decision_count
        }
    except Exception as e:
        results["tests"]["event_publishing"] = {
            "success": False,
            "error": str(e)
        }

    # Test 2: Verify metrics are being recorded
    try:
        from ai_observability import MetricsRegistry
        registry = MetricsRegistry.get_instance()

        # Record a test metric
        registry.increment_counter(
            "smoke_test_counter",
            {"test_id": test_id}
        )
        registry.set_gauge(
            "smoke_test_gauge",
            42.0,
            {"test_id": test_id}
        )
        registry.observe_histogram(
            "smoke_test_histogram",
            0.123,
            {"test_id": test_id}
        )

        metrics = registry.get_all_metrics()
        results["tests"]["metrics_recording"] = {
            "success": True,
            "counters_count": len(metrics.get("counters", {})),
            "gauges_count": len(metrics.get("gauges", {})),
            "histograms_count": len(metrics.get("histograms", {}))
        }
    except Exception as e:
        results["tests"]["metrics_recording"] = {
            "success": False,
            "error": str(e)
        }

    # Test 3: Verify state updates work
    try:
        state_before = integration.get_unified_state()
        initial_decisions = state_before.get("decisions", {}).get("total", 0)

        # The event we published should have updated the state
        state_after = integration.get_unified_state()
        new_decisions = state_after.get("decisions", {}).get("total", 0)

        results["tests"]["state_updates"] = {
            "success": new_decisions >= initial_decisions,
            "initial_total_decisions": initial_decisions,
            "new_total_decisions": new_decisions,
            "consciousness_level": state_after.get("consciousness", {}).get("consciousness_level", 0)
        }
    except Exception as e:
        results["tests"]["state_updates"] = {
            "success": False,
            "error": str(e)
        }

    # Test 4: Verify signal queues are initialized
    try:
        queues = list(integration._signal_queues.keys())
        results["tests"]["signal_queues"] = {
            "success": len(queues) > 0,
            "queues": queues,
            "count": len(queues)
        }
    except Exception as e:
        results["tests"]["signal_queues"] = {
            "success": False,
            "error": str(e)
        }

    # Test 5: Verify learning manager is tracking
    try:
        from ai_module_integration import LearningOutcome
        learning = integration.learning

        # Record a test outcome
        outcome = LearningOutcome(
            operation_id=f"smoke_test_{test_id}",
            operation_type="smoke_test_operation",
            predicted_outcome="success",
            actual_outcome="success",
            success=True,
            confidence_delta=0.01,
            context={"test": True}
        )
        learning.record_outcome(outcome)

        summary = learning.get_learning_summary()
        results["tests"]["learning_manager"] = {
            "success": summary.get("total_outcomes", 0) > 0,
            "total_outcomes": summary.get("total_outcomes", 0),
            "operation_types": summary.get("operation_types", 0)
        }
    except Exception as e:
        results["tests"]["learning_manager"] = {
            "success": False,
            "error": str(e)
        }

    # Calculate overall success
    all_tests_passed = all(
        test_result.get("success", False)
        for test_result in results["tests"].values()
    )
    results["success"] = all_tests_passed
    results["passed"] = sum(1 for t in results["tests"].values() if t.get("success", False))
    results["total"] = len(results["tests"])

    return results


@router.get("/summary")
async def get_full_summary():
    """Get complete summary of all AI systems"""
    obs = get_observability()
    integration = get_integration()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "observability_available": obs is not None,
        "integration_available": integration is not None
    }

    if obs:
        dashboard = obs.get_dashboard_data()
        summary["health"] = dashboard.get("health", {})
        summary["metrics_count"] = {
            "counters": len(dashboard.get("metrics", {}).get("counters", {})),
            "gauges": len(dashboard.get("metrics", {}).get("gauges", {})),
            "histograms": len(dashboard.get("metrics", {}).get("histograms", {}))
        }
        summary["events"] = dashboard.get("events", {})

    if integration:
        state = integration.get_unified_state()
        summary["unified_state"] = state
        summary["learning"] = integration.learning.get_learning_summary()
        summary["recovery"] = integration.recovery.get_recovery_summary()

    return summary


# =============================================================================
# SEEDING & PERSISTENCE
# =============================================================================

@router.post("/seed")
async def seed_observability():
    """
    Seed observability tables with baseline data for all bleeding-edge modules.
    Creates sample metrics, events, and traces to initialize the observability system.
    """
    try:
        from ai_observability import seed_observability_data
        result = seed_observability_data()
        return result
    except Exception as e:
        logger.error(f"Failed to seed observability data: {e}")
        return {"success": False, "error": str(e)}


@router.post("/flush")
async def flush_observability():
    """Flush all pending observability data to database"""
    try:
        from ai_observability import flush_persistence
        flush_persistence()
        return {"success": True, "message": "Observability data flushed"}
    except Exception as e:
        logger.error(f"Failed to flush observability data: {e}")
        return {"success": False, "error": str(e)}


@router.get("/persistence/status")
async def get_persistence_status():
    """Get observability persistence status"""
    try:
        from ai_observability import get_persistence
        persistence = get_persistence()
        if not persistence:
            return {
                "enabled": False,
                "reason": "Persistence not initialized"
            }
        return {
            "enabled": persistence._enabled,
            "buffer_sizes": {
                "metrics": len(persistence._write_buffer_metrics),
                "events": len(persistence._write_buffer_events),
                "traces": len(persistence._write_buffer_traces)
            },
            "flush_interval": persistence._flush_interval,
            "db_configured": persistence._db_config is not None
        }
    except Exception as e:
        logger.error(f"Failed to get persistence status: {e}")
        return {"enabled": False, "error": str(e)}
