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
from fastapi import APIRouter, Query
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

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

@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Get all metrics in Prometheus format.
    Compatible with Prometheus scraping.
    """
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return obs.get_prometheus_metrics()


@router.get("/metrics/json")
async def get_metrics_json():
    """Get all metrics in JSON format for dashboards"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    return obs.metrics.get_all_metrics()


@router.get("/metrics/histogram/{name}")
async def get_histogram(name: str):
    """Get specific histogram with percentiles"""
    obs = get_observability()
    if not obs:
        return {"error": "Observability not available"}

    histogram = obs.metrics.get_histogram(name)
    if histogram:
        return {
            "name": histogram.name,
            "count": histogram._count,
            "sum": histogram._sum,
            "avg": histogram.avg,
            "p50": histogram.p50,
            "p95": histogram.p95,
            "p99": histogram.p99
        }
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
