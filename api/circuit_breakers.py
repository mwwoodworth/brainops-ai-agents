"""
Circuit Breaker API Endpoints
=============================
Provides REST endpoints for circuit breaker management, monitoring, and control.

Endpoints:
- GET /circuit-breakers/status - Get all circuit breaker statuses
- GET /circuit-breakers/{service_name} - Get specific service circuit status
- POST /circuit-breakers/{service_name}/reset - Reset a circuit breaker
- GET /circuit-breakers/health - Get circuit breaker health summary
- GET /circuit-breakers/metrics - Get recent circuit breaker metrics

Author: BrainOps AI System
Version: 1.0.0 (2026-01-27)
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuit-breakers", tags=["Circuit Breakers"])

# Import circuit breaker manager
try:
    from service_circuit_breakers import (
        get_circuit_breaker_manager,
        get_circuit_breaker_health,
        get_all_circuit_statuses,
        CIRCUIT_BREAKER_CONFIG,
        ServiceType,
    )
    CB_AVAILABLE = True
except ImportError as e:
    CB_AVAILABLE = False
    logger.warning(f"Circuit breaker module not available: {e}")


def _ensure_available():
    """Ensure circuit breaker module is available"""
    if not CB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Circuit breaker module not available"
        )


@router.get("/status")
async def get_circuit_breaker_status(
    service_type: Optional[str] = Query(None, description="Filter by service type (ai_provider, database, webhook, api)")
) -> dict[str, Any]:
    """
    Get status of all circuit breakers.

    Optionally filter by service type:
    - ai_provider: OpenAI, Anthropic, Gemini, HuggingFace
    - database: Primary and backup database connections
    - webhook: Gumroad, Stripe, GitHub webhooks
    - api: Internal and external API calls
    """
    _ensure_available()

    status = get_all_circuit_statuses()

    # Filter by service type if specified
    if service_type:
        filtered_circuits = {}
        for name, circuit_status in status.get("circuits", {}).items():
            if circuit_status.get("service_type") == service_type:
                filtered_circuits[name] = circuit_status
        status["circuits"] = filtered_circuits
        status["total_circuits"] = len(filtered_circuits)

    return status


@router.get("/health")
async def get_health_summary() -> dict[str, Any]:
    """
    Get circuit breaker health summary for integration with health endpoints.

    Returns:
    - Total number of circuits
    - Count by state (open, half_open, closed)
    - List of critical circuits that are open
    - Overall health status
    """
    _ensure_available()

    return get_circuit_breaker_health()


@router.get("/config")
async def get_configuration() -> dict[str, Any]:
    """
    Get circuit breaker configuration for all services.

    Returns the configured thresholds, timeouts, and settings for each service.
    """
    _ensure_available()

    config_dict = {}
    for service_name, config in CIRCUIT_BREAKER_CONFIG.items():
        config_dict[service_name] = {
            "service_type": config.service_type.value,
            "failure_threshold": config.failure_threshold,
            "recovery_timeout": config.recovery_timeout,
            "half_open_max_requests": config.half_open_max_requests,
            "sliding_window_time": config.sliding_window_time,
            "priority": config.priority,
            "critical": config.critical,
            "description": config.description,
            "dependencies": config.dependencies
        }

    return {
        "total_services": len(config_dict),
        "service_types": list(set(c["service_type"] for c in config_dict.values())),
        "configurations": config_dict
    }


@router.get("/{service_name}")
async def get_service_circuit_status(service_name: str) -> dict[str, Any]:
    """
    Get circuit breaker status for a specific service.

    Args:
        service_name: Name of the service (e.g., 'openai', 'database', 'webhook_stripe')
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()
    status = manager.get_status(service_name)

    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])

    return status


@router.post("/{service_name}/reset")
async def reset_service_circuit(service_name: str) -> dict[str, Any]:
    """
    Force reset a circuit breaker to closed state.

    Use this to manually recover a circuit after the underlying issue is resolved.

    Args:
        service_name: Name of the service to reset
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()

    # Get current state first
    current_status = manager.get_status(service_name)
    if "error" in current_status:
        raise HTTPException(status_code=404, detail=current_status["error"])

    old_state = current_status.get("state", "unknown")

    # Reset the circuit
    manager.reset(service_name)

    # Get new state
    new_status = manager.get_status(service_name)

    return {
        "service_name": service_name,
        "action": "reset",
        "old_state": old_state,
        "new_state": new_status.get("state", "closed"),
        "message": f"Circuit breaker for {service_name} has been reset"
    }


@router.get("/metrics/recent")
async def get_recent_metrics(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent metrics to return")
) -> dict[str, Any]:
    """
    Get recent circuit breaker metrics for observability.

    Returns success/failure events with timing information.
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()
    metrics = manager.get_recent_metrics(limit)

    # Calculate summary stats
    success_count = sum(1 for m in metrics if m["outcome"] == "success")
    failure_count = sum(1 for m in metrics if m["outcome"] == "failure")

    # Group by service
    by_service: dict[str, dict[str, int]] = {}
    for m in metrics:
        service = m["service"]
        if service not in by_service:
            by_service[service] = {"success": 0, "failure": 0}
        by_service[service][m["outcome"]] += 1

    return {
        "total_events": len(metrics),
        "summary": {
            "successes": success_count,
            "failures": failure_count,
            "failure_rate": failure_count / len(metrics) if metrics else 0
        },
        "by_service": by_service,
        "recent_events": metrics[-50:]  # Return last 50 events
    }


@router.get("/check/{service_name}")
async def check_service_availability(service_name: str) -> dict[str, Any]:
    """
    Quick check if a service's circuit allows requests.

    Use this before making a service call to check if the circuit is open.

    Args:
        service_name: Name of the service to check

    Returns:
        - available: True if circuit allows requests
        - state: Current circuit state
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()
    allows = manager.allows_request(service_name)

    return {
        "service_name": service_name,
        "available": allows,
        "state": "closed" if allows else manager.get_status(service_name).get("state", "unknown"),
        "message": "Service available" if allows else "Circuit is open - requests blocked"
    }


@router.get("/ai-providers")
async def get_ai_provider_status() -> dict[str, Any]:
    """
    Get circuit breaker status for all AI providers.

    Convenient endpoint to check availability of OpenAI, Anthropic, Gemini, and HuggingFace.
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()

    ai_services = ["openai", "anthropic", "gemini", "huggingface"]
    statuses = {}
    available_count = 0

    for service in ai_services:
        status = manager.get_status(service)
        allows = manager.allows_request(service)
        if allows:
            available_count += 1
        statuses[service] = {
            "available": allows,
            "state": status.get("state", "unknown"),
            "failure_count": status.get("failure_count", 0),
            "health_score": status.get("health_score")
        }

    return {
        "summary": {
            "total": len(ai_services),
            "available": available_count,
            "unavailable": len(ai_services) - available_count
        },
        "providers": statuses
    }


@router.get("/databases")
async def get_database_status() -> dict[str, Any]:
    """
    Get circuit breaker status for database connections.
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()

    db_services = ["database", "database_backup"]
    statuses = {}

    for service in db_services:
        status = manager.get_status(service)
        allows = manager.allows_request(service)
        statuses[service] = {
            "available": allows,
            "state": status.get("state", "unknown"),
            "failure_count": status.get("failure_count", 0),
            "response_time_avg_ms": status.get("avg_response_time_ms", 0)
        }

    primary_available = statuses.get("database", {}).get("available", False)
    backup_available = statuses.get("database_backup", {}).get("available", False)

    return {
        "summary": {
            "primary_available": primary_available,
            "backup_available": backup_available,
            "any_available": primary_available or backup_available
        },
        "connections": statuses
    }


@router.post("/test/{service_name}/success")
async def simulate_success(
    service_name: str,
    response_time_ms: float = Query(100.0, description="Simulated response time in milliseconds")
) -> dict[str, Any]:
    """
    [DEBUG] Simulate a successful service call for testing.

    Records a success event to the circuit breaker.
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()
    manager.record_success(service_name, response_time_ms)

    status = manager.get_status(service_name)

    return {
        "service_name": service_name,
        "action": "recorded_success",
        "response_time_ms": response_time_ms,
        "current_state": status.get("state"),
        "success_count": status.get("success_count", 0)
    }


@router.post("/test/{service_name}/failure")
async def simulate_failure(
    service_name: str,
    response_time_ms: float = Query(5000.0, description="Simulated response time in milliseconds"),
    error: str = Query("Simulated failure", description="Error message")
) -> dict[str, Any]:
    """
    [DEBUG] Simulate a failed service call for testing.

    Records a failure event to the circuit breaker.
    """
    _ensure_available()

    manager = get_circuit_breaker_manager()
    manager.record_failure(service_name, response_time_ms, error)

    status = manager.get_status(service_name)

    return {
        "service_name": service_name,
        "action": "recorded_failure",
        "response_time_ms": response_time_ms,
        "error": error,
        "current_state": status.get("state"),
        "failure_count": status.get("failure_count", 0),
        "consecutive_failures": status.get("consecutive_failures", 0)
    }
