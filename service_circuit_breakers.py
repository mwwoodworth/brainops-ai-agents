#!/usr/bin/env python3
"""
SERVICE CIRCUIT BREAKERS MODULE
===============================
Centralized circuit breaker management for all external service calls.

This module provides:
- Per-service circuit breaker configuration
- Automatic failure tracking and recovery
- Health endpoint integration
- Observability metrics
- Decorators for easy integration

Based on the enhanced_circuit_breaker.py DynamicCircuitBreaker implementation.

Author: BrainOps AI System
Version: 1.0.0 (2026-01-27)
"""

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

# Import the enhanced circuit breaker
try:
    from enhanced_circuit_breaker import (
        DynamicCircuitBreaker,
        CircuitState,
        CircuitMetrics,
        CircuitAlert,
        AlertSeverity,
        get_self_healing_controller,
    )
    ENHANCED_CB_AVAILABLE = True
except ImportError:
    ENHANCED_CB_AVAILABLE = False
    # Fallback definitions
    class CircuitState(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

logger = logging.getLogger(__name__)

# Type variable for generic function signatures
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

class ServiceType(Enum):
    """Types of external services"""
    AI_PROVIDER = "ai_provider"
    DATABASE = "database"
    WEBHOOK = "webhook"
    API = "api"
    CACHE = "cache"
    QUEUE = "queue"


@dataclass
class ServiceCircuitConfig:
    """Configuration for a service's circuit breaker"""
    service_name: str
    service_type: ServiceType
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_requests: int = 3
    sliding_window_size: int = 100
    sliding_window_time: float = 300.0  # 5 minutes
    min_threshold: int = 2
    max_threshold: int = 20
    enable_prediction: bool = True
    priority: int = 5  # For deadlock resolution (higher = more important)
    dependencies: list[str] = field(default_factory=list)
    # Additional metadata
    description: str = ""
    critical: bool = False


# =============================================================================
# DEFAULT SERVICE CONFIGURATIONS
# =============================================================================

CIRCUIT_BREAKER_CONFIG: dict[str, ServiceCircuitConfig] = {
    # AI Providers - Slightly more lenient due to API variability
    "openai": ServiceCircuitConfig(
        service_name="openai",
        service_type=ServiceType.AI_PROVIDER,
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=8,
        description="OpenAI API (GPT models, embeddings)",
        critical=True
    ),
    "anthropic": ServiceCircuitConfig(
        service_name="anthropic",
        service_type=ServiceType.AI_PROVIDER,
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=8,
        description="Anthropic API (Claude models)",
        critical=True
    ),
    "gemini": ServiceCircuitConfig(
        service_name="gemini",
        service_type=ServiceType.AI_PROVIDER,
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=7,
        description="Google Gemini API",
        critical=False
    ),
    "huggingface": ServiceCircuitConfig(
        service_name="huggingface",
        service_type=ServiceType.AI_PROVIDER,
        failure_threshold=7,
        recovery_timeout=90.0,
        half_open_max_requests=3,
        sliding_window_time=300.0,
        priority=5,
        description="HuggingFace Inference API",
        critical=False
    ),

    # Database connections - More strict, faster recovery
    "database": ServiceCircuitConfig(
        service_name="database",
        service_type=ServiceType.DATABASE,
        failure_threshold=3,
        recovery_timeout=30.0,
        half_open_max_requests=1,
        sliding_window_time=120.0,
        min_threshold=2,
        priority=10,  # Highest priority
        description="Primary PostgreSQL database",
        critical=True
    ),
    "database_backup": ServiceCircuitConfig(
        service_name="database_backup",
        service_type=ServiceType.DATABASE,
        failure_threshold=3,
        recovery_timeout=30.0,
        half_open_max_requests=1,
        sliding_window_time=120.0,
        priority=9,
        description="Backup database connection",
        critical=True
    ),

    # External webhooks - More lenient, longer recovery
    "webhook_gumroad": ServiceCircuitConfig(
        service_name="webhook_gumroad",
        service_type=ServiceType.WEBHOOK,
        failure_threshold=8,
        recovery_timeout=120.0,
        half_open_max_requests=2,
        sliding_window_time=600.0,
        priority=4,
        description="Gumroad webhook processing",
        critical=False
    ),
    "webhook_stripe": ServiceCircuitConfig(
        service_name="webhook_stripe",
        service_type=ServiceType.WEBHOOK,
        failure_threshold=5,
        recovery_timeout=90.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=6,
        description="Stripe webhook processing",
        critical=True
    ),
    "webhook_github": ServiceCircuitConfig(
        service_name="webhook_github",
        service_type=ServiceType.WEBHOOK,
        failure_threshold=10,
        recovery_timeout=180.0,
        half_open_max_requests=3,
        sliding_window_time=600.0,
        priority=3,
        description="GitHub webhook processing",
        critical=False
    ),

    # Internal/ERP API calls
    "erp_api": ServiceCircuitConfig(
        service_name="erp_api",
        service_type=ServiceType.API,
        failure_threshold=5,
        recovery_timeout=45.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=7,
        dependencies=["database"],
        description="Weathercraft ERP API",
        critical=True
    ),
    "brainops_backend": ServiceCircuitConfig(
        service_name="brainops_backend",
        service_type=ServiceType.API,
        failure_threshold=5,
        recovery_timeout=45.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=8,
        dependencies=["database"],
        description="BrainOps Backend API",
        critical=True
    ),
    "mcp_bridge": ServiceCircuitConfig(
        service_name="mcp_bridge",
        service_type=ServiceType.API,
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=6,
        dependencies=["brainops_backend"],
        description="MCP Bridge Service",
        critical=False
    ),

    # External APIs
    "resend_email": ServiceCircuitConfig(
        service_name="resend_email",
        service_type=ServiceType.API,
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_requests=2,
        sliding_window_time=300.0,
        priority=5,
        description="Resend Email API",
        critical=False
    ),
    "render_api": ServiceCircuitConfig(
        service_name="render_api",
        service_type=ServiceType.API,
        failure_threshold=5,
        recovery_timeout=90.0,
        half_open_max_requests=2,
        sliding_window_time=600.0,
        priority=4,
        description="Render Deployment API",
        critical=False
    ),
    "vercel_api": ServiceCircuitConfig(
        service_name="vercel_api",
        service_type=ServiceType.API,
        failure_threshold=5,
        recovery_timeout=90.0,
        half_open_max_requests=2,
        sliding_window_time=600.0,
        priority=4,
        description="Vercel Deployment API",
        critical=False
    ),
}


# =============================================================================
# CIRCUIT BREAKER MANAGER
# =============================================================================

class ServiceCircuitBreakerManager:
    """
    Centralized manager for all service circuit breakers.

    Provides:
    - Automatic circuit breaker creation with service-specific configs
    - Unified status reporting for health endpoints
    - Alert aggregation
    - Metrics collection for observability
    """

    _instance: Optional['ServiceCircuitBreakerManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._circuits: dict[str, Any] = {}  # DynamicCircuitBreaker or fallback
        self._configs: dict[str, ServiceCircuitConfig] = {}
        self._alert_handlers: list[Callable[[str, dict], None]] = []
        self._metrics_history: list[dict] = []
        self._metrics_lock = threading.Lock()
        self._max_history = 1000

        # Initialize with default configs
        for service_name, config in CIRCUIT_BREAKER_CONFIG.items():
            self.register_service(config)

        self._initialized = True
        logger.info(f"ServiceCircuitBreakerManager initialized with {len(self._circuits)} circuits")

    def register_service(self, config: ServiceCircuitConfig) -> None:
        """Register a service with its circuit breaker configuration"""
        if ENHANCED_CB_AVAILABLE:
            circuit = DynamicCircuitBreaker(
                component_id=config.service_name,
                base_failure_threshold=config.failure_threshold,
                base_recovery_timeout=config.recovery_timeout,
                half_open_max_requests=config.half_open_max_requests,
                sliding_window_size=config.sliding_window_size,
                sliding_window_time=config.sliding_window_time,
                min_threshold=config.min_threshold,
                max_threshold=config.max_threshold,
                enable_prediction=config.enable_prediction
            )
            # Add alert handler
            circuit.add_alert_handler(lambda alert: self._on_circuit_alert(config.service_name, alert))
        else:
            # Fallback simple circuit breaker
            circuit = _SimpleFallbackCircuit(
                service_name=config.service_name,
                failure_threshold=config.failure_threshold,
                recovery_timeout=config.recovery_timeout
            )

        self._circuits[config.service_name] = circuit
        self._configs[config.service_name] = config
        logger.debug(f"Registered circuit breaker for {config.service_name}")

    def get_circuit(self, service_name: str) -> Optional[Any]:
        """Get circuit breaker for a service"""
        if service_name not in self._circuits:
            # Auto-create with defaults if not exists
            logger.warning(f"Circuit breaker for {service_name} not found, creating with defaults")
            config = ServiceCircuitConfig(
                service_name=service_name,
                service_type=ServiceType.API
            )
            self.register_service(config)

        return self._circuits.get(service_name)

    def record_success(
        self,
        service_name: str,
        response_time_ms: float = 0
    ) -> None:
        """Record a successful service call"""
        circuit = self.get_circuit(service_name)
        if circuit:
            circuit.record_success(response_time_ms)
            self._record_metric(service_name, "success", response_time_ms)

    def record_failure(
        self,
        service_name: str,
        response_time_ms: float = 0,
        error: Optional[str] = None
    ) -> None:
        """Record a failed service call"""
        circuit = self.get_circuit(service_name)
        if circuit:
            if ENHANCED_CB_AVAILABLE:
                circuit.record_failure(response_time_ms, error)
            else:
                circuit.record_failure(response_time_ms)
            self._record_metric(service_name, "failure", response_time_ms, error)

    def allows_request(self, service_name: str) -> bool:
        """Check if circuit allows the request"""
        circuit = self.get_circuit(service_name)
        if circuit:
            return circuit.allows_request
        return True  # Default to allowing if no circuit

    def is_open(self, service_name: str) -> bool:
        """Check if circuit is open (blocking requests)"""
        circuit = self.get_circuit(service_name)
        if circuit:
            return circuit.is_open
        return False

    def reset(self, service_name: str) -> None:
        """Force reset a circuit breaker"""
        circuit = self.get_circuit(service_name)
        if circuit:
            circuit.reset()
            logger.info(f"Circuit breaker reset for {service_name}")

    def _on_circuit_alert(self, service_name: str, alert: Any) -> None:
        """Handle circuit breaker alerts"""
        alert_dict = {
            "service": service_name,
            "severity": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity),
            "message": alert.message,
            "action": alert.auto_action_taken,
            "timestamp": datetime.now().isoformat()
        }

        for handler in self._alert_handlers:
            try:
                handler(service_name, alert_dict)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def _record_metric(
        self,
        service_name: str,
        outcome: str,
        response_time_ms: float,
        error: Optional[str] = None
    ) -> None:
        """Record metric for observability"""
        metric = {
            "service": service_name,
            "outcome": outcome,
            "response_time_ms": response_time_ms,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        with self._metrics_lock:
            self._metrics_history.append(metric)
            # Trim history if needed
            if len(self._metrics_history) > self._max_history:
                self._metrics_history = self._metrics_history[-self._max_history:]

    def add_alert_handler(self, handler: Callable[[str, dict], None]) -> None:
        """Add handler for circuit alerts"""
        self._alert_handlers.append(handler)

    def get_status(self, service_name: Optional[str] = None) -> dict[str, Any]:
        """Get status of circuit breaker(s)"""
        if service_name:
            return self._get_single_status(service_name)

        # Return status of all circuits
        return {
            "timestamp": datetime.now().isoformat(),
            "total_circuits": len(self._circuits),
            "summary": self._get_summary(),
            "circuits": {
                name: self._get_single_status(name)
                for name in self._circuits
            }
        }

    def _get_single_status(self, service_name: str) -> dict[str, Any]:
        """Get status of a single circuit"""
        # Be consistent with other call sites (allows_request/reset/etc) which
        # auto-create unknown circuits on first use.
        circuit = self.get_circuit(service_name)
        config = self._configs.get(service_name)

        if not circuit:
            return {"error": f"Circuit {service_name} not found"}

        if ENHANCED_CB_AVAILABLE and hasattr(circuit, 'get_metrics'):
            metrics = circuit.get_metrics()
            return {
                "service_name": service_name,
                "service_type": config.service_type.value if config else "unknown",
                "state": metrics.state.value,
                "failure_count": metrics.failure_count,
                "success_count": metrics.success_count,
                "total_requests": metrics.total_requests,
                "failure_rate": round(metrics.failure_rate, 4),
                "avg_response_time_ms": round(metrics.avg_response_time_ms, 2),
                "consecutive_failures": metrics.consecutive_failures,
                "dynamic_threshold": round(metrics.dynamic_threshold, 2),
                "health_score": round(circuit.get_health_score(), 2) if hasattr(circuit, 'get_health_score') else None,
                "critical": config.critical if config else False,
                "state_changed_at": metrics.state_changed_at.isoformat() if metrics.state_changed_at else None
            }
        else:
            # Fallback circuit status
            return {
                "service_name": service_name,
                "service_type": config.service_type.value if config else "unknown",
                "state": circuit.state if hasattr(circuit, 'state') else "unknown",
                "failure_count": getattr(circuit, 'failure_count', 0),
                "critical": config.critical if config else False
            }

    def _get_summary(self) -> dict[str, Any]:
        """Get summary statistics"""
        states = {"closed": 0, "open": 0, "half_open": 0}
        critical_open = []

        for name, circuit in self._circuits.items():
            config = self._configs.get(name)

            if ENHANCED_CB_AVAILABLE and hasattr(circuit, 'state'):
                state = circuit.state.value if hasattr(circuit.state, 'value') else str(circuit.state)
            else:
                state = getattr(circuit, 'state', 'closed')

            if state in states:
                states[state] += 1

            if state == "open" and config and config.critical:
                critical_open.append(name)

        return {
            "by_state": states,
            "critical_circuits_open": critical_open,
            "health_status": "degraded" if critical_open else "healthy"
        }

    def get_health_endpoint_data(self) -> dict[str, Any]:
        """Get data formatted for health endpoint integration"""
        summary = self._get_summary()

        return {
            "circuit_breakers": {
                "total": len(self._circuits),
                "open": summary["by_state"]["open"],
                "half_open": summary["by_state"]["half_open"],
                "closed": summary["by_state"]["closed"],
                "critical_open": summary["critical_circuits_open"],
                "overall_health": summary["health_status"]
            }
        }

    def get_recent_metrics(self, limit: int = 100) -> list[dict]:
        """Get recent metrics for observability"""
        with self._metrics_lock:
            return self._metrics_history[-limit:]


# =============================================================================
# FALLBACK CIRCUIT BREAKER (when enhanced not available)
# =============================================================================

class _SimpleFallbackCircuit:
    """Simple fallback circuit breaker when enhanced version is not available"""

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._check_recovery()
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    @property
    def is_closed(self) -> bool:
        return self.state == "closed"

    @property
    def allows_request(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open":
            return True  # Allow test requests
        return False

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def record_success(self, response_time_ms: float = 0):
        with self._lock:
            self._success_count += 1

            if self._state == "half_open":
                self._half_open_successes += 1
                if self._half_open_successes >= 3:
                    self._state = "closed"
                    self._failure_count = 0
                    self._half_open_successes = 0
                    logger.info(f"Circuit closed for {self.service_name}")

    def record_failure(self, response_time_ms: float = 0):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == "half_open":
                self._state = "open"
                self._half_open_successes = 0
                logger.warning(f"Circuit re-opened for {self.service_name}")
            elif self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(f"Circuit opened for {self.service_name} after {self._failure_count} failures")

    def _check_recovery(self):
        if self._state != "open":
            return

        if self._last_failure_time is None:
            return

        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.recovery_timeout:
            self._state = "half_open"
            self._half_open_successes = 0
            logger.info(f"Circuit half-open for {self.service_name}")

    def reset(self):
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._success_count = 0
            self._half_open_successes = 0
            logger.info(f"Circuit reset for {self.service_name}")


# =============================================================================
# DECORATORS FOR EASY INTEGRATION
# =============================================================================

def with_circuit_breaker(
    service_name: str,
    fallback: Optional[Callable[..., Any]] = None,
    raise_on_open: bool = True
):
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        service_name: Name of the service (must match CIRCUIT_BREAKER_CONFIG key)
        fallback: Optional fallback function to call when circuit is open
        raise_on_open: If True, raise CircuitOpenError when circuit is open

    Example:
        @with_circuit_breaker("openai")
        async def call_openai(prompt: str) -> str:
            # API call here
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()

            if not manager.allows_request(service_name):
                if fallback:
                    return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
                if raise_on_open:
                    raise CircuitOpenError(f"Circuit breaker open for {service_name}")
                return None

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                manager.record_success(service_name, response_time)
                return result
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                manager.record_failure(service_name, response_time, str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()

            if not manager.allows_request(service_name):
                if fallback:
                    return fallback(*args, **kwargs)
                if raise_on_open:
                    raise CircuitOpenError(f"Circuit breaker open for {service_name}")
                return None

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                manager.record_success(service_name, response_time)
                return result
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                manager.record_failure(service_name, response_time, str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class CircuitOpenError(Exception):
    """Raised when attempting to call a service with an open circuit"""
    pass


# =============================================================================
# CONTEXT MANAGER FOR MANUAL TRACKING
# =============================================================================

class CircuitBreakerContext:
    """
    Context manager for tracking service calls with circuit breaker.

    Example:
        async with CircuitBreakerContext("openai") as cb:
            response = await openai_client.chat.completions.create(...)
            cb.mark_success()
    """

    def __init__(self, service_name: str, raise_on_open: bool = True):
        self.service_name = service_name
        self.raise_on_open = raise_on_open
        self.manager = get_circuit_breaker_manager()
        self._start_time: float = 0
        self._completed = False

    async def __aenter__(self) -> 'CircuitBreakerContext':
        if not self.manager.allows_request(self.service_name):
            if self.raise_on_open:
                raise CircuitOpenError(f"Circuit breaker open for {self.service_name}")
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._completed:
            response_time = (time.time() - self._start_time) * 1000
            if exc_type is not None:
                self.manager.record_failure(self.service_name, response_time, str(exc_val))
            # If not explicitly marked, assume failure if there was an exception
        return False  # Don't suppress exceptions

    def __enter__(self) -> 'CircuitBreakerContext':
        if not self.manager.allows_request(self.service_name):
            if self.raise_on_open:
                raise CircuitOpenError(f"Circuit breaker open for {self.service_name}")
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._completed:
            response_time = (time.time() - self._start_time) * 1000
            if exc_type is not None:
                self.manager.record_failure(self.service_name, response_time, str(exc_val))
        return False

    def mark_success(self, response_time_ms: Optional[float] = None):
        """Explicitly mark the operation as successful"""
        if response_time_ms is None:
            response_time_ms = (time.time() - self._start_time) * 1000
        self.manager.record_success(self.service_name, response_time_ms)
        self._completed = True

    def mark_failure(self, error: Optional[str] = None, response_time_ms: Optional[float] = None):
        """Explicitly mark the operation as failed"""
        if response_time_ms is None:
            response_time_ms = (time.time() - self._start_time) * 1000
        self.manager.record_failure(self.service_name, response_time_ms, error)
        self._completed = True


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_manager_instance: Optional[ServiceCircuitBreakerManager] = None
_manager_lock = threading.Lock()


def get_circuit_breaker_manager() -> ServiceCircuitBreakerManager:
    """Get the singleton circuit breaker manager instance"""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = ServiceCircuitBreakerManager()
    return _manager_instance


# =============================================================================
# HEALTH INTEGRATION HELPER
# =============================================================================

def get_circuit_breaker_health() -> dict[str, Any]:
    """
    Get circuit breaker health data for health endpoint integration.

    Returns data suitable for including in /health endpoint response.
    """
    manager = get_circuit_breaker_manager()
    return manager.get_health_endpoint_data()


def get_all_circuit_statuses() -> dict[str, Any]:
    """
    Get complete status of all circuit breakers.

    Returns detailed status for each circuit including metrics.
    """
    manager = get_circuit_breaker_manager()
    return manager.get_status()


# =============================================================================
# SERVICE-SPECIFIC HELPER FUNCTIONS
# =============================================================================

def check_service_available(service_name: str) -> bool:
    """Quick check if a service's circuit allows requests"""
    manager = get_circuit_breaker_manager()
    return manager.allows_request(service_name)


def report_service_success(service_name: str, response_time_ms: float = 0) -> None:
    """Report a successful service call"""
    manager = get_circuit_breaker_manager()
    manager.record_success(service_name, response_time_ms)


def report_service_failure(service_name: str, response_time_ms: float = 0, error: str = "") -> None:
    """Report a failed service call"""
    manager = get_circuit_breaker_manager()
    manager.record_failure(service_name, response_time_ms, error)


def reset_service_circuit(service_name: str) -> None:
    """Force reset a service's circuit breaker"""
    manager = get_circuit_breaker_manager()
    manager.reset(service_name)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "CIRCUIT_BREAKER_CONFIG",
    "ServiceCircuitConfig",
    "ServiceType",

    # Manager
    "ServiceCircuitBreakerManager",
    "get_circuit_breaker_manager",

    # Decorators
    "with_circuit_breaker",
    "CircuitBreakerContext",
    "CircuitOpenError",

    # Health integration
    "get_circuit_breaker_health",
    "get_all_circuit_statuses",

    # Helper functions
    "check_service_available",
    "report_service_success",
    "report_service_failure",
    "reset_service_circuit",
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def example_usage():
        """Example of using the circuit breaker module"""

        print("\n=== Service Circuit Breakers Example ===\n")

        # Get manager
        manager = get_circuit_breaker_manager()

        # Check initial status
        print("1. Initial Circuit Status:")
        health = get_circuit_breaker_health()
        print(f"   Total circuits: {health['circuit_breakers']['total']}")
        print(f"   All closed: {health['circuit_breakers']['closed']}")
        print(f"   Overall health: {health['circuit_breakers']['overall_health']}")

        # Simulate some successes
        print("\n2. Recording 5 successes for 'openai':")
        for i in range(5):
            manager.record_success("openai", response_time_ms=100 + i * 10)
        status = manager.get_status("openai")
        print(f"   State: {status['state']}")
        print(f"   Success count: {status['success_count']}")

        # Simulate failures to open circuit
        print("\n3. Recording 5 failures for 'openai':")
        for i in range(5):
            manager.record_failure("openai", response_time_ms=5000, error="API timeout")
        status = manager.get_status("openai")
        print(f"   State: {status['state']}")
        print(f"   Failure count: {status['failure_count']}")
        print(f"   Allows request: {manager.allows_request('openai')}")

        # Using decorator example (simulated)
        print("\n4. Decorator usage example:")

        @with_circuit_breaker("anthropic")
        async def call_anthropic(prompt: str) -> str:
            # Simulated API call
            await asyncio.sleep(0.1)
            return f"Response to: {prompt}"

        try:
            result = await call_anthropic("Hello!")
            print(f"   Result: {result}")
        except CircuitOpenError as e:
            print(f"   Circuit open: {e}")

        # Context manager example
        print("\n5. Context manager usage:")
        try:
            with CircuitBreakerContext("database") as cb:
                # Simulated database call
                print("   Executing database query...")
                cb.mark_success(response_time_ms=25)
            print("   Database call successful")
        except CircuitOpenError as e:
            print(f"   Circuit open: {e}")

        # Reset circuit
        print("\n6. Resetting 'openai' circuit:")
        manager.reset("openai")
        status = manager.get_status("openai")
        print(f"   State after reset: {status['state']}")

        # Final status
        print("\n7. Final Health Status:")
        all_status = get_all_circuit_statuses()
        print(f"   Summary: {all_status['summary']}")

        print("\n=== Example Complete ===")

    asyncio.run(example_usage())
