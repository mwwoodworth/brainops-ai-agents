#!/usr/bin/env python3
"""
SELF-HEALING RECONCILIATION LOOP
2025 Cutting-Edge Pattern: Autonomous infrastructure healing

Based on Kubernetes operator/controller reconciliation pattern:
1. Observe - Detect anomalies via telemetry
2. Analyze - Determine root cause
3. Act - Execute remediation
4. Verify - Confirm healing success

Components:
- Health Observer: Collects latency, error rates, drift signals
- Anomaly Detector: Triggers incidents when thresholds exceeded
- Remediation Agent: Isolates, reroutes, retrains, rollbacks
- Circuit Breaker: Prevents cascading failures

Based on Perplexity research on self-healing AI infrastructure patterns 2025.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal
import psycopg2
from psycopg2.extras import Json
import httpx
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ============================================================================
# SHARED CONNECTION POOL - CRITICAL for preventing MaxClientsInSessionMode
# ============================================================================
try:
    from database.sync_pool import get_sync_pool
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False


@contextmanager
def _get_pooled_connection():
    """Get connection from shared pool - ALWAYS use this instead of psycopg2.connect()"""
    if _POOL_AVAILABLE:
        pool = get_sync_pool()
        with pool.get_connection() as conn:
            yield conn
    else:
        conn = psycopg2.connect(**_get_db_config())
        try:
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime, Decimal, and Enum types"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def json_safe_serialize(obj: Any) -> Any:
    """Recursively convert datetime/Decimal/Enum/bytes objects to JSON-serializable types"""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, dict):
        return {str(k): json_safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe_serialize(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return json_safe_serialize(obj.__dict__)
    else:
        return str(obj)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }

DB_CONFIG = None  # Lazy initialization - use _get_db_config() instead


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RemediationAction(Enum):
    """Types of remediation actions"""
    SCALE_DOWN = "scale_down"
    ROUTE_TO_FALLBACK = "route_to_fallback"
    TRIGGER_RETRAIN = "trigger_retrain"
    CONFIG_ROLLBACK = "config_rollback"
    CACHE_CLEAR = "cache_clear"
    CIRCUIT_BREAK = "circuit_break"
    ALERT_HUMAN = "alert_human"
    RESTART_SERVICE = "restart_service"


@dataclass
class HealthMetrics:
    """Health metrics for a component"""
    component_id: str
    component_type: str
    latency_p50_ms: float
    latency_p99_ms: float
    error_rate: float
    drift_score: float
    memory_usage_pct: float
    cpu_usage_pct: float
    request_count: int
    success_rate: float
    timestamp: datetime


@dataclass
class Incident:
    """Detected incident"""
    incident_id: str
    component_id: str
    severity: str
    incident_type: str
    description: str
    metrics: HealthMetrics
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    remediation_actions: List[str] = None
    human_escalated: bool = False


@dataclass
class ReconciliationResult:
    """Result of reconciliation loop"""
    cycle_id: str
    start_time: datetime
    end_time: datetime
    components_checked: int
    incidents_detected: int
    remediations_executed: int
    success: bool
    details: Dict[str, Any]


class CircuitBreaker:
    """Circuit breaker for component protection"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = {}
        self.open_circuits: Dict[str, datetime] = {}

    def is_open(self, component_id: str) -> bool:
        """Check if circuit is open (blocking requests)"""
        if component_id not in self.open_circuits:
            return False

        open_time = self.open_circuits[component_id]
        if datetime.now() - open_time > timedelta(seconds=self.recovery_timeout):
            # Recovery timeout passed, allow half-open test
            return False

        return True

    def record_failure(self, component_id: str):
        """Record a failure for component"""
        self.failures[component_id] = self.failures.get(component_id, 0) + 1

        if self.failures[component_id] >= self.failure_threshold:
            self.open_circuits[component_id] = datetime.now()
            logger.warning(f"Circuit breaker OPEN for {component_id}")

    def record_success(self, component_id: str):
        """Record success, potentially closing circuit"""
        self.failures[component_id] = 0
        if component_id in self.open_circuits:
            del self.open_circuits[component_id]
            logger.info(f"Circuit breaker CLOSED for {component_id}")

    def reset(self, component_id: str):
        """Force reset circuit"""
        self.failures[component_id] = 0
        if component_id in self.open_circuits:
            del self.open_circuits[component_id]


class SelfHealingReconciler:
    """
    Self-healing reconciliation loop for BrainOps infrastructure.
    Continuously monitors, detects, and heals system issues.
    """

    def __init__(self):
        # Thresholds for anomaly detection
        self.thresholds = {
            "latency_p99_ms": float(os.getenv("HEAL_LATENCY_THRESH", "5000")),
            "error_rate": float(os.getenv("HEAL_ERROR_THRESH", "0.05")),
            "drift_score": float(os.getenv("HEAL_DRIFT_THRESH", "0.2")),
            "memory_usage_pct": float(os.getenv("HEAL_MEMORY_THRESH", "90")),
            "cpu_usage_pct": float(os.getenv("HEAL_CPU_THRESH", "90")),
            "success_rate": float(os.getenv("HEAL_SUCCESS_THRESH", "0.95")),
        }

        # Components to monitor
        self.monitored_components = {
            "ai_agents": "https://brainops-ai-agents.onrender.com",
            "backend": "https://brainops-backend-prod.onrender.com",
            "mcp_bridge": "https://brainops-mcp-bridge.onrender.com",
        }

        # Fallback routes
        self.fallback_routes = {
            "ai_agents": "local_ai_fallback",
            "backend": "cached_response",
            "mcp_bridge": "direct_api",
        }

        self.circuit_breaker = CircuitBreaker()
        self.active_incidents: Dict[str, Incident] = {}
        self.running = False
        self.reconcile_interval = int(os.getenv("HEAL_INTERVAL_SECONDS", "120"))  # 2 min for stability

        self._init_database()

    def _init_database(self):
        """Initialize self-healing tables"""
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    logger.warning("Self-healing database init skipped - no connection available")
                    return
                cur = conn.cursor()

                cur.execute("""
            CREATE TABLE IF NOT EXISTS healing_incidents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                component_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                description TEXT,
                metrics JSONB,
                remediation_actions JSONB,
                human_escalated BOOLEAN DEFAULT FALSE,
                detected_at TIMESTAMP DEFAULT NOW(),
                resolved_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS healing_reconciliations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                cycle_id TEXT NOT NULL,
                components_checked INT,
                incidents_detected INT,
                remediations_executed INT,
                success BOOLEAN,
                details JSONB,
                started_at TIMESTAMP,
                ended_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS healing_metrics_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                component_id TEXT NOT NULL,
                component_type TEXT,
                metrics JSONB NOT NULL,
                recorded_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_healing_incidents_component ON healing_incidents(component_id);
            CREATE INDEX IF NOT EXISTS idx_healing_incidents_time ON healing_incidents(detected_at DESC);
            CREATE INDEX IF NOT EXISTS idx_healing_metrics_component ON healing_metrics_history(component_id);
            CREATE INDEX IF NOT EXISTS idx_healing_metrics_time ON healing_metrics_history(recorded_at DESC);
                """)

                conn.commit()
                cur.close()
                logger.info("✅ Self-healing database initialized")
        except Exception as e:
            logger.warning(f"Self-healing database init failed: {e}")

    async def start_reconciliation_loop(self):
        """Start the continuous reconciliation loop"""
        self.running = True
        logger.info("🔄 Self-healing reconciliation loop started")

        while self.running:
            try:
                result = await self.reconcile()
                logger.info(
                    f"Reconciliation cycle {result.cycle_id}: "
                    f"{result.incidents_detected} incidents, "
                    f"{result.remediations_executed} remediations"
                )
            except Exception as e:
                logger.error(f"Reconciliation cycle failed: {e}")

            await asyncio.sleep(self.reconcile_interval)

    def stop(self):
        """Stop the reconciliation loop"""
        self.running = False
        logger.info("Self-healing reconciliation loop stopped")

    async def reconcile(self) -> ReconciliationResult:
        """
        Execute one reconciliation cycle.
        Pattern: Observe -> Analyze -> Act -> Verify
        """
        cycle_id = f"heal_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        start_time = datetime.now()
        incidents_detected = 0
        remediations_executed = 0
        details: Dict[str, Any] = {"components": {}}

        # OBSERVE: Collect metrics from all components
        metrics_list = await self._observe_all_components()

        for metrics in metrics_list:
            component_details: Dict[str, Any] = {
                "status": HealthStatus.HEALTHY.value,
                "metrics": asdict(metrics),
                "incidents": [],
                "remediations": [],
            }

            # ANALYZE: Check for anomalies
            anomalies = self._detect_anomalies(metrics)

            if anomalies:
                incidents_detected += len(anomalies)
                component_details["status"] = HealthStatus.DEGRADED.value

                for anomaly in anomalies:
                    incident = self._create_incident(metrics, anomaly)
                    component_details["incidents"].append(incident.incident_type)

                    # ACT: Execute remediation
                    remediation = await self._remediate(incident)
                    if remediation:
                        remediations_executed += 1
                        component_details["remediations"].append(remediation)

                    # VERIFY: Check if remediation succeeded
                    healed = await self._verify_healing(metrics.component_id)
                    if healed:
                        self._resolve_incident(incident)
                        component_details["status"] = HealthStatus.HEALTHY.value
                    else:
                        # Escalate to human if not healed
                        await self._escalate_to_human(incident)
                        component_details["status"] = HealthStatus.CRITICAL.value

            details["components"][metrics.component_id] = component_details

        end_time = datetime.now()

        result = ReconciliationResult(
            cycle_id=cycle_id,
            start_time=start_time,
            end_time=end_time,
            components_checked=len(metrics_list),
            incidents_detected=incidents_detected,
            remediations_executed=remediations_executed,
            success=incidents_detected == 0 or remediations_executed > 0,
            details=details
        )

        self._store_reconciliation(result)
        return result

    async def _observe_all_components(self) -> List[HealthMetrics]:
        """Observe health metrics from all monitored components"""
        metrics_list = []

        async with httpx.AsyncClient(timeout=10) as client:
            for component_id, url in self.monitored_components.items():
                try:
                    start = datetime.now()
                    response = await client.get(f"{url}/health")
                    latency = (datetime.now() - start).total_seconds() * 1000

                    health_data = response.json() if response.status_code == 200 else {}

                    metrics = HealthMetrics(
                        component_id=component_id,
                        component_type=health_data.get("type", "service"),
                        latency_p50_ms=latency * 0.5,  # Estimate
                        latency_p99_ms=latency,
                        error_rate=0 if response.status_code == 200 else 1.0,
                        drift_score=health_data.get("drift_score", 0),
                        memory_usage_pct=health_data.get("memory_pct", 0),
                        cpu_usage_pct=health_data.get("cpu_pct", 0),
                        request_count=health_data.get("request_count", 0),
                        success_rate=1.0 if response.status_code == 200 else 0,
                        timestamp=datetime.now()
                    )

                    metrics_list.append(metrics)
                    self._store_metrics(metrics)
                    self.circuit_breaker.record_success(component_id)

                except Exception as e:
                    logger.warning(f"Failed to observe {component_id}: {e}")
                    self.circuit_breaker.record_failure(component_id)

                    # Create unhealthy metrics
                    metrics = HealthMetrics(
                        component_id=component_id,
                        component_type="service",
                        latency_p50_ms=9999,
                        latency_p99_ms=9999,
                        error_rate=1.0,
                        drift_score=0,
                        memory_usage_pct=0,
                        cpu_usage_pct=0,
                        request_count=0,
                        success_rate=0,
                        timestamp=datetime.now()
                    )
                    metrics_list.append(metrics)

        return metrics_list

    def _detect_anomalies(self, metrics: HealthMetrics) -> List[str]:
        """Detect anomalies in metrics"""
        anomalies = []

        if metrics.latency_p99_ms > self.thresholds["latency_p99_ms"]:
            anomalies.append("high_latency")

        if metrics.error_rate > self.thresholds["error_rate"]:
            anomalies.append("high_error_rate")

        if metrics.drift_score > self.thresholds["drift_score"]:
            anomalies.append("model_drift")

        if metrics.memory_usage_pct > self.thresholds["memory_usage_pct"]:
            anomalies.append("memory_pressure")

        if metrics.cpu_usage_pct > self.thresholds["cpu_usage_pct"]:
            anomalies.append("cpu_pressure")

        if metrics.success_rate < self.thresholds["success_rate"]:
            anomalies.append("low_success_rate")

        return anomalies

    def _create_incident(self, metrics: HealthMetrics, anomaly: str) -> Incident:
        """Create an incident from detected anomaly"""
        severity = self._determine_severity(anomaly, metrics)

        incident = Incident(
            incident_id=f"inc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{metrics.component_id}",
            component_id=metrics.component_id,
            severity=severity,
            incident_type=anomaly,
            description=f"{anomaly} detected in {metrics.component_id}",
            metrics=metrics,
            detected_at=datetime.now(),
            remediation_actions=[]
        )

        self.active_incidents[incident.incident_id] = incident
        self._store_incident(incident)

        return incident

    def _determine_severity(self, anomaly: str, metrics: HealthMetrics) -> str:
        """Determine incident severity"""
        if metrics.error_rate >= 0.5 or metrics.success_rate <= 0.5:
            return "critical"
        elif anomaly in ["high_error_rate", "model_drift"]:
            return "high"
        elif anomaly in ["high_latency", "memory_pressure"]:
            return "medium"
        else:
            return "low"

    async def _remediate(self, incident: Incident) -> Optional[str]:
        """Execute remediation action for incident"""
        component_id = incident.component_id

        # Check circuit breaker
        if self.circuit_breaker.is_open(component_id):
            logger.warning(f"Circuit breaker open for {component_id}, skipping remediation")
            return None

        # Select remediation action based on incident type
        action = self._select_remediation_action(incident)

        try:
            if action == RemediationAction.ROUTE_TO_FALLBACK:
                await self._route_to_fallback(component_id)
            elif action == RemediationAction.CACHE_CLEAR:
                await self._clear_cache(component_id)
            elif action == RemediationAction.CIRCUIT_BREAK:
                self.circuit_breaker.record_failure(component_id)
                self.circuit_breaker.record_failure(component_id)  # Force open
            elif action == RemediationAction.ALERT_HUMAN:
                await self._escalate_to_human(incident)

            incident.remediation_actions = incident.remediation_actions or []
            incident.remediation_actions.append(action.value)
            logger.info(f"Executed {action.value} for {component_id}")

            return action.value

        except Exception as e:
            logger.error(f"Remediation failed for {component_id}: {e}")
            return None

    def _select_remediation_action(self, incident: Incident) -> RemediationAction:
        """Select appropriate remediation action"""
        if incident.severity == "critical":
            return RemediationAction.CIRCUIT_BREAK

        if incident.incident_type == "high_error_rate":
            return RemediationAction.ROUTE_TO_FALLBACK

        if incident.incident_type == "model_drift":
            return RemediationAction.TRIGGER_RETRAIN

        if incident.incident_type == "high_latency":
            return RemediationAction.CACHE_CLEAR

        return RemediationAction.ALERT_HUMAN

    async def _route_to_fallback(self, component_id: str):
        """Route traffic to fallback"""
        fallback = self.fallback_routes.get(component_id)
        logger.info(f"Routing {component_id} to fallback: {fallback}")
        # In production, this would update load balancer or service mesh

    async def _clear_cache(self, component_id: str):
        """Clear component cache"""
        logger.info(f"Clearing cache for {component_id}")
        # In production, this would call cache invalidation API

    async def _verify_healing(self, component_id: str) -> bool:
        """Verify component has healed"""
        try:
            url = self.monitored_components.get(component_id)
            if not url:
                return False

            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{url}/health")
                return response.status_code == 200

        except Exception as exc:
            logger.warning("Health check failed for %s: %s", component_id, exc, exc_info=True)
            return False

    async def _escalate_to_human(self, incident: Incident):
        """Escalate incident to human operator"""
        incident.human_escalated = True
        logger.warning(f"HUMAN ESCALATION: {incident.incident_type} in {incident.component_id}")

        # Store escalation
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                cur.execute("""
                UPDATE healing_incidents
                SET human_escalated = TRUE
                WHERE component_id = %s AND resolved_at IS NULL
                """, (incident.component_id,))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to record escalation: {e}")

    def _resolve_incident(self, incident: Incident):
        """Mark incident as resolved"""
        incident.resolved_at = datetime.now()
        if incident.incident_id in self.active_incidents:
            del self.active_incidents[incident.incident_id]

        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                cur.execute("""
                UPDATE healing_incidents
                SET resolved_at = NOW()
                WHERE component_id = %s AND resolved_at IS NULL
                """, (incident.component_id,))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to resolve incident: {e}")

    def _store_incident(self, incident: Incident):
        """Store incident in database"""
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                # Convert metrics to safe dict
                safe_metrics = json_safe_serialize(asdict(incident.metrics))
                cur.execute("""
                INSERT INTO healing_incidents
                (component_id, severity, incident_type, description, metrics)
                VALUES (%s, %s, %s, %s, %s)
                """, (
                    incident.component_id,
                    incident.severity,
                    incident.incident_type,
                    incident.description,
                    Json(safe_metrics)
                ))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to store incident: {e}")

    def _store_metrics(self, metrics: HealthMetrics):
        """Store metrics history"""
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                safe_metrics = json_safe_serialize(asdict(metrics))
                cur.execute("""
                INSERT INTO healing_metrics_history
                (component_id, component_type, metrics)
                VALUES (%s, %s, %s)
                """, (
                    metrics.component_id,
                    metrics.component_type,
                    Json(safe_metrics)
                ))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.debug(f"Failed to store metrics: {e}")

    def _store_reconciliation(self, result: ReconciliationResult):
        """Store reconciliation result"""
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                # Use json_safe_serialize to handle datetime and enum types
                safe_details = json_safe_serialize(result.details)
                cur.execute("""
                INSERT INTO healing_reconciliations
                (cycle_id, components_checked, incidents_detected,
                 remediations_executed, success, details, started_at, ended_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    result.cycle_id,
                    result.components_checked,
                    result.incidents_detected,
                    result.remediations_executed,
                    result.success,
                    Json(safe_details),
                    result.start_time,
                    result.end_time
                ))
                conn.commit()
                cur.close()

            # Log to unified brain
            self._log_to_unified_brain('reconciliation_cycle', {
                'cycle_id': result.cycle_id,
                'components_checked': result.components_checked,
                'incidents_detected': result.incidents_detected,
                'remediations_executed': result.remediations_executed,
                'success': result.success
            })
        except Exception as e:
            logger.error(f"Failed to store reconciliation: {e}")

    def _log_to_unified_brain(self, action_type: str, data: Dict[str, Any]):
        """Log reconciliation actions to unified_brain table"""
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()

                # Serialize data safely
                safe_data = json_safe_serialize(data)

                cur.execute("""
                    INSERT INTO unified_brain (
                        agent_name, action_type, input_data, output_data,
                        success, metadata, executed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """, (
                    'self_healing_reconciler',
                    action_type,
                    Json(safe_data),
                    Json(safe_data),
                    data.get('success', True),
                    Json({
                        'timestamp': datetime.now().isoformat(),
                        'cycle': data.get('cycle_id', 'unknown')
                    })
                ))

                conn.commit()
                cur.close()
        except Exception as e:
            logger.debug(f"Failed to log to unified_brain: {e}")


# Singleton instance
_reconciler: Optional[SelfHealingReconciler] = None


def get_reconciler() -> SelfHealingReconciler:
    """Get or create reconciler instance"""
    global _reconciler
    if _reconciler is None:
        _reconciler = SelfHealingReconciler()
    return _reconciler


async def start_healing_loop():
    """Start the self-healing reconciliation loop"""
    reconciler = get_reconciler()
    await reconciler.start_reconciliation_loop()
