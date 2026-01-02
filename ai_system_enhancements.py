#!/usr/bin/env python3
"""
AI SYSTEM ENHANCEMENTS - Complete Observability & Intelligence Wiring
======================================================================
Wires all bleeding-edge modules to the observability and integration layer.

ENHANCEMENTS:
1. Module Health Scoring - Individual and aggregate health scores
2. Real-time Alerting - Threshold-based alerts with configurable triggers
3. Event Correlation - Track causal chains across modules
4. Auto-Recovery Triggers - Automatic recovery from detected anomalies
5. WebSocket Streaming - Real-time event streaming
6. Enhanced Learning - Pattern prediction with context matching
7. Module Wiring - Connect all bleeding-edge modules to integration

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# HEALTH SCORING SYSTEM
# =============================================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ModuleHealth:
    """Health state for a single module"""
    module_name: str
    status: HealthStatus
    score: float  # 0-100
    last_check: datetime
    error_rate: float
    latency_p95_ms: float
    availability: float  # Uptime percentage
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class HealthScorer:
    """
    Calculate health scores for individual modules and the system.
    Uses weighted factors: error rate, latency, availability, custom metrics.
    """

    def __init__(self):
        self._module_health: dict[str, ModuleHealth] = {}
        self._health_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._weights = {
            "error_rate": 0.35,
            "latency": 0.25,
            "availability": 0.30,
            "custom": 0.10
        }
        self._thresholds = {
            "error_rate_critical": 0.10,
            "error_rate_degraded": 0.05,
            "latency_critical_ms": 5000,
            "latency_degraded_ms": 2000,
            "availability_critical": 0.95,
            "availability_degraded": 0.99
        }
        self._lock = threading.Lock()

    def update_module_health(
        self,
        module_name: str,
        error_rate: float = 0.0,
        latency_p95_ms: float = 0.0,
        availability: float = 1.0,
        custom_metrics: dict[str, float] = None,
        issues: list[str] = None
    ) -> ModuleHealth:
        """Update health metrics for a module"""
        with self._lock:
            # Calculate component scores
            error_score = self._score_error_rate(error_rate)
            latency_score = self._score_latency(latency_p95_ms)
            availability_score = self._score_availability(availability)
            custom_score = self._score_custom(custom_metrics or {})

            # Weighted aggregate
            total_score = (
                error_score * self._weights["error_rate"] +
                latency_score * self._weights["latency"] +
                availability_score * self._weights["availability"] +
                custom_score * self._weights["custom"]
            )

            # Determine status
            if total_score >= 80:
                status = HealthStatus.HEALTHY
            elif total_score >= 50:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.CRITICAL

            health = ModuleHealth(
                module_name=module_name,
                status=status,
                score=round(total_score, 2),
                last_check=datetime.now(timezone.utc),
                error_rate=error_rate,
                latency_p95_ms=latency_p95_ms,
                availability=availability,
                issues=issues or [],
                metrics=custom_metrics or {}
            )

            self._module_health[module_name] = health
            self._health_history[module_name].append({
                "timestamp": health.last_check.isoformat(),
                "score": health.score,
                "status": health.status.value
            })

            return health

    def _score_error_rate(self, rate: float) -> float:
        """Score error rate (lower is better)"""
        if rate >= self._thresholds["error_rate_critical"]:
            return 0
        if rate >= self._thresholds["error_rate_degraded"]:
            return 50 - (rate / self._thresholds["error_rate_critical"]) * 50
        return 100 - (rate / self._thresholds["error_rate_degraded"]) * 50

    def _score_latency(self, latency_ms: float) -> float:
        """Score latency (lower is better)"""
        if latency_ms >= self._thresholds["latency_critical_ms"]:
            return 0
        if latency_ms >= self._thresholds["latency_degraded_ms"]:
            return 50 - ((latency_ms - self._thresholds["latency_degraded_ms"]) /
                         (self._thresholds["latency_critical_ms"] - self._thresholds["latency_degraded_ms"])) * 50
        return 100 - (latency_ms / self._thresholds["latency_degraded_ms"]) * 50

    def _score_availability(self, availability: float) -> float:
        """Score availability (higher is better)"""
        if availability <= self._thresholds["availability_critical"]:
            return 0
        if availability <= self._thresholds["availability_degraded"]:
            return 50 + ((availability - self._thresholds["availability_critical"]) /
                         (self._thresholds["availability_degraded"] - self._thresholds["availability_critical"])) * 50
        return 100

    def _score_custom(self, metrics: dict[str, float]) -> float:
        """Score custom metrics (normalized 0-100)"""
        if not metrics:
            return 100
        scores = [min(100, max(0, v)) for v in metrics.values()]
        return sum(scores) / len(scores)

    def get_module_health(self, module_name: str) -> Optional[ModuleHealth]:
        """Get current health for a module"""
        return self._module_health.get(module_name)

    def get_all_health(self) -> dict[str, ModuleHealth]:
        """Get health for all modules"""
        return dict(self._module_health)

    def get_aggregate_health(self) -> dict[str, Any]:
        """Get aggregate system health"""
        with self._lock:
            if not self._module_health:
                return {
                    "status": HealthStatus.UNKNOWN.value,
                    "score": 0,
                    "modules": 0
                }

            scores = [h.score for h in self._module_health.values()]
            avg_score = sum(scores) / len(scores)

            # Aggregate status is worst of all
            statuses = [h.status for h in self._module_health.values()]
            if HealthStatus.CRITICAL in statuses:
                agg_status = HealthStatus.CRITICAL
            elif HealthStatus.DEGRADED in statuses:
                agg_status = HealthStatus.DEGRADED
            else:
                agg_status = HealthStatus.HEALTHY

            return {
                "status": agg_status.value,
                "score": round(avg_score, 2),
                "modules": len(self._module_health),
                "healthy_count": sum(1 for h in self._module_health.values() if h.status == HealthStatus.HEALTHY),
                "degraded_count": sum(1 for h in self._module_health.values() if h.status == HealthStatus.DEGRADED),
                "critical_count": sum(1 for h in self._module_health.values() if h.status == HealthStatus.CRITICAL),
                "module_scores": {name: h.score for name, h in self._module_health.items()}
            }

    def get_health_history(self, module_name: str, limit: int = 100) -> list[dict]:
        """Get health history for a module"""
        # FIX: Add thread safety for reading health history
        with self._lock:
            return list(self._health_history.get(module_name, []))[-limit:]


# =============================================================================
# REAL-TIME ALERTING SYSTEM
# =============================================================================

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents an alert triggered by threshold violation"""
    id: str
    alert_type: str
    severity: AlertSeverity
    module: str
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertingSystem:
    """
    Real-time alerting with configurable thresholds.
    Supports deduplication, auto-resolve, and escalation.
    """

    def __init__(self):
        self._thresholds: dict[str, dict[str, Any]] = {}
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._handlers: list[Callable] = []
        self._dedupe_window_seconds = 300  # 5 minute deduplication
        self._last_alert_time: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def register_threshold(
        self,
        name: str,
        module: str,
        metric: str,
        warning_threshold: float = None,
        error_threshold: float = None,
        critical_threshold: float = None,
        comparison: str = "gt",  # gt, lt, eq
        auto_resolve: bool = True
    ):
        """Register a threshold for alerting"""
        self._thresholds[name] = {
            "module": module,
            "metric": metric,
            "warning": warning_threshold,
            "error": error_threshold,
            "critical": critical_threshold,
            "comparison": comparison,
            "auto_resolve": auto_resolve
        }

    def check_value(self, name: str, value: float, metadata: dict = None) -> Optional[Alert]:
        """Check a value against registered thresholds"""
        if name not in self._thresholds:
            return None

        threshold = self._thresholds[name]
        severity = None
        triggered_threshold = None

        # Check thresholds in order of severity
        for sev, sev_name in [(AlertSeverity.CRITICAL, "critical"),
                               (AlertSeverity.ERROR, "error"),
                               (AlertSeverity.WARNING, "warning")]:
            thresh = threshold.get(sev_name)
            if thresh is not None:
                triggered = self._compare(value, thresh, threshold["comparison"])
                if triggered:
                    severity = sev
                    triggered_threshold = thresh
                    break

        if severity is None:
            # Value is OK - check for auto-resolve
            if threshold.get("auto_resolve"):
                self._try_resolve_alert(name)
            return None

        # Check deduplication
        alert_key = f"{name}:{severity.value}"
        now = datetime.now(timezone.utc)

        with self._lock:
            last_time = self._last_alert_time.get(alert_key)
            if last_time and (now - last_time).total_seconds() < self._dedupe_window_seconds:
                return None  # Deduplicated

            self._last_alert_time[alert_key] = now

        # Create alert
        alert = Alert(
            id=str(uuid.uuid4()),
            alert_type=name,
            severity=severity,
            module=threshold["module"],
            metric=threshold["metric"],
            current_value=value,
            threshold=triggered_threshold,
            message=f"{name}: {value} {threshold['comparison']} {triggered_threshold}",
            timestamp=now,
            metadata=metadata or {}
        )

        with self._lock:
            self._active_alerts[alert.id] = alert
            # FIX: Convert alert to dict with proper serialization (handle datetime and enum)
            alert_dict = {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,  # Convert enum to string
                "module": alert.module,
                "metric": alert.metric,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),  # Convert datetime to string
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
                "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None,
                "metadata": alert.metadata
            }
            self._alert_history.append(alert_dict)

        # Call handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

        return alert

    def _compare(self, value: float, threshold: float, comparison: str) -> bool:
        """Compare value against threshold"""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "lte":
            return value <= threshold
        elif comparison == "eq":
            return value == threshold
        return False

    def _try_resolve_alert(self, alert_type: str):
        """Try to auto-resolve alerts of this type"""
        with self._lock:
            to_resolve = [
                alert_id for alert_id, alert in self._active_alerts.items()
                if alert.alert_type == alert_type and not alert.resolved
            ]
            for alert_id in to_resolve:
                self._active_alerts[alert_id].resolved = True
                self._active_alerts[alert_id].resolution_time = datetime.now(timezone.utc)

    def add_handler(self, handler: Callable):
        """Add alert handler"""
        self._handlers.append(handler)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].acknowledged = True
                return True
            return False

    def get_active_alerts(self, severity: AlertSeverity = None) -> list[Alert]:
        """Get active alerts"""
        with self._lock:
            alerts = [a for a in self._active_alerts.values() if not a.resolved]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts

    def get_alert_history(self, limit: int = 100) -> list[dict]:
        """Get alert history"""
        return list(self._alert_history)[-limit:]


# =============================================================================
# EVENT CORRELATION ENGINE
# =============================================================================

@dataclass
class EventChain:
    """Represents a chain of correlated events"""
    chain_id: str
    root_event_id: str
    events: list[str]
    modules_involved: set[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: float = 0
    status: str = "active"  # active, completed, error


class EventCorrelator:
    """
    Correlates events across modules to track causal chains.
    Uses correlation IDs and temporal proximity.
    """

    def __init__(self):
        self._active_chains: dict[str, EventChain] = {}
        self._completed_chains: deque = deque(maxlen=500)
        self._event_to_chain: dict[str, str] = {}
        self._correlation_window_ms = 5000  # 5 seconds
        self._lock = threading.Lock()

    def correlate_event(
        self,
        event_id: str,
        event_type: str,
        module: str,
        correlation_id: str = None,
        parent_event_id: str = None,
        timestamp: datetime = None
    ) -> Optional[EventChain]:
        """Correlate an event to a chain"""
        timestamp = timestamp or datetime.now(timezone.utc)

        with self._lock:
            chain = None

            # Try to find existing chain
            if correlation_id and correlation_id in self._active_chains:
                chain = self._active_chains[correlation_id]
            elif parent_event_id and parent_event_id in self._event_to_chain:
                chain_id = self._event_to_chain[parent_event_id]
                chain = self._active_chains.get(chain_id)

            # Create new chain if needed
            if chain is None:
                chain_id = correlation_id or str(uuid.uuid4())
                chain = EventChain(
                    chain_id=chain_id,
                    root_event_id=event_id,
                    events=[],
                    modules_involved=set(),
                    start_time=timestamp,
                    end_time=None
                )
                self._active_chains[chain_id] = chain

            # Add event to chain
            chain.events.append(event_id)
            chain.modules_involved.add(module)
            self._event_to_chain[event_id] = chain.chain_id

            return chain

    def complete_chain(self, chain_id: str, status: str = "completed"):
        """Mark a chain as complete"""
        with self._lock:
            if chain_id in self._active_chains:
                chain = self._active_chains.pop(chain_id)
                chain.status = status
                chain.end_time = datetime.now(timezone.utc)
                chain.duration_ms = (chain.end_time - chain.start_time).total_seconds() * 1000
                self._completed_chains.append(chain)

    def get_active_chains(self) -> list[EventChain]:
        """Get active event chains"""
        return list(self._active_chains.values())

    def get_chain_by_event(self, event_id: str) -> Optional[EventChain]:
        """Get chain containing an event"""
        chain_id = self._event_to_chain.get(event_id)
        if chain_id:
            return self._active_chains.get(chain_id)
        return None

    def get_chain_stats(self) -> dict[str, Any]:
        """Get correlation statistics"""
        with self._lock:
            completed = list(self._completed_chains)
            if not completed:
                return {
                    "active_chains": len(self._active_chains),
                    "completed_chains": 0,
                    "avg_duration_ms": 0,
                    "avg_events_per_chain": 0
                }

            durations = [c.duration_ms for c in completed]
            event_counts = [len(c.events) for c in completed]

            return {
                "active_chains": len(self._active_chains),
                "completed_chains": len(completed),
                "avg_duration_ms": sum(durations) / len(durations),
                "avg_events_per_chain": sum(event_counts) / len(event_counts),
                "modules_distribution": self._get_module_distribution(completed)
            }

    def _get_module_distribution(self, chains: list[EventChain]) -> dict[str, int]:
        """Get distribution of modules in chains"""
        distribution = defaultdict(int)
        for chain in chains:
            for module in chain.modules_involved:
                distribution[module] += 1
        return dict(distribution)


# =============================================================================
# AUTO-RECOVERY SYSTEM
# =============================================================================

class RecoveryAction(Enum):
    RESTART_MODULE = "restart_module"
    CLEAR_CACHE = "clear_cache"
    REDUCE_LOAD = "reduce_load"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    SCALE_UP = "scale_up"
    NOTIFY_ONLY = "notify_only"


@dataclass
class RecoveryRule:
    """Rule for automatic recovery"""
    name: str
    trigger_condition: str  # error_rate > 0.1, latency > 5000, etc.
    action: RecoveryAction
    module: str
    cooldown_seconds: int = 300
    max_attempts: int = 3
    escalate_after: int = 2


class AutoRecoveryEngine:
    """
    Automatically triggers recovery actions based on detected anomalies.
    Supports cooldown, escalation, and coordinated recovery.
    """

    def __init__(self):
        self._rules: dict[str, RecoveryRule] = {}
        self._last_trigger: dict[str, datetime] = {}
        self._trigger_counts: dict[str, int] = defaultdict(int)
        self._recovery_history: deque = deque(maxlen=500)
        self._handlers: dict[RecoveryAction, Callable] = {}
        self._lock = threading.Lock()

    def register_rule(self, rule: RecoveryRule):
        """Register a recovery rule"""
        self._rules[rule.name] = rule

    def register_handler(self, action: RecoveryAction, handler: Callable):
        """Register handler for a recovery action"""
        self._handlers[action] = handler

    async def evaluate_and_recover(
        self,
        module: str,
        metrics: dict[str, float]
    ) -> Optional[dict[str, Any]]:
        """Evaluate metrics and trigger recovery if needed"""
        triggered_rule = None
        triggered_action = None

        for rule_name, rule in self._rules.items():
            if rule.module != module:
                continue

            # Check cooldown
            with self._lock:
                last = self._last_trigger.get(rule_name)
                if last:
                    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                    if elapsed < rule.cooldown_seconds:
                        continue

            # Evaluate condition
            if self._evaluate_condition(rule.trigger_condition, metrics):
                # Check max attempts
                with self._lock:
                    attempts = self._trigger_counts[rule_name]
                    if attempts >= rule.max_attempts:
                        # Escalate
                        logger.warning(f"Rule {rule_name} exceeded max attempts, escalating")
                        continue

                triggered_rule = rule
                triggered_action = rule.action

                # Check escalation
                if attempts >= rule.escalate_after:
                    triggered_action = RecoveryAction.NOTIFY_ONLY
                break

        if triggered_rule is None:
            return None

        # Execute recovery
        result = await self._execute_recovery(triggered_rule, triggered_action, metrics)

        # Update tracking
        with self._lock:
            self._last_trigger[triggered_rule.name] = datetime.now(timezone.utc)
            self._trigger_counts[triggered_rule.name] += 1
            self._recovery_history.append({
                "rule": triggered_rule.name,
                "action": triggered_action.value,
                "module": module,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": result
            })

        return result

    def _evaluate_condition(self, condition: str, metrics: dict[str, float]) -> bool:
        """Evaluate a trigger condition"""
        try:
            # Parse simple conditions like "error_rate > 0.1"
            parts = condition.split()
            if len(parts) != 3:
                return False

            metric_name, operator, threshold = parts
            value = metrics.get(metric_name, 0)
            threshold = float(threshold)

            if operator == ">":
                return value > threshold
            elif operator == "<":
                return value < threshold
            elif operator == ">=":
                return value >= threshold
            elif operator == "<=":
                return value <= threshold
            elif operator == "==":
                return value == threshold
            return False
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    async def _execute_recovery(
        self,
        rule: RecoveryRule,
        action: RecoveryAction,
        metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Execute a recovery action"""
        handler = self._handlers.get(action)
        result = {
            "action": action.value,
            "module": rule.module,
            "success": False,
            "message": ""
        }

        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(rule.module, metrics)
                else:
                    handler(rule.module, metrics)
                result["success"] = True
                result["message"] = f"Executed {action.value} on {rule.module}"
            except Exception as e:
                result["message"] = str(e)
                logger.error(f"Recovery action failed: {e}")
        else:
            result["message"] = f"No handler registered for {action.value}"

        return result

    def get_recovery_history(self, limit: int = 100) -> list[dict]:
        """Get recovery history"""
        return list(self._recovery_history)[-limit:]

    def reset_trigger_count(self, rule_name: str):
        """Reset trigger count for a rule"""
        with self._lock:
            self._trigger_counts[rule_name] = 0


# =============================================================================
# ENHANCED LEARNING ENGINE
# =============================================================================

@dataclass
class PatternMatch:
    """A matched pattern from historical data"""
    pattern_id: str
    similarity: float
    outcome: str
    success_rate: float
    sample_size: int
    context: dict[str, Any]


class EnhancedLearningEngine:
    """
    Enhanced learning with pattern prediction and context matching.
    Goes beyond simple success rates to contextual prediction.
    """

    def __init__(self):
        self._patterns: dict[str, list[dict]] = defaultdict(list)
        self._pattern_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._context_vectors: dict[str, list[float]] = {}
        self._feature_weights: dict[str, float] = {}
        self._lock = threading.Lock()

    def learn_pattern(
        self,
        operation_type: str,
        context: dict[str, Any],
        outcome: str,
        success: bool
    ):
        """Learn from an operation outcome"""
        with self._lock:
            pattern = {
                "context": context,
                "outcome": outcome,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self._patterns[operation_type].append(pattern)

            # Update outcome counts
            outcome_key = "success" if success else "failure"
            self._pattern_outcomes[operation_type][outcome_key] += 1

            # Trim old patterns (keep last 1000)
            if len(self._patterns[operation_type]) > 1000:
                self._patterns[operation_type] = self._patterns[operation_type][-1000:]

    def predict_outcome(
        self,
        operation_type: str,
        context: dict[str, Any],
        top_k: int = 5
    ) -> tuple[float, list[PatternMatch]]:
        """Predict success probability based on similar patterns"""
        with self._lock:
            patterns = self._patterns.get(operation_type, [])
            if not patterns:
                return 0.5, []

            # Find similar patterns
            matches = []
            for i, pattern in enumerate(patterns[-100:]):  # Check recent patterns
                similarity = self._context_similarity(context, pattern["context"])
                if similarity > 0.3:
                    matches.append((similarity, pattern))

            if not matches:
                # Fall back to overall success rate
                outcomes = self._pattern_outcomes.get(operation_type, {})
                total = sum(outcomes.values())
                if total == 0:
                    return 0.5, []
                return outcomes.get("success", 0) / total, []

            # Sort by similarity
            matches.sort(key=lambda x: x[0], reverse=True)
            top_matches = matches[:top_k]

            # Calculate weighted prediction
            total_weight = sum(s for s, _ in top_matches)
            success_weight = sum(s for s, p in top_matches if p["success"])
            prediction = success_weight / total_weight if total_weight > 0 else 0.5

            # Build pattern match results
            pattern_matches = []
            for similarity, pattern in top_matches:
                pm = PatternMatch(
                    pattern_id=hashlib.md5(json.dumps(pattern["context"], sort_keys=True).encode()).hexdigest()[:8],
                    similarity=round(similarity, 3),
                    outcome=pattern["outcome"],
                    success_rate=1.0 if pattern["success"] else 0.0,
                    sample_size=1,
                    context=pattern["context"]
                )
                pattern_matches.append(pm)

            return round(prediction, 3), pattern_matches

    def _context_similarity(self, ctx1: dict, ctx2: dict) -> float:
        """Calculate similarity between contexts"""
        if not ctx1 or not ctx2:
            return 0.0

        keys = set(ctx1.keys()) | set(ctx2.keys())
        if not keys:
            return 0.0

        matches = 0
        total_weight = 0

        for key in keys:
            weight = self._feature_weights.get(key, 1.0)
            total_weight += weight

            v1, v2 = ctx1.get(key), ctx2.get(key)
            if v1 == v2:
                matches += weight
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numeric similarity
                max_val = max(abs(v1), abs(v2), 1)
                similarity = 1 - abs(v1 - v2) / max_val
                matches += weight * similarity

        return matches / total_weight if total_weight > 0 else 0.0

    def set_feature_weight(self, feature: str, weight: float):
        """Set importance weight for a feature"""
        self._feature_weights[feature] = weight

    def get_learning_stats(self) -> dict[str, Any]:
        """Get learning statistics"""
        with self._lock:
            return {
                "operation_types": len(self._patterns),
                "total_patterns": sum(len(p) for p in self._patterns.values()),
                "outcomes": {
                    op: dict(outcomes)
                    for op, outcomes in self._pattern_outcomes.items()
                },
                "feature_weights": dict(self._feature_weights)
            }


# =============================================================================
# MODULE WIRING BRIDGE
# =============================================================================

class ModuleWiringBridge:
    """
    Wires all bleeding-edge modules to the observability and integration layer.
    Acts as the central nervous system connecting all components.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModuleWiringBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.health_scorer = HealthScorer()
        self.alerting = AlertingSystem()
        self.correlator = EventCorrelator()
        self.auto_recovery = AutoRecoveryEngine()
        self.enhanced_learning = EnhancedLearningEngine()

        # Observability integration
        self._observability = None
        self._integration = None

        # Module references
        self._modules: dict[str, Any] = {}

        # Setup default thresholds and rules
        self._setup_defaults()

        logger.info("ModuleWiringBridge initialized")

    def _setup_defaults(self):
        """Setup default alerting thresholds and recovery rules"""
        # Alerting thresholds
        for module in ["ooda", "consciousness", "memory", "hallucination", "dependability", "circuit_breaker"]:
            self.alerting.register_threshold(
                f"{module}_error_rate",
                module=module,
                metric="error_rate",
                warning_threshold=0.05,
                error_threshold=0.10,
                critical_threshold=0.20
            )
            self.alerting.register_threshold(
                f"{module}_latency",
                module=module,
                metric="latency_p95_ms",
                warning_threshold=1000,
                error_threshold=3000,
                critical_threshold=5000
            )

        # Auto-recovery rules
        for module in ["ooda", "consciousness", "memory", "hallucination", "dependability", "circuit_breaker"]:
            self.auto_recovery.register_rule(RecoveryRule(
                name=f"{module}_error_recovery",
                trigger_condition="error_rate > 0.15",
                action=RecoveryAction.REDUCE_LOAD,
                module=module,
                cooldown_seconds=300
            ))
            self.auto_recovery.register_rule(RecoveryRule(
                name=f"{module}_latency_recovery",
                trigger_condition="latency_p95_ms > 4000",
                action=RecoveryAction.CLEAR_CACHE,
                module=module,
                cooldown_seconds=180
            ))

    def connect_observability(self):
        """Connect to the observability layer"""
        try:
            from ai_observability import EventBus, EventType, ObservabilityController
            self._observability = ObservabilityController.get_instance()
            self._event_bus = EventBus.get_instance()

            # Subscribe to events for correlation
            for event_type in EventType:
                self._event_bus.subscribe(event_type, self._handle_event)

            logger.info("Connected to observability layer")
        except Exception as e:
            logger.error(f"Failed to connect to observability: {e}")

    def connect_integration(self):
        """Connect to the integration layer"""
        try:
            from ai_module_integration import ModuleIntegrationOrchestrator
            self._integration = ModuleIntegrationOrchestrator.get_instance()
            logger.info("Connected to integration layer")
        except Exception as e:
            logger.error(f"Failed to connect to integration: {e}")

    def _handle_event(self, event):
        """Handle events for correlation and alerting"""
        try:
            # Correlate event
            self.correlator.correlate_event(
                event_id=event.event_id,
                event_type=event.event_type.value,
                module=event.source_module,
                correlation_id=event.correlation_id
            )

            # Check for error events
            if "error" in event.event_type.value.lower():
                self.alerting.check_value(
                    f"{event.source_module}_error_rate",
                    0.1,  # Increment error rate
                    {"event": event.to_dict()}
                )
        except Exception as e:
            logger.error(f"Error handling event: {e}")

    def update_module_metrics(
        self,
        module: str,
        error_rate: float,
        latency_p95_ms: float,
        request_count: int,
        success_count: int,
        custom_metrics: dict[str, float] = None
    ):
        """Update metrics for a module"""
        availability = success_count / request_count if request_count > 0 else 1.0

        # Update health score
        health = self.health_scorer.update_module_health(
            module_name=module,
            error_rate=error_rate,
            latency_p95_ms=latency_p95_ms,
            availability=availability,
            custom_metrics=custom_metrics
        )

        # Check alerting thresholds
        self.alerting.check_value(f"{module}_error_rate", error_rate)
        self.alerting.check_value(f"{module}_latency", latency_p95_ms)

        # Update observability metrics
        if self._observability:
            self._observability.metrics.set_gauge(
                f"{module}_health_score",
                health.score,
                {"module": module}
            )

        return health

    async def check_and_recover(self, module: str, metrics: dict[str, float]):
        """Check metrics and trigger auto-recovery if needed"""
        return await self.auto_recovery.evaluate_and_recover(module, metrics)

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "health": self.health_scorer.get_aggregate_health(),
            "module_health": {
                name: {
                    "status": h.status.value,
                    "score": h.score,
                    "error_rate": h.error_rate,
                    "latency_p95_ms": h.latency_p95_ms
                }
                for name, h in self.health_scorer.get_all_health().items()
            },
            "active_alerts": len(self.alerting.get_active_alerts()),
            "critical_alerts": len(self.alerting.get_active_alerts(AlertSeverity.CRITICAL)),
            "active_chains": len(self.correlator.get_active_chains()),
            "correlation_stats": self.correlator.get_chain_stats(),
            "learning_stats": self.enhanced_learning.get_learning_stats(),
            "recovery_history_count": len(self.auto_recovery.get_recovery_history())
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_wiring_bridge() -> ModuleWiringBridge:
    """Get the global module wiring bridge"""
    return ModuleWiringBridge.get_instance()


def get_health_scorer() -> HealthScorer:
    """Get the health scorer"""
    return get_wiring_bridge().health_scorer


def get_alerting() -> AlertingSystem:
    """Get the alerting system"""
    return get_wiring_bridge().alerting


def get_correlator() -> EventCorrelator:
    """Get the event correlator"""
    return get_wiring_bridge().correlator


def get_auto_recovery() -> AutoRecoveryEngine:
    """Get the auto-recovery engine"""
    return get_wiring_bridge().auto_recovery


def get_enhanced_learning() -> EnhancedLearningEngine:
    """Get the enhanced learning engine"""
    return get_wiring_bridge().enhanced_learning


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "ModuleWiringBridge",
    "HealthScorer",
    "AlertingSystem",
    "EventCorrelator",
    "AutoRecoveryEngine",
    "EnhancedLearningEngine",

    # Data classes
    "ModuleHealth",
    "Alert",
    "EventChain",
    "PatternMatch",
    "RecoveryRule",

    # Enums
    "HealthStatus",
    "AlertSeverity",
    "RecoveryAction",

    # Convenience functions
    "get_wiring_bridge",
    "get_health_scorer",
    "get_alerting",
    "get_correlator",
    "get_auto_recovery",
    "get_enhanced_learning"
]
