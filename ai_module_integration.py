#!/usr/bin/env python3
"""
AI MODULE INTEGRATION LAYER - Perfect Cross-Module Integration
==============================================================
Connects all bleeding-edge AI modules into a unified intelligent system.

INTEGRATION CAPABILITIES:
1. OODA → Consciousness (observations feed situational awareness)
2. Consciousness → Decisions (intentions guide decision making)
3. Decisions → Memory (outcomes recorded for learning)
4. Memory → OODA (learnings improve observations)
5. Error recovery coordination across all modules
6. Learning feedback loops for continuous improvement
7. State sharing and cross-module awareness

Author: BrainOps AI System
Version: 1.0.0
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading

from ai_observability import (
    EventBus, EventType, Event, MetricsRegistry,
    ObservabilityController, CorrelationContext,
    get_observability, get_event_bus, publish_event, publish_event_async,
    traced_operation, get_current_context, set_current_context
)

logger = logging.getLogger(__name__)


# =============================================================================
# FEEDBACK TYPES
# =============================================================================

class FeedbackType(Enum):
    """Types of feedback between modules"""
    OBSERVATION_OUTCOME = "observation_outcome"
    DECISION_OUTCOME = "decision_outcome"
    PREDICTION_ACCURACY = "prediction_accuracy"
    RECOVERY_SUCCESS = "recovery_success"
    LEARNING_SIGNAL = "learning_signal"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"


@dataclass
class FeedbackSignal:
    """Feedback signal between modules"""
    feedback_type: FeedbackType
    source_module: str
    target_module: str
    signal_value: float  # -1.0 to 1.0 (negative = bad, positive = good)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LearningOutcome:
    """Outcome of a learning-enabled operation"""
    operation_id: str
    operation_type: str
    predicted_outcome: Any
    actual_outcome: Any
    success: bool
    confidence_delta: float  # How much to adjust confidence
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CROSS-MODULE STATE
# =============================================================================

@dataclass
class UnifiedSystemState:
    """Unified state across all modules"""
    # OODA state
    current_observations: List[Dict] = field(default_factory=list)
    pending_decisions: List[Dict] = field(default_factory=list)
    active_actions: List[Dict] = field(default_factory=list)

    # Consciousness state
    awareness_level: str = "reactive"
    consciousness_level: float = 0.5
    active_intentions: List[Dict] = field(default_factory=list)

    # Memory state
    working_memory_count: int = 0
    long_term_memory_count: int = 0
    recent_contradictions: List[Dict] = field(default_factory=list)

    # Dependability state
    system_health: str = "OK"
    guard_violations: List[Dict] = field(default_factory=list)
    recovery_in_progress: bool = False

    # Circuit breaker state
    open_circuits: List[str] = field(default_factory=list)
    half_open_circuits: List[str] = field(default_factory=list)

    # Learning state
    total_decisions: int = 0
    successful_decisions: int = 0
    learning_accuracy: float = 0.5

    # Timestamps
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# LEARNING MANAGER
# =============================================================================

class LearningManager:
    """
    Manages learning across all modules.
    Tracks outcomes and adjusts confidence/thresholds.
    """

    def __init__(self):
        self._outcomes: deque = deque(maxlen=10000)
        self._confidence_adjustments: Dict[str, float] = defaultdict(float)
        self._success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._pattern_memory: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()
        self._metrics = get_observability()

    def record_outcome(self, outcome: LearningOutcome):
        """Record an outcome for learning"""
        with self._lock:
            self._outcomes.append(outcome)
            self._success_rates[outcome.operation_type].append(1.0 if outcome.success else 0.0)
            self._confidence_adjustments[outcome.operation_type] += outcome.confidence_delta

            # Store pattern for future learning
            pattern = {
                "context": outcome.context,
                "success": outcome.success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self._pattern_memory[outcome.operation_type].append(pattern)
            # Keep only last 1000 patterns per operation type
            if len(self._pattern_memory[outcome.operation_type]) > 1000:
                self._pattern_memory[outcome.operation_type] = self._pattern_memory[outcome.operation_type][-1000:]

        # Publish event
        publish_event(
            EventType.OBSERVATION_COMPLETE,
            "learning_manager",
            {
                "operation_id": outcome.operation_id,
                "operation_type": outcome.operation_type,
                "success": outcome.success,
                "confidence_delta": outcome.confidence_delta
            }
        )

        # Update metrics
        self._metrics.metrics.increment_counter(
            "learning_outcomes_total",
            {"operation_type": outcome.operation_type, "success": str(outcome.success)}
        )

    def get_success_rate(self, operation_type: str) -> float:
        """Get success rate for operation type"""
        with self._lock:
            rates = self._success_rates.get(operation_type, [])
            return sum(rates) / len(rates) if rates else 0.5

    def get_confidence_adjustment(self, operation_type: str) -> float:
        """Get cumulative confidence adjustment"""
        with self._lock:
            return self._confidence_adjustments.get(operation_type, 0.0)

    def should_adjust_threshold(self, operation_type: str, current_threshold: float) -> Optional[float]:
        """Determine if threshold should be adjusted based on outcomes"""
        success_rate = self.get_success_rate(operation_type)

        # Too many failures -> lower threshold (more sensitive)
        if success_rate < 0.7 and len(self._success_rates[operation_type]) >= 20:
            return current_threshold * 0.9

        # High success -> can raise threshold (less sensitive)
        if success_rate > 0.95 and len(self._success_rates[operation_type]) >= 50:
            return current_threshold * 1.1

        return None

    def predict_success(self, operation_type: str, context: Dict) -> float:
        """Predict success probability based on similar past contexts"""
        with self._lock:
            patterns = self._pattern_memory.get(operation_type, [])
            if not patterns:
                return 0.5

            # Simple similarity matching
            similar_patterns = []
            for pattern in patterns[-100:]:  # Check recent patterns
                pattern_context = pattern.get("context", {})
                similarity = self._context_similarity(context, pattern_context)
                if similarity > 0.5:
                    similar_patterns.append((similarity, pattern["success"]))

            if not similar_patterns:
                return self.get_success_rate(operation_type)

            # Weighted average by similarity
            total_weight = sum(s for s, _ in similar_patterns)
            if total_weight == 0:
                return 0.5

            weighted_success = sum(s * (1.0 if success else 0.0) for s, success in similar_patterns)
            return weighted_success / total_weight

    def _context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Calculate similarity between two contexts"""
        if not ctx1 or not ctx2:
            return 0.0

        keys = set(ctx1.keys()) | set(ctx2.keys())
        if not keys:
            return 0.0

        matches = sum(1 for k in keys if ctx1.get(k) == ctx2.get(k))
        return matches / len(keys)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning state"""
        with self._lock:
            return {
                "total_outcomes": len(self._outcomes),
                "operation_types": len(self._success_rates),
                "success_rates": {k: sum(v)/len(v) if v else 0 for k, v in self._success_rates.items()},
                "confidence_adjustments": dict(self._confidence_adjustments),
                "pattern_counts": {k: len(v) for k, v in self._pattern_memory.items()}
            }


# =============================================================================
# RECOVERY COORDINATOR
# =============================================================================

class RecoveryCoordinator:
    """
    Coordinates recovery actions across modules.
    Prevents conflicting recovery attempts.
    """

    def __init__(self):
        self._active_recoveries: Dict[str, Dict] = {}
        self._recovery_history: deque = deque(maxlen=1000)
        self._recovery_lock = asyncio.Lock()
        self._recovery_success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._module_priority = {
            "dependability": 1,
            "circuit_breaker": 2,
            "ooda": 3,
            "consciousness": 4,
            "memory": 5,
            "hallucination": 6
        }

    async def request_recovery(self, module: str, error_type: str,
                                context: Dict[str, Any]) -> bool:
        """
        Request permission to perform recovery.
        Returns True if recovery can proceed.
        """
        async with self._recovery_lock:
            # Check if conflicting recovery in progress
            for active_module, recovery in self._active_recoveries.items():
                if self._conflicts_with(module, error_type, active_module, recovery):
                    logger.warning(f"Recovery blocked: {module} conflicts with {active_module}")
                    return False

            # Register recovery
            self._active_recoveries[module] = {
                "error_type": error_type,
                "context": context,
                "started_at": datetime.now(timezone.utc)
            }

            publish_event(
                EventType.RECOVERY_ATTEMPTED,
                module,
                {"error_type": error_type, "status": "started"}
            )

            return True

    async def complete_recovery(self, module: str, success: bool,
                                 result: Dict[str, Any] = None):
        """Mark recovery as complete"""
        async with self._recovery_lock:
            if module in self._active_recoveries:
                recovery = self._active_recoveries.pop(module)
                recovery["completed_at"] = datetime.now(timezone.utc)
                recovery["success"] = success
                recovery["result"] = result
                self._recovery_history.append(recovery)

                # Track success rate
                error_type = recovery.get("error_type", "unknown")
                self._recovery_success_rates[error_type].append(1.0 if success else 0.0)

                publish_event(
                    EventType.RECOVERY_ATTEMPTED,
                    module,
                    {"error_type": error_type, "status": "completed", "success": success}
                )

    def get_recovery_success_rate(self, error_type: str) -> float:
        """Get historical success rate for error type recovery"""
        rates = self._recovery_success_rates.get(error_type, [])
        return sum(rates) / len(rates) if rates else 0.5

    def _conflicts_with(self, module: str, error_type: str,
                        active_module: str, active_recovery: Dict) -> bool:
        """Check if recovery conflicts with active recovery"""
        # Higher priority module always wins
        if self._module_priority.get(active_module, 99) < self._module_priority.get(module, 99):
            return True

        # Same module conflicts
        if module == active_module:
            return True

        # System-wide errors block all other recoveries
        if active_recovery.get("error_type") == "system_wide":
            return True

        return False

    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get summary of recovery state"""
        return {
            "active_recoveries": len(self._active_recoveries),
            "recovery_history_count": len(self._recovery_history),
            "success_rates": {k: sum(v)/len(v) if v else 0 for k, v in self._recovery_success_rates.items()}
        }


# =============================================================================
# MODULE INTEGRATION ORCHESTRATOR
# =============================================================================

class ModuleIntegrationOrchestrator:
    """
    Central orchestrator for all module integrations.
    Manages state, learning, recovery, and cross-module communication.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModuleIntegrationOrchestrator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.event_bus = get_event_bus()
        self.observability = get_observability()
        self.learning = LearningManager()
        self.recovery = RecoveryCoordinator()

        # Unified state
        self._state = UnifiedSystemState()
        self._state_lock = threading.Lock()

        # Module references (lazy loaded)
        self._modules: Dict[str, Any] = {}

        # Feedback handlers
        self._feedback_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Cross-module signal queues
        self._signal_queues: Dict[str, asyncio.Queue] = {}

        # Set up event subscriptions
        self._setup_event_handlers()

        logger.info("ModuleIntegrationOrchestrator initialized")

    def _setup_event_handlers(self):
        """Set up handlers for cross-module events"""

        # OODA events
        self.event_bus.subscribe(EventType.OBSERVATION_COMPLETE, self._on_observation_complete)
        self.event_bus.subscribe(EventType.DECISION_MADE, self._on_decision_made)
        self.event_bus.subscribe(EventType.ACTION_EXECUTED, self._on_action_executed)

        # Consciousness events
        self.event_bus.subscribe(EventType.INTENTION_GENERATED, self._on_intention_generated)
        self.event_bus.subscribe(EventType.AWARENESS_LEVEL_CHANGED, self._on_awareness_changed)

        # Memory events
        self.event_bus.subscribe(EventType.MEMORY_STORED, self._on_memory_stored)
        self.event_bus.subscribe(EventType.CONTRADICTION_DETECTED, self._on_contradiction_detected)

        # Error events
        self.event_bus.subscribe(EventType.GUARD_VIOLATION, self._on_guard_violation)
        self.event_bus.subscribe(EventType.CIRCUIT_STATE_CHANGED, self._on_circuit_state_changed)

        # Hallucination events
        self.event_bus.subscribe(EventType.HALLUCINATION_DETECTED, self._on_hallucination_detected)

    # =========================================================================
    # EVENT HANDLERS - Cross-Module Integration
    # =========================================================================

    def _on_observation_complete(self, event: Event):
        """Handle OODA observation -> feed to Consciousness and Memory"""
        with self._state_lock:
            self._state.current_observations.append(event.payload)
            self._state.current_observations = self._state.current_observations[-100:]

        # Feed to consciousness for situational awareness
        self._propagate_to_module("consciousness", "situational_update", {
            "observation": event.payload,
            "source": "ooda",
            "timestamp": event.timestamp.isoformat()
        })

        # Store significant observations in memory
        if event.payload.get("significance", 0) > 0.7:
            self._propagate_to_module("memory", "store_observation", {
                "observation": event.payload,
                "type": "episodic"
            })

    def _on_decision_made(self, event: Event):
        """Handle decision -> record to Memory for learning"""
        decision = event.payload

        with self._state_lock:
            self._state.total_decisions += 1
            self._state.pending_decisions.append({
                "decision_id": decision.get("id"),
                "timestamp": event.timestamp.isoformat(),
                "awaiting_outcome": True
            })

        # Store decision context in memory
        self._propagate_to_module("memory", "store_decision", {
            "decision": decision,
            "context": decision.get("context", {}),
            "confidence": decision.get("confidence", 0.5)
        })

        # Inform consciousness of decision
        self._propagate_to_module("consciousness", "decision_awareness", {
            "decision": decision,
            "intention_alignment": self._check_intention_alignment(decision)
        })

    def _on_action_executed(self, event: Event):
        """Handle action outcome -> update learning and consciousness"""
        action = event.payload
        success = action.get("success", False)
        decision_id = action.get("decision_id")

        # Record outcome for learning
        if decision_id:
            outcome = LearningOutcome(
                operation_id=decision_id,
                operation_type=action.get("action_type", "unknown"),
                predicted_outcome=action.get("expected_outcome"),
                actual_outcome=action.get("actual_outcome"),
                success=success,
                confidence_delta=0.05 if success else -0.1,
                context=action.get("context", {})
            )
            self.learning.record_outcome(outcome)

            # Update success rate
            with self._state_lock:
                if success:
                    self._state.successful_decisions += 1
                if self._state.total_decisions > 0:
                    self._state.learning_accuracy = self._state.successful_decisions / self._state.total_decisions

        # Remove from pending
        with self._state_lock:
            self._state.pending_decisions = [
                d for d in self._state.pending_decisions
                if d.get("decision_id") != decision_id
            ]

        # Feed outcome to memory for pattern learning
        self._propagate_to_module("memory", "store_outcome", {
            "action": action,
            "success": success,
            "learning_signal": outcome.confidence_delta if decision_id else 0
        })

    def _on_intention_generated(self, event: Event):
        """Handle consciousness intention -> inform OODA"""
        intention = event.payload

        with self._state_lock:
            self._state.active_intentions.append(intention)
            self._state.active_intentions = self._state.active_intentions[-20:]

        # Feed intention to OODA for decision guidance
        self._propagate_to_module("ooda", "intention_context", {
            "intention": intention,
            "priority": intention.get("priority", 0.5),
            "constraints": intention.get("constraints", [])
        })

    def _on_awareness_changed(self, event: Event):
        """Handle awareness level change -> adjust system behavior"""
        new_level = event.payload.get("level", "reactive")
        numeric = event.payload.get("numeric", 0.5)

        with self._state_lock:
            self._state.awareness_level = new_level
            self._state.consciousness_level = numeric

        # Adjust other modules based on awareness
        if numeric > 0.8:
            # High awareness - more proactive
            self._broadcast_to_all("awareness_update", {
                "level": new_level,
                "numeric": numeric,
                "mode": "proactive"
            })
        elif numeric < 0.3:
            # Low awareness - more conservative
            self._broadcast_to_all("awareness_update", {
                "level": new_level,
                "numeric": numeric,
                "mode": "conservative"
            })

    def _on_memory_stored(self, event: Event):
        """Handle memory storage -> update stats and predictions"""
        memory = event.payload

        with self._state_lock:
            if memory.get("type") == "working":
                self._state.working_memory_count += 1
            else:
                self._state.long_term_memory_count += 1

        # If wisdom crystallized, notify consciousness
        if memory.get("type") == "crystallized":
            self._propagate_to_module("consciousness", "wisdom_available", {
                "wisdom": memory,
                "source_count": memory.get("source_count", 1)
            })

    def _on_contradiction_detected(self, event: Event):
        """Handle memory contradiction -> trigger resolution"""
        contradiction = event.payload

        with self._state_lock:
            self._state.recent_contradictions.append(contradiction)
            self._state.recent_contradictions = self._state.recent_contradictions[-10:]

        # Alert consciousness
        self._propagate_to_module("consciousness", "contradiction_alert", {
            "contradiction": contradiction,
            "severity": contradiction.get("severity", "medium")
        })

        # If severe, trigger recovery
        if contradiction.get("severity") == "high":
            asyncio.create_task(self._coordinate_recovery("memory", contradiction))

    def _on_guard_violation(self, event: Event):
        """Handle guard violation -> coordinate recovery"""
        violation = event.payload

        with self._state_lock:
            self._state.guard_violations.append(violation)
            self._state.guard_violations = self._state.guard_violations[-20:]

        severity = violation.get("severity", "low")
        if severity in ["critical", "high"]:
            asyncio.create_task(self._coordinate_recovery("dependability", violation))

    def _on_circuit_state_changed(self, event: Event):
        """Handle circuit breaker state change"""
        component = event.payload.get("component")
        new_state = event.payload.get("new_state")

        with self._state_lock:
            if new_state == "open":
                if component not in self._state.open_circuits:
                    self._state.open_circuits.append(component)
                if component in self._state.half_open_circuits:
                    self._state.half_open_circuits.remove(component)
            elif new_state == "half_open":
                if component not in self._state.half_open_circuits:
                    self._state.half_open_circuits.append(component)
                if component in self._state.open_circuits:
                    self._state.open_circuits.remove(component)
            elif new_state == "closed":
                if component in self._state.open_circuits:
                    self._state.open_circuits.remove(component)
                if component in self._state.half_open_circuits:
                    self._state.half_open_circuits.remove(component)

        # Notify OODA of circuit state
        self._propagate_to_module("ooda", "circuit_state_update", {
            "component": component,
            "state": new_state,
            "open_circuits": list(self._state.open_circuits)
        })

    def _on_hallucination_detected(self, event: Event):
        """Handle hallucination detection -> alert consciousness and learning"""
        hallucination = event.payload

        # Record as negative learning outcome
        outcome = LearningOutcome(
            operation_id=hallucination.get("validation_id", "unknown"),
            operation_type="hallucination_prevention",
            predicted_outcome="valid",
            actual_outcome="hallucination",
            success=False,
            confidence_delta=-0.15,
            context=hallucination
        )
        self.learning.record_outcome(outcome)

        # Alert consciousness
        self._propagate_to_module("consciousness", "hallucination_alert", {
            "type": hallucination.get("type"),
            "confidence": hallucination.get("confidence"),
            "claims": hallucination.get("flagged_claims", [])
        })

    # =========================================================================
    # PROPAGATION METHODS
    # =========================================================================

    def _propagate_to_module(self, target: str, method: str, data: Dict):
        """Propagate data to specific module"""
        logger.debug(f"Propagating {method} to {target}")

        # Track metrics
        self.observability.metrics.increment_counter(
            "integration_propagations_total",
            {"target_module": target, "method": method}
        )

        # Add to signal queue if exists
        if target in self._signal_queues:
            try:
                self._signal_queues[target].put_nowait((method, data))
            except asyncio.QueueFull:
                logger.warning(f"Signal queue full for {target}")

    def _broadcast_to_all(self, method: str, data: Dict):
        """Broadcast to all modules"""
        for module in ["ooda", "consciousness", "memory", "dependability", "circuit_breaker", "hallucination"]:
            self._propagate_to_module(module, method, data)

    def _check_intention_alignment(self, decision: Dict) -> float:
        """Check how well decision aligns with current intentions"""
        with self._state_lock:
            if not self._state.active_intentions:
                return 0.5

            decision_type = decision.get("type", "")
            alignments = []

            for intention in self._state.active_intentions:
                intention_type = intention.get("type", "")
                if decision_type == intention_type:
                    alignments.append(1.0)
                elif decision_type in intention.get("constraints", []):
                    alignments.append(0.0)
                else:
                    alignments.append(0.5)

            return sum(alignments) / len(alignments) if alignments else 0.5

    # =========================================================================
    # RECOVERY COORDINATION
    # =========================================================================

    async def _coordinate_recovery(self, source: str, violation: Dict):
        """Coordinate recovery across modules"""
        error_type = violation.get("type", "unknown")

        if not await self.recovery.request_recovery(source, error_type, violation):
            logger.warning(f"Recovery request denied for {source}")
            return

        try:
            with self._state_lock:
                self._state.recovery_in_progress = True
                self._state.system_health = "RECOVERING"

            # Notify all modules
            self._broadcast_to_all("recovery_started", {
                "source": source,
                "error_type": error_type
            })

            # Execute recovery
            success = await self._execute_recovery(source, error_type, violation)

            await self.recovery.complete_recovery(source, success, {"violation": violation})

        finally:
            with self._state_lock:
                self._state.recovery_in_progress = False
                self._state.system_health = "OK" if success else "DEGRADED"

    async def _execute_recovery(self, source: str, error_type: str, violation: Dict) -> bool:
        """Execute recovery action"""
        recovery_strategies = {
            "dependability": self._recover_dependability,
            "circuit_breaker": self._recover_circuit,
            "memory": self._recover_memory,
            "hallucination": self._recover_hallucination
        }

        strategy = recovery_strategies.get(source)
        if strategy:
            return await strategy(violation)
        return False

    async def _recover_dependability(self, violation: Dict) -> bool:
        """Recover from dependability violation"""
        guard_type = violation.get("guard_type")

        if guard_type == "resource":
            self._broadcast_to_all("reduce_load", {"reason": "resource_violation"})
            return True
        if guard_type == "temporal":
            self._broadcast_to_all("extend_timeouts", {"reason": "temporal_violation"})
            return True
        return False

    async def _recover_circuit(self, violation: Dict) -> bool:
        """Recover from circuit breaker trigger"""
        component = violation.get("component")
        self._broadcast_to_all("use_fallback", {"component": component, "reason": "circuit_open"})
        return True

    async def _recover_memory(self, violation: Dict) -> bool:
        """Recover from memory contradiction"""
        self._propagate_to_module("memory", "consolidate_urgent", {"reason": "contradiction_detected"})
        return True

    async def _recover_hallucination(self, violation: Dict) -> bool:
        """Recover from hallucination detection"""
        self._propagate_to_module("hallucination", "increase_validation", {"reason": "hallucination_detected"})
        return True

    # =========================================================================
    # STATE ACCESS
    # =========================================================================

    def get_unified_state(self) -> Dict[str, Any]:
        """Get unified state across all modules"""
        with self._state_lock:
            return {
                "observations": {
                    "count": len(self._state.current_observations),
                    "recent": self._state.current_observations[-5:]
                },
                "decisions": {
                    "total": self._state.total_decisions,
                    "successful": self._state.successful_decisions,
                    "pending": len(self._state.pending_decisions),
                    "accuracy": round(self._state.learning_accuracy, 3)
                },
                "consciousness": {
                    "awareness_level": self._state.awareness_level,
                    "consciousness_level": round(self._state.consciousness_level, 3),
                    "active_intentions": len(self._state.active_intentions)
                },
                "memory": {
                    "working_count": self._state.working_memory_count,
                    "long_term_count": self._state.long_term_memory_count,
                    "contradictions": len(self._state.recent_contradictions)
                },
                "health": {
                    "status": self._state.system_health,
                    "recovery_in_progress": self._state.recovery_in_progress,
                    "open_circuits": self._state.open_circuits,
                    "guard_violations": len(self._state.guard_violations)
                },
                "learning": self.learning.get_learning_summary(),
                "recovery": self.recovery.get_recovery_summary(),
                "last_updated": self._state.last_updated.isoformat()
            }

    def get_adjusted_confidence(self, module: str, operation: str, base_confidence: float) -> float:
        """Get confidence adjusted by learning outcomes"""
        operation_key = f"{module}_{operation}"
        adjustment = self.learning.get_confidence_adjustment(operation_key)
        success_rate = self.learning.get_success_rate(operation_key)

        # Adjust based on historical success
        adjusted = base_confidence + (adjustment * 0.01)
        adjusted = adjusted * (0.5 + success_rate * 0.5)

        return max(0.1, min(0.99, adjusted))

    def predict_operation_success(self, module: str, operation: str, context: Dict) -> float:
        """Predict success of operation based on learning"""
        operation_key = f"{module}_{operation}"
        return self.learning.predict_success(operation_key, context)


# =============================================================================
# MODULE CONNECTORS
# =============================================================================

class OODAIntegrationConnector:
    """Connector for OODA module integration"""

    def __init__(self):
        self.orchestrator = ModuleIntegrationOrchestrator.get_instance()
        self.observability = get_observability()

    def report_observation(self, observation: Dict[str, Any]):
        """Report completed observation"""
        publish_event(EventType.OBSERVATION_COMPLETE, "ooda", observation)
        self.observability.ooda.record_observation(
            observation.get("duration_ms", 0),
            observation.get("source", "unknown")
        )

    def report_decision(self, decision: Dict[str, Any]):
        """Report decision made"""
        # Adjust confidence based on learning
        adjusted_confidence = self.orchestrator.get_adjusted_confidence(
            "ooda", decision.get("type", "unknown"),
            decision.get("confidence", 0.5)
        )
        decision["adjusted_confidence"] = adjusted_confidence

        publish_event(EventType.DECISION_MADE, "ooda", decision)
        self.observability.ooda.record_decision(adjusted_confidence, decision.get("type", "unknown"))

    def report_action(self, action: Dict[str, Any]):
        """Report action executed"""
        publish_event(EventType.ACTION_EXECUTED, "ooda", action)
        self.observability.ooda.record_action(
            action.get("duration_ms", 0),
            action.get("success", False),
            action.get("type", "unknown")
        )

    def get_intention_context(self) -> List[Dict]:
        """Get current intentions for decision guidance"""
        state = self.orchestrator.get_unified_state()
        return state.get("consciousness", {}).get("active_intentions", [])

    def get_open_circuits(self) -> List[str]:
        """Get list of open circuits to avoid"""
        state = self.orchestrator.get_unified_state()
        return state.get("health", {}).get("open_circuits", [])


class ConsciousnessIntegrationConnector:
    """Connector for Consciousness module integration"""

    def __init__(self):
        self.orchestrator = ModuleIntegrationOrchestrator.get_instance()
        self.observability = get_observability()

    def report_intention(self, intention: Dict[str, Any]):
        """Report generated intention"""
        publish_event(EventType.INTENTION_GENERATED, "consciousness", intention)
        self.observability.consciousness.record_intention(intention.get("type", "unknown"))

    def report_awareness_change(self, level: str, numeric: float):
        """Report awareness level change"""
        publish_event(
            EventType.AWARENESS_LEVEL_CHANGED, "consciousness",
            {"level": level, "numeric": numeric}
        )
        self.observability.consciousness.set_awareness_level(level, numeric)

    def get_recent_observations(self) -> List[Dict]:
        """Get recent observations for situational awareness"""
        state = self.orchestrator.get_unified_state()
        return state.get("observations", {}).get("recent", [])

    def get_learning_accuracy(self) -> float:
        """Get system learning accuracy for self-model"""
        state = self.orchestrator.get_unified_state()
        return state.get("decisions", {}).get("accuracy", 0.5)


class MemoryIntegrationConnector:
    """Connector for Memory module integration"""

    def __init__(self):
        self.orchestrator = ModuleIntegrationOrchestrator.get_instance()
        self.observability = get_observability()

    def report_store(self, memory: Dict[str, Any]):
        """Report memory stored"""
        publish_event(EventType.MEMORY_STORED, "memory", memory)
        self.observability.memory.record_store(
            memory.get("duration_ms", 0),
            memory.get("type", "unknown")
        )

    def report_contradiction(self, contradiction: Dict[str, Any]):
        """Report contradiction detected"""
        publish_event(EventType.CONTRADICTION_DETECTED, "memory", contradiction)
        self.observability.memory.record_contradiction(contradiction.get("resolution", "pending"))

    def get_decision_patterns(self) -> Dict[str, float]:
        """Get decision success patterns for prediction"""
        state = self.orchestrator.get_unified_state()
        return state.get("learning", {}).get("success_rates", {})


class HallucinationIntegrationConnector:
    """Connector for Hallucination Prevention module integration"""

    def __init__(self):
        self.orchestrator = ModuleIntegrationOrchestrator.get_instance()
        self.observability = get_observability()

    def report_validation(self, result: Dict[str, Any]):
        """Report validation result"""
        publish_event(EventType.VALIDATION_COMPLETE, "hallucination", result)
        self.observability.hallucination.record_validation(
            result.get("duration_ms", 0),
            result.get("result", "unknown"),
            result.get("method", "unknown")
        )

    def report_detection(self, hallucination: Dict[str, Any]):
        """Report hallucination detected"""
        publish_event(EventType.HALLUCINATION_DETECTED, "hallucination", hallucination)
        self.observability.hallucination.record_detection(
            hallucination.get("type", "unknown"),
            hallucination.get("confidence", 0.5)
        )

    def get_validation_success_rate(self) -> float:
        """Get historical validation success rate"""
        state = self.orchestrator.get_unified_state()
        return state.get("learning", {}).get("success_rates", {}).get("hallucination_validation", 0.5)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_module_integration() -> ModuleIntegrationOrchestrator:
    """Get the global module integration orchestrator"""
    return ModuleIntegrationOrchestrator.get_instance()


def get_ooda_integration() -> OODAIntegrationConnector:
    """Get OODA integration connector"""
    return OODAIntegrationConnector()


def get_consciousness_integration() -> ConsciousnessIntegrationConnector:
    """Get Consciousness integration connector"""
    return ConsciousnessIntegrationConnector()


def get_memory_integration() -> MemoryIntegrationConnector:
    """Get Memory integration connector"""
    return MemoryIntegrationConnector()


def get_hallucination_integration() -> HallucinationIntegrationConnector:
    """Get Hallucination integration connector"""
    return HallucinationIntegrationConnector()


# Export all
__all__ = [
    "ModuleIntegrationOrchestrator",
    "LearningManager",
    "RecoveryCoordinator",
    "UnifiedSystemState",
    "FeedbackSignal",
    "FeedbackType",
    "LearningOutcome",
    "OODAIntegrationConnector",
    "ConsciousnessIntegrationConnector",
    "MemoryIntegrationConnector",
    "HallucinationIntegrationConnector",
    "get_module_integration",
    "get_ooda_integration",
    "get_consciousness_integration",
    "get_memory_integration",
    "get_hallucination_integration"
]
