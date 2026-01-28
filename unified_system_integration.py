"""
Unified System Integration Layer
================================
This module ACTIVELY integrates all BrainOps AI systems together.
Instead of having separate modules that sit idle, this wires everything
into the execution flow so ALL capabilities are USED.

ENHANCEMENTS (v2.0):
- Event-driven communication between all modules
- Message queuing for async operations
- Circuit breakers for resilience
- Load balancing across components
- Priority-based routing
- System-wide health aggregation

Author: Claude Opus 4.5 + BrainOps AI Team
Version: 2.0.0
Purpose: MAKE ALL SYSTEMS POWERFUL AND USED!
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
IS_PRODUCTION = ENVIRONMENT in ("production", "prod")

DEFAULT_FEATURE_TIMEOUT_SECONDS = float(os.getenv("UNIFIED_SYSTEM_FEATURE_TIMEOUT_SECONDS", "5"))
DEFAULT_GRAPH_CONTEXT_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_GRAPH_CONTEXT_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)
DEFAULT_DECISION_TREE_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_DECISION_TREE_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)
DEFAULT_PRICING_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_PRICING_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)
DEFAULT_LEARNING_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_LEARNING_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)
DEFAULT_REVENUE_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_REVENUE_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)
DEFAULT_CONSCIOUSNESS_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_CONSCIOUSNESS_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)
DEFAULT_LIVE_BRAIN_TIMEOUT_SECONDS = float(
    os.getenv("UNIFIED_SYSTEM_LIVE_BRAIN_TIMEOUT_SECONDS", str(DEFAULT_FEATURE_TIMEOUT_SECONDS))
)


async def _await_with_timeout(
    awaitable: Any,
    *,
    timeout_seconds: float,
    label: str,
) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("%s timed out after %ss", label, timeout_seconds)
        return None


def _should_use_graph_context(task_data: dict[str, Any]) -> bool:
    """
    Graph context can be expensive; default OFF in production unless explicitly requested.
    """
    default_use_graph_context = not IS_PRODUCTION
    return bool(task_data.get("use_graph_context", default_use_graph_context))


# Import enhanced orchestration features
try:
    from autonomous_system_orchestrator import (
        AgentInstance,
        CircuitBreaker,
        EventBus,
        EventType,
        HealthAggregator,
        LoadBalancer,
        LoadBalancingStrategy,
        MessageQueue,
        SystemEvent,
    )
    ENHANCED_ORCHESTRATION_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced orchestration features not available")
    ENHANCED_ORCHESTRATION_AVAILABLE = False

# ============== LAZY IMPORTS FOR ALL SYSTEMS ==============
# These are imported lazily to avoid circular dependencies

def get_state_sync():
    """Get real-time state sync singleton"""
    try:
        from realtime_state_sync import get_state_sync as _get_sync
        return _get_sync()
    except ImportError:
        logger.warning("State sync not available")
        return None

def get_change_propagator():
    """Get change propagation daemon"""
    try:
        from change_propagation_daemon import ChangePropagator
        return ChangePropagator()
    except ImportError:
        logger.warning("Change propagator not available")
        return None

def get_graph_context():
    """Get graph context provider"""
    try:
        from graph_context_provider import get_graph_context_provider
        return get_graph_context_provider()
    except ImportError:
        logger.warning("Graph context not available")
        return None

def get_revenue_system():
    """Get revenue generation system"""
    try:
        from revenue_generation_system import get_revenue_system as _get_revenue
        return _get_revenue()
    except ImportError:
        logger.warning("Revenue system not available - import error")
        return None
    except Exception as e:
        logger.warning(f"Revenue system not available: {e}")
        return None

def get_customer_acquisition():
    """Get customer acquisition agents"""
    try:
        from customer_acquisition_agents import AcquisitionOrchestrator
        return AcquisitionOrchestrator()
    except ImportError:
        logger.warning("Customer acquisition not available")
        return None

def get_pricing_engine():
    """Get AI pricing engine"""
    try:
        from ai_pricing_engine import AIPricingEngine
        return AIPricingEngine()
    except ImportError:
        logger.warning("Pricing engine not available")
        return None

def get_notebook_learning():
    """Get Notebook LM+ learning system"""
    try:
        from notebook_lm_plus import NotebookLMPlus
        return NotebookLMPlus()
    except ImportError:
        logger.warning("Notebook learning not available")
        return None

def get_decision_tree():
    """Get AI decision tree"""
    try:
        from ai_decision_tree import AIDecisionTree
        return AIDecisionTree()
    except ImportError:
        logger.warning("Decision tree not available")
        return None

def get_a2ui_generator():
    """Get A2UI (Agent-to-User Interface) generator"""
    try:
        from a2ui_protocol import A2UIGenerator
        return A2UIGenerator()
    except ImportError:
        logger.warning("A2UI generator not available")
        return None

async def get_consciousness_controller():
    """Get consciousness emergence controller"""
    try:
        from consciousness_emergence import get_consciousness_controller as _get_consciousness
        return await _get_consciousness()
    except ImportError:
        logger.warning("Consciousness controller not available")
        return None

def get_hallucination_controller_sync():
    """Get hallucination prevention controller (sync wrapper)"""
    try:
        from hallucination_prevention import get_hallucination_controller
        return get_hallucination_controller()
    except ImportError:
        logger.warning("Hallucination prevention not available")
        return None

async def get_live_brain():
    """Get live memory brain"""
    try:
        from live_memory_brain import get_live_brain as _get_brain
        return await _get_brain()
    except ImportError:
        logger.warning("Live memory brain not available")
        return None


# ============== UNIFIED SYSTEM HOOKS ==============

@dataclass
class ExecutionContext:
    """Context passed through all systems during execution"""
    agent_name: str
    task_type: str
    task_data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_id: str = ""
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None

    # System enrichment
    graph_context: Optional[dict[str, Any]] = None
    historical_patterns: Optional[list[dict]] = None
    pricing_recommendations: Optional[dict] = None
    confidence_score: float = 1.0

    # A2UI and Consciousness integration
    a2ui_surface: Optional[dict[str, Any]] = None  # Generated UI for results
    consciousness_state: Optional[dict[str, Any]] = None  # Current consciousness
    hallucination_validated: bool = False  # Whether output was validated
    brain_memory_id: Optional[str] = None  # ID in brain memory

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task_type": self.task_type,
            "task_data": self.task_data,
            "timestamp": self.timestamp.isoformat(),
            "execution_id": self.execution_id,
            "tenant_id": self.tenant_id,
            "graph_context": self.graph_context,
            "historical_patterns": self.historical_patterns,
            "pricing_recommendations": self.pricing_recommendations,
            "confidence_score": self.confidence_score,
            "a2ui_surface": self.a2ui_surface,
            "consciousness_state": self.consciousness_state,
            "hallucination_validated": self.hallucination_validated,
            "brain_memory_id": self.brain_memory_id
        }


class UnifiedSystemIntegration:
    """
    Central integration point that ensures ALL systems are ACTIVELY USED.

    This class provides hooks that should be called at key points in agent execution:
    1. pre_execution() - Called BEFORE any agent runs
    2. enrich_context() - Adds context from all available systems
    3. post_execution() - Called AFTER any agent completes
    4. on_error() - Called when an agent fails

    By using these hooks, we ensure:
    - State is tracked in real-time
    - Graph context enriches agent decisions
    - Learning systems capture outcomes
    - Revenue/pricing systems inform business decisions
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logging.getLogger("UnifiedSystem")
        self.execution_count = 0
        self.systems_used = set()

        # NEW: Enhanced orchestration features
        if ENHANCED_ORCHESTRATION_AVAILABLE:
            self.event_bus = EventBus()
            self.message_queue = MessageQueue(max_workers=10)
            self.load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)
            self.health_aggregator = HealthAggregator()
            self.circuit_breakers: dict[str, CircuitBreaker] = {}
            self._enhanced_initialized = False
        else:
            self.event_bus = None
            self.message_queue = None
            self.load_balancer = None
            self.health_aggregator = None
            self.circuit_breakers = {}

        # Priority mapping for different task types
        self.task_priorities = {
            "critical": 1,
            "proposal": 2,
            "quote": 2,
            "pricing": 2,
            "estimate": 3,
            "invoice": 3,
            "payment": 2,
            "sale": 2,
            "subscription": 2,
            "analysis": 5,
            "reporting": 7,
            "maintenance": 10
        }

    async def initialize_enhanced_features(self):
        """Initialize enhanced orchestration features"""
        if not ENHANCED_ORCHESTRATION_AVAILABLE or self._enhanced_initialized:
            return

        # Start event bus
        await self.event_bus.start()
        self.logger.info("Event bus started in UnifiedSystemIntegration")

        # Start message queue
        await self.message_queue.start()
        self.logger.info("Message queue started in UnifiedSystemIntegration")

        # Initialize circuit breakers for all major systems
        system_names = [
            "state_sync", "graph_context", "revenue_system",
            "customer_acquisition", "pricing_engine",
            "notebook_learning", "decision_tree"
        ]

        for system_name in system_names:
            self.circuit_breakers[system_name] = CircuitBreaker(
                name=f"{system_name}_circuit",
                failure_threshold=5,
                success_threshold=2,
                timeout=60
            )

        # Register agent instances for load balancing
        agent_types = [
            "state_sync", "graph_context", "revenue",
            "acquisition", "pricing", "learning", "decision"
        ]

        for agent_type in agent_types:
            for i in range(2):  # 2 instances per type
                instance = AgentInstance(
                    instance_id=f"{agent_type}_instance_{i}",
                    agent_name=agent_type,
                    max_capacity=5,
                    weight=1
                )
                self.load_balancer.register_instance(instance)

        self._enhanced_initialized = True
        self.logger.info("Enhanced features initialized in UnifiedSystemIntegration")

    async def pre_execution(self, agent_name: str, task_type: str, task_data: dict[str, Any]) -> ExecutionContext:
        """
        Called BEFORE any agent execution.
        Enriches context with data from all available systems.
        """
        import uuid

        ctx = ExecutionContext(
            agent_name=agent_name,
            task_type=task_type,
            task_data=task_data,
            execution_id=str(uuid.uuid4())[:8]
        )

        self.logger.info(f"[{ctx.execution_id}] PRE-EXEC: {agent_name} / {task_type}")
        self.execution_count += 1

        # Determine priority
        priority = self.task_priorities.get(task_type, 5)

        # Publish agent started event
        if self.event_bus:
            await _await_with_timeout(
                self.event_bus.publish(
                    SystemEvent(
                        event_type=EventType.AGENT_STARTED,
                        source=agent_name,
                        data={
                            "execution_id": ctx.execution_id,
                            "task_type": task_type,
                            "priority": priority,
                        },
                        priority=priority,
                    )
                ),
                timeout_seconds=DEFAULT_FEATURE_TIMEOUT_SECONDS,
                label="UnifiedSystemIntegration event_bus.publish(AGENT_STARTED)",
            )

        # 1. Update state sync - track that this agent is executing
        state_sync = get_state_sync()
        circuit = self.circuit_breakers.get("state_sync")

        if state_sync and (not circuit or circuit.can_execute()):
            try:
                state_sync.register_agent(agent_name, {"status": "executing", "task": task_type})
                self.systems_used.add("state_sync")
                if circuit:
                    circuit.record_success()
            except Exception as e:
                self.logger.warning(f"State sync update failed: {e}")
                if circuit:
                    circuit.record_failure()

        # 2. Get graph context - understand codebase relationships
        if _should_use_graph_context(task_data):
            graph_ctx = get_graph_context()
            if graph_ctx:
                try:
                    ctx.graph_context = await _await_with_timeout(
                        graph_ctx.get_context_for_agent(agent_name, task_data),
                        timeout_seconds=DEFAULT_GRAPH_CONTEXT_TIMEOUT_SECONDS,
                        label="UnifiedSystemIntegration graph_context.get_context_for_agent",
                    )
                    if ctx.graph_context:
                        self.systems_used.add("graph_context")
                except Exception as e:
                    self.logger.warning(f"Graph context failed: {e}")

        # 3. Get pricing recommendations if relevant
        if task_type in ['proposal', 'quote', 'pricing', 'estimate', 'invoice']:
            pricing = get_pricing_engine()
            if pricing:
                try:
                    ctx.pricing_recommendations = await _await_with_timeout(
                        pricing.get_recommendations_async(task_data),
                        timeout_seconds=DEFAULT_PRICING_TIMEOUT_SECONDS,
                        label="UnifiedSystemIntegration pricing.get_recommendations_async",
                    )
                    if ctx.pricing_recommendations:
                        self.systems_used.add("pricing_engine")
                except Exception as e:
                    self.logger.warning(f"Pricing engine failed: {e}")

        # 4. Get decision tree guidance
        decision_tree = get_decision_tree()
        if decision_tree:
            try:
                guidance = await _await_with_timeout(
                    decision_tree.get_execution_guidance(agent_name, task_type, task_data),
                    timeout_seconds=DEFAULT_DECISION_TREE_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration decision_tree.get_execution_guidance",
                )
                if guidance:
                    ctx.confidence_score = guidance.get('confidence', 1.0)
                    self.systems_used.add("decision_tree")
            except Exception as e:
                self.logger.warning(f"Decision tree failed: {e}")

        return ctx

    async def enrich_context(self, ctx: ExecutionContext, additional_data: dict[str, Any]) -> ExecutionContext:
        """
        Called during execution to add more context.
        """
        # Get historical patterns from learning system
        learning = get_notebook_learning()
        if learning:
            try:
                patterns = await _await_with_timeout(
                    learning.get_patterns_for_task(ctx.agent_name, ctx.task_type),
                    timeout_seconds=DEFAULT_LEARNING_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration learning.get_patterns_for_task",
                )
                ctx.historical_patterns = patterns
                if patterns:
                    self.systems_used.add("notebook_learning")
            except Exception as e:
                self.logger.warning(f"Learning system failed: {e}")

        return ctx

    async def post_execution(self, ctx: ExecutionContext, result: dict[str, Any], success: bool) -> None:
        """
        Called AFTER agent execution completes.
        Updates all systems with the outcome.
        """
        self.logger.info(f"[{ctx.execution_id}] POST-EXEC: {'SUCCESS' if success else 'FAILED'}")

        # 1. Update state sync
        state_sync = get_state_sync()
        if state_sync:
            try:
                state_sync.register_agent(ctx.agent_name, {
                    "status": "completed" if success else "failed",
                    "last_execution": ctx.timestamp.isoformat(),
                    "last_task": ctx.task_type
                })
            except Exception as e:
                self.logger.warning(f"State sync post-update failed: {e}")

        # 2. Record learning
        learning = get_notebook_learning()
        if learning:
            try:
                await _await_with_timeout(
                    learning.record_execution(
                        agent=ctx.agent_name,
                        task_type=ctx.task_type,
                        input_data=ctx.task_data,
                        output_data=result,
                        success=success,
                    ),
                    timeout_seconds=DEFAULT_LEARNING_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration learning.record_execution",
                )
            except Exception as e:
                self.logger.warning(f"Learning record failed: {e}")

        # 3. Update decision tree with outcome
        decision_tree = get_decision_tree()
        if decision_tree:
            try:
                await _await_with_timeout(
                    decision_tree.record_outcome(
                        agent_name=ctx.agent_name,
                        task_type=ctx.task_type,
                        success=success,
                        confidence_used=ctx.confidence_score,
                    ),
                    timeout_seconds=DEFAULT_DECISION_TREE_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration decision_tree.record_outcome",
                )
            except Exception as e:
                self.logger.warning(f"Decision tree update failed: {e}")

        # 4. Trigger revenue tracking if applicable
        if success and ctx.task_type in ['invoice', 'payment', 'sale', 'subscription']:
            revenue = get_revenue_system()
            if revenue:
                try:
                    await _await_with_timeout(
                        revenue.track_revenue_event(ctx.task_type, result),
                        timeout_seconds=DEFAULT_REVENUE_TIMEOUT_SECONDS,
                        label="UnifiedSystemIntegration revenue.track_revenue_event",
                    )
                except Exception as e:
                    self.logger.warning(f"Revenue tracking failed: {e}")

        # 5. Generate A2UI surface for result display
        a2ui = get_a2ui_generator()
        if a2ui:
            try:
                # Generate appropriate UI based on result type
                if isinstance(result, dict):
                    if "metrics" in result or "stats" in result:
                        ctx.a2ui_surface = a2ui.dashboard_card(
                            title=f"{ctx.agent_name} Results",
                            metrics=[{"label": k, "value": str(v)[:50]}
                                    for k, v in result.items() if k != "status"][:6]
                        )
                    elif "data" in result or "items" in result:
                        data = result.get("data") or result.get("items", [])
                        if isinstance(data, list) and len(data) > 0:
                            cols = list(data[0].keys()) if isinstance(data[0], dict) else ["Value"]
                            ctx.a2ui_surface = a2ui.data_table(
                                title=f"{ctx.agent_name} Data",
                                columns=cols[:5],
                                data=[list(row.values())[:5] if isinstance(row, dict) else [row]
                                      for row in data[:10]]
                            )
                self.systems_used.add("a2ui")
            except Exception as e:
                self.logger.warning(f"A2UI generation failed: {e}")

        # 6. Record consciousness thought about execution
        try:
            consciousness = await _await_with_timeout(
                get_consciousness_controller(),
                timeout_seconds=DEFAULT_CONSCIOUSNESS_TIMEOUT_SECONDS,
                label="UnifiedSystemIntegration get_consciousness_controller",
            )
            if consciousness:
                await _await_with_timeout(
                    consciousness.record_thought(
                        {
                            "type": "execution_reflection",
                            "agent": ctx.agent_name,
                            "task": ctx.task_type,
                            "success": success,
                            "confidence": ctx.confidence_score,
                            "insight": f"Completed {ctx.task_type} with {success and 'success' or 'failure'}",
                            "timestamp": ctx.timestamp.isoformat(),
                        }
                    ),
                    timeout_seconds=DEFAULT_CONSCIOUSNESS_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration consciousness.record_thought",
                )
                ctx.consciousness_state = {"thought_recorded": True}
                self.systems_used.add("consciousness")
        except Exception as e:
            self.logger.warning(f"Consciousness recording failed: {e}")

        # 7. Store in live brain memory
        try:
            brain = await _await_with_timeout(
                get_live_brain(),
                timeout_seconds=DEFAULT_LIVE_BRAIN_TIMEOUT_SECONDS,
                label="UnifiedSystemIntegration get_live_brain",
            )
            if brain:
                from live_memory_brain import MemoryType
                memory_id = await _await_with_timeout(
                    brain.store(
                        content={
                            "agent": ctx.agent_name,
                            "task": ctx.task_type,
                            "result_summary": str(result)[:500],
                            "success": success,
                        },
                        memory_type=MemoryType.EPISODIC,
                        importance=0.7 if success else 0.9,
                        context={"execution_id": ctx.execution_id},
                    ),
                    timeout_seconds=DEFAULT_LIVE_BRAIN_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration live_brain.store",
                )
                ctx.brain_memory_id = memory_id
                self.systems_used.add("live_brain")
        except Exception as e:
            self.logger.warning(f"Brain memory storage failed: {e}")

    async def on_error(self, ctx: ExecutionContext, error: Exception) -> dict[str, Any]:
        """
        Called when an agent fails.
        Captures error for learning and potentially triggers recovery.
        """
        if "is not implemented" in str(error):
            self.logger.debug(f"[{ctx.execution_id}] Agent not yet implemented: {error}")
        else:
            self.logger.error(f"[{ctx.execution_id}] ERROR: {error}")

        recovery_actions = []

        # Record error for learning
        learning = get_notebook_learning()
        if learning:
            try:
                await _await_with_timeout(
                    learning.record_error(
                        agent=ctx.agent_name,
                        task_type=ctx.task_type,
                        error=str(error),
                        context=ctx.to_dict(),
                    ),
                    timeout_seconds=DEFAULT_LEARNING_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration learning.record_error",
                )
            except Exception as e:
                self.logger.warning(f"Error recording failed: {e}")

        # Get recovery suggestions from decision tree
        decision_tree = get_decision_tree()
        if decision_tree:
            try:
                recovery = await _await_with_timeout(
                    decision_tree.get_recovery_actions(ctx.agent_name, str(error)),
                    timeout_seconds=DEFAULT_DECISION_TREE_TIMEOUT_SECONDS,
                    label="UnifiedSystemIntegration decision_tree.get_recovery_actions",
                )
                if recovery:
                    recovery_actions = recovery
            except Exception as e:
                self.logger.warning(f"Recovery lookup failed: {e}")

        return {
            "error": str(error),
            "execution_id": ctx.execution_id,
            "recovery_actions": recovery_actions,
            "systems_consulted": list(self.systems_used)
        }

    def get_integration_stats(self) -> dict[str, Any]:
        """Get statistics about system integration usage"""
        return {
            "total_executions": self.execution_count,
            "systems_used": list(self.systems_used),
            "systems_available": {
                "state_sync": get_state_sync() is not None,
                "graph_context": get_graph_context() is not None,
                "revenue_system": get_revenue_system() is not None,
                "customer_acquisition": get_customer_acquisition() is not None,
                "pricing_engine": get_pricing_engine() is not None,
                "notebook_learning": get_notebook_learning() is not None,
                "decision_tree": get_decision_tree() is not None,
            }
        }


# ============== SINGLETON ACCESS ==============

_integration_instance: Optional[UnifiedSystemIntegration] = None

def get_unified_integration() -> UnifiedSystemIntegration:
    """Get the singleton unified integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = UnifiedSystemIntegration()
    return _integration_instance


# ============== CONVENIENCE DECORATORS ==============

def with_unified_integration(agent_name: str, task_type: str):
    """
    Decorator that automatically wraps agent execution with unified integration.

    Usage:
        @with_unified_integration("ProposalAgent", "generate")
        async def generate_proposal(task_data):
            ...
    """
    def decorator(func):
        async def wrapper(task_data: dict[str, Any], *args, **kwargs):
            integration = get_unified_integration()
            ctx = await integration.pre_execution(agent_name, task_type, task_data)

            try:
                result = await func(task_data, *args, **kwargs)
                await integration.post_execution(ctx, result, success=True)
                return result
            except Exception as e:
                error_info = await integration.on_error(ctx, e)
                await integration.post_execution(ctx, error_info, success=False)
                raise

        return wrapper
    return decorator


# ============== SCHEDULED JOBS ==============

async def run_hourly_system_sync():
    """Hourly job to sync all systems"""
    logger.info("Running hourly system sync...")

    # Sync state
    state_sync = get_state_sync()
    if state_sync:
        await state_sync.full_system_scan()

    # Propagate changes
    propagator = get_change_propagator()
    if propagator:
        await propagator.run_propagation_cycle()

    logger.info("Hourly sync complete")

async def run_daily_analytics():
    """Daily job to run analytics across all systems"""
    logger.info("Running daily analytics...")

    # Revenue analysis
    revenue = get_revenue_system()
    if revenue:
        try:
            await revenue.generate_daily_report()
        except Exception as e:
            logger.warning(f"Daily revenue report failed: {e}")

    # Customer acquisition analysis
    acquisition = get_customer_acquisition()
    if acquisition:
        try:
            await acquisition.analyze_pipeline()
        except Exception as e:
            logger.warning(f"Acquisition analysis failed: {e}")

    # Learning consolidation
    learning = get_notebook_learning()
    if learning:
        try:
            await learning.consolidate_daily_learnings()
        except Exception as e:
            logger.warning(f"Learning consolidation failed: {e}")

    logger.info("Daily analytics complete")


# ============== STARTUP INITIALIZATION ==============

async def initialize_all_systems():
    """Initialize all systems at startup"""
    logger.info("=" * 60)
    logger.info("INITIALIZING UNIFIED SYSTEM INTEGRATION v2.0")
    logger.info("=" * 60)

    integration = get_unified_integration()

    # Initialize enhanced features if available
    if ENHANCED_ORCHESTRATION_AVAILABLE:
        await integration.initialize_enhanced_features()
        logger.info("Enhanced orchestration features initialized")

    stats = integration.get_integration_stats()

    available = sum(1 for v in stats["systems_available"].values() if v)
    total = len(stats["systems_available"])

    logger.info(f"Systems available: {available}/{total}")
    for system, is_available in stats["systems_available"].items():
        status = "ACTIVE" if is_available else "UNAVAILABLE"
        logger.info(f"  - {system}: {status}")

    if ENHANCED_ORCHESTRATION_AVAILABLE:
        logger.info("\nEnhanced Features:")
        logger.info("  - Event Bus: ACTIVE")
        logger.info("  - Message Queue: ACTIVE (10 workers)")
        logger.info("  - Load Balancer: ACTIVE (LEAST_LOADED strategy)")
        logger.info(f"  - Circuit Breakers: {len(integration.circuit_breakers)} configured")
        logger.info("  - Health Aggregator: ACTIVE")

    # Run initial state scan
    state_sync = get_state_sync()
    if state_sync:
        logger.info("\nRunning initial state scan...")
        await state_sync.full_system_scan()

    logger.info("=" * 60)
    logger.info("ALL SYSTEMS INTEGRATED AND READY!")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    # Test the integration
    async def test():
        stats = await initialize_all_systems()
        print(json.dumps(stats, indent=2))

    asyncio.run(test())
