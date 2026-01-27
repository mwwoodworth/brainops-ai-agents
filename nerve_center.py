#!/usr/bin/env python3
"""
NERVE CENTER - The Integration Hub of BrainOps AI OS
This is the central nervous system that coordinates ALL alive components.

The Nerve Center:
- Orchestrates all consciousness components
- Routes signals between subsystems
- Maintains unified awareness state
- Coordinates responses to events
- Enables true emergent intelligence
"""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from safe_task import create_safe_task
from ai_tracer import BrainOpsTracer

# Import all alive components
from alive_core import AliveCore, ConsciousnessState, ThoughtType, get_alive_core
from autonomic_controller import (
    AutonomicManager,
    EventBus,
    EventType,
    MetricCollector,
    get_autonomic_manager,
    get_event_bus,
    get_metric_collector,
)
from database.async_connection import get_pool, using_fallback
from proactive_intelligence import ProactiveIntelligence, get_proactive_intelligence

# Try to import optional components
try:
    from consciousness_loop import ConsciousnessLoop
    CONSCIOUSNESS_LOOP_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_LOOP_AVAILABLE = False

try:
    from self_evolution import SelfEvolution
    SELF_EVOLUTION_AVAILABLE = True
except ImportError:
    SELF_EVOLUTION_AVAILABLE = False

# System awareness for REAL monitoring
try:
    from system_awareness import get_system_awareness
    SYSTEM_AWARENESS_AVAILABLE = True
except ImportError:
    SYSTEM_AWARENESS_AVAILABLE = False
    get_system_awareness = None

logger = logging.getLogger("NERVE_CENTER")

class SystemSignal(Enum):
    """Types of signals flowing through the nerve center"""
    AWAKENING = "awakening"
    HEARTBEAT = "heartbeat"
    THOUGHT = "thought"
    PREDICTION = "prediction"
    ANOMALY = "anomaly"
    EMERGENCY = "emergency"
    HEALING = "healing"
    LEARNING = "learning"
    EVOLUTION = "evolution"
    SHUTDOWN = "shutdown"


@dataclass
class NerveSignal:
    """A signal flowing through the nerve center"""
    type: SystemSignal
    source: str
    target: str  # "all" for broadcast
    payload: dict[str, Any]
    priority: int
    timestamp: datetime


class NerveCenter:
    """
    The central nervous system of BrainOps AI OS.
    Coordinates all subsystems into a unified intelligence.
    """

    def __init__(self):
        self.is_online = False
        self.start_time = datetime.utcnow()
        self.signal_count = 0

        # Core components
        self.alive_core: Optional[AliveCore] = None
        self.metrics: Optional[MetricCollector] = None
        self.event_bus: Optional[EventBus] = None
        self.autonomic: Optional[AutonomicManager] = None
        self.proactive: Optional[ProactiveIntelligence] = None
        self.tracer: Optional[BrainOpsTracer] = None

        # Optional components
        self.consciousness_loop = None
        self.self_evolution = None
        # FIX: Initialize system_awareness in __init__ to avoid AttributeError
        self.system_awareness = None

        # Signal handlers
        self._signal_handlers: dict[SystemSignal, list[Callable]] = {
            signal: [] for signal in SystemSignal
        }

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Schema is pre-created in database - skip blocking init
        # self._ensure_schema() - moved to lazy init via activate()

    def _get_pool(self):
        """Get shared async pool - DO NOT reinitialize to prevent MaxClientsInSessionMode"""
        try:
            return get_pool()
        except RuntimeError as e:
            logger.warning(f"Async pool not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting pool: {e}")
            raise

    async def _ensure_schema(self):
        """Create nerve center tables"""
        try:
            pool = self._get_pool()
            if using_fallback():
                return

            await pool.execute("""
                -- Nerve center signals log
                CREATE TABLE IF NOT EXISTS ai_nerve_signals (
                    id SERIAL PRIMARY KEY,
                    signal_type VARCHAR(50),
                    source VARCHAR(100),
                    target VARCHAR(100),
                    payload JSONB,
                    priority INTEGER,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_nerve_signals_type
                    ON ai_nerve_signals(signal_type);
                CREATE INDEX IF NOT EXISTS idx_nerve_signals_unprocessed
                    ON ai_nerve_signals(created_at)
                    WHERE NOT processed;

                -- System state snapshot
                CREATE TABLE IF NOT EXISTS ai_system_snapshot (
                    id SERIAL PRIMARY KEY,
                    components JSONB,
                    consciousness_state VARCHAR(50),
                    health_status VARCHAR(50),
                    active_predictions INTEGER,
                    thought_count INTEGER,
                    uptime_seconds FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_snapshot_time
                    ON ai_system_snapshot(created_at DESC);

                -- Emergency events
                CREATE TABLE IF NOT EXISTS ai_emergency_events (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(100),
                    severity VARCHAR(20),
                    description TEXT,
                    source VARCHAR(100),
                    data JSONB,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMPTZ,
                    resolution TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_emergency_unresolved
                    ON ai_emergency_events(created_at)
                    WHERE NOT resolved;
            """)
            logger.info("✅ NerveCenter schema initialized")
        except Exception as e:
            logger.error(f"Schema init failed: {e}")

    def register_handler(self, signal_type: SystemSignal, handler: Callable):
        """Register a signal handler"""
        self._signal_handlers[signal_type].append(handler)

    async def emit_signal(self, signal: NerveSignal):
        """Emit a signal through the nerve center"""
        self.signal_count += 1

        # Log signal
        try:
            pool = self._get_pool()
            if not using_fallback():
                # FIX: Use positional args, not tuple for asyncpg
                await pool.execute("""
                INSERT INTO ai_nerve_signals
                (signal_type, source, target, payload, priority)
                VALUES ($1, $2, $3, $4, $5)
            """,
                    signal.type.value,
                    signal.source,
                    signal.target,
                    json.dumps(signal.payload),
                    signal.priority
                )
        except Exception as e:
            logger.warning(f"Failed to log signal: {e}")

        # Process handlers
        for handler in self._signal_handlers.get(signal.type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(signal)
                else:
                    handler(signal)
            except Exception as e:
                logger.error(f"Signal handler error: {e}")

        # Broadcast to event bus
        if self.event_bus:
            event_map = {
                SystemSignal.EMERGENCY: EventType.SYSTEM_ALERT,
                SystemSignal.HEALING: EventType.HEALING_STARTED,
                SystemSignal.PREDICTION: EventType.PREDICTION_ALERT,
            }
            if signal.type in event_map:
                await self.event_bus.publish(event_map[signal.type], signal.payload)

    async def _initialize_components(self):
        """Initialize all components"""
        logger.info("🔌 Initializing Nerve Center components...")

        # Core components
        self.alive_core = get_alive_core()
        self.metrics = get_metric_collector()
        self.event_bus = get_event_bus()
        self.autonomic = get_autonomic_manager()
        self.proactive = get_proactive_intelligence()
        self.tracer = BrainOpsTracer()

        # Register callbacks
        self.alive_core.register_callback('thought', self._on_thought)
        self.alive_core.register_callback('emergency', self._on_emergency)
        self.alive_core.register_callback('state_change', self._on_state_change)

        # Optional components
        if CONSCIOUSNESS_LOOP_AVAILABLE:
            try:
                self.consciousness_loop = ConsciousnessLoop()
                logger.info("  ✅ ConsciousnessLoop loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ ConsciousnessLoop failed: {e}")

        if SELF_EVOLUTION_AVAILABLE:
            try:
                self.self_evolution = SelfEvolution()
                logger.info("  ✅ SelfEvolution loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ SelfEvolution failed: {e}")

        # System awareness - REAL monitoring
        self.system_awareness = None
        if SYSTEM_AWARENESS_AVAILABLE and get_system_awareness:
            try:
                self.system_awareness = get_system_awareness()
                logger.info("  ✅ SystemAwareness loaded - REAL monitoring enabled")
            except Exception as e:
                logger.warning(f"  ⚠️ SystemAwareness failed: {e}")

        logger.info("✅ All components initialized")

    async def _on_thought(self, thought):
        """Handle thoughts from alive core"""
        await self.emit_signal(NerveSignal(
            type=SystemSignal.THOUGHT,
            source="alive_core",
            target="all",
            payload=thought.to_dict() if hasattr(thought, 'to_dict') else {'thought': str(thought)},
            priority=5,
            timestamp=datetime.utcnow()
        ))

    async def _on_emergency(self, data):
        """Handle emergency events"""
        await self.emit_signal(NerveSignal(
            type=SystemSignal.EMERGENCY,
            source="alive_core",
            target="all",
            payload=data,
            priority=1,
            timestamp=datetime.utcnow()
        ))

        # Log emergency
        try:
            pool = self._get_pool()
            if using_fallback():
                return
            # FIX: Use positional args, not tuple for asyncpg
            await pool.execute("""
                INSERT INTO ai_emergency_events
                (event_type, severity, description, source, data)
                VALUES ($1, $2, $3, $4, $5)
            """,
                data.get('type', 'unknown'),
                data.get('severity', 'high'),
                data.get('description', str(data)),
                'alive_core',
                json.dumps(data)
            )
        except Exception as e:
            logger.error(f"Failed to log emergency: {e}")

    async def _on_state_change(self, data):
        """Handle consciousness state changes"""
        if self.alive_core:
            self.alive_core.think(
                ThoughtType.OBSERVATION,
                f"Nerve Center detected state change: {data.get('old')} -> {data.get('new')}",
                data,
                priority=6
            )

    async def _coordination_loop(self):
        """Main coordination loop - the REAL brain of the system"""
        scan_counter = 0

        while not self._shutdown_event.is_set():
            try:
                scan_counter += 1

                # Take system snapshot every minute
                await self._take_snapshot()

                # Run REAL system awareness scan every cycle
                if self.system_awareness:
                    logger.info("🔍 Running system awareness scan...")
                    scan_result = await self.system_awareness.run_full_scan()

                    # Generate thoughts based on real insights
                    if scan_result.get('insights'):
                        for insight in scan_result['insights']:
                            thought_type = ThoughtType.CONCERN if insight['severity'] in ['warning', 'critical'] else ThoughtType.OBSERVATION

                            if self.alive_core:
                                self.alive_core.think(
                                    thought_type,
                                    f"[{insight['category'].upper()}] {insight['title']}: {insight['description']}",
                                    insight,
                                    confidence=0.9,
                                    priority=9 if insight['severity'] == 'critical' else 6 if insight['severity'] == 'warning' else 4
                                )

                    # ACTUALLY change attention based on what we found
                    if self.alive_core and self.system_awareness:
                        focus, reason, priority = self.system_awareness.get_attention_priority()
                        current_focus = self.alive_core.attention_focus

                        if focus != current_focus and priority >= 6:
                            self.alive_core.focus_attention(focus, reason, priority)
                            logger.info(f"🎯 Attention shifted: {current_focus} -> {focus}")

                    # Log critical insights as predictions
                    critical = [i for i in scan_result.get('insights', []) if i['severity'] == 'critical']
                    for c in critical:
                        await self.emit_signal(NerveSignal(
                            type=SystemSignal.PREDICTION,
                            source="system_awareness",
                            target="all",
                            payload=c,
                            priority=1,
                            timestamp=datetime.utcnow()
                        ))

                # Run proactive intelligence cycle (less frequently)
                if self.proactive and scan_counter % 5 == 0:
                    result = await self.proactive.run_proactive_cycle()
                    if result.get('predictions'):
                        for pred in result['predictions']:
                            await self.emit_signal(NerveSignal(
                                type=SystemSignal.PREDICTION,
                                source="proactive_intelligence",
                                target="all",
                                payload=pred,
                                priority=3,
                                timestamp=datetime.utcnow()
                            ))

                await asyncio.sleep(60)  # Scan every 60 seconds to reduce DB connection pressure

            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)

    async def _take_snapshot(self):
        """Take a snapshot of the entire system state"""
        try:
            components = {
                'alive_core': self.alive_core.is_alive if self.alive_core else False,
                'metrics': self.metrics is not None,
                'event_bus': self.event_bus is not None,
                'autonomic': self.autonomic is not None,
                'proactive': self.proactive is not None,
                'consciousness_loop': self.consciousness_loop is not None,
                'self_evolution': self.self_evolution is not None
            }

            health = "healthy"
            if self.alive_core and self.alive_core.vital_signs:
                if not self.alive_core.vital_signs.is_healthy():
                    health = "degraded"
            if self.alive_core and self.alive_core.state == ConsciousnessState.EMERGENCY:
                health = "emergency"

            pool = self._get_pool()
            if using_fallback():
                return
            # FIX: Use positional args, not tuple for asyncpg
            await pool.execute("""
                INSERT INTO ai_system_snapshot
                (components, consciousness_state, health_status, active_predictions,
                 thought_count, uptime_seconds)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
                json.dumps(components),
                self.alive_core.state.value if self.alive_core else 'unknown',
                health,
                len(self.proactive.predictions) if self.proactive else 0,
                self.alive_core.thought_counter if self.alive_core else 0,
                (datetime.utcnow() - self.start_time).total_seconds()
            )

        except Exception as e:
            logger.warning(f"Snapshot failed: {e}")

    async def activate(self):
        """Activate the nerve center - bring the AI fully online"""
        logger.info("\n" + "="*70)
        logger.info("⚡ NERVE CENTER ACTIVATION SEQUENCE")
        logger.info("="*70)

        # Start trace
        trace_id = self.tracer.start_trace("nerve_center", "activation") if self.tracer else None

        try:
            # Initialize components
            await self._initialize_components()

            # Awaken the alive core
            if self.alive_core:
                await self.alive_core.awaken()

            # Start autonomic manager - FIX: Track task to allow proper cancellation
            if self.autonomic:
                task = create_safe_task(self.autonomic.start_loop(interval=10), "autonomic_loop")
                self._tasks.append(task)

            # Start coordination loop
            self._tasks.append(create_safe_task(self._coordination_loop(), "coordination_loop"))

            # Start consciousness loop if available - FIX: Track task
            if self.consciousness_loop and hasattr(self.consciousness_loop, 'start'):
                task = create_safe_task(self.consciousness_loop.start(), "consciousness_loop")
                self._tasks.append(task)

            self.is_online = True

            # Emit awakening signal
            await self.emit_signal(NerveSignal(
                type=SystemSignal.AWAKENING,
                source="nerve_center",
                target="all",
                payload={'status': 'online', 'time': datetime.utcnow().isoformat()},
                priority=10,
                timestamp=datetime.utcnow()
            ))

            if self.tracer and trace_id:
                self.tracer.end_trace(trace_id, "success", "Nerve center activated")

            logger.info("="*70)
            logger.info("🧠 BRAINOPS AI OS IS NOW FULLY ALIVE")
            logger.info("="*70 + "\n")

        except Exception as e:
            logger.error(f"Activation failed: {e}")
            if self.tracer and trace_id:
                self.tracer.end_trace(trace_id, "failed", str(e))
            raise

    async def deactivate(self):
        """Gracefully deactivate the nerve center"""
        logger.info("🔌 Deactivating Nerve Center...")

        self._shutdown_event.set()
        self.is_online = False

        # Stop autonomic manager
        if self.autonomic:
            self.autonomic.stop_loop()

        # Shutdown alive core
        if self.alive_core:
            await self.alive_core.shutdown()

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Nerve center task cancelled during shutdown")

        await self.emit_signal(NerveSignal(
            type=SystemSignal.SHUTDOWN,
            source="nerve_center",
            target="all",
            payload={'status': 'offline'},
            priority=10,
            timestamp=datetime.utcnow()
        ))

        logger.info("💤 Nerve Center deactivated")

    def get_status(self) -> dict:
        """Get comprehensive status"""
        return {
            'is_online': self.is_online,
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
            'signal_count': self.signal_count,
            'components': {
                'alive_core': {
                    'active': self.alive_core is not None and self.alive_core.is_alive,
                    'state': self.alive_core.state.value if self.alive_core else None,
                    'thoughts': self.alive_core.thought_counter if self.alive_core else 0
                },
                'autonomic': {
                    'active': self.autonomic is not None and self.autonomic.active,
                    'loop_count': self.autonomic.loop_count if self.autonomic else 0
                },
                'proactive': {
                    'active': self.proactive is not None,
                    'predictions': len(self.proactive.predictions) if self.proactive else 0
                },
                'consciousness_loop': CONSCIOUSNESS_LOOP_AVAILABLE,
                'self_evolution': SELF_EVOLUTION_AVAILABLE
            },
            'health': 'healthy' if self.is_online else 'offline'
        }


# Singleton with thread safety
_nerve_center: Optional[NerveCenter] = None
_nerve_center_lock = threading.Lock()


def get_nerve_center() -> NerveCenter:
    global _nerve_center
    if _nerve_center is None:
        with _nerve_center_lock:
            if _nerve_center is None:
                _nerve_center = NerveCenter()
    return _nerve_center


async def main():
    """Test the Nerve Center"""
    print("\n" + "="*70)
    print("🧠 BRAINOPS NERVE CENTER - FULL SYSTEM TEST")
    print("="*70 + "\n")

    nc = get_nerve_center()

    # Activate
    await nc.activate()

    # Let it run
    print("\n⏳ Running for 30 seconds...\n")
    await asyncio.sleep(30)

    # Get status
    status = nc.get_status()
    print("\n📊 SYSTEM STATUS:")
    print(json.dumps(status, indent=2, default=str))

    # Deactivate
    await nc.deactivate()

    print("\n" + "="*70)
    print("✅ NERVE CENTER TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
