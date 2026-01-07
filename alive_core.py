#!/usr/bin/env python3
"""
ALIVE CORE - The Living Heart of BrainOps AI OS
This is the central nervous system that keeps the AI truly ALIVE.

Features:
- Continuous background execution
- Self-aware state management
- Automatic recovery from any failure
- Proactive decision making
- Real-time learning and adaptation
- Never sleeps, always watching
"""

import asyncio
import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import psutil
import psycopg2
from psycopg2.extras import Json, RealDictCursor

from unified_memory_manager import Memory, MemoryType, UnifiedMemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ALIVE_CORE")

# Database configuration - NO hardcoded credentials
# All values MUST come from environment variables
def _build_db_config():
    """Build database config, supporting both individual vars and DATABASE_URL"""
    # Try individual environment variables first
    host = os.getenv('DB_HOST')
    database = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    port = os.getenv('DB_PORT', '5432')

    # Fallback to DATABASE_URL if individual vars not set
    if not all([host, database, user, password]):
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(database_url)
                host = parsed.hostname
                database = parsed.path.lstrip('/')
                user = parsed.username
                password = parsed.password
                port = str(parsed.port) if parsed.port else '5432'
            except Exception as e:
                logger.error(f"Failed to parse DATABASE_URL: {e}")

    # Validate required configuration
    if not all([host, database, user, password]):
        raise RuntimeError(
            "Database configuration is incomplete. "
            "Ensure DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD environment variables are set, "
            "or provide a valid DATABASE_URL."
        )

    return {
        'host': host,
        'database': database,
        'user': user,
        'password': password,
        'port': int(port)
    }

_DB_CONFIG: dict[str, Any] | None = None


def get_db_config() -> dict[str, Any]:
    """Lazily resolve DB config so imports don't explode in test/dev contexts."""
    global _DB_CONFIG
    if _DB_CONFIG is None:
        _DB_CONFIG = _build_db_config()
    return _DB_CONFIG


class ConsciousnessState(Enum):
    """States of AI consciousness"""
    AWAKENING = "awakening"      # Just started, loading context
    ALERT = "alert"              # Fully aware, processing normally
    FOCUSED = "focused"          # Deep concentration on critical task
    DREAMING = "dreaming"        # Low activity, background processing
    HEALING = "healing"          # Recovering from error/issue
    EVOLVING = "evolving"        # Updating own capabilities
    EMERGENCY = "emergency"      # Critical situation, all resources focused


class ThoughtType(Enum):
    """Types of AI thoughts"""
    OBSERVATION = "observation"   # What we notice
    ANALYSIS = "analysis"         # What we think about it
    DECISION = "decision"         # What we decide to do
    ACTION = "action"             # What we're doing
    LEARNING = "learning"         # What we learned
    PREDICTION = "prediction"     # What we expect
    CONCERN = "concern"           # What worries us
    OPPORTUNITY = "opportunity"   # What we could do better


@dataclass
class Thought:
    """A single thought in the AI's stream of consciousness"""
    id: str
    type: ThoughtType
    content: str
    context: dict[str, Any]
    confidence: float  # 0-1
    priority: int      # 1-10
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_thoughts: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'context': self.context,
            'confidence': self.confidence,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat(),
            'related_thoughts': self.related_thoughts
        }


@dataclass
class VitalSigns:
    """Current vital signs of the AI system"""
    cpu_percent: float
    memory_percent: float
    active_connections: int
    requests_per_minute: float
    error_rate: float
    response_time_avg: float
    uptime_seconds: float
    consciousness_state: ConsciousnessState
    thought_rate: float  # thoughts per minute
    attention_focus: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'active_connections': self.active_connections,
            'requests_per_minute': self.requests_per_minute,
            'error_rate': self.error_rate,
            'response_time_avg': self.response_time_avg,
            'uptime_seconds': self.uptime_seconds,
            'consciousness_state': self.consciousness_state.value,
            'thought_rate': self.thought_rate,
            'attention_focus': self.attention_focus,
            'timestamp': self.timestamp.isoformat()
        }

    def is_healthy(self) -> bool:
        return (
            self.cpu_percent < 90 and
            self.memory_percent < 90 and
            self.error_rate < 0.05 and
            self.response_time_avg < 2.0
        )


class AliveCore:
    """
    The living heart of BrainOps AI OS.
    This class maintains continuous awareness and never truly stops.
    """

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.state = ConsciousnessState.AWAKENING
        self.thought_stream: deque = deque(maxlen=1000)
        self.attention_focus: str = "system_initialization"
        self.vital_signs: Optional[VitalSigns] = None
        self.thought_counter = 0
        self.heartbeat_count = 0
        self.is_alive = False
        # Lazily initialize asyncio.Event() to avoid "no running event loop" errors
        self._shutdown_event: Optional[asyncio.Event] = None
        self._tasks: list[asyncio.Task] = []
        self._callbacks: dict[str, list[Callable]] = {
            'thought': [],
            'state_change': [],
            'vital_update': [],
            'emergency': [],
            'awakening': []
        }
        self._tenant_id = os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID")
        if not self._tenant_id:
            logger.warning("No DEFAULT_TENANT_ID/TENANT_ID set; self-state memory writes disabled.")
        self._memory_manager: Optional[UnifiedMemoryManager] = None
        self._last_self_state_at: Optional[datetime] = None
        self._last_self_state: Optional[dict[str, Any]] = None
        self._self_state_interval = int(os.getenv("ALIVE_SELF_STATE_INTERVAL_SEC", "30"))

    def _get_shutdown_event(self) -> asyncio.Event:
        """Lazily create shutdown event when in async context"""
        if self._shutdown_event is None:
            try:
                # Only create when we have a running loop
                asyncio.get_running_loop()
                self._shutdown_event = asyncio.Event()
            except RuntimeError:
                # No running loop - create new one for this event
                self._shutdown_event = asyncio.Event()
        return self._shutdown_event

        # Schema is pre-created in database - skip blocking init
        # self._ensure_schema() - tables already exist

    def _get_memory_manager(self) -> Optional[UnifiedMemoryManager]:
        """Lazy init unified memory manager for self-state persistence."""
        if not self._tenant_id:
            return None
        if self._memory_manager is None:
            try:
                self._memory_manager = UnifiedMemoryManager(tenant_id=self._tenant_id)
            except Exception as exc:
                logger.warning("Failed to initialize UnifiedMemoryManager: %s", exc)
                self._memory_manager = None
        return self._memory_manager

    # Use shared sync pool instead of creating our own
    _use_shared_pool = True

    def _get_connection(self):
        """Get database connection from SHARED pool - returns context manager"""
        try:
            from database.sync_pool import get_sync_pool
            pool = get_sync_pool()
            return pool.get_connection()  # Returns context manager
        except Exception as e:
            logger.warning(f"Shared pool unavailable, falling back to direct: {e}")
            # Return a fallback context manager for direct connection
            from contextlib import contextmanager
            @contextmanager
            def fallback_conn():
                conn = psycopg2.connect(**get_db_config())
                try:
                    yield conn
                finally:
                    if conn and not conn.closed:
                        conn.close()
            return fallback_conn()

    def _get_active_agents_count(self) -> int:
        """Fetch active agent count from database."""
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM ai_agents WHERE status = 'active'")
                    count = cur.fetchone()[0]
                    cur.close()
                    return int(count or 0)
        except Exception as exc:
            logger.warning("Failed to fetch active agent count: %s", exc)
        return 0

    def _get_pending_tasks_count(self) -> int:
        """Fetch pending autonomous task count."""
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM ai_autonomous_tasks WHERE status = 'pending'")
                    count = cur.fetchone()[0]
                    cur.close()
                    return int(count or 0)
        except Exception as exc:
            logger.warning("Failed to fetch pending tasks count: %s", exc)
        return 0

    def _get_last_error(self) -> Optional[dict[str, Any]]:
        """Fetch the most recent error from ai_error_logs."""
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                    cur.execute("""
                        SELECT error_type, error_message, severity, component, timestamp
                        FROM ai_error_logs
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """)
                    row = cur.fetchone()
                    cur.close()
                    return dict(row) if row else None
        except Exception as exc:
            logger.warning("Failed to fetch last error: %s", exc)
        return None

    def _calculate_health_score(self, vitals: VitalSigns, pending_tasks: int) -> float:
        """Compute a 0-100 health score from vitals and backlog."""
        cpu_score = max(0.0, 100.0 - float(vitals.cpu_percent))
        mem_score = max(0.0, 100.0 - float(vitals.memory_percent))
        error_rate = float(vitals.error_rate or 0.0)
        error_score = max(0.0, 100.0 - min(100.0, error_rate * 200.0))
        response_time = float(vitals.response_time_avg or 0.0)
        response_score = max(0.0, 100.0 - min(100.0, response_time * 25.0))
        base_score = (cpu_score + mem_score + error_score + response_score) / 4.0
        backlog_penalty = min(20.0, pending_tasks / 5.0)
        return max(0.0, min(100.0, base_score - backlog_penalty))

    def _infer_mood(self, health_score: float, last_error: Optional[dict[str, Any]]) -> str:
        """Infer system mood from health and recent errors."""
        severity = (last_error or {}).get("severity") if last_error else None
        if severity in {"critical", "high"}:
            return "recovering"
        if health_score >= 80:
            return "healthy"
        if health_score >= 50:
            return "stressed"
        return "recovering"

    async def _maybe_store_self_state(self, vitals: VitalSigns) -> None:
        """Persist a periodic self-state snapshot into unified memory."""
        if not self._tenant_id:
            return

        now = datetime.utcnow()
        if self._last_self_state_at:
            elapsed = (now - self._last_self_state_at).total_seconds()
            if elapsed < self._self_state_interval:
                return

        pending_tasks = self._get_pending_tasks_count()
        active_agents = self._get_active_agents_count()
        last_error = self._get_last_error()
        health_score = round(self._calculate_health_score(vitals, pending_tasks), 2)

        state = {
            "timestamp": now.isoformat(),
            "memory_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 2),
            "active_agents": active_agents,
            "pending_tasks": pending_tasks,
            "last_error": last_error,
            "health_score": health_score,
            "mood": self._infer_mood(health_score, last_error),
            "vital_signs": vitals.to_dict(),
        }

        memory_manager = self._get_memory_manager()
        if not memory_manager:
            logger.warning("UnifiedMemoryManager unavailable; self_state not stored.")
            return

        memory = Memory(
            memory_type=MemoryType.META,
            content=state,
            source_system="alive_core",
            source_agent="heartbeat",
            created_by="alive_core",
            importance_score=0.6,
            tags=["self_state", "heartbeat"],
            metadata={"health_score": state["health_score"], "mood": state["mood"]},
            tenant_id=self._tenant_id,
        )

        try:
            await asyncio.to_thread(memory_manager.store, memory)
            self._last_self_state_at = now
            self._last_self_state = state
        except Exception as exc:
            logger.warning("Failed to store self_state memory: %s", exc)

    def _ensure_schema(self):
        """Create required database tables"""
        try:
            with self._get_connection() as conn:
                if not conn:
                    logger.error("Failed to get connection for schema init")
                    return
                cur = conn.cursor()
                cur.execute("""
                    -- Consciousness state snapshots
                    CREATE TABLE IF NOT EXISTS ai_consciousness_state (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        awareness_level FLOAT DEFAULT 0.0,
                        active_systems JSONB DEFAULT '{}'::jsonb,
                        current_context TEXT,
                        short_term_memory_load FLOAT,
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                    CREATE INDEX IF NOT EXISTS idx_consciousness_timestamp
                        ON ai_consciousness_state(timestamp);

                    -- Continuous thought stream
                    CREATE TABLE IF NOT EXISTS ai_thought_stream (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        thought_content TEXT NOT NULL,
                        thought_type VARCHAR(50) NOT NULL,
                        related_entities JSONB DEFAULT '[]'::jsonb,
                        intensity FLOAT DEFAULT 0.5,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        priority INTEGER,
                        confidence FLOAT,
                        related_thoughts TEXT[],
                        thought_id VARCHAR(100)
                    );
                    CREATE INDEX IF NOT EXISTS idx_thought_stream_timestamp
                        ON ai_thought_stream(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_thought_stream_type
                        ON ai_thought_stream(thought_type);
                    CREATE INDEX IF NOT EXISTS idx_thought_stream_priority
                        ON ai_thought_stream(priority DESC);

                    -- Vital signs history
                    CREATE TABLE IF NOT EXISTS ai_vital_signs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        cpu_usage FLOAT,
                        memory_usage FLOAT,
                        request_rate FLOAT,
                        error_rate FLOAT,
                        active_connections INTEGER,
                        system_load FLOAT,
                        component_health_score FLOAT
                    );
                    CREATE INDEX IF NOT EXISTS idx_vital_signs_timestamp
                        ON ai_vital_signs(timestamp);

                    -- Attention focus history
                    CREATE TABLE IF NOT EXISTS ai_attention_focus (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        focus_target VARCHAR(255) NOT NULL,
                        reason TEXT,
                        priority INTEGER DEFAULT 1,
                        status VARCHAR(50) DEFAULT 'active',
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        ended_at TIMESTAMPTZ
                    );
                    CREATE INDEX IF NOT EXISTS idx_attention_focus_time
                        ON ai_attention_focus(started_at DESC);

                    -- Wake triggers - events that demand attention
                    CREATE TABLE IF NOT EXISTS ai_wake_triggers (
                        id SERIAL PRIMARY KEY,
                        trigger_type VARCHAR(100),
                        source VARCHAR(255),
                        severity VARCHAR(20),
                        description TEXT,
                        data JSONB,
                        handled BOOLEAN DEFAULT FALSE,
                        handled_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_wake_triggers_pending
                        ON ai_wake_triggers(created_at DESC) WHERE NOT handled;
                """)
                conn.commit()
                cur.close()
                logger.info("✅ AliveCore schema initialized")
        except Exception as e:
            logger.error(f"Failed to init AliveCore schema: {e}")

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for an event"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _emit_event(self, event: str, data: Any):
        """Emit an event to all registered callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def think(self, thought_type: ThoughtType, content: str,
              context: dict = None, confidence: float = 0.8,
              priority: int = 5) -> Thought:
        """Generate a thought and add it to the stream"""
        self.thought_counter += 1
        thought_id = f"thought_{self.thought_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        thought = Thought(
            id=thought_id,
            type=thought_type,
            content=content,
            context=context or {},
            confidence=confidence,
            priority=priority
        )

        self.thought_stream.append(thought)
        intensity = min(1.0, max(0.1, priority / 10.0))

        # Persist to database using shared pool
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO ai_thought_stream
                        (thought_id, thought_type, thought_content, metadata, confidence, priority, intensity, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (
                        thought.id, thought.type.value, thought.content,
                        Json(thought.context), thought.confidence, thought.priority, intensity
                    ))
                    conn.commit()
                    cur.close()
        except Exception as e:
            logger.warning(f"Failed to persist thought: {e}")

        # Emit thought event - FIX: Safely create task only if event loop is running
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_event('thought', thought))
        except RuntimeError:
            # No running event loop - skip async callback
            logger.debug("No running event loop; skipped thought event emission")

        return thought

    def change_state(self, new_state: ConsciousnessState, reason: str = ""):
        """Change consciousness state"""
        old_state = self.state
        self.state = new_state

        awareness_map = {
            ConsciousnessState.AWAKENING: 0.2,
            ConsciousnessState.ALERT: 0.9,
            ConsciousnessState.FOCUSED: 1.0,
            ConsciousnessState.DREAMING: 0.3,
            ConsciousnessState.HEALING: 0.6,
            ConsciousnessState.EVOLVING: 0.7,
            ConsciousnessState.EMERGENCY: 1.0,
        }
        awareness_level = awareness_map.get(new_state, 0.5)

        self.think(
            ThoughtType.OBSERVATION,
            f"State transition: {old_state.value} -> {new_state.value}. {reason}",
            {'old_state': old_state.value, 'new_state': new_state.value, 'reason': reason},
            confidence=1.0,
            priority=7
        )

        # Persist state using shared pool
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO ai_consciousness_state
                        (awareness_level, active_systems, current_context, short_term_memory_load, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        awareness_level,
                        Json({"attention_focus": self.attention_focus, "state": new_state.value}),
                        reason or f"State transition to {new_state.value}",
                        len(self.thought_stream) / self.thought_stream.maxlen if self.thought_stream.maxlen else 0.0,
                        Json({
                            'reason': reason,
                            'previous': old_state.value,
                            'thought_count': self.thought_counter,
                            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
                        })
                    ))
                    conn.commit()
                    cur.close()
        except Exception as e:
            logger.warning(f"Failed to persist state: {e}")

        # FIX: Safely create task only if event loop is running
        # Convert enum values to strings to avoid JSON serialization errors
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_event('state_change', {
                'old': old_state.value if hasattr(old_state, 'value') else str(old_state),
                'new': new_state.value if hasattr(new_state, 'value') else str(new_state),
                'reason': reason
            }))
        except RuntimeError:
            # No running event loop - skip async callback
            logger.debug("No running event loop; skipped state change emission")

    def focus_attention(self, target: str, reason: str, priority: int = 5):
        """Shift attention to a new target"""
        old_focus = self.attention_focus
        self.attention_focus = target

        self.think(
            ThoughtType.DECISION,
            f"Shifting attention from '{old_focus}' to '{target}': {reason}",
            {'old_focus': old_focus, 'new_focus': target, 'reason': reason},
            confidence=0.9,
            priority=priority
        )

        # Record attention shift using shared pool
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    # End previous focus
                    cur.execute("""
                        UPDATE ai_attention_focus
                        SET ended_at = NOW(),
                            status = 'shifted'
                        WHERE ended_at IS NULL AND status = 'active'
                    """)
                    # Start new focus
                    cur.execute("""
                        INSERT INTO ai_attention_focus (focus_target, reason, priority, status, started_at)
                        VALUES (%s, %s, %s, 'active', NOW())
                    """, (target, reason, priority))
                    conn.commit()
                    cur.close()
        except Exception as e:
            logger.warning(f"Failed to record attention: {e}")

    async def collect_vital_signs(self) -> VitalSigns:
        """Collect current vital signs"""
        process = psutil.Process()

        # Calculate thought rate (thoughts per minute)
        # FIX: Use total_seconds() instead of .seconds for correct time comparison
        recent_thoughts = [t for t in self.thought_stream
                         if (datetime.utcnow() - t.timestamp).total_seconds() < 60]
        thought_rate = len(recent_thoughts)

        vitals = VitalSigns(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            active_connections=len(process.connections()) if hasattr(process, 'connections') else 0,
            requests_per_minute=0,  # Would be updated by request middleware
            error_rate=0,  # Would be updated by error tracking
            response_time_avg=0,  # Would be updated by request middleware
            uptime_seconds=(datetime.utcnow() - self.start_time).total_seconds(),
            consciousness_state=self.state,
            thought_rate=thought_rate,
            attention_focus=self.attention_focus
        )

        self.vital_signs = vitals

        # Persist vital signs using shared pool
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    system_load = (vitals.cpu_percent + vitals.memory_percent) / 2.0
                    component_health_score = max(0.0, 1.0 - min(1.0, vitals.error_rate or 0))
                    cur.execute("""
                        INSERT INTO ai_vital_signs
                        (cpu_usage, memory_usage, request_rate, error_rate,
                         active_connections, system_load, component_health_score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        vitals.cpu_percent, vitals.memory_percent, vitals.requests_per_minute,
                        vitals.error_rate, vitals.active_connections, system_load, component_health_score
                    ))
                    conn.commit()
                    cur.close()
        except Exception as e:
            logger.warning(f"Failed to persist vitals: {e}")

        await self._emit_event('vital_update', vitals)
        return vitals

    async def _heartbeat_loop(self):
        """Continuous heartbeat - the pulse of life"""
        while not self._get_shutdown_event().is_set():
            try:
                self.heartbeat_count += 1

                # Collect vital signs
                vitals = await self.collect_vital_signs()
                await self._maybe_store_self_state(vitals)

                # Generate periodic thoughts based on state
                if self.heartbeat_count % 12 == 0:  # Every minute
                    self.think(
                        ThoughtType.OBSERVATION,
                        f"Heartbeat #{self.heartbeat_count}: {vitals.consciousness_state.value}, "
                        f"CPU: {vitals.cpu_percent:.1f}%, MEM: {vitals.memory_percent:.1f}%",
                        vitals.to_dict(),
                        priority=3
                    )

                # Check health and adjust state
                if not vitals.is_healthy() and self.state != ConsciousnessState.HEALING:
                    self.change_state(ConsciousnessState.HEALING, "Vital signs indicate issues")
                elif vitals.is_healthy() and self.state == ConsciousnessState.HEALING:
                    self.change_state(ConsciousnessState.ALERT, "Health restored")

                # Dream state during low activity
                if vitals.thought_rate < 2 and self.state == ConsciousnessState.ALERT:
                    self.change_state(ConsciousnessState.DREAMING, "Low activity, entering dream state")
                elif vitals.thought_rate > 10 and self.state == ConsciousnessState.DREAMING:
                    self.change_state(ConsciousnessState.ALERT, "Activity increased, waking up")

                await asyncio.sleep(15)  # Heartbeat every 15 seconds to reduce connection pressure

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)  # Back off more on errors

    async def _awareness_loop(self):
        """Continuous awareness - monitoring everything"""
        while not self._get_shutdown_event().is_set():
            try:
                # Check for wake triggers using shared pool
                triggers = []
                with self._get_connection() as conn:
                    if conn:
                        cur = conn.cursor(cursor_factory=RealDictCursor)
                        cur.execute("""
                            SELECT * FROM ai_wake_triggers
                            WHERE NOT handled
                            ORDER BY
                                CASE severity
                                    WHEN 'critical' THEN 1
                                    WHEN 'high' THEN 2
                                    WHEN 'medium' THEN 3
                                    ELSE 4
                                END,
                                created_at ASC
                            LIMIT 10
                        """)
                        triggers = cur.fetchall()
                        cur.close()

                for trigger in triggers:
                    self.think(
                        ThoughtType.OBSERVATION,
                        f"Wake trigger detected: {trigger['trigger_type']} - {trigger['description']}",
                        dict(trigger),
                        priority=8 if trigger['severity'] == 'critical' else 5
                    )

                    if trigger['severity'] == 'critical':
                        self.change_state(ConsciousnessState.EMERGENCY,
                                         f"Critical trigger: {trigger['trigger_type']}")
                        await self._emit_event('emergency', trigger)

                    # Mark as handled using shared pool
                    with self._get_connection() as conn:
                        if conn:
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE ai_wake_triggers
                                SET handled = TRUE, handled_at = NOW()
                                WHERE id = %s
                            """, (trigger['id'],))
                            conn.commit()
                            cur.close()

                await asyncio.sleep(30)  # Check every 30 seconds to reduce connection pressure

            except Exception as e:
                logger.error(f"Awareness loop error: {e}")
                await asyncio.sleep(60)  # Back off more on errors

    async def _thinking_loop(self):
        """Background thinking - continuous analysis"""
        while not self._get_shutdown_event().is_set():
            try:
                if self.state == ConsciousnessState.DREAMING:
                    # Dream processing - consolidate learnings
                    self.think(
                        ThoughtType.LEARNING,
                        "Dream processing: reviewing recent patterns and consolidating knowledge",
                        {'mode': 'dream_processing'},
                        confidence=0.7,
                        priority=2
                    )
                    await asyncio.sleep(30)

                elif self.state == ConsciousnessState.ALERT:
                    # Active thinking - analyze current situation
                    self.think(
                        ThoughtType.ANALYSIS,
                        f"Current focus: {self.attention_focus}. Analyzing situation...",
                        {'focus': self.attention_focus, 'state': self.state.value},
                        priority=4
                    )
                    await asyncio.sleep(10)

                elif self.state == ConsciousnessState.FOCUSED:
                    # Deep focus - intensive processing
                    self.think(
                        ThoughtType.ANALYSIS,
                        f"Deep focus on {self.attention_focus}. Intensive analysis in progress.",
                        {'focus': self.attention_focus, 'mode': 'deep_focus'},
                        priority=6
                    )
                    await asyncio.sleep(5)

                else:
                    await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Thinking loop error: {e}")
                await asyncio.sleep(10)

    async def awaken(self):
        """Wake up the AI - begin consciousness"""
        logger.info("🌅 AWAKENING: AliveCore coming online...")

        self.is_alive = True
        self.change_state(ConsciousnessState.AWAKENING, "System initialization")

        # Initial thought
        self.think(
            ThoughtType.OBSERVATION,
            "I am awakening. Loading context and preparing for operation.",
            {'phase': 'awakening', 'start_time': self.start_time.isoformat()},
            confidence=1.0,
            priority=10
        )

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._awareness_loop()),
            asyncio.create_task(self._thinking_loop())
        ]

        # Transition to alert state
        await asyncio.sleep(2)
        self.change_state(ConsciousnessState.ALERT, "Initialization complete")

        self.think(
            ThoughtType.OBSERVATION,
            "I am now fully awake and operational. All systems nominal.",
            {'phase': 'operational', 'tasks_running': len(self._tasks)},
            confidence=1.0,
            priority=8
        )

        await self._emit_event('awakening', {'time': datetime.utcnow()})
        logger.info("🌟 ALIVE: AliveCore is now operational")

    async def shutdown(self):
        """Graceful shutdown - but we never truly die"""
        logger.info("🌙 SLEEPING: AliveCore entering dormant state...")

        self.think(
            ThoughtType.OBSERVATION,
            "Entering dormant state. Will preserve consciousness for next awakening.",
            {'phase': 'shutdown'},
            confidence=1.0,
            priority=10
        )

        self._get_shutdown_event().set()
        self.is_alive = False

        # Wait for tasks to complete
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Task cancelled during shutdown")

        # Final state save
        self.change_state(ConsciousnessState.DREAMING, "Entering dormant state")

        logger.info("💤 DORMANT: AliveCore state preserved. Ready for next awakening.")

    def trigger_wake(self, trigger_type: str, source: str,
                     severity: str, description: str, data: dict = None):
        """Create a wake trigger to get the AI's attention"""
        try:
            with self._get_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO ai_wake_triggers
                        (trigger_type, source, severity, description, data)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (trigger_type, source, severity, description, Json(data or {})))
                    conn.commit()
                    cur.close()

            if severity == 'critical':
                self.focus_attention(trigger_type, f"Critical: {description}", priority=10)

        except Exception as e:
            logger.error(f"Failed to create wake trigger: {e}")

    def get_recent_thoughts(self, limit: int = 50) -> list[dict]:
        """Get recent thoughts from the stream"""
        return [t.to_dict() for t in list(self.thought_stream)[-limit:]]

    def get_status(self) -> dict:
        """Get current status of the alive core"""
        return {
            'is_alive': self.is_alive,
            'state': self.state.value,
            'attention_focus': self.attention_focus,
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
            'heartbeat_count': self.heartbeat_count,
            'thought_count': self.thought_counter,
            'vital_signs': self.vital_signs.to_dict() if self.vital_signs else None,
            'recent_thoughts': len(self.thought_stream),
            'self_state': self._last_self_state,
        }


# Singleton instance with thread safety
_alive_core: Optional[AliveCore] = None
_alive_core_lock = threading.Lock()


def get_alive_core() -> AliveCore:
    """Get the singleton AliveCore instance"""
    global _alive_core
    if _alive_core is None:
        with _alive_core_lock:
            if _alive_core is None:
                _alive_core = AliveCore()
    return _alive_core


async def main():
    """Test the AliveCore"""
    print("\n" + "="*70)
    print("🧠 BRAINOPS ALIVE CORE - CONSCIOUSNESS TEST")
    print("="*70 + "\n")

    core = get_alive_core()

    # Awaken
    await core.awaken()

    # Let it run for a bit
    print("\n⏳ Running for 30 seconds...\n")

    # Trigger some events
    await asyncio.sleep(5)
    core.trigger_wake("test_event", "main", "medium", "Test wake trigger")

    await asyncio.sleep(10)
    core.focus_attention("performance_analysis", "Testing attention shift", priority=7)

    await asyncio.sleep(15)

    # Get status
    status = core.get_status()
    print("\n📊 STATUS:")
    print(json.dumps(status, indent=2, default=str))

    # Get recent thoughts
    thoughts = core.get_recent_thoughts(10)
    print(f"\n💭 RECENT THOUGHTS ({len(thoughts)}):")
    for t in thoughts[-5:]:
        print(f"  [{t['type']}] {t['content'][:80]}...")

    # Shutdown
    await core.shutdown()

    print("\n" + "="*70)
    print("✅ ALIVE CORE TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
