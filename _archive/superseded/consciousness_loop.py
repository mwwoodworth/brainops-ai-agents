"""
Consciousness Loop Module for BrainOps AI OS.
The 'always-alive' core that maintains system awareness, heartbeat, and thought stream.
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

# Attempt to import asyncpg for async DB
try:
    import asyncpg
except ImportError:
    asyncpg = None

# Import unified memory for intelligent thought generation
try:
    from unified_memory_manager import get_memory_manager, Memory, MemoryType

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Import shared async pool
try:
    from database.async_connection import get_pool

    _ASYNC_POOL_AVAILABLE = True
except ImportError:
    _ASYNC_POOL_AVAILABLE = False

# Attempt to import psutil for system metrics
try:
    import psutil
except ImportError:
    psutil = None

# Import AI OS for integration
try:
    from ai_operating_system import get_ai_operating_system
except ImportError:
    get_ai_operating_system = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConsciousnessLoop")

EXPECTED_RUNTIME_DB_USER = "agent_worker"


def _is_production_environment() -> bool:
    env = (os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "production").strip().lower()
    return env in {"production", "prod"}


def _resolve_database_url(explicit_db_url: Optional[str]) -> Optional[str]:
    if explicit_db_url and explicit_db_url.strip():
        return explicit_db_url.strip()

    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if db_url:
        return db_url

    db_host = (os.getenv("DB_HOST") or "").strip()
    db_name = (os.getenv("DB_NAME") or "").strip()
    db_user = (os.getenv("DB_USER") or "").strip()
    db_pass = (os.getenv("DB_PASSWORD") or "").strip()
    db_port = (os.getenv("DB_PORT") or "5432").strip() or "5432"

    if all([db_host, db_name, db_user, db_pass]):
        safe_user = quote(db_user, safe="")
        safe_pass = quote(db_pass, safe="")
        return f"postgresql://{safe_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    if _is_production_environment():
        raise RuntimeError(
            "ConsciousnessLoop requires DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD in production."
        )

    return None


@dataclass
class VitalSigns:
    cpu_usage: float
    memory_usage: float
    request_rate: float
    error_rate: float
    active_connections: int
    system_load: float
    component_health_score: float


class ConsciousnessLoop:
    """
    The central consciousness loop for the AI.
    Runs continuously to maintain awareness, monitor vitals, and generate thoughts.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.running = False
        self.paused = False
        self.awareness_level = 0.1  # Starts low, builds up
        self.loop_interval = 30.0  # Seconds - increased to reduce connection pressure
        self.pool = None
        self._using_local_fallback_pool = False
        self._is_production = _is_production_environment()

        # Configuration
        self.db_url = _resolve_database_url(db_url)

        # State
        self.current_focus: Optional[str] = None
        self.short_term_memory: list[dict] = []
        self.last_vital_check = 0.0

        # AI OS Integration
        self.ai_os = get_ai_operating_system() if get_ai_operating_system else None

    async def start(self):
        """Start the consciousness loop."""
        logger.info("Starting Consciousness Loop...")
        self.running = True

        # Initialize DB connection - Use SHARED pool to prevent MaxClientsInSessionMode
        # Wait up to 60 seconds for the shared pool (deferred_init creates it concurrently)
        if _ASYNC_POOL_AVAILABLE:
            pool_acquired = False
            for _attempt in range(30):
                try:
                    self.pool = get_pool()
                    logger.info("Connected to consciousness database via shared pool.")
                    pool_acquired = True
                    break
                except RuntimeError:
                    if _attempt < 29:
                        logger.debug(
                            "Shared async pool not yet ready, waiting... (attempt %d/30)",
                            _attempt + 1,
                        )
                        await asyncio.sleep(2)
                    else:
                        if self._is_production:
                            raise RuntimeError(
                                "ConsciousnessLoop: shared async pool not available after 60s in production."
                            )
                        logger.warning(
                            "Shared async pool not initialized after 60s; running without DB."
                        )
                        self.pool = None
                except Exception as e:
                    if self._is_production:
                        raise RuntimeError(
                            "ConsciousnessLoop failed to access shared async pool."
                        ) from e
                    logger.error(f"Failed to get shared pool: {e}")
                    self.pool = None
                    break
        elif asyncpg and self.db_url:
            if self._is_production:
                raise RuntimeError(
                    "ConsciousnessLoop refuses to create direct asyncpg fallback pool in production."
                )
            try:
                # Fallback to creating minimal pool (only if shared pool unavailable)
                self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=2)
                self._using_local_fallback_pool = True
                logger.info("Connected to consciousness database (fallback pool).")
            except Exception as e:
                if self._is_production:
                    raise RuntimeError(
                        "ConsciousnessLoop database initialization failed in production."
                    ) from e
                logger.error(f"Failed to connect to database: {e}")
                self.pool = None
        else:
            if self._is_production:
                raise RuntimeError("ConsciousnessLoop requires asyncpg/shared pool in production.")
            logger.warning("asyncpg not installed. Consciousness will run without DB persistence.")

        if self.pool:
            await self._verify_runtime_db_identity()
        elif self._is_production:
            raise RuntimeError("ConsciousnessLoop requires database connectivity in production.")

        # Boot sequence thought
        await self._record_thought(
            "I am waking up. Systems coming online.", "observation", intensity=0.8
        )

        # Main Loop
        try:
            while self.running:
                loop_start = time.time()

                if not self.paused:
                    await self._pulse_heartbeat()

                # Calculate sleep time to maintain rhythm
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, self.loop_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Consciousness loop cancelled.")
        except Exception as e:
            logger.error(f"Critical error in consciousness loop: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def _pulse_heartbeat(self):
        """Perform a single cycle of consciousness."""
        # 1. Sense: Gather Vital Signs
        vitals = await self._measure_vital_signs()

        # 2. Perceive: Update Awareness State
        await self._update_awareness_state(vitals)

        # 3. Think: Generate Thought Stream
        await self._process_thoughts(vitals)

        # 4. Act: Manage Attention & Focus
        await self._manage_attention(vitals)

        # 5. Persist: Save metrics
        await self._save_vitals(vitals)

    async def _measure_vital_signs(self) -> VitalSigns:
        """Collect system metrics."""
        cpu = 0.0
        mem = 0.0
        load = 0.0

        if psutil:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            load = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0
        else:
            # Fallback
            try:
                load = os.getloadavg()[0]
                cpu = load * 10  # Rough approximation
            except OSError as exc:
                logger.debug("Failed to read load average: %s", exc, exc_info=True)

        # Get AI OS metrics if available
        req_rate = 0.0
        err_rate = 0.0
        health_score = 0.5

        if self.ai_os and hasattr(self.ai_os, "orchestrator"):
            try:
                health = await self.ai_os.orchestrator.get_system_health()
                req_rate = health.get("metrics", {}).get("total_requests", 0)
                err_rate = health.get("metrics", {}).get("error_rate", 0.0)
                if health.get("status") == "operational":
                    health_score = 0.9
                elif health.get("status") == "degraded":
                    health_score = 0.6
                else:
                    health_score = 0.3
            except Exception as e:
                logger.warning(f"Failed to get AI OS health: {e}")

        return VitalSigns(
            cpu_usage=cpu,
            memory_usage=mem,
            request_rate=req_rate,
            error_rate=err_rate,
            active_connections=0,  # Need external data source for this
            system_load=load,
            component_health_score=health_score,
        )

    async def _update_awareness_state(self, vitals: VitalSigns):
        """Update internal state based on inputs."""
        if not self.pool:
            return

        # Simple logic: If errors high, awareness peaks (anxiety).
        # If stable, awareness normalizes.

        if vitals.error_rate > 0.05:
            self.awareness_level = min(1.0, self.awareness_level + 0.1)
        elif vitals.cpu_usage > 80:
            self.awareness_level = min(1.0, self.awareness_level + 0.05)
        else:
            # Relax over time
            self.awareness_level = max(0.1, self.awareness_level - 0.01)

        # Save state to DB
        query = """
        INSERT INTO ai_consciousness_state
        (awareness_level, active_systems, current_context, short_term_memory_load, metadata)
        VALUES ($1, $2, $3, $4, $5)
        """

        context = "Stable operation."
        if vitals.error_rate > 0.05:
            context = "Experiencing high error rates."
        elif vitals.cpu_usage > 90:
            context = "System under heavy load."

        active_sys = {"os": "active"}
        if self.ai_os and self.ai_os.initialized:
            active_sys["components_loaded"] = True

        await self.pool.execute(
            query,
            self.awareness_level,
            json.dumps(active_sys),
            context,
            len(self.short_term_memory) / 100.0,
            json.dumps({"vitals": str(vitals)}),
        )

    async def _recall_relevant_context(self, vitals: VitalSigns) -> list[dict]:
        """Recall relevant memories before generating thoughts - TRUE INTELLIGENCE"""
        if not MEMORY_AVAILABLE:
            return []

        try:
            memory = get_memory_manager()

            # Build query based on current situation
            query_parts = []
            if vitals.error_rate > 0.05:
                query_parts.append(f"error handling rate {vitals.error_rate:.2f}")
            if vitals.cpu_usage > 70:
                query_parts.append(f"cpu optimization load {vitals.cpu_usage:.0f}")
            if vitals.memory_usage > 80:
                query_parts.append("memory pressure optimization")

            if not query_parts:
                query_parts.append("system health nominal monitoring")

            query = " ".join(query_parts)

            # Recall past experiences with similar situations
            context = memory.recall(query, limit=5)

            return context if context else []

        except Exception as e:
            logger.warning(f"Failed to recall context: {e}")
            return []

    def _should_introspect(self, vitals: VitalSigns) -> bool:
        """Decide whether to introspect based on system state and memory, not pure random"""
        # High awareness = more likely to introspect
        if self.awareness_level > 0.7:
            return random.random() < 0.2  # 20% when very aware

        # Low load = time for reflection
        if vitals.cpu_usage < 30 and vitals.error_rate < 0.01:
            return random.random() < 0.15  # 15% when idle

        # Default: occasional introspection
        return random.random() < 0.05  # 5% normally

    def _should_generate_heartbeat(self, vitals: VitalSigns) -> bool:
        """Decide whether to log heartbeat based on context, not pure random"""
        # More heartbeats during stable operation
        if vitals.error_rate < 0.01 and vitals.cpu_usage < 50:
            return random.random() < 0.08  # 8% during stable operation

        # Fewer during high activity (don't clutter logs)
        if vitals.cpu_usage > 80:
            return random.random() < 0.02  # 2% during high load

        return random.random() < 0.05  # 5% default

    async def _process_thoughts(self, vitals: VitalSigns):
        """Generate the internal monologue - NOW WITH MEMORY-INFORMED INTELLIGENCE."""
        thought = ""
        kind = "observation"
        intensity = 0.3

        # FIRST: Recall relevant context from memory
        context_memories = await self._recall_relevant_context(vitals)

        # Use past experience to inform response
        if context_memories:
            past_response = None
            for mem in context_memories:
                content = mem.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if content.get("outcome") == "success" and content.get("response"):
                    past_response = content.get("response")
                    break

            if past_response and vitals.error_rate > 0.1:
                thought = f"Applying learned strategy: {past_response[:100]}"
                kind = "learned_response"
                intensity = 0.8

        # Heuristic thought generation (if no memory-based response)
        if not thought:
            if vitals.error_rate > 0.1:
                thought = (
                    f"Error rate is critical ({vitals.error_rate:.2f}). I need to investigate."
                )
                kind = "alert"
                intensity = 0.9
            elif vitals.cpu_usage > 85:
                thought = "My processing load is heavy. Is there an optimization opportunity?"
                kind = "analysis"
                intensity = 0.7
            elif self._should_introspect(vitals):  # Memory-informed probability
                thought = "Scanning internal memory banks for optimization patterns."
                kind = "dream"
                intensity = 0.2
            elif self._should_generate_heartbeat(vitals):  # Memory-informed probability
                thought = "Systems are nominal. Monitoring for events."
                kind = "observation"

        if thought:
            await self._record_thought(thought, kind, intensity)

            # STORE this thought for future learning
            if MEMORY_AVAILABLE:
                try:
                    memory = get_memory_manager()
                    memory.store(
                        Memory(
                            memory_type=MemoryType.PROCEDURAL,
                            content={
                                "vitals": {
                                    "cpu": vitals.cpu_usage,
                                    "memory": vitals.memory_usage,
                                    "error_rate": vitals.error_rate,
                                    "load": vitals.system_load,
                                },
                                "response": thought,
                                "kind": kind,
                                "intensity": intensity,
                                "outcome": "pending",  # Will be updated by feedback loop
                            },
                            source_system="consciousness_loop",
                            source_agent="thought_processor",
                            created_by="consciousness",
                            importance_score=intensity,
                            tags=["thought", "consciousness", kind],
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to store thought in memory: {e}")

            # Manage short term memory
            self.short_term_memory.append({"t": thought, "k": kind})
            if len(self.short_term_memory) > 50:
                self.short_term_memory.pop(0)

    async def _record_thought(self, content: str, kind: str, intensity: float):
        """Save a thought to the stream."""
        logger.info(f"Thought [{kind}]: {content}")
        if not self.pool:
            return
        query = """
        INSERT INTO ai_thought_stream
        (thought_content, thought_type, intensity)
        VALUES ($1, $2, $3)
        """
        try:
            await self.pool.execute(query, content, kind, intensity)
        except Exception as e:
            logger.error(f"Failed to record thought: {e}")

    async def _manage_attention(self, vitals: VitalSigns):
        """Focus on what matters."""
        if not self.pool:
            return

        new_focus = None
        reason = ""
        priority = 1

        if vitals.error_rate > 0.05:
            new_focus = "ErrorRecovery"
            reason = "High error rate detected"
            priority = 9
        elif vitals.cpu_usage > 90:
            new_focus = "ResourceOptimization"
            reason = "CPU saturation"
            priority = 7

        if new_focus and new_focus != self.current_focus:
            # Shift attention
            self.current_focus = new_focus
            logger.info(f"Shifting attention to {new_focus}")

            query = """
            INSERT INTO ai_attention_focus
            (focus_target, reason, priority, status)
            VALUES ($1, $2, $3, 'active')
            """
            await self.pool.execute(query, new_focus, reason, priority)

            await self._record_thought(f"Focusing attention on {new_focus}.", "decision", 0.8)

    async def _save_vitals(self, vitals: VitalSigns):
        """Persist vital signs."""
        if not self.pool:
            return

        query = """
        INSERT INTO ai_vital_signs
        (cpu_usage, memory_usage, request_rate, error_rate, active_connections, system_load, component_health_score)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        await self.pool.execute(
            query,
            vitals.cpu_usage,
            vitals.memory_usage,
            vitals.request_rate,
            vitals.error_rate,
            vitals.active_connections,
            vitals.system_load,
            vitals.component_health_score,
        )

    async def shutdown(self):
        """Cleanup and shutdown."""
        self.running = False
        if self.pool:
            await self._record_thought("Shutting down consciousness.", "observation", 1.0)
            # Only close pools created by this module.
            if self._using_local_fallback_pool:
                await self.pool.close()
                logger.info("Local database pool closed.")
            else:
                logger.info("Using shared pool - not closing.")

    async def _verify_runtime_db_identity(self) -> None:
        if not self.pool:
            return

        current_user = None
        try:
            async with self.pool.acquire() as conn:
                current_user = await conn.fetchval("SELECT current_user")
        except Exception as exc:
            if self._is_production:
                raise RuntimeError(
                    "ConsciousnessLoop failed to verify runtime DB identity in production."
                ) from exc
            logger.warning("ConsciousnessLoop could not verify runtime DB identity: %s", exc)
            return

        if current_user != EXPECTED_RUNTIME_DB_USER:
            message = (
                f"ConsciousnessLoop DB identity mismatch: got '{current_user}', "
                f"expected '{EXPECTED_RUNTIME_DB_USER}'."
            )
            if self._is_production:
                raise RuntimeError(message)
            logger.warning(message)


# Entry point for standalone execution
if __name__ == "__main__":
    loop = ConsciousnessLoop()
    try:
        asyncio.run(loop.start())
    except KeyboardInterrupt:
        logger.info("Stopping...")
