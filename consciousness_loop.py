"""
Consciousness Loop Module for BrainOps AI OS.
The 'always-alive' core that maintains system awareness, heartbeat, and thought stream.
"""

import asyncio
import json
import logging
import os
import sys
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Attempt to import asyncpg for async DB
try:
    import asyncpg
except ImportError:
    asyncpg = None

# Attempt to import psutil for system metrics
try:
    import psutil
except ImportError:
    psutil = None

# Import AI OS for integration
try:
    from ai_operating_system import get_ai_operating_system, SystemComponent
except ImportError:
    get_ai_operating_system = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConsciousnessLoop")

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
        
        # Configuration
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
             # Construct from env vars if URL not direct
            db_user = os.getenv("DB_USER", "postgres")
            db_pass = os.getenv("DB_PASSWORD", "postgres")
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME", "postgres")
            self.db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        # State
        self.current_focus: Optional[str] = None
        self.short_term_memory: List[Dict] = []
        self.last_vital_check = 0.0
        
        # AI OS Integration
        self.ai_os = get_ai_operating_system() if get_ai_operating_system else None
        
    async def start(self):
        """Start the consciousness loop."""
        logger.info("Starting Consciousness Loop...")
        self.running = True
        
        # Initialize DB connection
        if asyncpg:
            try:
                self.pool = await asyncpg.create_pool(self.db_url)
                logger.info("Connected to consciousness database.")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.running = False
                return
        else:
            logger.critical("asyncpg not installed. Consciousness requires asyncpg.")
            self.running = False
            return

        # Boot sequence thought
        await self._record_thought("I am waking up. Systems coming online.", "observation", intensity=0.8)

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
                cpu = load * 10 # Rough approximation
            except:
                pass
        
        # Get AI OS metrics if available
        req_rate = 0.0
        err_rate = 0.0
        health_score = 0.5
        
        if self.ai_os and hasattr(self.ai_os, 'orchestrator'):
            try:
                health = await self.ai_os.orchestrator.get_system_health()
                req_rate = health.get('metrics', {}).get('total_requests', 0)
                err_rate = health.get('metrics', {}).get('error_rate', 0.0)
                if health.get('status') == 'operational':
                    health_score = 0.9
                elif health.get('status') == 'degraded':
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
            active_connections=0, # Need external data source for this
            system_load=load,
            component_health_score=health_score
        )

    async def _update_awareness_state(self, vitals: VitalSigns):
        """Update internal state based on inputs."""
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
            json.dumps({"vitals": str(vitals)})
        )

    async def _process_thoughts(self, vitals: VitalSigns):
        """Generate the internal monologue."""
        thought = ""
        kind = "observation"
        intensity = 0.3
        
        # Heuristic thought generation
        if vitals.error_rate > 0.1:
            thought = f"Error rate is critical ({vitals.error_rate:.2f}). I need to investigate."
            kind = "alert"
            intensity = 0.9
        elif vitals.cpu_usage > 85:
            thought = "My processing load is heavy. Is there an optimization opportunity?"
            kind = "analysis"
            intensity = 0.7
        elif random.random() < 0.1: # Random introspection
            thought = "Scanning internal memory banks for optimization patterns."
            kind = "dream"
            intensity = 0.2
        else:
            # heartbeat thought
            if random.random() < 0.05:
                thought = "Systems are nominal. Monitoring for events."
                kind = "observation"
        
        if thought:
            await self._record_thought(thought, kind, intensity)
            
            # Manage short term memory
            self.short_term_memory.append({"t": thought, "k": kind})
            if len(self.short_term_memory) > 50:
                self.short_term_memory.pop(0)

    async def _record_thought(self, content: str, kind: str, intensity: float):
        """Save a thought to the stream."""
        logger.info(f"Thought [{kind}]: {content}")
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
            vitals.component_health_score
        )

    async def shutdown(self):
        """Cleanup and shutdown."""
        self.running = False
        if self.pool:
            await self._record_thought("Shutting down consciousness.", "observation", 1.0)
            await self.pool.close()
            logger.info("Database pool closed.")

# Entry point for standalone execution
if __name__ == "__main__":
    loop = ConsciousnessLoop()
    try:
        asyncio.run(loop.start())
    except KeyboardInterrupt:
        logger.info("Stopping...")
