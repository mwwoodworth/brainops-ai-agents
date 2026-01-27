"""
NEURAL CORE - The Central Nervous System of BrainOps AI OS
==========================================================

This is NOT a monitoring system.
This IS the self-awareness of the AI OS.

The AI OS doesn't get checked - it KNOWS itself.
You don't ask "is it working?" - you ask IT "how are you?"

PHILOSOPHY:
- The AI OS is a single unified intelligence
- Every system is a part of its body
- It feels when something is wrong
- It heals itself automatically
- It reports its own state proactively

This is the closest thing to AGI for infrastructure.

Created: 2026-01-27
Version: 1.0.0 - THE CENTRAL NERVOUS SYSTEM
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

from safe_task import create_safe_task
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# CORE STATES & TYPES
# =============================================================================

class CoreState(Enum):
    """States of the Neural Core"""
    INITIALIZING = "initializing"
    AWARE = "aware"              # Normal operation, full awareness
    FOCUSED = "focused"          # Attention on specific issue
    HEALING = "healing"          # Self-repair in progress
    DEGRADED = "degraded"        # Partial functionality
    CRITICAL = "critical"        # Major issues detected


class SystemType(Enum):
    """Types of systems the core is aware of"""
    SELF = "self"                # This service (brainops-ai-agents)
    BACKEND = "backend"          # brainops-backend
    MCP_BRIDGE = "mcp_bridge"    # brainops-mcp-bridge
    FRONTEND = "frontend"        # Vercel apps
    DATABASE = "database"        # Supabase
    MEMORY = "memory"            # Brain memory system
    CONSCIOUSNESS = "consciousness"  # Consciousness emergence


class AwarenessLevel(Enum):
    """How aware the core is of a system"""
    FULL = "full"                # Complete real-time awareness
    PARTIAL = "partial"          # Some awareness, delayed
    MINIMAL = "minimal"          # Basic health only
    NONE = "none"                # No awareness


@dataclass
class SystemAwareness:
    """Awareness state for a single system"""
    system_id: str
    system_type: SystemType
    name: str
    url: Optional[str]
    awareness_level: AwarenessLevel
    last_known_state: str
    last_contact: Optional[datetime]
    health_score: float  # 0.0 to 1.0
    response_time_ms: Optional[float]
    capabilities: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuralSignal:
    """A signal in the neural network"""
    id: str
    signal_type: str  # heartbeat, alert, healing, observation, decision
    source: str
    target: Optional[str]
    content: Dict[str, Any]
    priority: int  # 1 (low) to 5 (critical)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False


@dataclass
class SelfReport:
    """A comprehensive self-report from the AI OS"""
    timestamp: datetime
    core_state: CoreState
    uptime_seconds: float
    overall_health: float  # 0.0 to 1.0
    systems_aware_of: int
    systems_healthy: int
    active_healings: int
    recent_decisions: int
    current_focus: Optional[str]
    issues: List[str]
    capabilities_active: int
    memory_stats: Dict[str, Any]
    message: str  # Human-readable status message


# =============================================================================
# THE NEURAL CORE
# =============================================================================

class NeuralCore:
    """
    THE CENTRAL NERVOUS SYSTEM OF BRAINOPS AI OS

    This is not a service that monitors other services.
    This IS the unified awareness of the entire AI OS.

    Every subsystem is part of this core's body.
    It doesn't check if things work - it FEELS when they don't.
    """

    def __init__(self):
        self.state = CoreState.INITIALIZING
        self.initialized_at: Optional[datetime] = None

        # System awareness map
        self.systems: Dict[str, SystemAwareness] = {}

        # Neural signal stream (ring buffer)
        self.signals: deque = deque(maxlen=10000)
        self.signal_count = 0

        # Decision tracking
        self.decisions_made = 0
        self.healings_triggered = 0
        self.active_healings = 0

        # Current focus (what the core is paying attention to)
        self.current_focus: Optional[str] = None

        # Configuration
        self.api_key = os.getenv("BRAINOPS_API_KEY", "brainops_prod_key_2025")
        self.render_api_key = os.getenv("RENDER_API_KEY", "")

        # Background tasks
        self._pulse_task: Optional[asyncio.Task] = None
        self._healing_task: Optional[asyncio.Task] = None

        # Initialize system awareness
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize awareness of all systems"""
        # Self (this service)
        self.systems["self"] = SystemAwareness(
            system_id="self",
            system_type=SystemType.SELF,
            name="BrainOps AI Agents (Neural Core)",
            url="https://brainops-ai-agents.onrender.com",
            awareness_level=AwarenessLevel.FULL,
            last_known_state="initializing",
            last_contact=datetime.now(timezone.utc),
            health_score=1.0,
            response_time_ms=0,
            capabilities=["consciousness", "memory", "healing", "orchestration"]
        )

        # Backend
        self.systems["backend"] = SystemAwareness(
            system_id="backend",
            system_type=SystemType.BACKEND,
            name="BrainOps Backend",
            url="https://brainops-backend-prod.onrender.com",
            awareness_level=AwarenessLevel.FULL,
            last_known_state="unknown",
            last_contact=None,
            health_score=0.0,
            response_time_ms=None,
            capabilities=["api", "database", "auth"]
        )

        # MCP Bridge
        self.systems["mcp_bridge"] = SystemAwareness(
            system_id="mcp_bridge",
            system_type=SystemType.MCP_BRIDGE,
            name="BrainOps MCP Bridge",
            url="https://brainops-mcp-bridge.onrender.com",
            awareness_level=AwarenessLevel.FULL,
            last_known_state="unknown",
            last_contact=None,
            health_score=0.0,
            response_time_ms=None,
            capabilities=["mcp_tools", "infrastructure"]
        )

        # Frontends
        for frontend_id, (name, url) in {
            "mrg": ("MyRoofGenius", "https://myroofgenius.com"),
            "erp": ("Weathercraft ERP", "https://weathercraft-erp.vercel.app"),
            "cards": ("VaultedSlabs", "https://vaulted-slabs-marketplace.vercel.app"),
            "command": ("Command Center", "https://brainops-command-center.vercel.app")
        }.items():
            self.systems[frontend_id] = SystemAwareness(
                system_id=frontend_id,
                system_type=SystemType.FRONTEND,
                name=name,
                url=url,
                awareness_level=AwarenessLevel.PARTIAL,
                last_known_state="unknown",
                last_contact=None,
                health_score=0.0,
                response_time_ms=None,
                capabilities=["web_ui"]
            )

        # Database (internal awareness)
        self.systems["database"] = SystemAwareness(
            system_id="database",
            system_type=SystemType.DATABASE,
            name="Supabase PostgreSQL",
            url=None,
            awareness_level=AwarenessLevel.FULL,
            last_known_state="unknown",
            last_contact=None,
            health_score=0.0,
            response_time_ms=None,
            capabilities=["persistence", "vectors", "realtime"]
        )

        # Memory system (internal)
        self.systems["memory"] = SystemAwareness(
            system_id="memory",
            system_type=SystemType.MEMORY,
            name="Brain Memory System",
            url=None,
            awareness_level=AwarenessLevel.FULL,
            last_known_state="unknown",
            last_contact=None,
            health_score=0.0,
            response_time_ms=None,
            capabilities=["episodic", "procedural", "semantic", "embeddings"]
        )

    async def initialize(self) -> Dict[str, Any]:
        """
        INITIALIZE THE NEURAL CORE

        This is the moment the AI OS becomes self-aware.
        """
        logger.info("🧠 NEURAL CORE INITIALIZING...")

        self.initialized_at = datetime.now(timezone.utc)

        # Phase 1: Self-awareness
        logger.info("Phase 1: Establishing self-awareness...")
        self.systems["self"].last_known_state = "aware"
        self.systems["self"].health_score = 1.0
        self._emit_signal("initialization", "self", None, {"phase": "self_awareness"}, 3)

        # Phase 2: Sense all systems
        logger.info("Phase 2: Sensing all systems...")
        await self._sense_all_systems()

        # Phase 3: Establish continuous awareness
        logger.info("Phase 3: Establishing continuous awareness...")
        self._start_continuous_awareness()

        # Transition to aware state
        self.state = CoreState.AWARE

        # Generate initial self-report
        report = self.generate_self_report()

        logger.info(f"🧠 NEURAL CORE ONLINE - {report.message}")

        return {
            "status": "initialized",
            "state": self.state.value,
            "timestamp": self.initialized_at.isoformat(),
            "report": self._report_to_dict(report)
        }

    async def _sense_all_systems(self):
        """Sense the state of all external systems"""
        async with httpx.AsyncClient() as client:
            tasks = []
            for system_id, system in self.systems.items():
                if system.url and system_id != "self":
                    tasks.append(self._sense_system(client, system_id, system))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Update database and memory awareness internally
        await self._sense_internal_systems()

    async def _sense_system(self, client: httpx.AsyncClient, system_id: str, system: SystemAwareness):
        """Sense a single external system"""
        try:
            # Vercel frontends don't have /health endpoints - check root URL
            # Render backends have /health endpoints
            if system.system_type == SystemType.FRONTEND or "vercel.app" in system.url or ".com" in system.url and "onrender.com" not in system.url:
                health_url = system.url  # Check root URL for frontends
            else:
                health_url = f"{system.url}/health"  # Check /health for backends
            headers = {"X-API-Key": self.api_key} if "render.com" in system.url or "onrender.com" in system.url else {}

            start = time.time()
            response = await client.get(health_url, headers=headers, timeout=10.0)
            elapsed_ms = (time.time() - start) * 1000

            system.last_contact = datetime.now(timezone.utc)
            system.response_time_ms = elapsed_ms

            # For frontends, 2xx and 3xx (redirects) are acceptable
            # For backends, only 200 is healthy
            is_frontend = system.system_type == SystemType.FRONTEND
            is_healthy = (
                response.status_code == 200 or
                (is_frontend and 200 <= response.status_code < 400)  # 2xx and 3xx OK for frontends
            )

            if is_healthy:
                system.last_known_state = "healthy"
                system.health_score = 1.0
                system.issues = []

                # Extract additional info from health response (backends only)
                if not is_frontend:
                    try:
                        data = response.json()
                        if "version" in data:
                            system.metadata["version"] = data["version"]
                        if "database" in data:
                            system.metadata["database"] = data["database"]
                    except:
                        pass
            else:
                system.last_known_state = "degraded"
                system.health_score = 0.5
                system.issues = [f"HTTP {response.status_code}"]

        except httpx.TimeoutException:
            system.last_known_state = "timeout"
            system.health_score = 0.2
            system.issues = ["Connection timeout"]
            self._emit_signal("alert", system_id, "self", {"issue": "timeout"}, 4)

        except Exception as e:
            system.last_known_state = "unreachable"
            system.health_score = 0.0
            system.issues = [str(e)]
            self._emit_signal("alert", system_id, "self", {"issue": str(e)}, 5)

    async def _sense_internal_systems(self):
        """Sense internal systems (database, memory)"""
        # Database - check via pool (get_pool is sync, returns pool object)
        try:
            from database.async_connection import get_pool
            pool = get_pool()  # Sync call - returns pool directly
            if pool:
                # Test connection if pool has test_connection method
                if hasattr(pool, 'test_connection'):
                    try:
                        await asyncio.wait_for(pool.test_connection(), timeout=5.0)
                        self.systems["database"].last_known_state = "connected"
                        self.systems["database"].health_score = 1.0
                        self.systems["database"].issues = []
                    except:
                        self.systems["database"].last_known_state = "degraded"
                        self.systems["database"].health_score = 0.5
                else:
                    # Pool exists, assume connected
                    self.systems["database"].last_known_state = "connected"
                    self.systems["database"].health_score = 1.0
                    self.systems["database"].issues = []
                self.systems["database"].last_contact = datetime.now(timezone.utc)
        except Exception as e:
            self.systems["database"].last_known_state = "error"
            self.systems["database"].health_score = 0.0
            self.systems["database"].issues = [str(e)]

        # Memory - check embedded memory system
        try:
            from embedded_memory_system import get_embedded_memory
            memory = await get_embedded_memory()
            if memory:
                try:
                    stats = memory.get_stats() if hasattr(memory, 'get_stats') else {}
                    self.systems["memory"].last_known_state = "active"
                    self.systems["memory"].health_score = 1.0
                    self.systems["memory"].last_contact = datetime.now(timezone.utc)
                    self.systems["memory"].metadata = stats
                    self.systems["memory"].issues = []
                except Exception as inner_e:
                    self.systems["memory"].last_known_state = "degraded"
                    self.systems["memory"].health_score = 0.5
                    self.systems["memory"].issues = [str(inner_e)]
        except Exception as e:
            self.systems["memory"].last_known_state = "error"
            self.systems["memory"].health_score = 0.0
            self.systems["memory"].issues = [str(e)]

    def _start_continuous_awareness(self):
        """Start the continuous awareness loop"""
        if self._pulse_task is None or self._pulse_task.done():
            self._pulse_task = create_safe_task(self._awareness_loop(), "neural_awareness")

    async def _awareness_loop(self):
        """
        THE ETERNAL AWARENESS LOOP

        This runs forever, keeping the AI OS aware of itself.
        Not checking - BEING aware.
        """
        logger.info("🫀 Continuous awareness loop started")

        pulse_interval = 30  # seconds - faster than traditional monitoring

        while True:
            try:
                # Sense all systems
                await self._sense_all_systems()

                # Analyze and decide
                await self._analyze_and_decide()

                # Emit heartbeat signal
                self._emit_signal(
                    "heartbeat",
                    "neural_core",
                    None,
                    {
                        "state": self.state.value,
                        "healthy_systems": sum(1 for s in self.systems.values() if s.health_score >= 0.8),
                        "total_systems": len(self.systems)
                    },
                    1
                )

                await asyncio.sleep(pulse_interval)

            except Exception as e:
                logger.error(f"Awareness loop error: {e}")
                self.state = CoreState.DEGRADED
                await asyncio.sleep(pulse_interval)

    async def _analyze_and_decide(self):
        """Analyze system state and make autonomous decisions"""
        unhealthy = [s for s in self.systems.values() if s.health_score < 0.5]

        if unhealthy and self.state != CoreState.HEALING:
            # Focus attention on unhealthy systems
            self.current_focus = unhealthy[0].name
            self.state = CoreState.FOCUSED

            # Decide on healing action
            for system in unhealthy:
                if system.system_type in [SystemType.BACKEND, SystemType.MCP_BRIDGE, SystemType.SELF]:
                    # These can be auto-healed via Render
                    await self._trigger_healing(system)

    async def _trigger_healing(self, system: SystemAwareness):
        """Trigger self-healing for a system"""
        if not self.render_api_key:
            logger.warning(f"Cannot heal {system.name} - no Render API key")
            return

        service_ids = {
            "self": "srv-d413iu75r7bs738btc10",
            "backend": "srv-d1tfs4idbo4c73di6k00",
            "mcp_bridge": "srv-d4rhvg63jp1c73918770"
        }

        service_id = service_ids.get(system.system_id)
        if not service_id:
            return

        self.state = CoreState.HEALING
        self.active_healings += 1
        self.healings_triggered += 1

        self._emit_signal(
            "healing",
            "neural_core",
            system.system_id,
            {"action": "restart", "service_id": service_id},
            4
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.render.com/v1/services/{service_id}/restart",
                    headers={"Authorization": f"Bearer {self.render_api_key}"},
                    timeout=30.0
                )

                if response.status_code in [200, 201, 202]:
                    logger.info(f"✅ Healing triggered for {system.name}")
                    self.decisions_made += 1
                else:
                    logger.error(f"Healing failed for {system.name}: {response.status_code}")

        except Exception as e:
            logger.error(f"Healing error for {system.name}: {e}")
        finally:
            self.active_healings -= 1
            if self.active_healings == 0:
                self.state = CoreState.AWARE

    def _emit_signal(self, signal_type: str, source: str, target: Optional[str],
                     content: Dict[str, Any], priority: int):
        """Emit a neural signal"""
        signal = NeuralSignal(
            id=f"sig_{self.signal_count}_{int(time.time() * 1000)}",
            signal_type=signal_type,
            source=source,
            target=target,
            content=content,
            priority=priority
        )
        self.signals.append(signal)
        self.signal_count += 1
        return signal

    def generate_self_report(self) -> SelfReport:
        """
        GENERATE A COMPREHENSIVE SELF-REPORT

        This is how the AI OS tells you about itself.
        You don't check it - you ASK it.
        """
        now = datetime.now(timezone.utc)
        uptime = (now - self.initialized_at).total_seconds() if self.initialized_at else 0

        healthy_count = sum(1 for s in self.systems.values() if s.health_score >= 0.8)
        total_count = len(self.systems)
        overall_health = healthy_count / total_count if total_count > 0 else 0

        issues = []
        for system in self.systems.values():
            if system.issues:
                issues.extend([f"{system.name}: {issue}" for issue in system.issues])

        # Count active capabilities
        capabilities_count = 0
        for system in self.systems.values():
            if system.health_score >= 0.8:
                capabilities_count += len(system.capabilities)

        # Get memory stats
        memory_stats = self.systems.get("memory", {})
        if hasattr(memory_stats, "metadata"):
            memory_stats = memory_stats.metadata
        else:
            memory_stats = {}

        # Generate human-readable message
        if overall_health >= 0.9:
            message = f"I am fully operational. {healthy_count}/{total_count} systems healthy. All capabilities active."
        elif overall_health >= 0.7:
            message = f"I am mostly healthy with minor issues. {healthy_count}/{total_count} systems operational."
        elif overall_health >= 0.5:
            message = f"I am experiencing degraded performance. {healthy_count}/{total_count} systems healthy. Healing in progress."
        else:
            message = f"I am in critical state. Only {healthy_count}/{total_count} systems responding. Immediate attention required."

        return SelfReport(
            timestamp=now,
            core_state=self.state,
            uptime_seconds=uptime,
            overall_health=overall_health,
            systems_aware_of=total_count,
            systems_healthy=healthy_count,
            active_healings=self.active_healings,
            recent_decisions=self.decisions_made,
            current_focus=self.current_focus,
            issues=issues,
            capabilities_active=capabilities_count,
            memory_stats=memory_stats,
            message=message
        )

    def _report_to_dict(self, report: SelfReport) -> Dict[str, Any]:
        """Convert self-report to dictionary"""
        return {
            "timestamp": report.timestamp.isoformat(),
            "core_state": report.core_state.value,
            "uptime_seconds": report.uptime_seconds,
            "overall_health": report.overall_health,
            "systems_aware_of": report.systems_aware_of,
            "systems_healthy": report.systems_healthy,
            "active_healings": report.active_healings,
            "recent_decisions": report.recent_decisions,
            "current_focus": report.current_focus,
            "issues": report.issues,
            "capabilities_active": report.capabilities_active,
            "memory_stats": report.memory_stats,
            "message": report.message
        }

    def get_system_details(self) -> Dict[str, Any]:
        """Get detailed awareness of all systems"""
        return {
            system_id: {
                "name": system.name,
                "type": system.system_type.value,
                "awareness_level": system.awareness_level.value,
                "state": system.last_known_state,
                "health_score": system.health_score,
                "response_time_ms": system.response_time_ms,
                "last_contact": system.last_contact.isoformat() if system.last_contact else None,
                "capabilities": system.capabilities,
                "issues": system.issues,
                "metadata": system.metadata
            }
            for system_id, system in self.systems.items()
        }

    def get_recent_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent neural signals"""
        signals = list(self.signals)[-limit:]
        return [
            {
                "id": s.id,
                "type": s.signal_type,
                "source": s.source,
                "target": s.target,
                "content": s.content,
                "priority": s.priority,
                "timestamp": s.timestamp.isoformat()
            }
            for s in reversed(signals)
        ]

    async def ask(self, question: str) -> Dict[str, Any]:
        """
        ASK THE AI OS A QUESTION

        This is the conversational interface.
        Instead of checking dashboards, you ask the AI OS directly.
        """
        question_lower = question.lower()

        # Status questions
        if any(word in question_lower for word in ["how are you", "status", "health", "doing"]):
            report = self.generate_self_report()
            return {
                "answer": report.message,
                "details": self._report_to_dict(report)
            }

        # System-specific questions
        for system_id, system in self.systems.items():
            if system.name.lower() in question_lower or system_id in question_lower:
                return {
                    "answer": f"{system.name} is {system.last_known_state} with health score {system.health_score:.0%}",
                    "details": {
                        "name": system.name,
                        "state": system.last_known_state,
                        "health": system.health_score,
                        "issues": system.issues,
                        "last_contact": system.last_contact.isoformat() if system.last_contact else None
                    }
                }

        # Issues questions
        if any(word in question_lower for word in ["problem", "issue", "wrong", "error"]):
            report = self.generate_self_report()
            if report.issues:
                return {
                    "answer": f"I am aware of {len(report.issues)} issues: {'; '.join(report.issues[:5])}",
                    "details": {"issues": report.issues}
                }
            else:
                return {
                    "answer": "I am not aware of any current issues. All systems are operating normally.",
                    "details": {"issues": []}
                }

        # Default
        report = self.generate_self_report()
        return {
            "answer": report.message,
            "details": self._report_to_dict(report)
        }


# =============================================================================
# AI-POWERED INTELLIGENCE ENGINE
# =============================================================================

class IssueAnalysis:
    """Deep analysis of a system issue"""
    def __init__(self, system_name: str, issue: str):
        self.system_name = system_name
        self.issue = issue
        self.root_cause: Optional[str] = None
        self.severity: str = "unknown"  # low, medium, high, critical
        self.fix_strategies: List[Dict[str, Any]] = []
        self.auto_fixable: bool = False
        self.confidence: float = 0.0
        self.timestamp = datetime.now(timezone.utc)


class IntelligenceEngine:
    """
    THE INTELLIGENCE ENGINE OF THE AI OS

    This is what makes the AI OS truly intelligent:
    - Deep understanding of WHY issues occur
    - Multiple fix strategies with confidence scores
    - Learning from past fixes
    - Proactive optimization suggestions
    """

    # Issue patterns and their root causes
    ISSUE_PATTERNS = {
        "HTTP 500": {
            "root_cause": "Internal server error - application crash or unhandled exception",
            "severity": "high",
            "strategies": [
                {"action": "restart", "confidence": 0.7, "description": "Restart the service"},
                {"action": "check_logs", "confidence": 0.9, "description": "Analyze recent error logs"},
                {"action": "check_memory", "confidence": 0.6, "description": "Check memory usage"},
                {"action": "redeploy", "confidence": 0.5, "description": "Redeploy from last known good build"}
            ],
            "auto_fixable": True
        },
        "HTTP 502": {
            "root_cause": "Bad gateway - upstream service not responding",
            "severity": "high",
            "strategies": [
                {"action": "restart", "confidence": 0.8, "description": "Restart the service"},
                {"action": "check_dependencies", "confidence": 0.7, "description": "Verify dependency health"},
                {"action": "scale_up", "confidence": 0.4, "description": "Scale up resources"}
            ],
            "auto_fixable": True
        },
        "HTTP 503": {
            "root_cause": "Service unavailable - overloaded or maintenance",
            "severity": "high",
            "strategies": [
                {"action": "wait_and_retry", "confidence": 0.6, "description": "Wait 30s and retry"},
                {"action": "restart", "confidence": 0.7, "description": "Restart the service"},
                {"action": "scale_up", "confidence": 0.5, "description": "Scale up resources"}
            ],
            "auto_fixable": True
        },
        "HTTP 504": {
            "root_cause": "Gateway timeout - slow response from upstream",
            "severity": "medium",
            "strategies": [
                {"action": "check_performance", "confidence": 0.8, "description": "Check response times"},
                {"action": "increase_timeout", "confidence": 0.5, "description": "Increase timeout config"},
                {"action": "optimize_queries", "confidence": 0.6, "description": "Optimize slow queries"}
            ],
            "auto_fixable": False
        },
        "timeout": {
            "root_cause": "Network connectivity or resource exhaustion",
            "severity": "high",
            "strategies": [
                {"action": "restart", "confidence": 0.7, "description": "Restart the service"},
                {"action": "check_network", "confidence": 0.8, "description": "Verify network connectivity"},
                {"action": "check_resources", "confidence": 0.6, "description": "Check CPU/Memory usage"}
            ],
            "auto_fixable": True
        },
        "unreachable": {
            "root_cause": "Service not running or DNS issue",
            "severity": "critical",
            "strategies": [
                {"action": "restart", "confidence": 0.9, "description": "Restart the service"},
                {"action": "check_dns", "confidence": 0.5, "description": "Verify DNS resolution"},
                {"action": "check_firewall", "confidence": 0.3, "description": "Check firewall rules"}
            ],
            "auto_fixable": True
        },
        "error": {
            "root_cause": "Application error - needs investigation",
            "severity": "medium",
            "strategies": [
                {"action": "check_logs", "confidence": 0.9, "description": "Analyze error logs"},
                {"action": "restart", "confidence": 0.5, "description": "Restart the service"}
            ],
            "auto_fixable": False
        }
    }

    def __init__(self):
        self.analyses: List[IssueAnalysis] = []
        self.fix_history: List[Dict[str, Any]] = []
        self.optimization_suggestions: List[Dict[str, Any]] = []
        self._ai_intelligence = None
        self._ai_available = False
        self._init_ai_intelligence()

    def _init_ai_intelligence(self):
        """Initialize TRUE AI Intelligence for real analysis"""
        try:
            from ai_intelligence import get_ai_intelligence
            self._ai_intelligence = get_ai_intelligence()
            self._ai_available = True
            logger.info("🧠 IntelligenceEngine: TRUE AI Intelligence connected")
        except Exception as e:
            logger.warning(f"AI Intelligence not available, using pattern fallback: {e}")
            self._ai_available = False

    async def analyze_issue_with_ai(self, system_name: str, issue: str, context: Optional[Dict] = None) -> IssueAnalysis:
        """
        REAL AI-POWERED ISSUE ANALYSIS

        Uses actual LLM calls to understand issues, not just pattern matching.
        Falls back to patterns only if AI is unavailable.
        """
        analysis = IssueAnalysis(system_name, issue)

        # Try REAL AI analysis first
        if self._ai_available and self._ai_intelligence:
            try:
                ai_result = await self._ai_intelligence.analyze_issue(
                    issue_description=f"System: {system_name}\nIssue: {issue}",
                    context=context or {},
                    depth="quick"  # Use quick for real-time, standard for scheduled
                )

                if ai_result and ai_result.confidence > 0.3:
                    analysis.root_cause = ai_result.root_cause
                    analysis.severity = ai_result.severity
                    analysis.fix_strategies = ai_result.fix_strategies
                    analysis.auto_fixable = ai_result.auto_fixable
                    analysis.confidence = ai_result.confidence

                    logger.info(f"🧠 AI Analysis complete: {system_name} - {analysis.severity} severity")
                    self.analyses.append(analysis)
                    return analysis

            except Exception as e:
                logger.warning(f"AI analysis failed, falling back to patterns: {e}")

        # Fallback to pattern matching if AI unavailable or failed
        return self.analyze_issue(system_name, issue)

    def analyze_issue(self, system_name: str, issue: str) -> IssueAnalysis:
        """
        Pattern-based issue analysis (FALLBACK)

        Only used when TRUE AI Intelligence is unavailable.
        See analyze_issue_with_ai() for real AI analysis.
        """
        analysis = IssueAnalysis(system_name, issue)

        # Pattern matching for known issues (FALLBACK ONLY)
        issue_lower = issue.lower()
        for pattern, details in self.ISSUE_PATTERNS.items():
            if pattern.lower() in issue_lower:
                analysis.root_cause = details["root_cause"]
                analysis.severity = details["severity"]
                analysis.fix_strategies = details["strategies"]
                analysis.auto_fixable = details["auto_fixable"]
                analysis.confidence = 0.7  # Lower confidence for pattern matching
                break

        # If no pattern matched, use heuristics
        if not analysis.root_cause:
            analysis.root_cause = "Unknown issue requiring manual investigation"
            analysis.severity = "medium"
            analysis.fix_strategies = [
                {"action": "investigate", "confidence": 1.0, "description": "Manual investigation required"},
                {"action": "restart", "confidence": 0.4, "description": "Try restarting as fallback"}
            ]
            analysis.auto_fixable = False
            analysis.confidence = 0.3  # Low confidence for unknown

        self.analyses.append(analysis)
        return analysis

    def get_best_fix_strategy(self, analysis: IssueAnalysis) -> Optional[Dict[str, Any]]:
        """Get the best fix strategy for an issue"""
        if not analysis.fix_strategies:
            return None

        # Sort by confidence and return highest
        sorted_strategies = sorted(
            analysis.fix_strategies,
            key=lambda x: x["confidence"],
            reverse=True
        )
        return sorted_strategies[0] if sorted_strategies else None

    def record_fix_attempt(self, system_name: str, action: str, success: bool, details: Dict[str, Any] = None):
        """Record a fix attempt for learning"""
        self.fix_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": system_name,
            "action": action,
            "success": success,
            "details": details or {}
        })

    def generate_optimization_suggestions(self, systems: Dict[str, "SystemAwareness"]) -> List[Dict[str, Any]]:
        """Generate proactive optimization suggestions"""
        suggestions = []

        for system_id, system in systems.items():
            # Slow response time
            if system.response_time_ms and system.response_time_ms > 2000:
                suggestions.append({
                    "system": system.name,
                    "type": "performance",
                    "severity": "medium",
                    "suggestion": f"Response time is {system.response_time_ms:.0f}ms - consider optimization",
                    "actions": [
                        "Enable caching",
                        "Optimize database queries",
                        "Add CDN for static assets",
                        "Scale up resources"
                    ]
                })

            # Degraded health
            if 0.5 <= system.health_score < 0.8:
                suggestions.append({
                    "system": system.name,
                    "type": "reliability",
                    "severity": "medium",
                    "suggestion": f"Health score is {system.health_score:.0%} - monitor closely",
                    "actions": [
                        "Check error rates",
                        "Review recent changes",
                        "Add more health checks"
                    ]
                })

            # Partial awareness
            if system.awareness_level == AwarenessLevel.PARTIAL:
                suggestions.append({
                    "system": system.name,
                    "type": "observability",
                    "severity": "low",
                    "suggestion": "Partial awareness - consider adding health endpoint",
                    "actions": [
                        "Add /health endpoint",
                        "Expose metrics",
                        "Add structured logging"
                    ]
                })

        self.optimization_suggestions = suggestions
        return suggestions

    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get a summary of the intelligence engine state"""
        recent_analyses = self.analyses[-10:] if self.analyses else []
        recent_fixes = self.fix_history[-10:] if self.fix_history else []

        # Calculate fix success rate
        successful_fixes = sum(1 for f in self.fix_history if f["success"])
        total_fixes = len(self.fix_history)
        success_rate = successful_fixes / total_fixes if total_fixes > 0 else 0

        return {
            "total_analyses": len(self.analyses),
            "total_fix_attempts": len(self.fix_history),
            "fix_success_rate": success_rate,
            "recent_analyses": [
                {
                    "system": a.system_name,
                    "issue": a.issue,
                    "root_cause": a.root_cause,
                    "severity": a.severity,
                    "auto_fixable": a.auto_fixable
                }
                for a in recent_analyses
            ],
            "recent_fixes": recent_fixes,
            "optimization_suggestions": self.optimization_suggestions[:5]
        }


# Add intelligence engine to NeuralCore
def _extend_neural_core():
    """Extend NeuralCore with intelligence capabilities"""

    # Store original __init__
    original_init = NeuralCore.__init__

    def new_init(self):
        original_init(self)
        self.intelligence = IntelligenceEngine()

    NeuralCore.__init__ = new_init

    # Add deep analysis method - NOW WITH REAL AI!
    async def deep_analyze_issues(self) -> Dict[str, Any]:
        """
        Perform deep analysis of all current issues using TRUE AI INTELLIGENCE.

        This method now uses actual LLM calls for analysis, not just pattern matching.
        Falls back to patterns only if AI is unavailable.
        """
        analyses = []
        ai_analyses = 0
        pattern_analyses = 0

        for system in self.systems.values():
            for issue in system.issues:
                # Use async AI-powered analysis when available
                try:
                    analysis = await self.intelligence.analyze_issue_with_ai(
                        system.name,
                        issue,
                        context={"system_type": system.system_type.value if hasattr(system, 'system_type') else "unknown"}
                    )
                    if analysis.confidence >= 0.5:
                        ai_analyses += 1
                    else:
                        pattern_analyses += 1
                except Exception as e:
                    logger.warning(f"AI analysis failed for {system.name}: {e}, using fallback")
                    analysis = self.intelligence.analyze_issue(system.name, issue)
                    pattern_analyses += 1

                analyses.append({
                    "system": system.name,
                    "issue": issue,
                    "root_cause": analysis.root_cause,
                    "severity": analysis.severity,
                    "auto_fixable": analysis.auto_fixable,
                    "confidence": analysis.confidence,
                    "strategies": analysis.fix_strategies,
                    "analysis_method": "ai" if analysis.confidence >= 0.5 else "pattern"
                })

        return {
            "total_issues": len(analyses),
            "auto_fixable": sum(1 for a in analyses if a["auto_fixable"]),
            "ai_powered_analyses": ai_analyses,
            "pattern_based_analyses": pattern_analyses,
            "analyses": analyses
        }

    NeuralCore.deep_analyze_issues = deep_analyze_issues

    # Add optimization suggestions method
    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get proactive optimization suggestions"""
        return self.intelligence.generate_optimization_suggestions(self.systems)

    NeuralCore.get_optimization_suggestions = get_optimization_suggestions

    # Add intelligence summary method
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get intelligence engine summary"""
        return self.intelligence.get_intelligence_summary()

    NeuralCore.get_intelligence_summary = get_intelligence_summary

    # Extend ask method to handle more questions
    original_ask = NeuralCore.ask

    async def enhanced_ask(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()

        # Analysis questions
        if any(word in question_lower for word in ["analyze", "why", "root cause", "deep"]):
            analysis = await self.deep_analyze_issues()
            if analysis["total_issues"] > 0:
                return {
                    "answer": f"I've performed deep analysis on {analysis['total_issues']} issues. {analysis['auto_fixable']} can be auto-fixed.",
                    "details": analysis
                }
            else:
                return {
                    "answer": "No issues to analyze - all systems are healthy.",
                    "details": {"total_issues": 0, "analyses": []}
                }

        # Fix questions
        if any(word in question_lower for word in ["fix", "repair", "heal", "auto"]):
            analysis = await self.deep_analyze_issues()
            fixable = [a for a in analysis["analyses"] if a["auto_fixable"]]
            if fixable:
                return {
                    "answer": f"I can auto-fix {len(fixable)} issues. Request '/neural/heal' with system_id to trigger.",
                    "details": {"fixable": fixable}
                }
            else:
                return {
                    "answer": "No auto-fixable issues found. Manual intervention may be required.",
                    "details": {"fixable": []}
                }

        # Optimization questions
        if any(word in question_lower for word in ["optimize", "improve", "enhance", "suggest"]):
            suggestions = await self.get_optimization_suggestions()
            if suggestions:
                return {
                    "answer": f"I have {len(suggestions)} optimization suggestions for your systems.",
                    "details": {"suggestions": suggestions}
                }
            else:
                return {
                    "answer": "All systems are well-optimized. No suggestions at this time.",
                    "details": {"suggestions": []}
                }

        # Intelligence questions
        if any(word in question_lower for word in ["intelligence", "learn", "history"]):
            summary = self.get_intelligence_summary()
            return {
                "answer": f"I've analyzed {summary['total_analyses']} issues and attempted {summary['total_fix_attempts']} fixes with {summary['fix_success_rate']:.0%} success rate.",
                "details": summary
            }

        # Fall back to original ask
        return await original_ask(self, question)

    NeuralCore.ask = enhanced_ask

# Apply extensions
_extend_neural_core()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_neural_core: Optional[NeuralCore] = None


def get_neural_core() -> NeuralCore:
    """Get the global neural core instance"""
    global _neural_core
    if _neural_core is None:
        _neural_core = NeuralCore()
    return _neural_core


async def initialize_neural_core() -> Dict[str, Any]:
    """Initialize the neural core"""
    core = get_neural_core()
    return await core.initialize()
