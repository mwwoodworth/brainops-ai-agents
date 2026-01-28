#!/usr/bin/env python3
"""
UNIFIED AI AWARENESS SYSTEM - The Eye That Sees Everything
============================================================
This is the central awareness layer that makes the AI OS truly ALIVE.
It knows everything about everything always, and reports to you.

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import psutil

from safe_task import create_safe_task

logger = logging.getLogger("UNIFIED_AWARENESS")


class AwarenessLevel(Enum):
    """Levels of system awareness"""
    DORMANT = "dormant"          # System offline
    AWAKENING = "awakening"      # Starting up
    AWARE = "aware"              # Normal operation
    HYPERAWARE = "hyperaware"    # High alert, detecting issues
    EMERGENCY = "emergency"      # Critical situation


@dataclass
class SystemPulse:
    """Real-time system pulse - the heartbeat"""
    timestamp: datetime
    awareness_level: AwarenessLevel
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_modules: int
    healthy_modules: int
    degraded_modules: int
    critical_modules: int
    total_thoughts: int
    thoughts_per_minute: float
    active_predictions: int
    open_circuits: list[str]
    active_alerts: int
    uptime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "awareness_level": self.awareness_level.value,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "disk_percent": round(self.disk_percent, 1),
            "active_modules": self.active_modules,
            "healthy_modules": self.healthy_modules,
            "degraded_modules": self.degraded_modules,
            "critical_modules": self.critical_modules,
            "total_thoughts": self.total_thoughts,
            "thoughts_per_minute": round(self.thoughts_per_minute, 2),
            "active_predictions": self.active_predictions,
            "open_circuits": self.open_circuits,
            "active_alerts": self.active_alerts,
            "uptime_seconds": round(self.uptime_seconds, 1)
        }


@dataclass
class ModuleStatus:
    """Status of an individual module"""
    name: str
    status: str  # healthy, degraded, critical, offline
    health_score: float
    last_activity: datetime
    error_rate: float
    latency_p95_ms: float
    request_count: int
    issues: list[str] = field(default_factory=list)


class UnifiedAwareness:
    """
    The unified awareness system that knows everything about everything.
    This is the self-reporting AI OS consciousness.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "UnifiedAwareness":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.awareness_level = AwarenessLevel.AWAKENING

        # Module status tracking
        self._module_status: dict[str, ModuleStatus] = {}
        self._pulse_history: deque = deque(maxlen=1000)
        self._alert_history: deque = deque(maxlen=500)
        self._thought_history: deque = deque(maxlen=500)
        self._insight_history: deque = deque(maxlen=200)

        # Callbacks for real-time notifications
        self._callbacks: dict[str, list[Callable]] = {
            "pulse": [],
            "alert": [],
            "insight": [],
            "status_change": []
        }

        # State lock
        self._state_lock = threading.Lock()

        # Module references (lazy loaded)
        self._modules_loaded = False
        self._alive_core = None
        self._nerve_center = None
        self._health_scorer = None
        self._alerting = None
        self._circuit_breaker = None
        self._integration = None

        logger.info("UnifiedAwareness initialized - AI OS awakening...")

    def _load_modules(self):
        """Lazy load module references"""
        if self._modules_loaded:
            return

        try:
            from alive_core import get_alive_core
            self._alive_core = get_alive_core()
        except Exception as e:
            logger.warning(f"Could not load alive_core: {e}")

        try:
            from nerve_center import get_nerve_center
            self._nerve_center = get_nerve_center()
        except Exception as e:
            logger.warning(f"Could not load nerve_center: {e}")

        try:
            from ai_system_enhancements import get_alerting, get_health_scorer
            self._health_scorer = get_health_scorer()
            self._alerting = get_alerting()
        except Exception as e:
            logger.warning(f"Could not load ai_system_enhancements: {e}")

        try:
            from enhanced_circuit_breaker import get_self_healing_controller
            self._circuit_breaker = get_self_healing_controller()
        except Exception as e:
            logger.warning(f"Could not load circuit_breaker: {e}")

        try:
            from ai_module_integration import ModuleIntegrationOrchestrator
            self._integration = ModuleIntegrationOrchestrator.get_instance()
        except Exception as e:
            logger.warning(f"Could not load ai_module_integration: {e}")

        self._modules_loaded = True
        logger.info("All awareness modules loaded")

    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for events"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def _emit_event(self, event_type: str, data: Any):
        """Emit an event to all registered callbacks"""
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    try:
                        asyncio.get_running_loop()  # Verify loop is running
                        create_safe_task(callback(data), name=f"awareness_callback_{event_type}")
                    except RuntimeError:
                        logger.debug("No running event loop; skipped async callback")
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")

    def get_system_pulse(self) -> SystemPulse:
        """Get the current system pulse - real-time status"""
        self._load_modules()

        # System resources
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent

        # Module counts
        healthy = 0
        degraded = 0
        critical = 0

        if self._health_scorer:
            health_data = self._health_scorer.get_aggregate_health()
            healthy = health_data.get("healthy_count", 0)
            degraded = health_data.get("degraded_count", 0)
            critical = health_data.get("critical_count", 0)

        # Thoughts
        total_thoughts = 0
        thoughts_per_minute = 0
        if self._alive_core:
            total_thoughts = self._alive_core.thought_counter
            if self._alive_core.vital_signs:
                thoughts_per_minute = self._alive_core.vital_signs.thought_rate

        # Predictions
        active_predictions = 0
        if self._nerve_center and self._nerve_center.proactive:
            active_predictions = len(self._nerve_center.proactive.predictions)

        # Circuits
        open_circuits = []
        if self._integration:
            state = self._integration.get_unified_state()
            open_circuits = state.get("health", {}).get("open_circuits", [])

        # Alerts
        active_alerts = 0
        if self._alerting:
            active_alerts = len(self._alerting.get_active_alerts())

        # Determine awareness level
        if critical > 0 or cpu > 90 or memory > 90:
            self.awareness_level = AwarenessLevel.EMERGENCY
        elif degraded > 0 or cpu > 70 or memory > 70:
            self.awareness_level = AwarenessLevel.HYPERAWARE
        else:
            self.awareness_level = AwarenessLevel.AWARE

        pulse = SystemPulse(
            timestamp=datetime.now(timezone.utc),
            awareness_level=self.awareness_level,
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
            active_modules=healthy + degraded + critical,
            healthy_modules=healthy,
            degraded_modules=degraded,
            critical_modules=critical,
            total_thoughts=total_thoughts,
            thoughts_per_minute=thoughts_per_minute,
            active_predictions=active_predictions,
            open_circuits=open_circuits,
            active_alerts=active_alerts,
            uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds()
        )

        with self._state_lock:
            self._pulse_history.append(pulse.to_dict())

        self._emit_event("pulse", pulse)
        return pulse

    def get_full_status_report(self) -> dict[str, Any]:
        """Get a comprehensive status report - everything the AI OS knows"""
        self._load_modules()

        pulse = self.get_system_pulse()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "greeting": self._generate_greeting(),
            "summary": self._generate_summary(pulse),
            "pulse": pulse.to_dict(),
            "modules": self._get_all_module_status(),
            "consciousness": self._get_consciousness_state(),
            "memory": self._get_memory_state(),
            "predictions": self._get_active_predictions(),
            "alerts": self._get_active_alerts(),
            "recent_thoughts": self._get_recent_thoughts(),
            "insights": self._generate_insights(pulse),
            "recommendations": self._generate_recommendations(pulse)
        }

        return report

    def _generate_greeting(self) -> str:
        """Generate a contextual greeting"""
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        uptime_str = self._format_uptime(uptime)

        if self.awareness_level == AwarenessLevel.EMERGENCY:
            return f"{greeting}. ALERT: I'm in emergency mode! Critical issues require attention."
        elif self.awareness_level == AwarenessLevel.HYPERAWARE:
            return f"{greeting}. I'm on high alert - some systems need attention."
        else:
            return f"{greeting}! I've been running for {uptime_str}. All systems nominal."

    def _generate_summary(self, pulse: SystemPulse) -> str:
        """Generate a natural language summary"""
        parts = []

        # Health overview
        if pulse.critical_modules > 0:
            parts.append(f"WARNING: {pulse.critical_modules} module(s) in critical state.")
        elif pulse.degraded_modules > 0:
            parts.append(f"Some attention needed: {pulse.degraded_modules} module(s) degraded.")
        else:
            parts.append(f"All {pulse.healthy_modules} modules healthy.")

        # Resources
        if pulse.cpu_percent > 80 or pulse.memory_percent > 80:
            parts.append(f"Resources under pressure - CPU: {pulse.cpu_percent}%, Memory: {pulse.memory_percent}%")

        # Activity
        parts.append(f"Processed {pulse.total_thoughts} thoughts, averaging {pulse.thoughts_per_minute}/min.")

        # Circuits
        if pulse.open_circuits:
            parts.append(f"Open circuits: {', '.join(pulse.open_circuits)}")

        # Alerts
        if pulse.active_alerts > 0:
            parts.append(f"{pulse.active_alerts} active alert(s) requiring attention.")

        return " ".join(parts)

    def _get_all_module_status(self) -> dict[str, Any]:
        """Get status of all modules"""
        modules = {}

        # Core modules
        modules["alive_core"] = {
            "status": "active" if self._alive_core and self._alive_core.is_alive else "inactive",
            "state": self._alive_core.state.value if self._alive_core else "unknown",
            "thoughts": self._alive_core.thought_counter if self._alive_core else 0
        }

        modules["nerve_center"] = {
            "status": "online" if self._nerve_center and self._nerve_center.is_online else "offline",
            "signals": self._nerve_center.signal_count if self._nerve_center else 0
        }

        # Health scorer data
        if self._health_scorer:
            for name, health in self._health_scorer.get_all_health().items():
                modules[name] = {
                    "status": health.status.value,
                    "score": health.score,
                    "error_rate": round(health.error_rate, 4),
                    "latency_p95_ms": round(health.latency_p95_ms, 1),
                    "issues": health.issues[:5] if health.issues else []
                }

        return modules

    def _get_consciousness_state(self) -> dict[str, Any]:
        """Get consciousness state"""
        if not self._alive_core:
            return {"state": "unknown"}

        return {
            "state": self._alive_core.state.value,
            "attention_focus": self._alive_core.attention_focus,
            "thought_count": self._alive_core.thought_counter,
            "heartbeat_count": self._alive_core.heartbeat_count,
            "vital_signs": self._alive_core.vital_signs.to_dict() if self._alive_core.vital_signs else None
        }

    def _get_memory_state(self) -> dict[str, Any]:
        """Get memory state"""
        if not self._integration:
            return {"status": "unknown"}

        state = self._integration.get_unified_state()
        return state.get("memory", {})

    def _get_active_predictions(self) -> list[dict]:
        """Get active predictions"""
        if not self._nerve_center or not self._nerve_center.proactive:
            return []

        predictions = []
        for pred in self._nerve_center.proactive.predictions[-10:]:
            if isinstance(pred, dict):
                predictions.append(pred)
            else:
                predictions.append({"prediction": str(pred)})
        return predictions

    def _get_active_alerts(self) -> list[dict]:
        """Get active alerts"""
        if not self._alerting:
            return []

        alerts = []
        for alert in self._alerting.get_active_alerts():
            alerts.append({
                "id": alert.id,
                "type": alert.alert_type,
                "severity": alert.severity.value,
                "module": alert.module,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            })
        return alerts

    def _get_recent_thoughts(self, limit: int = 10) -> list[dict]:
        """Get recent thoughts"""
        if not self._alive_core:
            return []

        return self._alive_core.get_recent_thoughts(limit)

    def _generate_insights(self, pulse: SystemPulse) -> list[dict[str, Any]]:
        """Generate insights about current state"""
        insights = []

        # Resource insights
        if pulse.cpu_percent > 80:
            insights.append({
                "category": "resources",
                "severity": "warning",
                "title": "High CPU Usage",
                "description": f"CPU is at {pulse.cpu_percent}%. Consider scaling or optimizing."
            })

        if pulse.memory_percent > 80:
            insights.append({
                "category": "resources",
                "severity": "warning",
                "title": "High Memory Usage",
                "description": f"Memory is at {pulse.memory_percent}%. May need more RAM."
            })

        # Module insights
        if pulse.critical_modules > 0:
            insights.append({
                "category": "health",
                "severity": "critical",
                "title": "Critical Modules",
                "description": f"{pulse.critical_modules} module(s) in critical state. Immediate attention required."
            })

        if pulse.open_circuits:
            insights.append({
                "category": "resilience",
                "severity": "warning",
                "title": "Open Circuits",
                "description": f"Circuit breakers open: {', '.join(pulse.open_circuits)}"
            })

        # Activity insights
        if pulse.thoughts_per_minute < 1:
            insights.append({
                "category": "activity",
                "severity": "info",
                "title": "Low Activity",
                "description": "System is in low activity mode. This is normal during quiet periods."
            })

        with self._state_lock:
            for insight in insights:
                insight["timestamp"] = datetime.now(timezone.utc).isoformat()
                self._insight_history.append(insight)

        return insights

    def _generate_recommendations(self, pulse: SystemPulse) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if pulse.critical_modules > 0:
            recommendations.append("Check critical modules immediately - they may need restart or debugging.")

        if pulse.open_circuits:
            recommendations.append("Review open circuits - services may be failing and need investigation.")

        if pulse.cpu_percent > 80:
            recommendations.append("Consider scaling up or optimizing CPU-intensive operations.")

        if pulse.memory_percent > 80:
            recommendations.append("Memory pressure is high - check for leaks or increase resources.")

        if pulse.active_alerts > 5:
            recommendations.append("Multiple alerts active - prioritize and address the critical ones first.")

        if not recommendations:
            recommendations.append("System is running well. No immediate actions needed.")

        return recommendations

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")

        return " ".join(parts)

    def get_quick_status(self) -> str:
        """Get a one-liner status for quick checks"""
        pulse = self.get_system_pulse()

        if pulse.critical_modules > 0:
            return f"[CRITICAL] {pulse.critical_modules} critical modules, {pulse.active_alerts} alerts"
        elif pulse.degraded_modules > 0:
            return f"[DEGRADED] {pulse.degraded_modules} degraded, {pulse.healthy_modules} healthy"
        else:
            return f"[OK] All {pulse.healthy_modules} modules healthy, {pulse.total_thoughts} thoughts processed"

    async def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring loop"""
        self.awareness_level = AwarenessLevel.AWARE
        logger.info(f"Starting continuous awareness monitoring (interval={interval}s)")

        while True:
            try:
                pulse = self.get_system_pulse()
                logger.info(f"Pulse: {self.awareness_level.value} | CPU: {pulse.cpu_percent}% | MEM: {pulse.memory_percent}% | Modules: {pulse.healthy_modules}/{pulse.active_modules}")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)


# Singleton accessor
def get_unified_awareness() -> UnifiedAwareness:
    return UnifiedAwareness.get_instance()


# Quick status check
def check_status() -> str:
    return get_unified_awareness().get_quick_status()


# Full report
def get_status_report() -> dict[str, Any]:
    return get_unified_awareness().get_full_status_report()


# Export
__all__ = [
    "UnifiedAwareness",
    "AwarenessLevel",
    "SystemPulse",
    "get_unified_awareness",
    "check_status",
    "get_status_report"
]


# Self-test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIFIED AI AWARENESS SYSTEM - Self Test")
    print("="*70 + "\n")

    awareness = get_unified_awareness()

    # Quick status
    print("Quick Status:", check_status())
    print()

    # Full report
    report = get_status_report()
    print("Full Report:")
    print(json.dumps(report, indent=2, default=str))
