"""
Background Task Monitoring API
==============================
Monitors all background tasks (AUREA, Scheduler, etc.) with heartbeats.
No more fire-and-forget - we know if tasks are alive.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor/background", tags=["Background Task Monitoring"])


@dataclass
class TaskHeartbeat:
    """Represents a heartbeat from a background task."""
    task_name: str
    last_heartbeat: datetime
    status: str  # "running", "stopped", "error", "unknown"
    iteration_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackgroundTaskMonitor:
    """
    Monitors all background tasks with heartbeats.
    Tasks register and send periodic heartbeats.
    The monitor detects stale tasks and raises alerts.
    """

    def __init__(self, heartbeat_timeout_seconds: int = 60):
        self._tasks: Dict[str, TaskHeartbeat] = {}
        self._timeout = heartbeat_timeout_seconds
        self._monitor_task: Optional[asyncio.Task] = None
        self._alerts: List[Dict[str, Any]] = []
        self._started = False

    def register_task(self, task_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Register a new background task for monitoring."""
        self._tasks[task_name] = TaskHeartbeat(
            task_name=task_name,
            last_heartbeat=datetime.now(timezone.utc),
            status="registered",
            metadata=metadata or {}
        )
        logger.info(f"📋 Background task registered: {task_name}")

    def heartbeat(self, task_name: str, status: str = "running", metadata: Optional[Dict[str, Any]] = None):
        """Record a heartbeat from a background task."""
        now = datetime.now(timezone.utc)

        if task_name not in self._tasks:
            self.register_task(task_name)

        task = self._tasks[task_name]
        task.last_heartbeat = now
        task.status = status
        task.iteration_count += 1
        if metadata:
            task.metadata.update(metadata)

        logger.debug(f"💓 Heartbeat from {task_name}: {status}")

    def report_error(self, task_name: str, error: str):
        """Report an error from a background task."""
        if task_name in self._tasks:
            self._tasks[task_name].status = "error"
            self._tasks[task_name].last_error = error
            self._add_alert(task_name, "error", error)
        else:
            self.register_task(task_name)
            self._tasks[task_name].status = "error"
            self._tasks[task_name].last_error = error

    def _add_alert(self, task_name: str, alert_type: str, message: str):
        """Add an alert."""
        alert = {
            "task_name": task_name,
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False
        }
        self._alerts.append(alert)
        logger.warning(f"⚠️ Alert for {task_name}: {alert_type} - {message}")

    async def _monitor_loop(self):
        """Background loop that checks for stale tasks."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._check_tasks()
            except asyncio.CancelledError:
                logger.info("Background monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    async def _check_tasks(self):
        """Check all tasks for staleness."""
        now = datetime.now(timezone.utc)
        timeout_delta = timedelta(seconds=self._timeout)

        for task_name, task in self._tasks.items():
            if task.status in ["running", "registered"]:
                age = now - task.last_heartbeat
                if age > timeout_delta:
                    # Task is stale
                    task.status = "stale"
                    self._add_alert(
                        task_name,
                        "stale",
                        f"No heartbeat for {age.seconds} seconds"
                    )

    async def start(self):
        """Start the monitoring loop."""
        if self._started:
            return

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._started = True
        logger.info("✅ Background Task Monitor started")

    async def stop(self):
        """Stop the monitoring loop."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Background monitor task cancelled")
        self._started = False

    def get_status(self) -> Dict[str, Any]:
        """Get the status of all monitored tasks."""
        now = datetime.now(timezone.utc)

        tasks_status = {}
        for name, task in self._tasks.items():
            age = (now - task.last_heartbeat).total_seconds()
            tasks_status[name] = {
                "status": task.status,
                "last_heartbeat": task.last_heartbeat.isoformat(),
                "seconds_since_heartbeat": round(age, 2),
                "iteration_count": task.iteration_count,
                "last_error": task.last_error,
                "is_healthy": age < self._timeout and task.status in ["running", "registered"],
                "metadata": task.metadata
            }

        running = sum(1 for t in tasks_status.values() if t["status"] == "running")
        stale = sum(1 for t in tasks_status.values() if t["status"] == "stale")
        error = sum(1 for t in tasks_status.values() if t["status"] == "error")

        return {
            "monitoring_active": self._started,
            "total_tasks": len(tasks_status),
            "running": running,
            "stale": stale,
            "errors": error,
            "tasks": tasks_status,
            "unresolved_alerts": len([a for a in self._alerts if not a["resolved"]]),
            "timestamp": now.isoformat()
        }

    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get all alerts."""
        if include_resolved:
            return self._alerts[-100:]
        return [a for a in self._alerts if not a["resolved"]][-50:]

    def resolve_alert(self, task_name: str):
        """Resolve all alerts for a task."""
        for alert in self._alerts:
            if alert["task_name"] == task_name:
                alert["resolved"] = True


# Singleton instance
_monitor: Optional[BackgroundTaskMonitor] = None


def get_monitor() -> BackgroundTaskMonitor:
    """Get or create the background task monitor."""
    global _monitor
    if _monitor is None:
        _monitor = BackgroundTaskMonitor(heartbeat_timeout_seconds=120)
    return _monitor


async def start_monitoring():
    """Start the background task monitor."""
    monitor = get_monitor()
    await monitor.start()


# =============================================================================
# ROUTER ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_monitoring_status():
    """Get status of all monitored background tasks."""
    monitor = get_monitor()
    return monitor.get_status()


@router.get("/alerts")
async def get_alerts(include_resolved: bool = False):
    """Get alerts from monitored tasks."""
    monitor = get_monitor()
    return {
        "alerts": monitor.get_alerts(include_resolved),
        "total": len(monitor._alerts)
    }


@router.post("/heartbeat/{task_name}")
async def send_heartbeat(task_name: str, status: str = "running"):
    """Send a heartbeat for a task (for external tasks)."""
    monitor = get_monitor()
    monitor.heartbeat(task_name, status)
    return {"status": "recorded", "task": task_name}


@router.post("/register/{task_name}")
async def register_task(task_name: str):
    """Register a new task for monitoring."""
    monitor = get_monitor()
    monitor.register_task(task_name)
    return {"status": "registered", "task": task_name}


@router.post("/resolve/{task_name}")
async def resolve_alerts(task_name: str):
    """Resolve all alerts for a task."""
    monitor = get_monitor()
    monitor.resolve_alert(task_name)
    return {"status": "resolved", "task": task_name}


# =============================================================================
# AUREA INTEGRATION
# =============================================================================

async def monitor_aurea(app_state):
    """Monitor AUREA orchestration loop with heartbeats."""
    monitor = get_monitor()
    monitor.register_task("aurea_orchestrator", {"type": "orchestration_loop"})

    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds

            aurea = getattr(app_state, "aurea", None)
            if aurea:
                is_running = getattr(aurea, "_orchestrating", False) or getattr(aurea, "is_running", False)
                decision_count = getattr(aurea, "decision_count", 0)

                if is_running:
                    monitor.heartbeat("aurea_orchestrator", "running", {
                        "decision_count": decision_count
                    })
                else:
                    monitor.heartbeat("aurea_orchestrator", "stopped")
            else:
                monitor.heartbeat("aurea_orchestrator", "not_initialized")
        except asyncio.CancelledError:
            break
        except Exception as e:
            monitor.report_error("aurea_orchestrator", str(e))


async def monitor_scheduler(app_state):
    """Monitor Agent Scheduler with heartbeats."""
    monitor = get_monitor()
    monitor.register_task("agent_scheduler", {"type": "job_scheduler"})

    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds

            scheduler = getattr(app_state, "scheduler", None)
            if scheduler:
                is_running = getattr(scheduler, "_running", False) or getattr(scheduler, "running", False)
                job_count = len(getattr(scheduler, "jobs", []))

                if is_running:
                    monitor.heartbeat("agent_scheduler", "running", {
                        "job_count": job_count
                    })
                else:
                    monitor.heartbeat("agent_scheduler", "stopped")
            else:
                monitor.heartbeat("agent_scheduler", "not_initialized")
        except asyncio.CancelledError:
            break
        except Exception as e:
            monitor.report_error("agent_scheduler", str(e))


async def monitor_self_healing(app_state):
    """Monitor Self-Healing Reconciler with heartbeats."""
    monitor = get_monitor()
    monitor.register_task("self_healing_reconciler", {"type": "healing_loop"})

    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds

            reconciler = getattr(app_state, "reconciler", None)
            if reconciler:
                is_running = getattr(reconciler, "is_running", False) or getattr(reconciler, "_running", False)
                heal_count = getattr(reconciler, "heal_count", 0)

                if is_running:
                    monitor.heartbeat("self_healing_reconciler", "running", {
                        "heal_count": heal_count
                    })
                else:
                    monitor.heartbeat("self_healing_reconciler", "stopped")
            else:
                monitor.heartbeat("self_healing_reconciler", "not_initialized")
        except asyncio.CancelledError:
            break
        except Exception as e:
            monitor.report_error("self_healing_reconciler", str(e))


async def start_all_monitoring(app_state):
    """Start monitoring all background tasks."""
    monitor = get_monitor()
    await monitor.start()

    # Start monitoring tasks
    asyncio.create_task(monitor_aurea(app_state))
    asyncio.create_task(monitor_scheduler(app_state))
    asyncio.create_task(monitor_self_healing(app_state))

    logger.info("✅ All background task monitoring started")
