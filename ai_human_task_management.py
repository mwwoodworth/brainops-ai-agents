#!/usr/bin/env python3
"""
AI-Human Task Management System
================================

Complete bidirectional task management between AI agents and human operators.
Enables true collaboration with:
- Human-to-AI task assignment
- AI-to-Human escalation and approval requests
- Real-time progress tracking
- Priority-based routing
- Handoff protocols
- Audit trail

This is a REAL system for business operations, not a demo.

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import json
import os
import logging
import hashlib
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"          # Complete within 1 hour
    MEDIUM = "medium"      # Complete within 4 hours
    LOW = "low"            # Complete within 24 hours
    BACKGROUND = "background"  # No deadline


class TaskStatus(Enum):
    """Task lifecycle status"""
    PENDING = "pending"                # Created, not started
    ASSIGNED = "assigned"              # Assigned to agent/human
    IN_PROGRESS = "in_progress"        # Currently being worked on
    AWAITING_APPROVAL = "awaiting_approval"  # AI needs human approval
    ESCALATED = "escalated"            # Escalated to human
    COMPLETED = "completed"            # Successfully done
    FAILED = "failed"                  # Could not complete
    CANCELLED = "cancelled"            # Cancelled


class TaskType(Enum):
    """Types of tasks"""
    HUMAN_TO_AI = "human_to_ai"        # Human assigns to AI
    AI_TO_HUMAN = "ai_to_human"        # AI escalates to human
    AI_INTERNAL = "ai_internal"        # AI-to-AI task
    APPROVAL_REQUEST = "approval"       # AI requests human approval
    NOTIFICATION = "notification"       # Informational
    SCHEDULED = "scheduled"            # Scheduled task


@dataclass
class Task:
    """A task in the system"""
    id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime

    # Assignment
    created_by: str  # "human:<user_id>" or "agent:<agent_name>"
    assigned_to: Optional[str] = None  # Who is working on it

    # Timeline
    due_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Content
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Tracking
    progress_percent: int = 0
    progress_notes: List[str] = field(default_factory=list)

    # Relations
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)

    # Approval workflow
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approval_notes: Optional[str] = None

    # Metadata
    tenant_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskUpdate:
    """Update to a task"""
    task_id: str
    timestamp: datetime
    update_type: str  # "progress", "status", "escalation", "completion"
    from_entity: str  # Who made the update
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class AIHumanTaskManager:
    """
    Central Task Management System for AI-Human Collaboration

    This is the bridge between human operators and AI agents, enabling:
    - Structured task delegation
    - Progress visibility
    - Escalation handling
    - Approval workflows
    - Complete audit trail
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.updates: Dict[str, List[TaskUpdate]] = defaultdict(list)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Queue by priority for processing
        self.priority_queues: Dict[TaskPriority, List[str]] = {p: [] for p in TaskPriority}

        # Handlers
        self._ai_handlers: Dict[str, Callable] = {}
        self._human_handlers: Dict[str, Callable] = {}

        # Metrics
        self.metrics = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_escalated": 0,
            "avg_completion_time_minutes": 0,
            "ai_handled": 0,
            "human_handled": 0
        }

        logger.info("AI-Human Task Manager initialized")

    # ==================== TASK CREATION ====================

    async def create_task(
        self,
        title: str,
        description: str,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.MEDIUM,
        created_by: str = "system",
        assigned_to: Optional[str] = None,
        payload: Optional[Dict] = None,
        due_at: Optional[datetime] = None,
        requires_approval: bool = False,
        parent_task_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Task:
        """Create a new task"""
        task_id = hashlib.md5(
            f"{title}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        task = Task(
            id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            assigned_to=assigned_to,
            due_at=due_at,
            payload=payload or {},
            requires_approval=requires_approval,
            parent_task_id=parent_task_id,
            tenant_id=tenant_id,
            tags=tags or []
        )

        self.tasks[task_id] = task
        self.priority_queues[priority].append(task_id)
        self.metrics["tasks_created"] += 1

        # If already assigned, set status
        if assigned_to:
            task.status = TaskStatus.ASSIGNED

        # Link to parent
        if parent_task_id and parent_task_id in self.tasks:
            self.tasks[parent_task_id].subtask_ids.append(task_id)

        # Persist to database
        await self._persist_task(task)

        # Notify subscribers
        await self._notify_subscribers(task_id, "created", task)

        logger.info(f"Task created: {task_id} - {title} [{priority.value}]")
        return task

    # ==================== HUMAN-TO-AI DELEGATION ====================

    async def delegate_to_ai(
        self,
        title: str,
        description: str,
        agent_name: str,
        user_id: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        payload: Optional[Dict] = None,
        due_at: Optional[datetime] = None
    ) -> Task:
        """Human delegates a task to an AI agent"""
        task = await self.create_task(
            title=title,
            description=description,
            task_type=TaskType.HUMAN_TO_AI,
            priority=priority,
            created_by=f"human:{user_id}",
            assigned_to=f"agent:{agent_name}",
            payload=payload,
            due_at=due_at
        )

        # Trigger AI agent execution
        await self._dispatch_to_ai_agent(task)

        return task

    async def _dispatch_to_ai_agent(self, task: Task):
        """Dispatch task to the assigned AI agent"""
        if not task.assigned_to or not task.assigned_to.startswith("agent:"):
            return

        agent_name = task.assigned_to.split(":")[1]

        # Update status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)

        try:
            # Get agent executor
            from agent_executor import get_agent_executor
            executor = get_agent_executor()

            # Execute the task
            result = await executor.execute(
                agent_name,
                {
                    "type": "delegated_task",
                    "task_id": task.id,
                    "title": task.title,
                    "description": task.description,
                    **task.payload
                }
            )

            # Handle result
            if result.get("status") in ["completed", "ok", "success"]:
                await self.complete_task(task.id, result, completed_by=task.assigned_to)
            else:
                task.status = TaskStatus.FAILED
                task.error = result.get("error", "Task execution failed")

        except Exception as e:
            logger.error(f"AI agent dispatch failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)

            # Escalate to human
            await self.escalate_to_human(
                task.id,
                reason=f"AI agent failed: {e}",
                agent_name=agent_name
            )

    # ==================== AI-TO-HUMAN ESCALATION ====================

    async def escalate_to_human(
        self,
        task_id: str,
        reason: str,
        agent_name: str,
        suggested_action: Optional[str] = None,
        urgency: TaskPriority = TaskPriority.HIGH
    ) -> Task:
        """AI escalates a task to human attention"""
        original_task = self.tasks.get(task_id)

        escalation_task = await self.create_task(
            title=f"[ESCALATED] {original_task.title if original_task else 'Unknown task'}",
            description=f"Agent {agent_name} escalated this task.\n\nReason: {reason}",
            task_type=TaskType.AI_TO_HUMAN,
            priority=urgency,
            created_by=f"agent:{agent_name}",
            payload={
                "original_task_id": task_id,
                "escalation_reason": reason,
                "suggested_action": suggested_action,
                "original_payload": original_task.payload if original_task else {}
            },
            tags=["escalated", "needs_human_attention"]
        )

        # Update original task
        if original_task:
            original_task.status = TaskStatus.ESCALATED
            self._add_update(task_id, "escalation", agent_name, f"Escalated: {reason}")

        self.metrics["tasks_escalated"] += 1

        # Notify human operators
        await self._notify_human_operators(escalation_task)

        return escalation_task

    async def request_approval(
        self,
        title: str,
        description: str,
        agent_name: str,
        action_to_approve: Dict[str, Any],
        risk_level: str = "medium"
    ) -> Task:
        """AI requests human approval for an action"""
        task = await self.create_task(
            title=f"[APPROVAL NEEDED] {title}",
            description=description,
            task_type=TaskType.APPROVAL_REQUEST,
            priority=TaskPriority.HIGH if risk_level == "high" else TaskPriority.MEDIUM,
            created_by=f"agent:{agent_name}",
            payload={
                "action_to_approve": action_to_approve,
                "risk_level": risk_level,
                "agent": agent_name
            },
            requires_approval=True,
            tags=["approval_required", f"risk_{risk_level}"]
        )

        return task

    async def approve_task(
        self,
        task_id: str,
        approved_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """Human approves a task"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.approved_by = approved_by
        task.approval_notes = notes
        task.status = TaskStatus.ASSIGNED  # Ready for AI to proceed

        self._add_update(task_id, "approval", approved_by, f"Approved: {notes or 'No notes'}")

        # Notify the requesting agent
        await self._notify_subscribers(task_id, "approved", task)

        return True

    async def reject_task(
        self,
        task_id: str,
        rejected_by: str,
        reason: str
    ) -> bool:
        """Human rejects a task"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.CANCELLED
        task.approval_notes = f"REJECTED: {reason}"

        self._add_update(task_id, "rejection", rejected_by, f"Rejected: {reason}")

        return True

    # ==================== PROGRESS TRACKING ====================

    async def update_progress(
        self,
        task_id: str,
        percent: int,
        note: str,
        updated_by: str
    ):
        """Update task progress"""
        task = self.tasks.get(task_id)
        if not task:
            return

        task.progress_percent = min(100, max(0, percent))
        task.progress_notes.append(f"[{datetime.now().strftime('%H:%M')}] {note}")

        self._add_update(task_id, "progress", updated_by, note, {"percent": percent})

        # Notify watchers
        await self._notify_subscribers(task_id, "progress", {"percent": percent, "note": note})

    async def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
        completed_by: str
    ):
        """Mark task as completed"""
        task = self.tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.result = result
        task.progress_percent = 100

        # Calculate metrics
        if task.started_at:
            duration_minutes = (task.completed_at - task.started_at).total_seconds() / 60
            self._update_avg_completion_time(duration_minutes)

        if completed_by.startswith("agent:"):
            self.metrics["ai_handled"] += 1
        else:
            self.metrics["human_handled"] += 1

        self.metrics["tasks_completed"] += 1

        self._add_update(task_id, "completion", completed_by, "Task completed")
        await self._notify_subscribers(task_id, "completed", result)
        await self._persist_task(task)

        # Complete parent if all subtasks done
        if task.parent_task_id:
            await self._check_parent_completion(task.parent_task_id)

    # ==================== TASK QUERIES ====================

    def get_pending_for_human(self, user_id: Optional[str] = None) -> List[Task]:
        """Get tasks awaiting human attention"""
        pending = []
        for task in self.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.ESCALATED, TaskStatus.AWAITING_APPROVAL]:
                if task.task_type in [TaskType.AI_TO_HUMAN, TaskType.APPROVAL_REQUEST]:
                    if not user_id or task.assigned_to == f"human:{user_id}":
                        pending.append(task)

        # Sort by priority and time
        return sorted(pending, key=lambda t: (
            list(TaskPriority).index(t.priority),
            t.created_at
        ))

    def get_pending_for_ai(self, agent_name: Optional[str] = None) -> List[Task]:
        """Get tasks awaiting AI processing"""
        pending = []
        for task in self.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                if task.task_type in [TaskType.HUMAN_TO_AI, TaskType.AI_INTERNAL, TaskType.SCHEDULED]:
                    if not agent_name or task.assigned_to == f"agent:{agent_name}":
                        pending.append(task)

        return sorted(pending, key=lambda t: (
            list(TaskPriority).index(t.priority),
            t.created_at
        ))

    def get_task_history(self, task_id: str) -> List[TaskUpdate]:
        """Get complete history of task updates"""
        return self.updates.get(task_id, [])

    def get_metrics(self) -> Dict[str, Any]:
        """Get task management metrics"""
        active_count = len([t for t in self.tasks.values() if t.status in [
            TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS
        ]])

        return {
            **self.metrics,
            "active_tasks": active_count,
            "total_tasks": len(self.tasks),
            "pending_human_tasks": len(self.get_pending_for_human()),
            "pending_ai_tasks": len(self.get_pending_for_ai()),
            "completion_rate": (
                self.metrics["tasks_completed"] / max(self.metrics["tasks_created"], 1) * 100
            )
        }

    # ==================== SUBSCRIPTIONS ====================

    def subscribe(self, task_id: str, callback: Callable):
        """Subscribe to task updates"""
        self.subscribers[task_id].append(callback)

    def subscribe_to_type(self, task_type: TaskType, callback: Callable):
        """Subscribe to all tasks of a type"""
        self.subscribers[f"type:{task_type.value}"].append(callback)

    async def _notify_subscribers(self, task_id: str, event: str, data: Any):
        """Notify subscribers of an event"""
        task = self.tasks.get(task_id)

        # Direct subscribers
        for callback in self.subscribers.get(task_id, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, task_id, data)
                else:
                    callback(event, task_id, data)
            except Exception as e:
                logger.warning(f"Subscriber notification failed: {e}")

        # Type subscribers
        if task:
            for callback in self.subscribers.get(f"type:{task.task_type.value}", []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event, task_id, data)
                    else:
                        callback(event, task_id, data)
                except Exception as e:
                    logger.warning(f"Type subscriber notification failed: {e}")

    async def _notify_human_operators(self, task: Task):
        """Notify human operators of a task needing attention"""
        # This would integrate with notification systems (email, Slack, etc.)
        logger.info(f"HUMAN ATTENTION NEEDED: {task.title} [{task.priority.value}]")

        # Store in database for dashboard visibility
        await self._persist_task(task)

    # ==================== INTERNAL HELPERS ====================

    def _add_update(
        self,
        task_id: str,
        update_type: str,
        from_entity: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """Add an update to task history"""
        update = TaskUpdate(
            task_id=task_id,
            timestamp=datetime.now(timezone.utc),
            update_type=update_type,
            from_entity=from_entity,
            message=message,
            data=data or {}
        )
        self.updates[task_id].append(update)

    def _update_avg_completion_time(self, duration_minutes: float):
        """Update average completion time"""
        current = self.metrics["avg_completion_time_minutes"]
        count = self.metrics["tasks_completed"]
        if count > 0:
            self.metrics["avg_completion_time_minutes"] = (
                (current * (count - 1) + duration_minutes) / count
            )
        else:
            self.metrics["avg_completion_time_minutes"] = duration_minutes

    async def _check_parent_completion(self, parent_id: str):
        """Check if parent task can be completed"""
        parent = self.tasks.get(parent_id)
        if not parent:
            return

        all_done = all(
            self.tasks[sid].status == TaskStatus.COMPLETED
            for sid in parent.subtask_ids
            if sid in self.tasks
        )

        if all_done and parent.status == TaskStatus.IN_PROGRESS:
            # Aggregate subtask results
            subtask_results = [
                self.tasks[sid].result
                for sid in parent.subtask_ids
                if sid in self.tasks
            ]

            await self.complete_task(
                parent_id,
                {"subtask_results": subtask_results},
                completed_by="system"
            )

    async def _persist_task(self, task: Task):
        """Persist task to database"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO ai_human_tasks (
                            id, title, description, task_type, priority, status,
                            created_at, created_by, assigned_to, due_at,
                            started_at, completed_at, payload, result, error,
                            progress_percent, requires_approval, approved_by,
                            parent_task_id, tenant_id, tags
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                        ON CONFLICT (id) DO UPDATE SET
                            status = EXCLUDED.status,
                            completed_at = EXCLUDED.completed_at,
                            result = EXCLUDED.result,
                            error = EXCLUDED.error,
                            progress_percent = EXCLUDED.progress_percent,
                            approved_by = EXCLUDED.approved_by
                    """,
                        task.id,
                        task.title,
                        task.description,
                        task.task_type.value,
                        task.priority.value,
                        task.status.value,
                        task.created_at,
                        task.created_by,
                        task.assigned_to,
                        task.due_at,
                        task.started_at,
                        task.completed_at,
                        json.dumps(task.payload),
                        json.dumps(task.result) if task.result else None,
                        task.error,
                        task.progress_percent,
                        task.requires_approval,
                        task.approved_by,
                        task.parent_task_id,
                        task.tenant_id,
                        task.tags
                    )
        except Exception as e:
            logger.warning(f"Task persistence failed: {e}")


# Singleton instance
_task_manager: Optional[AIHumanTaskManager] = None


def get_task_manager() -> AIHumanTaskManager:
    """Get or create the task manager"""
    global _task_manager
    if _task_manager is None:
        _task_manager = AIHumanTaskManager()
    return _task_manager


# Agent wrapper for integration
class AIHumanTaskAgent:
    """Agent wrapper for AI-Human Task Management"""

    def __init__(self):
        self.name = "AIHumanTaskManager"
        self.agent_type = "ai_human_task_management"

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task management operations"""
        manager = get_task_manager()
        action = task.get("action", "status")

        if action == "delegate_to_ai":
            result = await manager.delegate_to_ai(
                title=task.get("title"),
                description=task.get("description"),
                agent_name=task.get("agent_name"),
                user_id=task.get("user_id", "unknown"),
                payload=task.get("payload")
            )
            return {"status": "delegated", "task_id": result.id}

        elif action == "escalate":
            result = await manager.escalate_to_human(
                task_id=task.get("task_id"),
                reason=task.get("reason"),
                agent_name=task.get("agent_name", "unknown")
            )
            return {"status": "escalated", "escalation_id": result.id}

        elif action == "approve":
            success = await manager.approve_task(
                task_id=task.get("task_id"),
                approved_by=task.get("approved_by"),
                notes=task.get("notes")
            )
            return {"status": "approved" if success else "failed"}

        elif action == "pending_for_human":
            tasks = manager.get_pending_for_human(task.get("user_id"))
            return {
                "status": "ok",
                "count": len(tasks),
                "tasks": [{"id": t.id, "title": t.title, "priority": t.priority.value}
                         for t in tasks[:20]]
            }

        elif action == "metrics":
            return {"status": "ok", "metrics": manager.get_metrics()}

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


if __name__ == "__main__":
    async def main():
        print("=" * 70)
        print("AI-HUMAN TASK MANAGEMENT SYSTEM")
        print("=" * 70)

        manager = get_task_manager()

        # Human delegates to AI
        print("\n1. Human delegating task to AI...")
        task1 = await manager.delegate_to_ai(
            title="Analyze Q4 sales data",
            description="Analyze sales data and generate insights",
            agent_name="DataAnalyst",
            user_id="user_123",
            priority=TaskPriority.HIGH
        )
        print(f"   Created task: {task1.id}")

        # AI requests approval
        print("\n2. AI requesting approval...")
        task2 = await manager.request_approval(
            title="Deploy new pricing model",
            description="The new pricing model will affect all customer quotes",
            agent_name="PricingAgent",
            action_to_approve={"model": "dynamic_v2", "effective_date": "2025-01-01"},
            risk_level="high"
        )
        print(f"   Approval task: {task2.id}")

        # Get metrics
        print("\n3. Task Management Metrics:")
        metrics = manager.get_metrics()
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        print("\nAI-Human Task Management System Ready!")

    asyncio.run(main())
