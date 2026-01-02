#!/usr/bin/env python3
"""
Distributed Agent Coordination System
Multi-agent orchestration, task distribution, and synchronization
"""

import asyncio
import json
import logging
import os
from urllib.parse import urlparse as _urlparse
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class AgentState(Enum):
    """Possible states for an agent"""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskStatus(Enum):
    """Status of a distributed task"""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CoordinationMode(Enum):
    """Coordination modes for multi-agent work"""
    INDEPENDENT = "independent"  # Agents work independently
    SEQUENTIAL = "sequential"   # Agents work in sequence
    PARALLEL = "parallel"       # Agents work in parallel
    CONSENSUS = "consensus"     # Agents must reach consensus
    LEADER_FOLLOWER = "leader_follower"  # One leader, others follow


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AgentRegistration:
    """Registration information for an agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    capabilities: list[str]
    state: AgentState = AgentState.IDLE
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_task_id: Optional[str] = None
    max_concurrent_tasks: int = 1
    current_task_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """A task to be distributed among agents"""
    task_id: str
    task_type: str
    payload: dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    required_capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGroup:
    """A group of related tasks"""
    group_id: str
    name: str
    mode: CoordinationMode
    tasks: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    leader_agent_id: Optional[str] = None
    participating_agents: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationMessage:
    """Message for inter-agent communication"""
    message_id: str
    from_agent_id: str
    to_agent_id: Optional[str]  # None for broadcast
    message_type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    requires_ack: bool = False
    acked: bool = False


# ============================================================================
# DISTRIBUTED AGENT COORDINATOR
# ============================================================================

class DistributedAgentCoordinator:
    """
    Coordinates multiple AI agents for distributed task execution
    """

    def __init__(self):
        self._initialized = False
        self._db_config = None

        # Agent registry
        self._agents: dict[str, AgentRegistration] = {}

        # Task management
        self._tasks: dict[str, DistributedTask] = {}
        self._task_queue: list[str] = []  # Priority queue of task IDs
        self._task_groups: dict[str, TaskGroup] = {}

        # Message passing
        self._message_queue: dict[str, list[CoordinationMessage]] = defaultdict(list)
        self._broadcast_messages: list[CoordinationMessage] = []

        # Coordination state
        self._locks: dict[str, asyncio.Lock] = {}
        self._leader_elections: dict[str, str] = {}  # group_id -> leader_agent_id
        self._consensus_votes: dict[str, dict[str, Any]] = {}

        # Callbacks
        self._task_callbacks: dict[str, Callable] = {}
        self._event_handlers: dict[str, list[Callable]] = defaultdict(list)

        # Monitoring
        self._stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "messages_sent": 0,
            "agents_registered": 0
        }

        self._main_lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _get_db_config(self) -> dict[str, Any]:
        """Get database configuration lazily with validation"""
        if not self._db_config:
            required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                
        # DATABASE_URL fallback
        _db_url = os.getenv('DATABASE_URL', '')
        if _db_url:
            try:
                _p = _urlparse(_db_url)
                globals().update({'_DB_HOST': _p.hostname, '_DB_NAME': _p.path.lstrip('/'), '_DB_USER': _p.username, '_DB_PASSWORD': _p.password, '_DB_PORT': str(_p.port or 5432)})
            except: pass
        missing = [v for v in required_vars if not os.getenv(v) and not globals().get('_' + v)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

            self._db_config = {
                'host': os.getenv('DB_HOST'),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'port': int(os.getenv('DB_PORT', '5432'))
            }
        return self._db_config

    async def initialize(self):
        """Initialize the coordination system"""
        if self._initialized:
            return

        async with self._main_lock:
            if self._initialized:
                return

            try:
                await self._initialize_database()
                await self._load_persisted_state()
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self._initialized = True
                logger.info("Distributed agent coordination system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize coordination system: {e}")

    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Agent registry table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_agent_registry (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    agent_id VARCHAR(255) UNIQUE NOT NULL,
                    agent_name VARCHAR(255) NOT NULL,
                    agent_type VARCHAR(100) NOT NULL,
                    capabilities JSONB DEFAULT '[]'::jsonb,
                    state VARCHAR(50) DEFAULT 'idle',
                    registered_at TIMESTAMPTZ DEFAULT NOW(),
                    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
                    current_task_id VARCHAR(255),
                    max_concurrent_tasks INT DEFAULT 1,
                    current_task_count INT DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Distributed tasks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_distributed_tasks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id VARCHAR(255) UNIQUE NOT NULL,
                    task_type VARCHAR(100) NOT NULL,
                    payload JSONB NOT NULL,
                    priority INT DEFAULT 2,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    assigned_at TIMESTAMPTZ,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    assigned_agent_id VARCHAR(255),
                    required_capabilities JSONB DEFAULT '[]'::jsonb,
                    dependencies JSONB DEFAULT '[]'::jsonb,
                    timeout_seconds INT DEFAULT 300,
                    retry_count INT DEFAULT 0,
                    max_retries INT DEFAULT 3,
                    result JSONB,
                    error TEXT,
                    correlation_id VARCHAR(255),
                    tenant_id VARCHAR(255),
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)

            # Task groups table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_task_groups (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    group_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    mode VARCHAR(50) NOT NULL,
                    tasks JSONB DEFAULT '[]'::jsonb,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    leader_agent_id VARCHAR(255),
                    participating_agents JSONB DEFAULT '[]'::jsonb,
                    results JSONB DEFAULT '{}'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)

            # Coordination messages table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_coordination_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    message_id VARCHAR(255) UNIQUE NOT NULL,
                    from_agent_id VARCHAR(255) NOT NULL,
                    to_agent_id VARCHAR(255),
                    message_type VARCHAR(100) NOT NULL,
                    payload JSONB NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    correlation_id VARCHAR(255),
                    requires_ack BOOLEAN DEFAULT false,
                    acked BOOLEAN DEFAULT false,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_distributed_tasks_status
                ON ai_distributed_tasks(status)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_distributed_tasks_priority
                ON ai_distributed_tasks(priority DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_registry_state
                ON ai_agent_registry(state)
            """)

            conn.commit()
            conn.close()
            logger.info("Coordination database tables initialized")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    async def _load_persisted_state(self):
        """Load persisted state from database"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Load agents
            cur.execute("SELECT * FROM ai_agent_registry WHERE state != 'offline'")
            for row in cur.fetchall():
                self._agents[row['agent_id']] = AgentRegistration(
                    agent_id=row['agent_id'],
                    agent_name=row['agent_name'],
                    agent_type=row['agent_type'],
                    capabilities=row['capabilities'] or [],
                    state=AgentState(row['state']),
                    registered_at=row['registered_at'],
                    last_heartbeat=row['last_heartbeat'],
                    current_task_id=row['current_task_id'],
                    max_concurrent_tasks=row['max_concurrent_tasks'],
                    current_task_count=row['current_task_count'],
                    metadata=row['metadata'] or {}
                )

            # Load pending tasks
            cur.execute("""
                SELECT * FROM ai_distributed_tasks
                WHERE status IN ('pending', 'queued', 'assigned', 'in_progress')
                ORDER BY priority DESC, created_at ASC
            """)
            for row in cur.fetchall():
                task = DistributedTask(
                    task_id=row['task_id'],
                    task_type=row['task_type'],
                    payload=row['payload'],
                    priority=TaskPriority(row['priority']),
                    status=TaskStatus(row['status']),
                    created_at=row['created_at'],
                    assigned_at=row['assigned_at'],
                    started_at=row['started_at'],
                    assigned_agent_id=row['assigned_agent_id'],
                    required_capabilities=row['required_capabilities'] or [],
                    dependencies=row['dependencies'] or [],
                    timeout_seconds=row['timeout_seconds'],
                    retry_count=row['retry_count'],
                    max_retries=row['max_retries'],
                    correlation_id=row['correlation_id'],
                    tenant_id=row['tenant_id'],
                    metadata=row['metadata'] or {}
                )
                self._tasks[task.task_id] = task
                if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                    self._task_queue.append(task.task_id)

            conn.close()
            logger.info(f"Loaded {len(self._agents)} agents and {len(self._tasks)} tasks")

        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")

    async def _heartbeat_loop(self):
        """Background heartbeat and cleanup loop"""
        while True:
            try:
                await asyncio.sleep(30)
                await self._check_agent_health()
                await self._check_task_timeouts()
                await self._process_task_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    # ========================================================================
    # AGENT MANAGEMENT
    # ========================================================================

    async def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        capabilities: list[str],
        max_concurrent_tasks: int = 1,
        metadata: Optional[dict[str, Any]] = None
    ) -> AgentRegistration:
        """Register an agent with the coordinator"""
        await self.initialize()

        registration = AgentRegistration(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            capabilities=capabilities,
            state=AgentState.IDLE,
            max_concurrent_tasks=max_concurrent_tasks,
            metadata=metadata or {}
        )

        self._agents[agent_id] = registration
        self._stats["agents_registered"] += 1

        await self._persist_agent(registration)
        await self._emit_event("agent_registered", {"agent_id": agent_id})

        logger.info(f"Agent registered: {agent_id} ({agent_type})")
        return registration

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.state = AgentState.OFFLINE

            # Reassign any tasks
            if agent.current_task_id:
                await self._reassign_task(agent.current_task_id)

            await self._persist_agent(agent)
            await self._emit_event("agent_unregistered", {"agent_id": agent_id})

            logger.info(f"Agent unregistered: {agent_id}")

    async def update_agent_heartbeat(self, agent_id: str, state: Optional[AgentState] = None):
        """Update agent heartbeat"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.last_heartbeat = datetime.now(timezone.utc)
            if state:
                agent.state = state
            await self._persist_agent(agent)

    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID"""
        return self._agents.get(agent_id)

    async def get_available_agents(
        self,
        capabilities: Optional[list[str]] = None
    ) -> list[AgentRegistration]:
        """Get available agents, optionally filtered by capabilities"""
        available = []
        for agent in self._agents.values():
            if agent.state != AgentState.IDLE:
                continue
            if agent.current_task_count >= agent.max_concurrent_tasks:
                continue

            if capabilities:
                if not all(cap in agent.capabilities for cap in capabilities):
                    continue

            available.append(agent)

        return available

    async def _check_agent_health(self):
        """Check agent health and mark offline if stale"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)

        for agent in list(self._agents.values()):
            if agent.state == AgentState.OFFLINE:
                continue

            if agent.last_heartbeat < cutoff:
                logger.warning(f"Agent {agent.agent_id} missed heartbeat, marking offline")
                agent.state = AgentState.OFFLINE

                if agent.current_task_id:
                    await self._reassign_task(agent.current_task_id)

                await self._persist_agent(agent)

    async def _persist_agent(self, agent: AgentRegistration):
        """Persist agent to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_agent_registry (
                    agent_id, agent_name, agent_type, capabilities, state,
                    registered_at, last_heartbeat, current_task_id,
                    max_concurrent_tasks, current_task_count, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    last_heartbeat = EXCLUDED.last_heartbeat,
                    current_task_id = EXCLUDED.current_task_id,
                    current_task_count = EXCLUDED.current_task_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (
                agent.agent_id,
                agent.agent_name,
                agent.agent_type,
                json.dumps(agent.capabilities),
                agent.state.value,
                agent.registered_at,
                agent.last_heartbeat,
                agent.current_task_id,
                agent.max_concurrent_tasks,
                agent.current_task_count,
                json.dumps(agent.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist agent: {e}")

    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: Optional[list[str]] = None,
        dependencies: Optional[list[str]] = None,
        timeout_seconds: int = 300,
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> DistributedTask:
        """Submit a task for distributed execution"""
        await self.initialize()

        task_id = f"task_{uuid.uuid4().hex[:12]}"

        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            status=TaskStatus.PENDING,
            required_capabilities=required_capabilities or [],
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
            correlation_id=correlation_id,
            tenant_id=tenant_id,
            metadata=metadata or {}
        )

        self._tasks[task_id] = task
        self._stats["tasks_created"] += 1

        # Check dependencies
        if dependencies:
            all_complete = all(
                self._tasks.get(dep_id, {}).status == TaskStatus.COMPLETED
                for dep_id in dependencies
                if dep_id in self._tasks
            )
            if not all_complete:
                task.status = TaskStatus.QUEUED
        else:
            # Add to queue
            self._task_queue.append(task_id)
            self._task_queue.sort(
                key=lambda tid: self._tasks[tid].priority.value,
                reverse=True
            )

        await self._persist_task(task)
        await self._emit_event("task_submitted", {"task_id": task_id})

        # Try immediate assignment
        await self._process_task_queue()

        logger.info(f"Task submitted: {task_id} ({task_type})")
        return task

    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent"""
        if task_id not in self._tasks:
            return False
        if agent_id not in self._agents:
            return False

        task = self._tasks[task_id]
        agent = self._agents[agent_id]

        if agent.current_task_count >= agent.max_concurrent_tasks:
            return False

        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent_id
        task.assigned_at = datetime.now(timezone.utc)

        agent.current_task_count += 1
        agent.current_task_id = task_id
        if agent.current_task_count >= agent.max_concurrent_tasks:
            agent.state = AgentState.BUSY

        # Remove from queue
        if task_id in self._task_queue:
            self._task_queue.remove(task_id)

        await self._persist_task(task)
        await self._persist_agent(agent)
        await self._emit_event("task_assigned", {"task_id": task_id, "agent_id": agent_id})

        logger.info(f"Task {task_id} assigned to agent {agent_id}")
        return True

    async def start_task(self, task_id: str) -> bool:
        """Mark a task as started"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)

        await self._persist_task(task)
        await self._emit_event("task_started", {"task_id": task_id})

        return True

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
        success: bool = True
    ) -> bool:
        """Complete a task"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.completed_at = datetime.now(timezone.utc)

        if success:
            task.status = TaskStatus.COMPLETED
            task.result = result
            self._stats["tasks_completed"] += 1
        else:
            task.status = TaskStatus.FAILED
            task.error = result.get("error", "Unknown error")
            self._stats["tasks_failed"] += 1

        # Release agent
        if task.assigned_agent_id and task.assigned_agent_id in self._agents:
            agent = self._agents[task.assigned_agent_id]
            agent.current_task_count = max(0, agent.current_task_count - 1)
            if agent.current_task_id == task_id:
                agent.current_task_id = None
            if agent.current_task_count < agent.max_concurrent_tasks:
                agent.state = AgentState.IDLE
            await self._persist_agent(agent)

        await self._persist_task(task)
        await self._emit_event("task_completed", {"task_id": task_id, "success": success})

        # Process dependent tasks
        await self._process_dependent_tasks(task_id)

        # Execute callback if registered
        if task_id in self._task_callbacks:
            try:
                callback = self._task_callbacks.pop(task_id)
                await callback(task)
            except Exception as e:
                logger.error(f"Task callback error: {e}")

        logger.info(f"Task {task_id} completed: {'success' if success else 'failed'}")
        return True

    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]

        if task.retry_count >= task.max_retries:
            logger.warning(f"Task {task_id} exceeded max retries")
            return False

        task.retry_count += 1
        task.status = TaskStatus.PENDING
        task.assigned_agent_id = None
        task.assigned_at = None
        task.started_at = None
        task.error = None

        self._task_queue.append(task_id)
        self._task_queue.sort(
            key=lambda tid: self._tasks[tid].priority.value,
            reverse=True
        )

        await self._persist_task(task)
        await self._emit_event("task_retried", {"task_id": task_id})

        logger.info(f"Task {task_id} queued for retry (attempt {task.retry_count})")
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now(timezone.utc)

        if task_id in self._task_queue:
            self._task_queue.remove(task_id)

        if task.assigned_agent_id and task.assigned_agent_id in self._agents:
            agent = self._agents[task.assigned_agent_id]
            agent.current_task_count = max(0, agent.current_task_count - 1)
            if agent.current_task_id == task_id:
                agent.current_task_id = None
            agent.state = AgentState.IDLE
            await self._persist_agent(agent)

        await self._persist_task(task)
        await self._emit_event("task_cancelled", {"task_id": task_id})

        logger.info(f"Task {task_id} cancelled")
        return True

    async def _reassign_task(self, task_id: str):
        """Reassign a task from a failed agent"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.assigned_agent_id = None
        task.status = TaskStatus.PENDING

        self._task_queue.append(task_id)
        self._task_queue.sort(
            key=lambda tid: self._tasks[tid].priority.value,
            reverse=True
        )

        await self._persist_task(task)
        logger.info(f"Task {task_id} queued for reassignment")

    async def _check_task_timeouts(self):
        """Check for timed out tasks"""
        now = datetime.now(timezone.utc)

        for task in list(self._tasks.values()):
            if task.status != TaskStatus.IN_PROGRESS:
                continue

            if task.started_at:
                elapsed = (now - task.started_at).total_seconds()
                if elapsed > task.timeout_seconds:
                    logger.warning(f"Task {task.task_id} timed out")
                    task.status = TaskStatus.TIMEOUT
                    task.error = "Task timed out"
                    await self._persist_task(task)
                    await self._emit_event("task_timeout", {"task_id": task.task_id})

                    # Try retry
                    if task.retry_count < task.max_retries:
                        await self.retry_task(task.task_id)

    async def _process_task_queue(self):
        """Process the task queue and assign tasks to available agents"""
        if not self._task_queue:
            return

        assigned = []

        for task_id in list(self._task_queue):
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                assigned.append(task_id)
                continue

            # Find available agent
            agents = await self.get_available_agents(task.required_capabilities)
            if not agents:
                continue

            # Assign to first available
            agent = agents[0]
            success = await self.assign_task(task_id, agent.agent_id)
            if success:
                assigned.append(task_id)

        # Remove assigned tasks from queue
        for task_id in assigned:
            if task_id in self._task_queue:
                self._task_queue.remove(task_id)

    async def _process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that depend on a completed task"""
        for task in self._tasks.values():
            if task.status != TaskStatus.QUEUED:
                continue
            if completed_task_id not in task.dependencies:
                continue

            # Check if all dependencies are complete
            all_complete = all(
                self._tasks.get(dep_id, {}).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self._tasks
            )

            if all_complete:
                task.status = TaskStatus.PENDING
                self._task_queue.append(task.task_id)

        # Sort queue
        self._task_queue.sort(
            key=lambda tid: self._tasks[tid].priority.value,
            reverse=True
        )

        await self._process_task_queue()

    async def _persist_task(self, task: DistributedTask):
        """Persist task to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_distributed_tasks (
                    task_id, task_type, payload, priority, status,
                    created_at, assigned_at, started_at, completed_at,
                    assigned_agent_id, required_capabilities, dependencies,
                    timeout_seconds, retry_count, max_retries, result,
                    error, correlation_id, tenant_id, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    assigned_at = EXCLUDED.assigned_at,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    assigned_agent_id = EXCLUDED.assigned_agent_id,
                    retry_count = EXCLUDED.retry_count,
                    result = EXCLUDED.result,
                    error = EXCLUDED.error,
                    metadata = EXCLUDED.metadata
            """, (
                task.task_id,
                task.task_type,
                json.dumps(task.payload),
                task.priority.value,
                task.status.value,
                task.created_at,
                task.assigned_at,
                task.started_at,
                task.completed_at,
                task.assigned_agent_id,
                json.dumps(task.required_capabilities),
                json.dumps(task.dependencies),
                task.timeout_seconds,
                task.retry_count,
                task.max_retries,
                json.dumps(task.result) if task.result else None,
                task.error,
                task.correlation_id,
                task.tenant_id,
                json.dumps(task.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist task: {e}")

    # ========================================================================
    # TASK GROUPS & COORDINATION
    # ========================================================================

    async def create_task_group(
        self,
        name: str,
        mode: CoordinationMode,
        tasks: list[DistributedTask],
        metadata: Optional[dict[str, Any]] = None
    ) -> TaskGroup:
        """Create a group of coordinated tasks"""
        group_id = f"group_{uuid.uuid4().hex[:12]}"

        # Submit all tasks
        task_ids = []
        for task in tasks:
            submitted = await self.submit_task(
                task_type=task.task_type,
                payload=task.payload,
                priority=task.priority,
                required_capabilities=task.required_capabilities,
                timeout_seconds=task.timeout_seconds,
                correlation_id=group_id,
                metadata={**task.metadata, "group_id": group_id}
            )
            task_ids.append(submitted.task_id)

        group = TaskGroup(
            group_id=group_id,
            name=name,
            mode=mode,
            tasks=task_ids,
            metadata=metadata or {}
        )

        self._task_groups[group_id] = group
        await self._persist_task_group(group)

        logger.info(f"Task group created: {group_id} with {len(task_ids)} tasks")
        return group

    async def elect_leader(self, group_id: str) -> Optional[str]:
        """Elect a leader for a task group"""
        if group_id not in self._task_groups:
            return None

        group = self._task_groups[group_id]

        # Get participating agents
        participating = set()
        for task_id in group.tasks:
            task = self._tasks.get(task_id)
            if task and task.assigned_agent_id:
                participating.add(task.assigned_agent_id)

        if not participating:
            return None

        # Simple election: choose agent with most capabilities
        best_agent = None
        best_score = 0

        for agent_id in participating:
            agent = self._agents.get(agent_id)
            if agent:
                score = len(agent.capabilities)
                if score > best_score:
                    best_score = score
                    best_agent = agent_id

        if best_agent:
            group.leader_agent_id = best_agent
            group.participating_agents = list(participating)
            self._leader_elections[group_id] = best_agent
            await self._persist_task_group(group)
            logger.info(f"Leader elected for group {group_id}: {best_agent}")

        return best_agent

    async def _persist_task_group(self, group: TaskGroup):
        """Persist task group to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_task_groups (
                    group_id, name, mode, tasks, status, created_at,
                    completed_at, leader_agent_id, participating_agents,
                    results, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (group_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    completed_at = EXCLUDED.completed_at,
                    leader_agent_id = EXCLUDED.leader_agent_id,
                    participating_agents = EXCLUDED.participating_agents,
                    results = EXCLUDED.results,
                    metadata = EXCLUDED.metadata
            """, (
                group.group_id,
                group.name,
                group.mode.value,
                json.dumps(group.tasks),
                group.status.value,
                group.created_at,
                group.completed_at,
                group.leader_agent_id,
                json.dumps(group.participating_agents),
                json.dumps(group.results),
                json.dumps(group.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist task group: {e}")

    # ========================================================================
    # MESSAGING
    # ========================================================================

    async def send_message(
        self,
        from_agent_id: str,
        to_agent_id: Optional[str],
        message_type: str,
        payload: dict[str, Any],
        requires_ack: bool = False,
        correlation_id: Optional[str] = None
    ) -> CoordinationMessage:
        """Send a message between agents"""
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        message = CoordinationMessage(
            message_id=message_id,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message_type=message_type,
            payload=payload,
            requires_ack=requires_ack,
            correlation_id=correlation_id
        )

        if to_agent_id:
            self._message_queue[to_agent_id].append(message)
        else:
            self._broadcast_messages.append(message)

        self._stats["messages_sent"] += 1

        await self._persist_message(message)
        logger.debug(f"Message sent: {message_id} from {from_agent_id} to {to_agent_id or 'broadcast'}")

        return message

    async def get_messages(self, agent_id: str, limit: int = 100) -> list[CoordinationMessage]:
        """Get messages for an agent"""
        messages = []

        # Get direct messages
        if agent_id in self._message_queue:
            messages.extend(self._message_queue[agent_id][:limit])
            self._message_queue[agent_id] = self._message_queue[agent_id][limit:]

        # Get broadcast messages
        for msg in self._broadcast_messages:
            if msg.from_agent_id != agent_id:
                messages.append(msg)

        return messages[:limit]

    async def acknowledge_message(self, message_id: str):
        """Acknowledge a message"""
        # Find and mark as acked
        for queue in self._message_queue.values():
            for msg in queue:
                if msg.message_id == message_id:
                    msg.acked = True
                    return

        for msg in self._broadcast_messages:
            if msg.message_id == message_id:
                msg.acked = True
                return

    async def _persist_message(self, message: CoordinationMessage):
        """Persist message to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_coordination_messages (
                    message_id, from_agent_id, to_agent_id, message_type,
                    payload, timestamp, correlation_id, requires_ack, acked
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                message.message_id,
                message.from_agent_id,
                message.to_agent_id,
                message.message_type,
                json.dumps(message.payload),
                message.timestamp,
                message.correlation_id,
                message.requires_ack,
                message.acked
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist message: {e}")

    # ========================================================================
    # EVENT HANDLING
    # ========================================================================

    def on_event(self, event_type: str, handler: Callable):
        """Register an event handler"""
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, data: dict[str, Any]):
        """Emit an event to handlers"""
        for handler in self._event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def register_task_callback(self, task_id: str, callback: Callable):
        """Register a callback for task completion"""
        self._task_callbacks[task_id] = callback

    # ========================================================================
    # MONITORING & STATS
    # ========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get coordination statistics"""
        return {
            **self._stats,
            "active_agents": len([a for a in self._agents.values() if a.state != AgentState.OFFLINE]),
            "total_agents": len(self._agents),
            "pending_tasks": len(self._task_queue),
            "in_progress_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
            "task_groups": len(self._task_groups),
            "pending_messages": sum(len(q) for q in self._message_queue.values())
        }

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status"""
        stats = await self.get_stats()

        return {
            "status": "healthy" if stats["active_agents"] > 0 else "degraded",
            "initialized": self._initialized,
            "stats": stats
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_coordinator_instance: Optional[DistributedAgentCoordinator] = None


def get_agent_coordinator() -> DistributedAgentCoordinator:
    """Get or create the agent coordinator instance"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = DistributedAgentCoordinator()
    return _coordinator_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def register_agent(
    agent_id: str,
    agent_name: str,
    agent_type: str,
    capabilities: list[str]
) -> AgentRegistration:
    """Register an agent"""
    coordinator = get_agent_coordinator()
    return await coordinator.register_agent(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        capabilities=capabilities
    )


async def submit_task(
    task_type: str,
    payload: dict[str, Any],
    priority: TaskPriority = TaskPriority.NORMAL
) -> DistributedTask:
    """Submit a task"""
    coordinator = get_agent_coordinator()
    return await coordinator.submit_task(
        task_type=task_type,
        payload=payload,
        priority=priority
    )


async def complete_task(task_id: str, result: dict[str, Any], success: bool = True) -> bool:
    """Complete a task"""
    coordinator = get_agent_coordinator()
    return await coordinator.complete_task(task_id, result, success)


async def get_coordinator_stats() -> dict[str, Any]:
    """Get coordinator statistics"""
    coordinator = get_agent_coordinator()
    return await coordinator.get_stats()


if __name__ == "__main__":
    async def test():
        coordinator = get_agent_coordinator()
        await coordinator.initialize()

        # Register agents
        agent1 = await coordinator.register_agent(
            agent_id="agent_1",
            agent_name="Test Agent 1",
            agent_type="worker",
            capabilities=["processing", "analysis"]
        )
        print(f"Registered: {agent1.agent_id}")

        agent2 = await coordinator.register_agent(
            agent_id="agent_2",
            agent_name="Test Agent 2",
            agent_type="worker",
            capabilities=["processing"]
        )
        print(f"Registered: {agent2.agent_id}")

        # Submit task
        task = await coordinator.submit_task(
            task_type="test_task",
            payload={"data": "test"},
            priority=TaskPriority.HIGH,
            required_capabilities=["processing"]
        )
        print(f"Task submitted: {task.task_id}")

        # Wait for assignment
        await asyncio.sleep(1)

        # Complete task
        await coordinator.complete_task(task.task_id, {"result": "success"}, True)

        # Get stats
        stats = await coordinator.get_stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")

    asyncio.run(test())
